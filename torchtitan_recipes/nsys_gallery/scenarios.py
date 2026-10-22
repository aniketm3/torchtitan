"""Small PyTorch training loops, one per Nsight Systems gallery scenario.

Each run trains for NUM_STEPS steps and calls torch.cuda.profiler.start()/stop()
around steps CAPTURE_FIRST..CAPTURE_LAST, so `nsys profile
--capture-range=cudaProfilerApi` records only steady-state steps (no warmup,
no torch.compile time). Every step is wrapped in NVTX ranges:

    step N
      data       get the batch (host prep, DataLoader, or nothing)
      forward
      backward
      optimizer

Usage (normally via run_gallery.sh):
    nsys profile ... python scenarios.py --scenario healthy
    nsys profile ... torchrun --standalone --nproc_per_node 2 scenarios.py --scenario fsdp_2gpu
"""

import argparse
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import nvtx
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

NUM_STEPS = 20
CAPTURE_FIRST, CAPTURE_LAST = 12, 16  # inclusive: 5 captured steps


@dataclass
class Config:
    dim: int
    num_layers: int
    vocab: int
    batch: int
    seq: int
    # Simulated host-side batch prep per step (tokenizing, packing, masking).
    # It overlaps with GPU work unless something forces the CPU to wait.
    host_prep_ms: float = 0.0


# ~50 ms of GPU work per step on an H100: mostly big matmuls.
BIG = Config(dim=2048, num_layers=16, vocab=32768, batch=8, seq=1024, host_prep_ms=10.0)
# Each matmul is a few microseconds: launch cost dominates.
TINY = Config(dim=256, num_layers=16, vocab=1024, batch=4, seq=64)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class MLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, 4 * dim, bias=False)
        self.down = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        return x + self.down(F.gelu(self.up(self.norm(x))))


class TinyLM(nn.Module):
    """Embedding -> MLP blocks -> LM head. Attention is left out on purpose:
    the scenarios are about the step's structure, not the layer's."""

    def __init__(self, cfg: Config, recompute: bool = False):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab, cfg.dim)
        self.layers = nn.ModuleList(MLPBlock(cfg.dim) for _ in range(cfg.num_layers))
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab, bias=False)
        self.recompute = recompute

    def forward(self, tokens):
        x = self.embed(tokens)
        for layer in self.layers:
            if self.recompute:
                # Full activation checkpointing: keep only the layer input,
                # rerun the layer's forward during backward.
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        logits = self.head(self.norm(x))
        return F.cross_entropy(logits.flatten(0, 1).float(), tokens.flatten())


class LoopedExperts(nn.Module):
    """MoE expert layer the naive way: one small matmul pair per expert."""

    def __init__(self, num_experts, dim, ffn):
        super().__init__()
        self.up = nn.ModuleList(nn.Linear(dim, ffn, bias=False) for _ in range(num_experts))
        self.down = nn.ModuleList(nn.Linear(ffn, dim, bias=False) for _ in range(num_experts))

    def forward(self, x):  # x: (experts, tokens_per_expert, dim), already routed
        xs = x.unbind(0)
        return torch.stack([self.down[e](F.gelu(self.up[e](xs[e]))) for e in range(len(xs))])


class GroupedExperts(nn.Module):
    """Same math as LoopedExperts, as one batched matmul over all experts."""

    def __init__(self, num_experts, dim, ffn):
        super().__init__()
        self.up = nn.Parameter(torch.randn(num_experts, dim, ffn) * dim**-0.5)
        self.down = nn.Parameter(torch.randn(num_experts, ffn, dim) * ffn**-0.5)

    def forward(self, x):
        return torch.bmm(F.gelu(torch.bmm(x, self.up)), self.down)


class MoEStack(nn.Module):
    def __init__(self, experts_cls, num_layers=4, num_experts=64, dim=1024, ffn=512):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(num_layers))
        self.experts = nn.ModuleList(experts_cls(num_experts, dim, ffn) for _ in range(num_layers))

    def forward(self, x):
        for norm, experts in zip(self.norms, self.experts):
            x = x + experts(norm(x))
        return x.float().pow(2).mean()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class HostPrepBatches:
    """Random token batches made on the CPU, copied with non_blocking=True from
    pinned memory. host_prep_ms stands in for real CPU work per batch."""

    def __init__(self, cfg: Config, device):
        self.cfg, self.device = cfg, device
        self.pinned = [torch.empty(cfg.batch, cfg.seq, dtype=torch.long).pin_memory() for _ in range(2)]
        self.i = 0

    def next(self):
        if self.cfg.host_prep_ms:
            time.sleep(self.cfg.host_prep_ms / 1000)
        buf = self.pinned[self.i % 2]
        self.i += 1
        buf.random_(0, self.cfg.vocab)
        return buf.to(self.device, non_blocking=True)


class SlowTokenDataset(Dataset):
    """Each sample takes delay_ms of CPU time (think: decode + tokenize)."""

    def __init__(self, cfg: Config, delay_ms: float):
        self.cfg, self.delay = cfg, delay_ms / 1000

    def __len__(self):
        return 1_000_000

    def __getitem__(self, idx):
        time.sleep(self.delay)
        g = torch.Generator().manual_seed(idx)
        return torch.randint(0, self.cfg.vocab, (self.cfg.seq,), generator=g)


class LoaderBatches:
    def __init__(self, loader, device, non_blocking):
        self.it, self.device, self.non_blocking = iter(loader), device, non_blocking

    def next(self):
        return next(self.it).to(self.device, non_blocking=self.non_blocking)


class FixedActivations:
    """MoE scenarios: activations already on the GPU, routed evenly."""

    def __init__(self, x):
        self.x = x

    def next(self):
        return self.x


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

DESCRIPTIONS = {
    "healthy": "BIG model, host prep overlapped with GPU work, no syncs",
    "dataloader_bound": "DataLoader num_workers=0, 10 ms per sample: the GPU waits for data",
    "dataloader_fixed": "same dataset, 8 workers + pin_memory + prefetch_factor 4",
    "launch_overhead": "TINY model: thousands of microsecond kernels, CPU launch-bound",
    "launch_overhead_compiled": "TINY model with torch.compile(mode='reduce-overhead'): fused kernels + CUDA graphs",
    "hidden_sync": "healthy + logging loss.item() at the end of every step",
    "no_recompute": "BIG model, no activation checkpointing (same as healthy)",
    "full_recompute": "BIG model, full activation checkpointing on every block",
    "looped_matmul": "4 MoE layers, 64 experts as a Python loop of small matmuls",
    "grouped_matmul": "4 MoE layers, 64 experts as one batched matmul (torch.bmm)",
    "fsdp_2gpu": "BIG model, FSDP2 across 2 GPUs: NCCL all-gather/reduce-scatter",
}


def build(scenario, device):
    """Returns (model_or_fn, params, batches, extra_flags)."""
    sync_every_step = scenario == "hidden_sync"

    if scenario in ("looped_matmul", "grouped_matmul"):
        cls = LoopedExperts if scenario == "looped_matmul" else GroupedExperts
        model = MoEStack(cls).to(device, torch.bfloat16)
        # 64 experts x 256 tokens each = 16,384 tokens, routed evenly. Real
        # routers are uneven; grouped GEMM kernels handle that with offsets.
        x = torch.randn(64, 256, 1024, device=device, dtype=torch.bfloat16)
        return model, list(model.parameters()), FixedActivations(x), sync_every_step

    cfg = TINY if scenario.startswith("launch_overhead") else BIG
    model = TinyLM(cfg, recompute=scenario == "full_recompute").to(device, torch.bfloat16)
    params = list(model.parameters())

    if scenario == "fsdp_2gpu":
        from torch.distributed.fsdp import fully_shard

        for layer in model.layers:
            fully_shard(layer)
        fully_shard(model)
        params = list(model.parameters())

    if scenario == "launch_overhead_compiled":
        model = torch.compile(model, mode="reduce-overhead")

    if scenario == "dataloader_bound":
        loader = DataLoader(SlowTokenDataset(cfg, delay_ms=10), batch_size=cfg.batch, num_workers=0)
        batches = LoaderBatches(loader, device, non_blocking=False)
    elif scenario == "dataloader_fixed":
        loader = DataLoader(
            SlowTokenDataset(cfg, delay_ms=10),
            batch_size=cfg.batch,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )
        batches = LoaderBatches(loader, device, non_blocking=True)
    else:
        batches = HostPrepBatches(cfg, device)
    return model, params, batches, sync_every_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=sorted(DESCRIPTIONS))
    args = parser.parse_args()

    distributed = "LOCAL_RANK" in os.environ
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if distributed:
        torch.distributed.init_process_group("nccl", device_id=torch.device("cuda", rank))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.manual_seed(0)

    model, params, batches, sync_every_step = build(args.scenario, device)
    opt = torch.optim.AdamW(params, lr=1e-4, fused=True)
    compiled = args.scenario == "launch_overhead_compiled"

    if rank == 0:
        n = sum(p.numel() for p in params) * (2 if args.scenario == "fsdp_2gpu" else 1)
        print(f"[{args.scenario}] {DESCRIPTIONS[args.scenario]} | params {n / 1e6:.1f}M", flush=True)

    for step in range(NUM_STEPS):
        if step == CAPTURE_FIRST:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
        nvtx.range_push(f"step {step}")

        nvtx.range_push("data")
        batch = batches.next()
        nvtx.range_pop()

        if compiled:
            torch.compiler.cudagraph_mark_step_begin()
        nvtx.range_push("forward")
        loss = model(batch)
        nvtx.range_pop()

        nvtx.range_push("backward")
        loss.backward()
        nvtx.range_pop()

        nvtx.range_push("optimizer")
        opt.step()
        opt.zero_grad(set_to_none=True)
        nvtx.range_pop()

        if sync_every_step:
            # Logging the loss: .item() makes the CPU wait until the GPU has
            # finished all queued work, so next step's host prep can't overlap.
            nvtx.range_push("log loss.item()")
            print(f"step {step} loss {loss.item():.4f}", flush=True)
            nvtx.range_pop()

        nvtx.range_pop()  # step
        if step == CAPTURE_LAST:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            if rank == 0:
                print(f"[{args.scenario}] captured steps {CAPTURE_FIRST}-{CAPTURE_LAST}", flush=True)

    torch.cuda.synchronize()
    if rank == 0:
        print(f"[{args.scenario}] done, peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
