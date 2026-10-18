# Perf log: Qwen3 throughput on torchtitan

Running record of every run, its result, and what it taught us.
Target: maximize tokens/sec per GPU (TPS); also report MFU and memory %.

## Phase 0: what "good" looks like (Qwen3-1.7B, 1x H100, seq 4096)

### Model size (from `torchtitan/models/qwen3/flavors.py`, flavor `1.7B`)
dim 2048, 28 layers, 16 query heads / 8 KV heads (head_dim 128), MLP hidden 6144,
vocab 151,936.

| Part | Params |
|---|---|
| per layer: attention q/k/v/o | 12.6M |
| per layer: MLP (gate, up, down) | 37.7M |
| 28 layers | 1.409B |
| embedding table | 0.311B |
| LM head (shares its matrix with the embedding) | (same 0.311B) |
| **total** | **1.72B** |

Correction after the first real run: torchtitan reported 1,720,574,976 params, so
the embedding and LM head are one shared ("tied") matrix. I first assumed 2.03B.
FLOPs are unchanged, because the LM head matmul still runs once per token.

### FLOPs per token
torchtitan's formula (`torchtitan/models/qwen3/model.py:115`):
`6 * active_params + attention_flops`.

- **6 x params:** each weight does a multiply and an add (2 FLOPs) in the forward,
  and twice that in the backward (gradient w.r.t. input and w.r.t. weight): 2 + 4 = 6.
- **Active params** exclude the embedding table: a lookup, not a matmul.
  1.409B + 0.311B = 1.72B -> **10.32 GFLOPs/token**.
- **Attention** (the q@K^T and attn@V matmuls, which have no weights):
  `6 * heads * (2 * head_dim) * seq_len` per layer = **2.82 GFLOPs/token**.
  This counts the full 4096x4096 score matrix, even though causal masking skips
  half of it. That's the standard convention.
- **Total: 13.14 GFLOPs/token.** Layers 64%, LM head 14%, attention 21%.

### Ceiling on an H100
Peak dense BF16 = 989 TFLOPS (the value torchtitan uses for H100 SXM).

| MFU | tokens/sec |
|---|---|
| 100% (impossible) | 75,259 |
| 50% (excellent) | 37,630 |
| 40% (good) | 30,104 |

Ideal step time = tokens per step x 13.14 GFLOPs / 989 TFLOPS:

| LBS | tokens/step | 100% MFU | 50% MFU |
|---|---|---|---|
| 1 | 4,096 | 54 ms | 109 ms |
| 4 | 16,384 | 218 ms | 435 ms |
| 8 | 32,768 | 435 ms | 871 ms |
| 16 | 65,536 | 871 ms | 1,742 ms |

**Expectation:** 40-50% MFU is realistic for a model this size. The matrices
are small (2048 wide), so a larger share of time goes to non-matmul work
(norms, activations, optimizer, kernel launches) than in a big model.

### Memory budget (80 GB H100)
- **Fixed cost, regardless of batch size:** fp32 weights (4 B) + fp32 grads (4 B)
  + Adam's two moment estimates (8 B) = 16 B/param x 1.72B = **25.6 GiB**.
- **Left for activations:** roughly 45 GiB, after the CUDA context, bf16
  compute copies and allocator slack.
- **Activations grow linearly with LBS.** That's why AC exists: storing less per
  token lets the batch grow.

### Predictions to check in Phase 1
1. no-AC runs out of memory at a small LBS (low single digits).
2. full-AC reaches the largest LBS but pays roughly +33% compute: it reruns the
   whole forward pass during backward.
3. selective-AC reaches the best TPS: it recomputes only cheap ops.
4. TPS rises with LBS (larger, more efficient matmuls) and flattens once the
   GPU is saturated.

## Phase 1: AC x LBS sweep

### Test run (2026-10-17): `aa_1p7b_selective_lbs4_fsdp1_dp1`
Pod `0g2crqztbhal4p` (1x H100 SXM, US-MO-1), torch 2.16.0.dev20261017+cu130, FA3.

| step | tps | MFU | mem |
|---|---|---|---|
| 1 | 1,367 | 1.8% | 38.5 GiB (48.7%) |
| 2 | 23,402 | 31.1% | 51.7 GiB (65.3%) |
| 10 | 30,231 | 40.2% | 51.7 GiB |
| 40 | 28,688 | 38.1% | 51.7 GiB |
| **avg steps 11-40** | **29,319** | **39.0%** | **65.3% peak** |

What it shows:
- **Phase 0 checks out.** It predicted 30.1k tok/s at 40% MFU; measured 29.3k at
  39.0%. The log's own tflops (397 at step 10) matches 30,231 x 13.14 GFLOPs.
- **Correction (found while setting up Phase 2):** these runs were NOT
  compile-free. Qwen3's config compiles four small regions by default (loss,
  SwiGLU activation, RoPE, fp32->bf16 split). Phase 2's compile ablation turns
  them off (`_eager`) instead of on.
- **Why discard warmup:** step 1 is 20x slower (CUDA/cuDNN autotuning, allocator
  warming up, first-time kernel loads). Including it would drag the average down.
- **Memory jumps between step 1 and 2 (38.5 -> 51.7 GiB):** Adam's two moment
  buffers (2 x 4 B x 1.72B = 12.8 GiB) are created lazily at the first optimizer
  step. 51.7 - 38.5 = 13.2 GiB.
- **Each run takes ~1.5 min** (startup + 40 steps), so the full sweep is ~45 min.

### Full sweep (2026-10-17, pod `0g2crqztbhal4p`, ~64 min, ~$4.25)
Data: `sweep_logs/phase1_1p7b/results.csv`, plot `tps_vs_mem.png`. TPS/MFU = mean
of steps 11-40; mem = peak reserved.

| AC | max LBS (OOM at) | best TPS | at LBS | MFU | mem |
|---|---|---|---|---|---|
| none | 4 (6) | 28,934 | 4 | 38.4% | 80% |
| selective | 8 (10) | **31,151** | 8 | **41.4%** | **91%** |
| full | 48 (64) | 30,569 | 24 | 40.6% | 64% |

#### Where memory goes (linear fit of mem vs LBS)
| AC | memory per extra sequence (4096 tok) | per token |
|---|---|---|
| none | 8.3 GiB | 2.0 MiB |
| selective | 5.0 GiB | 1.2 MiB |
| full | 0.85 GiB | 0.2 MiB |

Every mode has the same ~31 GiB fixed cost at LBS -> 0: 25.6 GiB of fp32 weights,
grads and Adam state, plus ~5 GiB of CUDA context, cuBLAS/FA workspaces and bf16
working copies. So on 79 GiB, ~48 GiB is the activation budget, and
**max LBS ~= 48 GiB / (memory per sequence)**: none -> 5.8 (measured 4-5),
selective -> 9.6 (measured 8-9), full -> 56 (measured 48-63).

#### Cost of recomputation (same LBS)
full vs none: -8% (LBS 1), -9% (2), -9% (3), -14% (4). full vs selective at LBS 8: -10%.
Full AC reruns each block's forward: +3.76 GFLOPs/token (+29%), yet it loses only
~10%. **Open question for Phase 2:** why is the recompute cheaper than its FLOPs?
Hypothesis: a large share of the step isn't block matmuls (LM head + loss on a
152k vocab, optimizer, elementwise ops), and none of that is recomputed.
The profile of full vs selective at LBS 8 should show it.

#### Predictions vs results
1. **no-AC OOMs at a small LBS:** yes, max LBS 4.
2. **full-AC reaches the largest LBS but pays for recompute:** yes (LBS 48), but
   the cost is ~10%, not the ~25% I expected (see the open question).
3. **selective wins:** nominally (31.2k vs 30.6k), but the gap is 2% and identical
   runs differed by 7% (selective LBS 4: 29.3k test vs 27.4k sweep).
   **Not distinguishable from full-AC until noise is measured.**
4. **TPS rises with LBS, then flattens:** yes. 18-19k at LBS 1 -> 29-31k. Full-AC
   flattens past LBS ~16 and dips slightly at 32+.

#### The blog's claims
- **Selective wins:** consistent, but within noise here (see 3).
- **Peak near ~92% memory:** selective peaked at 91%, but only because LBS 8 is
  the largest batch that fits. Full-AC peaked at 64% (LBS 24); filling memory
  further (LBS 32-48, up to 91%) did not help. So "peak near full memory" holds
  for the mode whose curve is still rising when memory runs out. Bigger batches
  only help until the GPU is saturated.
- **Degrades with allocation retries:** not observed. No completed run logged
  retries; they appeared only in the three OOM runs. The LBS grid jumped from
  fitting (selective 8 = 91%) straight to OOM (10). Selective LBS 9 is the run
  that could show the retry regime.

## Phase 2: diagnose beyond the knobs (1x H100)

Run 2026-10-18 on pod `wu6vmrnauk85ed` (~15 min). Logs: `sweep_logs/phase2_1p7b/`;
TensorBoard, traces and memory snapshots: `outputs/phase2/<run>/`.
Trace numbers come from `trace_summary.py`.

### Ablations (one change each from selective LBS 8)
| run | TPS | MFU | mem | data-wait % | note |
|---|---|---|---|---|---|
| baseline rep1 / rep2 / rep3 | 33,827 / 32,282 / 31,888 | 42-45% | 91.0% | 3.3 / 1.1 / 2.2 | mean 32.7k, spread +-3% |
| baseline (profiled) | 32,066 | 42.6% | 91.0% | 1.3 | |
| eager (compile off) rep1 / rep2 | 24,588 / 25,026 | 33% | 98.1% | 2.4 / 1.5 | **-24% TPS, +7 pts memory** |
| LBS 9 | 31,050 | 41.3% | 97.2% | 2.6 | -5% vs LBS 8, no retries logged |
| full-AC LBS 8 (profiled) | 28,727 | 38.2% | 46.4% | 4.1 | -12% |
| selective LBS 1 (profiled, "worst") | 19,491 | 25.9% | 45.9% | 1.0 | |

- **Noise:** identical runs spread +-3% on one machine. Yesterday's Phase 1 machine
  ran the same config at 31.2k, 5% below today's mean. Compare only within a
  session, and treat <5% differences as noise.
- **Compile is worth +31%** (25k -> 32.7k), and saves 7 points of memory. The
  compiled regions fuse chains of elementwise ops (loss, SwiGLU, RoPE) into
  single kernels that don't write intermediates to HBM. Eager does more memory
  round-trips (slower) and keeps more intermediate tensors (more memory).
- **Data:** the dataloader wait is 1-4% of the step. Not the bottleneck by
  that metric, but see the trace: CPU-side batch handling is.
- **LBS 9 (97% memory) is slower than LBS 8 (91%)** even though it moves 12.5%
  more tokens per step. That reproduces the blog's "degrades near full memory",
  but without logged allocation retries. Per-step TPS dropped from ~31k to
  ~25k over steps 31-40. Mechanism not yet confirmed (no profile of LBS 9).

### Trace: where the step goes (selective LBS 8, step 19)
GPU step 902 ms, GPU busy 80.8% (step 18: 70.7%). Kernel time 730 ms:
matmul 428 (59%), RMSNorm 76 (10.5%), compiled Triton 68 (9.3%), eager
elementwise 53 (7.3%), attention 49 (6.8%), other 31, optimizer 24.

**The idle ~20-30% is almost all ONE gap of ~151 ms at the start of each
step.** What happens:
1. End of step 18: the CPU calls `aten::item` -> `cudaStreamSynchronize` and
   blocks for 684 ms until the GPU finishes all queued work. This is the
   metrics logging (`loss.item()`, `grad_norm.item()`, trainer.py ~450-462),
   which runs only on logging steps. I set `log_freq=1`, so **every step syncs.**
2. Step 19 starts with the GPU drained. The CPU then spends ~150 ms on batch
   handling before the first kernel launches: `aten::view` 56 ms, `aten::to`
   (host->device copy) 43 ms, `aten::reshape` 42 ms. A view should take
   microseconds, so the main thread is likely blocked by the GIL (grain
   dataloader threads). Not confirmed; the trace only records the main thread.
3. The GPU idles for that whole time.

**Classification: host/CPU-bound at step boundaries, caused by a per-step sync
in the measurement harness.** Without the sync, the CPU would prepare batch N+1
while the GPU still runs step N.

Upper bound if the gap vanished: ~730 ms of kernels for 32,768 tokens = ~45k
tok/s (~59% MFU). The rest of the headroom is non-matmul kernels (41% of GPU
time).

**The open question from Phase 1 is answered.** full-AC LBS 8 matmul time is
553 ms vs 428 ms for selective: +29%, exactly the predicted extra FLOPs. Total
kernel time is +19%. But end-to-end TPS was only -10 to -12%, because each step
also carries the ~150 ms fixed idle gap, which dilutes any GPU-side
difference. **The idle gap made the AC modes look closer than they are.**

**Worst config (selective LBS 1):** GPU step 165-219 ms, kernel time 154 ms.
The optimizer (23.5 ms, a fixed cost per step) is 15% of kernel time, vs 3% at
LBS 8. Matmuls are only 38%. With few tokens per step, fixed per-step costs
(optimizer, launches, the sync) dominate. That's why small batches have low MFU.

### Fixes to try
1. `log_freq=10`: one sync per 10 steps, so the CPU can run ahead.
   Prediction: GPU busy -> 95%+, TPS +15-25% at LBS 8.
2. Re-run selective vs full-AC with fix 1: the full-AC penalty should grow
   toward the +19% kernel-time gap, since the idle time no longer hides it.

### Phase 2b: fix 1, log every 10 steps (2026-10-18, pod `mf35in367j6ugy`, ~31 min, ~$2)
All on one machine; baseline = the same config with per-step logging.

| run | TPS | MFU | step time | mem |
|---|---|---|---|---|
| selective LBS 8, log every step (baseline) | 30,577 | 40.6% | 1.073 s | 91% |
| **selective LBS 8, log every 10** rep1 / rep2 | **42,268 / 42,346** | **56.3%** | **0.775 s** | 91% |
| selective LBS 8, log every 10, profiled | 41,661 | 55.4% | 0.787 s | 91% |
| full-AC LBS 8, log every 10 | 35,858 | 47.6% | 0.914 s | 46% |
| full-AC LBS 24, log every 10 | 37,036 | 49.2% | 2.654 s | 64% |

(With log_freq 10, TPS is the mean of the step 20/30/40 log windows. Each is a
10-step average.)

- **+38% TPS** from removing the per-step sync (predicted +15-25%).
- **Trace confirms it** (`outputs/phase2b/..._log10_prof__fixed`): GPU busy 98.8%
  (was 71-81%); idle gaps 6 ms per step (was 170-300 ms). The GPU step is now
  730-750 ms, the same as the kernel time. The run is now **GPU-bound**.
- **Selective vs full-AC, now resolved:** full is -15% at LBS 8 and -12% at
  LBS 24, outside the +-3% noise. With the idle gap gone, the recompute cost shows
  (the kernel-time gap was +19%). **Selective wins, confirmed.** Full-AC's best
  batch (24) still can't make up the recompute.
- **Phase 1's absolute numbers were all depressed** by the sync. The gap was a
  roughly fixed time per step, so it hurt short steps (small LBS) most. Rankings
  and the "peak near full memory" shape should be re-checked with log_freq 10.

**Classification after fix 1:** GPU compute-bound. Of 735 ms of kernels per step:
matmul 59% (running at ~776 TFLOPS = **78% of H100 peak** while they run), RMSNorm
10.5% (an unfused eager layer_norm kernel, fwd + bwd), compiled Triton 9.4%,
eager elementwise 7.3%, attention 6.5%, other 4%, optimizer 1-3%.
Remaining headroom is in the ~40% of GPU time that isn't matmul. Next targets:
RMSNorm (77 ms) and eager elementwise ops (53 ms), by fusing or compiling them.

**Interview takeaway:** before tuning anything, check the GPU is actually busy.
A per-step `.item()` (logging, NaN checks, grad-norm prints) is a classic hidden
sync. The fix is fewer syncs or async metrics, not a different parallelism.

## Phase 3 plan (8 GPUs, one node), not run yet
Script: `phase3.sh dense` then `phase3.sh moe`. Every run uses `_log10`.
Hardware: **8x H200** (141 GB) recommended over 8x H100 (80 GB). For the 30B,
FSDP=8 leaves ~10 GiB/GPU for activations on H100 (full-AC only, LBS 1-2) vs
~71 GiB on H200 (a real AC x LBS sweep, closer to the trial's 288 GB B300s).
H200 compute is the same as H100 (989 TFLOPS BF16), but HBM is faster.

### Part A: Qwen3-1.7B dense (DDP vs FSDP vs HSDP, scaling 1 -> 8)
| run | what it tests |
|---|---|
| 1 GPU, selective LBS 8 | same-node baseline for scaling efficiency |
| DDP=8 (`fsdp1_dp8`) | full model copy per GPU; one gradient all-reduce per step |
| FSDP=8 (`fsdp8_dp1`) | each GPU holds 1/8 of weights/grads/optimizer; all-gather weights per layer, reduce-scatter grads |
| HSDP 4x2 (`fsdp4_dp2`) | shard within groups of 4, replicate across 2 groups (normally used for shard-in-node, replicate-across-nodes) |
| FSDP=8, selective LBS 12 / 16, none LBS 8 | spend the memory FSDP frees |
| profiles: DDP=8 and FSDP=8 at LBS 8 | see communication kernels overlap compute |

Predictions:
- **Scaling efficiency (TPS/GPU at 8 / at 1) ~97-99% for both DDP and FSDP.**
  FSDP moves ~27 ms of data per step over NVLink vs ~740 ms of compute, and it
  runs on a separate stream overlapped with compute.
- **FSDP=8 frees ~22 GiB/GPU** (static 25.6 -> 3.2 GiB), so selective LBS 16 or
  no-AC LBS 8 should fit. They may beat the 1-GPU TPS/GPU slightly.
- HSDP ~= FSDP here. It only matters when part of the network is slow (across nodes).

### Part B: Qwen3-30B-A3B MoE (30.5B total, 3.0B active per token)
FLOPs/token = 27.9 G (6 x 3.04B active + attention) -> 40% MFU = 14.2k tok/s/GPU.

| run | what it tests |
|---|---|
| full-AC LBS 1 | does it fit; first number |
| full LBS 4/8, default LBS 2/4, selective LBS 2/4 | AC x LBS for an MoE |
| default LBS 4 + EP=8 | expert parallelism: each GPU keeps 16 experts; tokens move via all-to-all |
| profiles: default LBS 4 with and without EP | where MoE time goes |

**The key prediction: at small batch the MoE is communication-bound under FSDP.**
FSDP must gather ALL 30.5B params each step (0.24 s all-gather + 0.24 s
reduce-scatter at ~450 GB/s), but each token only computes with 3B of them:

| LBS | compute (40% MFU) | FSDP comm |
|---|---|---|
| 1 | 0.29 s | 0.47 s  <- comm-bound |
| 2 | 0.58 s | 0.47 s |
| 4 | 1.16 s | 0.47 s  <- hidden by overlap |
| 8 | 2.31 s | 0.47 s |

So batch size matters more for MoE than for dense. EP changes what moves: expert
weights stay put (no all-gather for 96% of the params), and token activations
move instead (all-to-all). That's the standard Megatron MoE trade-off.
Also expect lower MFU than dense: per-expert GEMMs are small (each expert sees
~256 x LBS tokens), and routing/permutation adds non-matmul work.

Cost: ~55-60 min total (setup ~10, dense ~15, MoE ~30) x $42.32/hr = ~$40-45.
