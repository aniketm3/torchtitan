# Nsight Systems gallery: labeled example profiles

> The reports themselves (`.nsys-rep`, about 3 MB) are not committed. Regenerate them
> with `bash run_gallery.sh` on a GPU pod (after `bash install_nsys.sh`). The paths
> `reports/` and `stats/` below are relative to its output folder.

Ten small PyTorch training loops on 1x H100 SXM (torch 2.8, CUDA 12.8,
Nsight Systems CLI 2026.5.1), each built to show one pattern. Nothing is
hidden: each scenario changes one thing, named in its title, and every number
below comes from that scenario's report.

Code: `torchtitan_recipes/nsys_gallery/` (`scenarios.py` holds the loops,
`run_gallery.sh` profiles them all, `analyze.py` computes the numbers).

## How the profiles were taken

```
nsys profile -t cuda,nvtx,osrt,cublas,cudnn \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  -o reports/<scenario> python scenarios.py --scenario <scenario>
```

- Each run trains 20 steps. The script calls `torch.cuda.profiler.start()`
  before step 12 and `stop()` after step 16, so every report holds **5
  steady-state steps**: no warmup, no torch.compile time.
- The script brackets the capture with `torch.cuda.synchronize()`, so the
  first and last ~100 ms of a report can show a long `cudaDeviceSynchronize`.
  That's setup, outside the steps; ignore it.
- NVTX ranges label every step: `step N` > `data`, `forward`, `backward`,
  `optimizer` (and `log loss.item()` in hidden_sync).

## Opening a report

1. Install **Nsight Systems for macOS** (free, from developer.nvidia.com),
   version **2026.5 or newer**. An older GUI may refuse reports from a newer CLI.
2. File > Open > `reports/<scenario>.nsys-rep`.
3. The rows you'll use:
   - **CUDA HW (... H100 80GB HBM3)**: what the GPU actually ran. Expand it to
     see per-stream rows. Gaps here are GPU idle time.
   - Under CUDA HW, **NVTX** (the "GPU projection"): your NVTX ranges shifted
     to when the GPU ran their work. Use it to see which phase the GPU is on.
   - **Processes > python > [main thread]**: the CPU side.
     - **NVTX**: when the CPU executed each phase.
     - **CUDA API**: every `cudaLaunchKernel`, `cudaMemcpyAsync`,
       `cudaStreamSynchronize` call. A long sync here means the CPU was waiting.
     - **OS runtime libraries**: sleeps, locks and waits on the CPU.
4. Zoom: drag across the time ruler at the top to select a range, then
   right-click > Zoom into selection. Click any block to see its name and duration.
5. Numbers: the bottom pane's dropdown has **Stats System View** > *CUDA GPU
   Kernel Summary* and *NVTX Range Summary*. These match `stats/<scenario>.stats.txt`.

## Summary

| scenario | GPU step | GPU idle | kernels/step | median kernel | CPU blocked in sync/step |
|---|---|---|---|---|---|
| **healthy** | **56.1 ms** | **1.3%** | 278 | 140 us | 0 |
| dataloader_bound | 97.6 ms | 45.4% | 278 | 135 us | 0 * |
| dataloader_fixed | 55.8 ms | 1.6% | 278 | 140 us | 0 |
| launch_overhead | 11.2 ms | 93.6% | 228 | 2.8 us | 0 |
| launch_overhead_compiled | 5.9 ms | 87.0% | 226 | 2.8 us | 0 |
| hidden_sync | 63.0 ms | 14.7% | 278 | 136 us | 36.9 ms |
| no_recompute | 55.5 ms | 1.2% | 278 | 140 us | 0 |
| full_recompute | 64.5 ms | 1.4% | 326 | 138 us | 0 |
| looped_matmul | 61.5 ms | 83.5% | 2,102 | 3.6 us | 0 |
| grouped_matmul | 4.7 ms | 13.7% | 72 | 43 us | 0 |

- **GPU step:** GPU time from first to last activity in the capture, divided by 5 steps.
- **GPU idle:** share of that time with no kernel, memcpy or memset running.
- **CPU blocked in sync:** time per step inside `cuda*Synchronize` or a blocking
  `cudaMemcpy`, counted only inside `step N` ranges.
- \* dataloader_bound's sync comes from a pageable host-to-device copy, which
  shows up as `cudaMemcpyAsync` and isn't counted here. See its entry.

The two models:
- **BIG:** an MLP-only LM. Embedding (vocab 32,768), then 16 blocks of LayerNorm
  + 2048->8192->2048 MLP with GELU, then the LM head. Batch 8 x 1,024 tokens,
  bf16, fused AdamW. 604M parameters in matmuls.
- **TINY:** the same shape at width 256, vocab 1,024, batch 4 x 64 tokens.

Attention is left out on purpose: these scenarios are about the shape of a step,
not about one layer.

---

## 1. healthy: the reference for "good"

**What it is:** the BIG model, with batches made on the CPU. Each step does
10 ms of simulated host prep (standing in for tokenizing and packing), then
copies the batch from pinned memory with `non_blocking=True`. Nothing in the
step forces the CPU to wait for the GPU.

**Rows to look at:**
- CUDA HW: kernels packed back to back.
- The CPU thread's NVTX row, against the NVTX GPU projection.

**Pattern:**
- **The GPU row is solid:** one gap over 50 us every couple of steps.
- **The CPU runs ahead of the GPU.** A step takes the CPU only ~30 ms to issue
  (data 10.4, forward 4.9, backward 10.6, optimizer 3.6) against 56 ms of GPU
  work. So the CPU's `step 15` range sits well to the left of the GPU
  projection of `step 15`.
- **Long `cudaLaunchKernel` calls (up to ~1.4 ms) are harmless here.** Once the
  CPU is far enough ahead, the GPU's launch queue fills, and each launch waits
  for a free slot. That's backpressure, not a problem.

**Key numbers:**
- 56.1 ms per step, 1.3% idle.
- 278 kernels per step, median 140 us: big matmuls.
- About 530 TFLOPS: 3.62 GFLOPs/token x 8,192 tokens / 56.1 ms, which is 54% of
  the H100's 989 TFLOPS peak.

**This is what compute-bound looks like.** The only way to go faster is faster
kernels.

## 2. dataloader_bound: the GPU waits for data

**What it is:** the same model, fed by a `DataLoader` with `num_workers=0`. Each
sample sleeps 10 ms (standing in for decode and tokenize), so one batch of 8
costs the main thread ~80 ms. There's no `pin_memory`, and the copy is a plain
`.to(device)`.

**Rows to look at:**
- CUDA HW: a large gap in every step.
- The CPU thread's NVTX `data` range: 83 ms, lined up with the gap.
- OS runtime libraries: 8 `clock_nanosleep` calls inside each `data` range, one
  per sample.
- CUDA API: the batch copy is a `cudaMemcpyAsync` from pageable memory. CUDA
  stages that through a bounce buffer, so the call blocks the CPU until the copy
  is done. It takes microseconds here because the batch is tiny token ids.

**Pattern:**
- The GPU finishes the step's work, then sits idle while the CPU is still in
  `data`.
- The GPU projection of `forward` starts only after the CPU's `data` range ends.

**Key numbers:**
- 97.6 ms per step, 45.4% idle.
- 43.6 ms of gaps per step.
- The kernels themselves are unchanged from healthy: 278 per step, median 135 us.

**vs healthy:** the same GPU work takes 1.74x longer. Every millisecond lost is
a gap, not a slower kernel. The fix is upstream of the GPU: see the next entry.

## 3. dataloader_fixed: same data, loaded in parallel

**What it is:** the same dataset, with these DataLoader settings:
- `num_workers=8`: 8 processes prepare batches in parallel.
- `pin_memory=True`: batches land in page-locked memory, so the copy can be async.
- `prefetch_factor=4`: each worker keeps up to 4 batches ready ahead of time.
- `persistent_workers=True`: workers stay alive between epochs.
- The copy to the GPU uses `non_blocking=True`.

**Rows to look at:**
- CUDA HW: solid again.
- The main thread's `data` range is now about 1 ms.
- The main thread's OS runtime row: the 40 `clock_nanosleep` calls from
  dataloader_bound are gone. Only a short `poll`/`read` remains, which is the
  main process picking up a finished batch from a worker.
- The worker processes aren't traced in this report, so their sleeps don't
  appear anywhere. Their effect shows only in what's missing from the main thread.

**Pattern:** looks like healthy. The 80 ms of per-batch work still happens,
but in the workers, in parallel, ahead of time.

**Key numbers:** 55.8 ms per step, 1.6% idle, `data` 1.1 ms on the main thread.

**vs healthy:** the same, within noise. **Rule of thumb:** if the `data` range
on the main thread is a large share of the step and lines up with GPU gaps,
you're input-bound.

## 4. launch_overhead: kernels too small to keep the GPU busy

**What it is:** the TINY model in eager mode. Each matmul is 256 tokens x 256
wide: microseconds of work.

**Rows to look at:**
- Zoom deep into CUDA HW: thin slivers of kernels with gaps between them.
- The CPU thread's CUDA API row: a dense comb of `cudaLaunchKernel` calls.

**Pattern:**
- **The GPU runs each kernel faster than the CPU can launch the next one**, so
  the GPU timeline is mostly empty.
- The CPU is 100% busy launching, while the GPU is ~6% busy.

**Key numbers:**
- 11.2 ms per step, 93.6% idle.
- 228 kernels per step, median **2.8 us**: shorter than the launch itself.
- 60 gaps per step.
- The CPU NVTX ranges (forward 3.5 ms + backward 6.6 ms + optimizer 1.0 ms)
  add up to the step time.

**vs healthy:**
- healthy: median kernel 140 us, the CPU ahead, the GPU full.
- launch_overhead: median kernel 2.8 us, the CPU behind, the GPU empty.

**Tell-tale:** the step time tracks the CPU's NVTX ranges, not the GPU's work.

## 5. launch_overhead_compiled: torch.compile + CUDA graphs

**What it is:** the TINY model wrapped in `torch.compile(mode="reduce-overhead")`.
Inductor fuses some elementwise ops, and the forward and backward are each
captured as a **CUDA graph**: one CPU call replays hundreds of kernels.

**Rows to look at:**
- CUDA API: per step, there are now **2 `cudaGraphLaunch` calls** (forward and
  backward) plus ~6 `cudaLaunchKernel` calls (the optimizer), instead of ~230
  launches.
- CUDA HW: the kernels inside each graph run back to back. The profile was
  recorded with `--cuda-graph-trace=node`, so you see each kernel. Without that
  flag, each graph shows as one block.

**Pattern:** dense bursts (the graph replays), then gaps where the CPU does
everything else: the Python loop, compile guard checks, the eager optimizer.

**Key numbers:**
- 5.9 ms per step (1.9x faster than eager), 87% idle.
- Gaps per step drop from 60 to 8.
- CPU time per step: forward 1.7 ms (was 3.5), backward 2.5 ms (was 6.6).

**vs launch_overhead:** the per-kernel launch cost is gone. The GPU is still
mostly idle, because 0.8 ms of GPU work per step can't hide even a few
milliseconds of Python. The remaining fixes:
- Give the GPU more work per step (a bigger batch).
- Capture more of the step, such as the optimizer, in the graph.

Two caveats:
- Kernel count barely changes (226 vs 228). Fusion only removes the small
  elementwise ops; the matmuls stay separate cuBLAS calls. CUDA graphs are the
  bigger win here.
- `--cuda-graph-trace=node` adds tracing overhead to each graph launch, so this
  run is a little faster without the profiler.

## 6. hidden_sync: a `.item()` every step

**What it is:** healthy plus one line at the end of each step:
`print(f"... {loss.item()}")`. That's the same mistake Phase 2 found in
torchtitan's logging.

**Rows to look at:**
- The CPU thread's NVTX `log loss.item()` range (~37 ms).
- Inside it, on CUDA API: a `cudaMemcpyAsync` (device to host), then
  **`cudaStreamSynchronize`**.
- CUDA HW: a gap at the start of every step.

**Pattern:**
- **`.item()` needs the loss value on the CPU**, so the CPU waits until the GPU
  has finished everything queued: the whole backward and the optimizer.
- **The GPU's queue is then empty.** The next step's 10 ms of host prep runs
  while the GPU sits idle.
- In healthy, that prep overlaps the previous step's GPU work. Here it can't.
- The CPU NVTX ranges and the GPU projection now line up step by step: the CPU
  no longer runs ahead.

**Key numbers:**
- 63.0 ms per step, 14.7% idle.
- 8.6 ms of gaps per step: roughly the 10 ms host prep.
- 36.9 ms per step of the CPU blocked in sync.

**vs healthy:** +12% step time, from one innocent-looking line. **How to find
it:** look for long `cudaStreamSynchronize` or `cudaMemcpy` calls on the CPU
inside the step, followed by a GPU gap.
- **The fix:** log every N steps, or keep metrics on the GPU and read them
  rarely.
- **A sync with no CPU work after it costs much less.** The damage comes from
  CPU work that can no longer overlap.

## 7. no_recompute vs full_recompute: activation checkpointing

**What they are:** the BIG model without and with
`torch.utils.checkpoint(..., use_reentrant=False)` around each of the 16 blocks.

**Rows to look at:**
- CUDA HW with the NVTX GPU projection: look inside the **backward** range.
- In full_recompute, forward kernels (LayerNorm, an up-projection matmul, GELU)
  appear again, block by block, interleaved with the backward kernels.

**Pattern:**
- **Forward:** unchanged.
- **Backward:** longer, and contains recomputed forward work.

**Key numbers:**

| | step | kernels/step | GPU idle |
|---|---|---|---|
| no_recompute | 55.5 ms | 278 | 1.2% |
| full_recompute | 64.5 ms (**+16%**) | 326 (+48) | 1.4% |

**Why only +16% and not ~+30%?** The kernel counts show it: +48 kernels = 3 per
block (norm, up matmul, GELU), not 4. PyTorch's non-reentrant checkpoint
**stops recomputing early**. Backward needs the norm output, the up output and
the GELU output, but never the down projection's output, so that matmul is
skipped. Only half of each block's matmul FLOPs are recomputed.

**vs healthy:** the GPU is still fully busy. Recompute makes the step longer
with more work, not idle. That's the AC trade-off from Phase 1: spend compute to
save activation memory, then spend that memory on a bigger batch.

## 8. looped_matmul vs grouped_matmul: why grouped GEMM matters for MoE

**What they are:** 4 MoE layers, each with 64 experts (1,024 -> 512 -> 1,024,
GELU). There are 16,384 tokens, already routed **evenly**: 256 per expert.
- **looped_matmul:** a Python loop over 64 `nn.Linear` pairs. 64 x 2 small
  matmuls per layer, in forward and again in backward.
- **grouped_matmul:** the same weights stacked into (64, 1024, 512) tensors.
  One `torch.bmm` per projection does all 64 experts at once.

**Rows to look at:**
- Zoom into CUDA HW in looped_matmul: thousands of ~3-5 us `nvjet_tst_64x64...`
  kernels with gaps between them.
- In grouped_matmul: a few ~45 us `nvjet_tst_256x128...` kernels.
- The CUDA API row shows the same contrast: a dense comb of launches vs a handful.

**Pattern:** looped is scenario 4 again, inside one layer. Each expert's matmul
is too small to fill the GPU, and the CPU can't launch them fast enough.

**Key numbers:**

| | step | kernels/step | median kernel | GPU idle |
|---|---|---|---|---|
| looped_matmul | 61.5 ms | 2,102 | 3.6 us | 83.5% |
| grouped_matmul | 4.7 ms (**13x faster**) | 72 | 43 us | 13.7% |

- About 42% of grouped's remaining GPU time is the fused AdamW kernel
  (`multi_tensor_apply_kernel`).
- The kernel names show the shape: cuBLAS picks small 64x64 tiles for the
  looped matmuls and large 256x128 tiles for the batched ones. Larger tiles get
  more work done per launch.

**Caveat:** even routing is what lets `bmm` work. Real routers send different
token counts to each expert, so production MoE code uses **grouped GEMM**
kernels that take per-expert offsets: `torch._grouped_mm`, Transformer
Engine's GroupedLinear in Megatron, CUTLASS grouped GEMM. The timeline lesson is
the same: one launch for all experts instead of one per expert.

## 9. fsdp_2gpu: not run

Skipped by choice to keep the pod at 1 GPU. `scenarios.py --scenario fsdp_2gpu`
and `run_gallery.sh` already support it; it runs automatically on a pod with 2 or
more GPUs. Your Phase 3 PyTorch-profiler traces (`outputs/phase3/*_prof__*`)
show the same thing in Perfetto: `ncclDevKernel_AllGather` and `ReduceScatter`
on their own streams, overlapping compute.

---

## Files

```
reports/<scenario>.nsys-rep      open in the Nsight Systems GUI
stats/<scenario>.stats.txt       nsys stats: cuda_gpu_kern_sum, nvtx_sum, cuda_api_sum
stats/<scenario>.analysis.txt    analyze.py output (the numbers above)
stats/<scenario>.json            the same, machine-readable
stats/<scenario>.run.log         the script's console output
summary.md                       the summary table
```

Profiled on pod `wgfgv68d0pdn4q` (1x H100 SXM, US-MO-1) on 2026-10-22.
The pod ran ~8 minutes, about $0.55.
