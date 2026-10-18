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

