# Throughput study: reproducing Aleph Alpha's hierarchical sweep

A learning reproduction of
[Scaling Pre-Training in Practice: A Hierarchical Approach](https://aleph-alpha.com/en/blog/scaling-pre-training-in-practice-a-hierarchical-approach/):
sweep activation checkpointing (AC) x local batch size, profile the result, and
find out what limits tokens/sec per GPU. Qwen3-1.7B on 1x H100 first, then
8 GPUs and Qwen3-30B-A3B.

**Results and reasoning live in [LOG.md](LOG.md)**: per-phase numbers,
predictions vs outcomes, and where to look in each trace.

## Files

| File | Purpose |
|---|---|
| `recipes.py` | One config per run, encoded in the name: `aa_{1p7b,30b_a3b}_{none,selective,full,default}_lbs{N}_fsdp{F}_dp{D}[_ep8][_eager][_log10][_prof]` |
| `run_sweep.sh` | Phase 1: AC x LBS sweep, each AC mode until OOM |
| `phase2.sh`, `phase2b.sh` | Phase 2: noise, compile, LBS 9, profiles; then the log_freq fix |
| `phase3.sh` | Phase 3: `dense` (DDP/FSDP/HSDP, 1 -> 8 GPUs) and `moe` (30B-A3B, EP) |
| `summarize.py` | Logs -> TPS / MFU / peak memory table, CSV, TPS-vs-memory plot |
| `tb_summary.py` | TensorBoard -> data-loading share and step time per run |
| `trace_summary.py` | Profiler trace -> GPU busy %, idle gaps, kernel time by category |
| `pod_setup.sh` | Environment setup on a fresh RunPod pod |
| `stop_pod_after_copy.sh`, `fetch_and_stop.sh` | Pod waits for the laptop to copy results, then stops itself |

## Running on a RunPod pod

```bash
# laptop: copy the repo over, then on the pod:
bash torchtitan_recipes/aleph_alpha_scaling/pod_setup.sh
NGPU=1 MODULE=torchtitan_recipes.aleph_alpha_scaling.recipes \
  CONFIG=aa_1p7b_selective_lbs8_fsdp1_dp1_log10 ./run_train.sh
```

Setup notes that cost time to find:
- `requirements.txt` replaces the torch nightly with a stable build; `pod_setup.sh`
  reinstalls the nightly last.
- On H100, torchtitan uses FlashAttention 3, which needs the `flash_attn_3` wheel.
- Logging every step (`log_freq=1`) forces a CPU-GPU sync each step and cost
  ~28% of throughput here. Use `_log10` for measurements.
- Stopping a pod wipes its container disk. Copy results off first
  (`fetch_and_stop.sh`).
