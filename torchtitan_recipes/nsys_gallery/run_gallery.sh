#!/usr/bin/env bash
# Profile every scenario under nsys, then write stats and key numbers.
# Run from this folder on the pod:  bash run_gallery.sh
# Output: $OUT/reports/<scenario>.nsys-rep, $OUT/stats/<scenario>.*, $OUT/summary.md
set -u
cd "$(dirname "$0")"
OUT=${OUT:-/root/nsys_gallery_out}
SCENARIOS=${SCENARIOS:-"healthy dataloader_bound dataloader_fixed launch_overhead launch_overhead_compiled hidden_sync no_recompute full_recompute looped_matmul grouped_matmul"}
mkdir -p "$OUT/reports" "$OUT/stats"
log() { echo "$(date -u +%T) $*" | tee -a "$OUT/progress.log"; }

NUM_GPUS=$(nvidia-smi -L | wc -l)
if [[ $NUM_GPUS -ge 2 ]]; then SCENARIOS="$SCENARIOS fsdp_2gpu"; fi

for s in $SCENARIOS; do
    log "[run ] $s"
    if [[ $s == fsdp_2gpu ]]; then
        launch=(torchrun --standalone --nproc_per_node 2 scenarios.py --scenario "$s")
    else
        launch=(python scenarios.py --scenario "$s")
    fi
    # Only the steps between torch.cuda.profiler.start()/stop() are recorded.
    # cuda-graph-trace=node shows each kernel inside a CUDA graph, not one block.
    if ! nsys profile \
        -t cuda,nvtx,osrt,cublas,cudnn \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        --cuda-graph-trace=node \
        --sample=none --cpuctxsw=none \
        --force-overwrite=true \
        -o "$OUT/reports/$s" \
        "${launch[@]}" >"$OUT/stats/$s.run.log" 2>&1; then
        log "[fail] $s (see stats/$s.run.log)"
        tail -5 "$OUT/stats/$s.run.log" | tee -a "$OUT/progress.log"
        continue
    fi
    rep="$OUT/reports/$s.nsys-rep"
    db="$OUT/reports/$s.sqlite"  # nsys stats reuses this export
    nsys export --type sqlite --force-overwrite true -o "$db" "$rep" >/dev/null 2>&1
    {
        for report in cuda_gpu_kern_sum nvtx_sum cuda_api_sum; do
            nsys stats --report "$report" --format column "$rep" 2>&1 | grep -v "^Processing\|^Generating SQLite"
        done
    } >"$OUT/stats/$s.stats.txt"
    python analyze.py "$db" --name "$s" --json "$OUT/stats/$s.json" | tee -a "$OUT/progress.log"
    rm -f "$db"
done

python analyze.py --summary "$OUT"/stats/*.json >"$OUT/summary.md"
cat "$OUT/summary.md" | tee -a "$OUT/progress.log"
log "DONE"
touch "$OUT/DONE"
