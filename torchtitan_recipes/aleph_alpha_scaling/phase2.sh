#!/usr/bin/bash
# Phase 2 runs on 1x H100: noise, compile, the retry gap, and profiles.
# Each line is "<config> <tag>"; the log is $LOG_DIR/<config>__<tag>.log and
# TensorBoard/profiler output goes to outputs/phase2/<config>__<tag>/.
# Run from the repo root.
set -u

LOG_DIR=${LOG_DIR:-sweep_logs/phase2_1p7b}
STOP_POD_WHEN_DONE=${STOP_POD_WHEN_DONE:-0}
mkdir -p "$LOG_DIR"

RUNS="
aa_1p7b_selective_lbs8_fsdp1_dp1          rep1
aa_1p7b_selective_lbs8_fsdp1_dp1          rep2
aa_1p7b_selective_lbs8_fsdp1_dp1          rep3
aa_1p7b_selective_lbs8_fsdp1_dp1_eager    rep1
aa_1p7b_selective_lbs8_fsdp1_dp1_eager    rep2
aa_1p7b_selective_lbs9_fsdp1_dp1          rep1
aa_1p7b_selective_lbs8_fsdp1_dp1_prof     best
aa_1p7b_selective_lbs1_fsdp1_dp1_prof     worst
aa_1p7b_full_lbs8_fsdp1_dp1_prof          fullac
"

echo "$RUNS" | while read -r config tag; do
    [[ -z "$config" ]] && continue
    run="${config}__${tag}"
    log="$LOG_DIR/$run.log"
    if [[ -f "$log" ]] && grep -q "Training completed" "$log"; then
        echo "[skip] $run"
        continue
    fi
    echo "[run ] $run"
    NGPU=1 MODULE=torchtitan_recipes.aleph_alpha_scaling.recipes CONFIG=$config \
        ./run_train.sh --output-dir "outputs/phase2/$run" >"$log" 2>&1 </dev/null ||
        echo "[fail] $run (see $log)"
done

python torchtitan_recipes/aleph_alpha_scaling/summarize.py "$LOG_DIR" --csv "$LOG_DIR/results.csv"
python torchtitan_recipes/aleph_alpha_scaling/tb_summary.py outputs/phase2

if [[ "$STOP_POD_WHEN_DONE" == "1" ]]; then
    bash torchtitan_recipes/aleph_alpha_scaling/stop_pod_after_copy.sh "$LOG_DIR"
fi
