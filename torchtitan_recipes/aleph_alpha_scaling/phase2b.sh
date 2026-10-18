#!/usr/bin/bash
# Phase 2b: test fix 1 (log every 10 steps instead of every step).
# Same runner as phase2.sh, different run list. Run from the repo root.
set -u
LOG_DIR=${LOG_DIR:-sweep_logs/phase2b_1p7b}
STOP_POD_WHEN_DONE=${STOP_POD_WHEN_DONE:-0}
mkdir -p "$LOG_DIR"

RUNS="
aa_1p7b_selective_lbs8_fsdp1_dp1              rep1
aa_1p7b_selective_lbs8_fsdp1_dp1_log10        rep1
aa_1p7b_selective_lbs8_fsdp1_dp1_log10        rep2
aa_1p7b_full_lbs8_fsdp1_dp1_log10             rep1
aa_1p7b_full_lbs24_fsdp1_dp1_log10            rep1
aa_1p7b_selective_lbs8_fsdp1_dp1_log10_prof   fixed
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
        ./run_train.sh --output-dir "outputs/phase2b/$run" >"$log" 2>&1 </dev/null ||
        echo "[fail] $run (see $log)"
done

python torchtitan_recipes/aleph_alpha_scaling/summarize.py "$LOG_DIR" --csv "$LOG_DIR/results.csv"
python torchtitan_recipes/aleph_alpha_scaling/tb_summary.py outputs/phase2b

if [[ "$STOP_POD_WHEN_DONE" == "1" ]]; then
    bash torchtitan_recipes/aleph_alpha_scaling/stop_pod_after_copy.sh "$LOG_DIR"
fi
