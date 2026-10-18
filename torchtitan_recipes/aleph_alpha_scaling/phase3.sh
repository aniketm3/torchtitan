#!/usr/bin/bash
# Phase 3 runs on one 8-GPU node. Each line: "<num_gpus> <config> <tag>".
# Logs: $LOG_DIR/<config>__<tag>.log; outputs: outputs/phase3/<config>__<tag>/.
# All runs log every 10 steps (_log10), the Phase 2b fix.
#   ./phase3.sh dense   # Qwen3-1.7B: DDP vs FSDP vs HSDP, 1 -> 8 GPU scaling
#   ./phase3.sh moe     # Qwen3-30B-A3B: fit, AC x LBS, expert parallelism
# Run from the repo root.
set -u
SECTION=$1
LOG_DIR=${LOG_DIR:-sweep_logs/phase3_$SECTION}
STOP_POD_WHEN_DONE=${STOP_POD_WHEN_DONE:-0}
mkdir -p "$LOG_DIR"

DENSE="
1 aa_1p7b_selective_lbs8_fsdp1_dp1_log10        base1gpu
8 aa_1p7b_selective_lbs8_fsdp1_dp8_log10        ddp
8 aa_1p7b_selective_lbs8_fsdp8_dp1_log10        fsdp
8 aa_1p7b_selective_lbs8_fsdp4_dp2_log10        hsdp
8 aa_1p7b_selective_lbs12_fsdp8_dp1_log10       fsdp
8 aa_1p7b_selective_lbs16_fsdp8_dp1_log10       fsdp
8 aa_1p7b_none_lbs8_fsdp8_dp1_log10             fsdp
8 aa_1p7b_selective_lbs8_fsdp1_dp8_log10_prof   ddp
8 aa_1p7b_selective_lbs8_fsdp8_dp1_log10_prof   fsdp
"
MOE="
8 aa_30b_a3b_full_lbs1_fsdp8_dp1_log10          fit
8 aa_30b_a3b_full_lbs4_fsdp8_dp1_log10          sweep
8 aa_30b_a3b_full_lbs8_fsdp8_dp1_log10          sweep
8 aa_30b_a3b_default_lbs2_fsdp8_dp1_log10       sweep
8 aa_30b_a3b_default_lbs4_fsdp8_dp1_log10       sweep
8 aa_30b_a3b_selective_lbs2_fsdp8_dp1_log10     sweep
8 aa_30b_a3b_selective_lbs4_fsdp8_dp1_log10     sweep
8 aa_30b_a3b_default_lbs4_fsdp8_dp1_ep8_log10   ep
8 aa_30b_a3b_default_lbs4_fsdp8_dp1_log10_prof  noep
8 aa_30b_a3b_default_lbs4_fsdp8_dp1_ep8_log10_prof ep
"
case $SECTION in
    dense) RUNS=$DENSE ;;
    moe) RUNS=$MOE ;;
    *) echo "usage: $0 dense|moe"; exit 1 ;;
esac

echo "$RUNS" | while read -r ngpu config tag; do
    [[ -z "$config" ]] && continue
    run="${config}__${tag}"
    log="$LOG_DIR/$run.log"
    if [[ -f "$log" ]] && grep -q "Training completed" "$log"; then
        echo "[skip] $run"
        continue
    fi
    echo "[run ] $run"
    NGPU=$ngpu MODULE=torchtitan_recipes.aleph_alpha_scaling.recipes CONFIG=$config \
        ./run_train.sh --output-dir "outputs/phase3/$run" >"$log" 2>&1 </dev/null ||
        echo "[fail] $run (see $log)"
done

python torchtitan_recipes/aleph_alpha_scaling/summarize.py "$LOG_DIR" --csv "$LOG_DIR/results.csv"
python torchtitan_recipes/aleph_alpha_scaling/tb_summary.py outputs/phase3

if [[ "$STOP_POD_WHEN_DONE" == "1" ]]; then
    bash torchtitan_recipes/aleph_alpha_scaling/stop_pod_after_copy.sh "$LOG_DIR"
fi
