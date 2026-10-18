#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# AC mode x local batch size sweep. For each AC mode, run LBS values in order
# until one fails (normally OOM), then move to the next mode. Run from repo root.
#
#   ./torchtitan_recipes/aleph_alpha_scaling/run_sweep.sh
#   AC_MODES=selective LBS_LIST="10 11 12" ./torchtitan_recipes/aleph_alpha_scaling/run_sweep.sh
#   STOP_POD_WHEN_DONE=1 ./torchtitan_recipes/aleph_alpha_scaling/run_sweep.sh

set -u

MODEL=${MODEL:-1p7b}
NGPU=${NGPU:-1}
FSDP=${FSDP:-$NGPU}
DP=${DP:-1}
AC_MODES=${AC_MODES:-"none selective full"}
LBS_LIST=${LBS_LIST:-"1 2 3 4 6 8 10 12 16 20 24 32 40 48 64"}
LOG_DIR=${LOG_DIR:-sweep_logs/phase1_${MODEL}}
STOP_POD_WHEN_DONE=${STOP_POD_WHEN_DONE:-0}

mkdir -p "$LOG_DIR"

for ac in $AC_MODES; do
    for lbs in $LBS_LIST; do
        name="aa_${MODEL}_${ac}_lbs${lbs}_fsdp${FSDP}_dp${DP}"
        log="$LOG_DIR/$name.log"
        if [[ -f "$log" ]] && grep -q "Training completed" "$log"; then
            echo "[skip] $name (already done)"
            continue
        fi
        # Save the fully resolved config next to the log (CPU only, no GPU).
        python -m torchtitan.train --module torchtitan_recipes.aleph_alpha_scaling.recipes \
            --config "$name" --print-config >"$LOG_DIR/$name.config.txt" 2>&1
        echo "[run ] $name"
        NGPU=$NGPU \
        MODULE=torchtitan_recipes.aleph_alpha_scaling.recipes \
        CONFIG=$name \
        ./run_train.sh >"$log" 2>&1
        status=$?
        if [[ $status -ne 0 ]]; then
            if grep -qi "out of memory\|OutOfMemoryError" "$log"; then
                echo "[oom ] $name -- stopping $ac sweep"
            else
                echo "[fail] $name (exit $status) -- see $log; stopping $ac sweep"
            fi
            break
        fi
    done
done

python torchtitan_recipes/aleph_alpha_scaling/summarize.py "$LOG_DIR" \
    --csv "$LOG_DIR/results.csv" --plot "$LOG_DIR/tps_vs_mem.png"

if [[ "$STOP_POD_WHEN_DONE" == "1" ]]; then
    bash torchtitan_recipes/aleph_alpha_scaling/stop_pod_after_copy.sh "$LOG_DIR"
fi
