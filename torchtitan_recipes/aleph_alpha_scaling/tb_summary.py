# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-run data-loading share from TensorBoard, which the console log omits.

    python torchtitan_recipes/aleph_alpha_scaling/tb_summary.py outputs/phase2

For each run folder, averages over steps after warmup: the fraction of step
time spent waiting on the dataloader, and the mean end-to-end step time.
"""

import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

NUM_WARMUP_STEPS = 10
TAGS = {
    "data%": "time_metrics/data_loading(%)",
    "step_s": "time_metrics/end_to_end(s)",
}


def summarize_run(run_dir: Path) -> dict[str, float]:
    out = {}
    for events in run_dir.rglob("events.out.tfevents.*"):
        acc = EventAccumulator(str(events))
        acc.Reload()
        for name, tag in TAGS.items():
            if tag in acc.Tags()["scalars"]:
                vals = [e.value for e in acc.Scalars(tag) if e.step > NUM_WARMUP_STEPS]
                if vals:
                    out[name] = sum(vals) / len(vals)
    return out


def main() -> None:
    root = Path(sys.argv[1])
    print(f"{'run':<55}{'data%':>8}{'step_s':>9}")
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        stats = summarize_run(run_dir)
        data = f"{stats['data%']:.2f}" if "data%" in stats else "n/a"
        step = f"{stats['step_s']:.3f}" if "step_s" in stats else "n/a"
        print(f"{run_dir.name:<55}{data:>8}{step:>9}")


if __name__ == "__main__":
    main()
