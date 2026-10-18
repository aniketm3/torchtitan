# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Summarize sweep logs into the blog's metrics: TPS/GPU, MFU, mem%.

    python torchtitan_recipes/aleph_alpha_scaling/summarize.py sweep_logs [--plot out.png]

Per run, drops the first NUM_WARMUP_STEPS steps, averages tps and mfu over the
rest, and reports the peak reserved-memory fraction.
"""

import argparse
import csv
import re
from pathlib import Path

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_STEP_RE = re.compile(
    r"step:\s*(\d+).*?memory:\s*[\d.]+GiB\(([\d.]+)%\).*?tps:\s*([\d,]+)"
    r".*?mfu:\s*([\d.]+)%"
)
_NAME_RE = re.compile(r"aa_(1p7b|30b_a3b)_(\w+?)_lbs(\d+)_fsdp(\d+)_dp(\d+)")
NUM_WARMUP_STEPS = 10


def parse_log(path: Path) -> dict | None:
    steps = []
    for line in path.read_text(errors="replace").splitlines():
        m = _STEP_RE.search(_ANSI_RE.sub("", line))
        if m:
            step, mem, tps, mfu = m.groups()
            steps.append((int(step), float(mem), int(tps.replace(",", "")), float(mfu)))
    name = _NAME_RE.search(path.stem)
    if not steps or name is None:
        return None
    window = [s for s in steps if s[0] > NUM_WARMUP_STEPS]
    if not window:
        return None
    model, ac, lbs, fsdp, dp = name.groups()
    return {
        "model": model,
        "ac": ac,
        "lbs": int(lbs),
        "fsdp": int(fsdp),
        "dp": int(dp),
        # Whatever follows the base name, e.g. "_compile__rep2" or "_prof__best".
        "variant": path.stem[name.end():].lstrip("_"),
        "num_steps": len(steps),
        "num_avg_steps": len(window),
        "tps": sum(s[2] for s in window) / len(window),
        "mfu": sum(s[3] for s in window) / len(window),
        "mem_pct": max(s[1] for s in steps),
        # The blog flags "CUDA allocation retry" warnings as the memory-saturation signal.
        "alloc_retries": "memory allocation retries" in path.read_text(errors="replace"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--csv", type=Path, help="Write results to this CSV.")
    parser.add_argument("--plot", type=Path, help="Save a TPS-vs-memory plot.")
    args = parser.parse_args()

    rows = [r for p in sorted(args.log_dir.glob("*.log")) if (r := parse_log(p))]
    if not rows:
        raise SystemExit(f"No parsable runs in {args.log_dir}")
    rows.sort(key=lambda r: (r["ac"], r["lbs"], r["variant"]))

    header = f"{'AC':<10}{'LBS':>5}{'FSDP/DP':>9}{'steps':>7}{'TPS/GPU':>10}{'MFU%':>7}{'mem%':>7}  retries  variant"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['ac']:<10}{r['lbs']:>5}{r['fsdp']:>6}/{r['dp']:<2}{r['num_steps']:>7}"
            f"{r['tps']:>10,.0f}{r['mfu']:>7.1f}{r['mem_pct']:>7.1f}"
            f"  {'yes' if r['alloc_retries'] else '':<7}  {r['variant']}"
        )
    best = max(rows, key=lambda r: r["tps"])
    print(
        f"\nBest: {best['ac']} LBS={best['lbs']} -> {best['tps']:,.0f} TPS/GPU, "
        f"{best['mfu']:.1f}% MFU, {best['mem_pct']:.0f}% mem"
    )
    print("Blog (16x B200): selective LBS=18 -> 28.4k TPS/GPU, 37.5% MFU, 92% mem")

    if args.csv:
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for ac in sorted({r["ac"] for r in rows}):
            pts = [r for r in rows if r["ac"] == ac]
            ax.plot(
                [r["mem_pct"] for r in pts],
                [r["tps"] for r in pts],
                marker="o",
                label=f"{ac}-AC",
            )
            for r in pts:
                ax.annotate(str(r["lbs"]), (r["mem_pct"], r["tps"]), fontsize=7)
        ax.set_xlabel("peak reserved memory (%)")
        ax.set_ylabel("tokens/s/GPU")
        ax.set_title(f"Qwen3 {rows[0]['model']}, seq 4096 (labels = LBS)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)


if __name__ == "__main__":
    main()
