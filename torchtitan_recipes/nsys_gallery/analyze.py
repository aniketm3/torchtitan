"""Key numbers from an Nsight Systems report exported to SQLite.

    nsys export --type sqlite -o healthy.sqlite healthy.nsys-rep
    python analyze.py healthy.sqlite --name healthy --json healthy.json
    python analyze.py --summary stats/*.json        # markdown table

Per GPU, over the captured window (first to last GPU activity):
  step time    window / number of captured "step N" NVTX ranges
  idle %       share of the window with no kernel, memcpy or memset running
  gaps         idle stretches longer than 50 us
  kernels      launches per step and median kernel duration
  nccl         NCCL kernel time per step and how much of it overlaps compute
CPU side (from the NVTX ranges and CUDA API calls):
  phase ms     mean CPU duration of data / forward / backward / optimizer
  sync ms      time per step the CPU spent blocked in cuda*Synchronize or
               synchronous cudaMemcpy
"""

import argparse
import json
import re
import sqlite3
import statistics
import sys

GAP_US = 50
PHASES = ["data", "forward", "backward", "optimizer", "log loss.item()"]


def union_length(intervals):
    """Total length covered by (start, end) intervals, and the merged list."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(e - s for s, e in merged), merged


def overlap_length(intervals, merged_other):
    """Length of `intervals` covered by the merged intervals `merged_other`."""
    total, j = 0, 0
    for start, end in sorted(intervals):
        while j < len(merged_other) and merged_other[j][1] <= start:
            j += 1
        k = j
        while k < len(merged_other) and merged_other[k][0] < end:
            total += min(end, merged_other[k][1]) - max(start, merged_other[k][0])
            k += 1
    return total


def has_table(db, name):
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()


def analyze(path, name):
    db = sqlite3.connect(path)
    kernels = db.execute(
        "SELECT k.start, k.end, k.deviceId, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.shortName = s.id"
    ).fetchall()
    other = []
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if has_table(db, table):
            other += db.execute(f"SELECT start, end, deviceId FROM {table}").fetchall()

    nvtx = []
    if has_table(db, "NVTX_EVENTS"):
        nvtx = db.execute(
            "SELECT n.start, n.end, n.globalTid, COALESCE(n.text, s.value) FROM NVTX_EVENTS n "
            "LEFT JOIN StringIds s ON n.textId = s.id WHERE n.end IS NOT NULL"
        ).fetchall()
    steps = {text for _, _, _, text in nvtx if text and re.fullmatch(r"step \d+", text)}
    num_steps = max(len(steps), 1)

    result = {"name": name, "num_steps": len(steps), "gpus": {}}
    for device in sorted({d for _, _, d, _ in kernels}):
        dev_kernels = [(s, e, n) for s, e, d, n in kernels if d == device]
        activity = [(s, e) for s, e, _ in dev_kernels] + [(s, e) for s, e, d in other if d == device]
        window_start = min(s for s, _ in activity)
        window_end = max(e for _, e in activity)
        window = window_end - window_start
        busy, merged = union_length(activity)
        gaps = [b[0] - a[1] for a, b in zip(merged, merged[1:]) if b[0] - a[1] > GAP_US * 1000]

        nccl = [(s, e) for s, e, n in dev_kernels if "nccl" in n.lower()]
        compute = [(s, e) for s, e, n in dev_kernels if "nccl" not in n.lower()]
        nccl_total = sum(e - s for s, e in nccl)
        _, compute_merged = union_length(compute)
        durations_us = [(e - s) / 1000 for s, e, _ in dev_kernels]

        result["gpus"][str(device)] = {
            "step_ms": window / num_steps / 1e6,
            "idle_pct": 100 * (1 - busy / window),
            "num_gaps_over_50us_per_step": len(gaps) / num_steps,
            "gap_ms_per_step": sum(gaps) / num_steps / 1e6,
            "kernels_per_step": len(dev_kernels) / num_steps,
            "median_kernel_us": statistics.median(durations_us),
            "nccl_ms_per_step": nccl_total / num_steps / 1e6,
            "nccl_overlap_pct": (
                100 * overlap_length(nccl, compute_merged) / nccl_total if nccl_total else None
            ),
        }

    phases = {}
    for phase in PHASES:
        durs = [(e - s) / 1e6 for s, e, _, text in nvtx if text == phase]
        if durs:
            phases[phase] = statistics.mean(durs)
    result["cpu_phase_ms"] = phases

    # Only syncs inside a "step N" range count: the script's own synchronize()
    # before profiler.stop() is outside every step.
    # globalTid packs the process id in bits 24-47.
    def pid(tid):
        return (tid >> 24) & 0xFFFFFF

    step_ranges = [(s, e, pid(tid)) for s, e, tid, text in nvtx if text in steps]
    sync_ns = 0
    if has_table(db, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        for start, end, tid, api in db.execute(
            "SELECT r.start, r.end, r.globalTid, s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON r.nameId = s.id"
        ):
            if not ("Synchronize" in api or re.match(r"cudaMemcpy(_v\d+)?$", api)):
                continue
            if any(s <= start < e and p == pid(tid) for s, e, p in step_ranges):
                sync_ns += end - start
    # Divide by processes too: FSDP has one rank per GPU, each with its own syncs.
    num_procs = max(len({p for _, _, p in step_ranges}), 1)
    result["cpu_sync_ms_per_step"] = sync_ns / num_steps / num_procs / 1e6
    return result


def print_result(r):
    print(f"== {r['name']}  ({r['num_steps']} captured steps)")
    for device, g in r["gpus"].items():
        line = (
            f"  GPU {device}: step {g['step_ms']:.1f} ms | idle {g['idle_pct']:.1f}% | "
            f"gaps>50us {g['num_gaps_over_50us_per_step']:.1f}/step = {g['gap_ms_per_step']:.1f} ms | "
            f"{g['kernels_per_step']:.0f} kernels/step, median {g['median_kernel_us']:.1f} us"
        )
        if g["nccl_overlap_pct"] is not None:
            line += f" | NCCL {g['nccl_ms_per_step']:.1f} ms/step, {g['nccl_overlap_pct']:.0f}% overlapped"
        print(line)
    phases = ", ".join(f"{k} {v:.1f}" for k, v in r["cpu_phase_ms"].items())
    print(f"  CPU ms per step: {phases} | blocked in sync {r['cpu_sync_ms_per_step']:.1f}")


def summary(paths):
    rows = [json.load(open(p)) for p in paths]
    print("| scenario | GPU step (ms) | GPU idle | kernels/step | median kernel | CPU blocked in sync (ms/step) | NCCL overlapped |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        g = r["gpus"]["0"] if "0" in r["gpus"] else next(iter(r["gpus"].values()))
        nccl = f"{g['nccl_overlap_pct']:.0f}%" if g["nccl_overlap_pct"] is not None else "-"
        print(
            f"| {r['name']} | {g['step_ms']:.1f} | {g['idle_pct']:.1f}% | {g['kernels_per_step']:.0f} | "
            f"{g['median_kernel_us']:.1f} us | {r['cpu_sync_ms_per_step']:.1f} | {nccl} |"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--name")
    parser.add_argument("--json")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        summary(args.paths)
        return
    r = analyze(args.paths[0], args.name or args.paths[0])
    print_result(r)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(r, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
