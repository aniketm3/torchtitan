# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Summarize a PyTorch profiler trace: how busy the GPU is, and on what.

    python torchtitan_recipes/aleph_alpha_scaling/trace_summary.py <rank0_trace.json.gz> ...

Per profiled step, using the GPU-side step window (the ProfilerStep#N
annotation on the main CUDA stream, i.e. the stream with the most kernels):
- busy%: fraction of the window with at least one GPU kernel running (any
  stream). 100 - busy% is GPU idle time: the GPU waiting on the CPU.
- idle gaps: GPU-idle stretches > 20 us, the visible holes in Perfetto.
- CPU step: the same step on the CPU timeline. The CPU queues kernels ahead of
  the GPU, so this is not GPU time; if it is longer than the GPU step, the CPU
  is the bottleneck.
- GPU kernel time split into categories by kernel name (matmul, attention,
  elementwise/fused, optimizer, memory copies, other).
"""

import gzip
import json
import re
import sys
from collections import defaultdict

CATEGORIES = [
    ("attention", re.compile(r"flash|fmha|attn|sdpa", re.I)),
    ("norm", re.compile(r"layer_norm|rms_norm|GammaBeta", re.I)),
    ("matmul", re.compile(r"gemm|nvjet|cutlass|cublas|sm90_xmma|wgmma", re.I)),
    ("optimizer", re.compile(r"multi_tensor|foreach|adam", re.I)),
    ("triton (compiled)", re.compile(r"^triton_", re.I)),
    ("elementwise (eager)", re.compile(r"elementwise|vectorized|reduce|unrolled|index|cat|copy_kernel|fill", re.I)),
]


def category(name: str) -> str:
    for cat, pattern in CATEGORIES:
        if pattern.search(name):
            return cat
    return "other"


def union_length(intervals):
    total, end = 0.0, float("-inf")
    for s, e in sorted(intervals):
        if s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def summarize(path: str) -> None:
    with gzip.open(path) if path.endswith(".gz") else open(path) as f:
        events = json.load(f)["traceEvents"]
    kernels = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    stream_counts = defaultdict(int)
    for k in kernels:
        stream_counts[k["tid"]] += 1
    main_stream = max(stream_counts, key=stream_counts.get)
    gpu_steps = [
        e for e in events
        if e.get("cat") == "gpu_user_annotation" and e.get("tid") == main_stream
        and e.get("name", "").startswith("ProfilerStep#")
    ]
    cpu_steps = {
        e["name"]: e["dur"] for e in events
        if e.get("cat") == "user_annotation" and e.get("name", "").startswith("ProfilerStep#")
    }
    print(f"\n== {path.split('outputs/')[-1]}")
    print(f"  streams (kernel count): {dict(stream_counts)}; main stream = {main_stream}")
    for step in sorted(gpu_steps, key=lambda e: e["ts"]):
        s0, s1 = step["ts"], step["ts"] + step["dur"]
        ks = sorted((k for k in kernels if s0 <= k["ts"] < s1), key=lambda k: k["ts"])
        busy = union_length([(k["ts"], k["ts"] + k["dur"]) for k in ks])
        gaps, end = [], s0
        for k in ks:
            if k["ts"] - end > 20:
                gaps.append(k["ts"] - end)
            end = max(end, k["ts"] + k["dur"])
        by_cat = defaultdict(float)
        for k in ks:
            by_cat[category(k["name"]) if k["cat"] == "kernel" else "memcpy/memset"] += k["dur"]
        total = sum(by_cat.values())
        print(f"  {step['name']}: GPU step {step['dur']/1e3:7.1f} ms | CPU step "
              f"{cpu_steps.get(step['name'], 0)/1e3:7.1f} ms | GPU busy {100*busy/step['dur']:5.1f}% | "
              f"{len(gaps)} idle gaps >20us = {sum(gaps)/1e3:.1f} ms | {len(ks)} kernels")
        for cat, t in sorted(by_cat.items(), key=lambda x: -x[1]):
            print(f"      {cat:<22}{t/1e3:8.1f} ms  {100*t/total:5.1f}%")
    top = defaultdict(lambda: [0.0, 0])
    for k in kernels:
        top[k["name"][:80]][0] += k["dur"]
        top[k["name"][:80]][1] += 1
    print("  top kernels (both profiled steps):")
    for name, (t, n) in sorted(top.items(), key=lambda x: -x[1][0])[:8]:
        print(f"    {t/1e3:8.1f} ms  x{n:<5} [{category(name)}] {name}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
