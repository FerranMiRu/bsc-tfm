"""Single-thread per-sample read time (ms) per loader arm, across the sample-timing probes.

The headline is merged_big (xarray) vs merged_big_raw (no-xarray): the raw arm is the no-xarray
reader, the merged arm is the xarray path it replaces. The 42441485/86 probes are the clean
(serial, non-contended) Run 45 measurement; the earlier two ran alongside other jobs on the same
store, so their xarray numbers carry contention noise.
"""

from __future__ import annotations

import statistics

from analysis.common import load_sample_timing


JOBS = [
    ("42417946", "early"),
    ("42419687", "contended"),
    ("42441485", "clean"),
    ("42441486", "clean"),
]

ARMS = ["single_small", "single_big", "merged_small", "merged_big", "merged_big_raw", "multi"]


def render() -> None:
    measured = {jid: load_sample_timing(jid) for jid, _ in JOBS}

    print("=" * 96)
    print("SAMPLE TIMING — single-thread median ms/sample by arm")
    print("=" * 96)
    header = f"{'arm':>16}" + "".join(f"{jid:>12}" for jid, _ in JOBS) + f"{'clean_mean':>12}"
    print(header)
    print(f"{'(tag)':>16}" + "".join(f"{tag:>12}" for _, tag in JOBS))
    print("-" * 96)

    for arm in ARMS:
        cells = ""

        for jid, _ in JOBS:
            value = measured[jid].get(arm)
            cells += f"{value.median_ms:>12.1f}" if value else f"{'-':>12}"

        clean = [
            measured[jid][arm].median_ms
            for jid, tag in JOBS
            if tag == "clean" and arm in measured[jid]
        ]
        clean_mean = f"{statistics.mean(clean):>12.1f}" if clean else f"{'-':>12}"
        print(f"{arm:>16}{cells}{clean_mean}")

    _render_speedup(measured)


def _render_speedup(measured: dict) -> None:
    print()
    print("xarray -> raw speedup on the merged-big store (median ms):")

    for jid, tag in JOBS:
        arms = measured[jid]

        if "merged_big" not in arms or "merged_big_raw" not in arms:
            continue

        xarray = arms["merged_big"].median_ms
        raw = arms["merged_big_raw"].median_ms
        print(f"  {jid} ({tag:>9}):  {xarray:7.1f}  ->  {raw:5.1f}  =  {xarray / raw:.1f}x")


def main() -> None:
    render()


if __name__ == "__main__":
    main()
