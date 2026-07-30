"""Multi-worker DataLoader throughput (samples/s) per arm, across the throughput probes.

Each arm is ``<loader>_w<workers>``. The end-to-end raw/merged speedup measured here is far smaller
than the single-thread sample-timing speedup, because with many workers the merged loader is already
near the GPU-feed ceiling — so the question these probes answer is whether the clean (Run 45,
serial) reps reproduce the contended 42419688 numbers.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from analysis.common import load_throughput


JOBS = [
    ("42419688", "contended"),
    ("42441487", "clean"),
    ("42441488", "clean"),
]


def render() -> None:
    measured = {jid: load_throughput(jid) for jid, _ in JOBS}
    arms = _ordered_arms(measured)

    print("=" * 96)
    print("THROUGHPUT — steady-state samples/s by loader/worker arm")
    print("=" * 96)
    header = f"{'arm':>14}" + "".join(f"{jid:>12}" for jid, _ in JOBS) + f"{'clean_mean':>12}"
    print(header)
    print(f"{'(tag)':>14}" + "".join(f"{tag:>12}" for _, tag in JOBS))
    print("-" * 96)

    for arm in arms:
        cells = ""

        for jid, _ in JOBS:
            value = measured[jid].get(arm)
            cells += f"{value.sps:>12.1f}" if value else f"{'-':>12}"

        clean = [
            measured[jid][arm].sps for jid, tag in JOBS if tag == "clean" and arm in measured[jid]
        ]
        clean_mean = f"{statistics.mean(clean):>12.1f}" if clean else f"{'-':>12}"
        print(f"{arm:>14}{cells}{clean_mean}")

    _render_speedup(measured)


def _ordered_arms(measured: dict) -> list[str]:
    seen = []

    for arms in measured.values():
        for arm in arms:
            if arm not in seen:
                seen.append(arm)

    return seen


def _render_speedup(measured: dict) -> None:
    clean = [jid for jid, tag in JOBS if tag == "clean"]
    paired: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for jid in clean:
        for arm, value in measured[jid].items():
            loader = "raw" if arm.startswith("raw") else "merged"
            paired[value.workers][loader].append(value.sps)

    print()
    print("clean raw/merged throughput ratio (mean over clean reps):")

    for workers in sorted(paired):
        merged = paired[workers].get("merged")
        raw = paired[workers].get("raw")

        if not merged or not raw:
            continue

        ratio = statistics.mean(raw) / statistics.mean(merged)
        print(
            f"  w{workers}:  merged {statistics.mean(merged):6.1f}  ->  "
            f"raw {statistics.mean(raw):6.1f}  =  {ratio:.2f}x"
        )


def main() -> None:
    render()


if __name__ == "__main__":
    main()
