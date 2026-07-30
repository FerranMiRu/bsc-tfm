"""Cold/warm decomposition + LRU miss-rate model from a cache_miss.csv trace.

Reads the per-chunk-read latency trace (arm,store,sample_idx,chunk_key,wall_us) and, per arm:

  1. classifies every read cold vs warm by a latency threshold (bimodal valley),
  2. reports the miss rate m, mean cold/warm cost, per-sample read time, and the warming
     trajectory (first-quartile vs last-quartile samples),
  3. runs an LRU cache simulation over the real access sequence, sweeping capacity (in files),
     to predict m independently of timing.

The verification: at the matched 14-yr window, do single_small and single_big touch the same N
chunk files per sample yet differ in measured miss rate m and cold cost -- which would pin the
store-size effect on the inode/dentry cache-miss rate alone.

Usage:
  python tests/analyze_cache_miss.py path/to/cache_miss.csv
"""

import csv
import statistics
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path


COLD_US = 1000.0
CAPACITIES = [10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000]
HIST_EDGES = [100, 500, 2_000, 10_000, 50_000]


def load_rows(path):
    rows = []
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            rows.append(
                (
                    row["arm"],
                    row["store"],
                    int(row["sample_idx"]),
                    row["chunk_key"],
                    float(row["wall_us"]),
                )
            )
    return rows


def histogram(latencies):
    counts = [0] * (len(HIST_EDGES) + 1)

    for value in latencies:
        bucket = len(HIST_EDGES)

        for i, edge in enumerate(HIST_EDGES):
            if value < edge:
                bucket = i
                break

        counts[bucket] += 1

    labels = [f"<{HIST_EDGES[0]}"]
    labels += [f"{HIST_EDGES[i - 1]}-{HIST_EDGES[i]}" for i in range(1, len(HIST_EDGES))]
    labels += [f">{HIST_EDGES[-1]}"]
    return list(zip(labels, counts, strict=True))


def summarize(latencies):
    cold = [value for value in latencies if value >= COLD_US]
    warm = [value for value in latencies if value < COLD_US]
    n = len(latencies)
    return {
        "n": n,
        "m": len(cold) / n if n else 0.0,
        "cold_mean": statistics.mean(cold) if cold else 0.0,
        "warm_mean": statistics.mean(warm) if warm else 0.0,
        "p50": statistics.median(latencies) if n else 0.0,
    }


def per_sample_read_ms(rows):
    by_sample = defaultdict(float)

    for _arm, _store, sample_idx, _key, wall_us in rows:
        by_sample[sample_idx] += wall_us

    return [by_sample[s] / 1000.0 for s in sorted(by_sample)]


def lru_miss_rate(file_ids, capacity):
    cache = OrderedDict()
    misses = 0

    for file_id in file_ids:
        if file_id in cache:
            cache.move_to_end(file_id)
            continue

        misses += 1
        cache[file_id] = True

        if len(cache) > capacity:
            cache.popitem(last=False)

    return misses / len(file_ids) if file_ids else 0.0


def report_arm(arm, rows) -> dict:
    print(f"\n{'=' * 72}\nARM: {arm}   (reads={len(rows)})\n{'=' * 72}")

    latencies = [wall_us for *_, wall_us in rows]
    summary = summarize(latencies)
    print(
        f"  miss rate m = {summary['m']:.3f}   cold_mean = {summary['cold_mean'] / 1000:.2f} ms"
        f"   warm_mean = {summary['warm_mean'] / 1000:.3f} ms   p50 = {summary['p50']:.1f} us"
    )

    print("  latency histogram (us):")
    for label, count in histogram(latencies):
        print(f"    {label:>14} : {count:>7}  ({100 * count / len(latencies):5.1f}%)")

    samples = per_sample_read_ms(rows)
    quartile = max(1, len(samples) // 4)
    first_q = statistics.mean(samples[:quartile])
    last_q = statistics.mean(samples[-quartile:])
    print(
        f"  per-sample read: median {statistics.median(samples):.1f} ms"
        f"   first-quartile {first_q:.1f} ms -> last-quartile {last_q:.1f} ms"
        f"   (warming x{first_q / last_q:.2f})"
    )

    n_per_sample = len(rows) / len(samples)
    cold_ms = summary["cold_mean"] / 1000.0
    warm_ms = summary["warm_mean"] / 1000.0
    m = summary["m"]
    predicted = n_per_sample * (m * cold_ms + (1 - m) * warm_ms)
    print(
        f"  analytical N*[m c + (1-m) w] = {n_per_sample:.1f} * "
        f"[{m:.3f}*{cold_ms:.2f} + {1 - m:.3f}*{warm_ms:.3f}] = {predicted:.1f} ms/sample"
    )

    file_ids = [f"{store}/{key}" for _arm, store, _s, key, _us in rows]
    distinct = len(set(file_ids))
    print(f"  LRU sim (distinct files touched = {distinct:,}):")
    for capacity in CAPACITIES:
        print(
            f"    capacity {capacity:>9,} files -> m_sim = {lru_miss_rate(file_ids, capacity):.3f}"
        )

    return {
        "m": m,
        "cold_ms": cold_ms,
        "n_per_sample": n_per_sample,
        "distinct": distinct,
        "per_sample_median": statistics.median(samples),
    }


def report_comparison(stats) -> None:
    if "single_small" not in stats or "single_big" not in stats:
        return

    small = stats["single_small"]
    big = stats["single_big"]

    def ratio(numerator, denominator) -> str:
        return f"{numerator / denominator:.2f}" if denominator else "n/a"

    print(f"\n{'#' * 72}")
    print("MATCHED-WINDOW CONTROL: single_small vs single_big (same 14-yr window)")
    print(f"{'#' * 72}")
    print(
        f"  files/sample      small={small['n_per_sample']:.1f}   "
        f"big={big['n_per_sample']:.1f}   (expect equal)"
    )
    print(f"  distinct touched  small={small['distinct']:,}   big={big['distinct']:,}")
    print(
        f"  miss rate m       small={small['m']:.3f}   big={big['m']:.3f}   "
        f"ratio={ratio(big['m'], small['m'])}"
    )
    print(
        f"  cold_mean ms      small={small['cold_ms']:.2f}   big={big['cold_ms']:.2f}   "
        f"ratio={ratio(big['cold_ms'], small['cold_ms'])}"
    )
    print(
        f"  per-sample median small={small['per_sample_median']:.1f} ms   "
        f"big={big['per_sample_median']:.1f} ms   "
        f"ratio={ratio(big['per_sample_median'], small['per_sample_median'])}"
    )


def main() -> None:
    rows = load_rows(sys.argv[1])

    rows_by_arm = OrderedDict()
    for row in rows:
        rows_by_arm.setdefault(row[0], []).append(row)

    stats = {}
    for arm, arm_rows in rows_by_arm.items():
        stats[arm] = report_arm(arm, arm_rows)

    report_comparison(stats)


if __name__ == "__main__":
    main()
