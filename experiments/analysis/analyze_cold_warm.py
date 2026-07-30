"""Cold/warm decomposition + LRU miss-rate model from a cold_warm.csv trace.

Reads the per-chunk-read latency trace (arm,store,sample_idx,chunk_key,wall_us)
and, per arm:

  1. classifies every read cold vs warm by a latency threshold (bimodal valley),
  2. reports the miss rate m, mean cold/warm cost, per-sample read time, and the
     warming trajectory (first-quartile vs last-quartile samples),
  3. runs an LRU cache simulation over the real access sequence, sweeping the
     capacity (in files), to predict m independently of timing.

The verification: does a single capacity reproduce the measured m across arms,
and does N x [m c + (1-m) w] match the measured per-sample read time?
"""

import csv
import statistics
from collections import Counter, OrderedDict, defaultdict

CSV_PATH = "/Users/fmr/development/bsc/ai4land-tfm/jobs-map/42095663/cold_warm.csv"

COLD_US = 1000.0
CAPACITIES = [10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000]
HIST_EDGES = [100, 500, 2_000, 10_000, 50_000]

LUH2_VARS = {
    "c3ann", "c3nfx", "c3per", "c4ann", "c4per", "pastr", "primf",
    "primn", "range", "secdf", "secdn", "secma", "secmb", "urban",
}


def load_rows(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(
                (row["arm"], row["store"], int(row["sample_idx"]), row["chunk_key"], float(row["wall_us"]))
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
    return list(zip(labels, counts))


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

    ordered = [by_sample[s] / 1000.0 for s in sorted(by_sample)]
    return ordered


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


def report_arm(arm, rows):
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
    c = summary["cold_mean"] / 1000.0
    w = summary["warm_mean"] / 1000.0
    m = summary["m"]
    predicted = n_per_sample * (m * c + (1 - m) * w)
    print(
        f"  analytical N*[m c + (1-m) w] = {n_per_sample:.1f} * "
        f"[{m:.3f}*{c:.2f} + {1 - m:.3f}*{w:.3f}] = {predicted:.1f} ms/sample"
    )

    file_ids = [f"{store}/{key}" for _arm, store, _s, key, _us in rows]
    distinct = len(set(file_ids))
    print(f"  LRU sim (distinct files touched = {distinct:,}):")
    for capacity in CAPACITIES:
        print(f"    capacity {capacity:>9,} files -> m_sim = {lru_miss_rate(file_ids, capacity):.3f}")


def report_luh2(rows_by_arm):
    print(f"\n{'#' * 72}\nLUH2-only per-open (single bundled store vs multizarr luh2 store)\n{'#' * 72}")

    for arm, rows in rows_by_arm.items():
        luh2 = [wall_us for _a, _store, _s, key, wall_us in rows if key.split("/", 1)[0] in LUH2_VARS]

        if not luh2:
            continue

        summary = summarize(luh2)
        print(
            f"  {arm:>14}: reads={summary['n']:>6}  m={summary['m']:.3f}"
            f"  cold_mean={summary['cold_mean'] / 1000:.2f} ms  p50={summary['p50']:.1f} us"
        )


def main():
    rows = load_rows(CSV_PATH)

    rows_by_arm = OrderedDict()
    for row in rows:
        rows_by_arm.setdefault(row[0], []).append(row)

    store_counts = Counter((arm, store) for arm, store, *_ in rows)
    print("rows per (arm, store):")
    for (arm, store), count in store_counts.most_common():
        print(f"  {arm:>14}  {store:<50} {count:>8}")

    for arm, arm_rows in rows_by_arm.items():
        report_arm(arm, arm_rows)

    report_luh2(rows_by_arm)


if __name__ == "__main__":
    main()
