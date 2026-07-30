"""Compare metrics across multiple chain CSVs.

Groups rows by `(dataset, ranks, num_workers)` and reports mean ± std (n=N reps)
for each numeric metric. Useful for sanity-checking repeatability across the
three independent reps the chain script produces.

Usage:
    uv run python scripts/compare_runs.py run_a.csv run_b.csv run_c.csv
"""

import csv
import statistics
import sys
from collections import defaultdict

METRICS = ["wall_s", "startup_s", "sps_wall", "b_med", "b_mean", "b_p95"]


def main() -> None:
    csv_files = sys.argv[1:]
    grouped: dict[tuple[str, int, int], list[dict[str, float]]] = defaultdict(list)

    for path in csv_files:
        with open(path) as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = (row["dataset"], int(row["ranks"]), int(row["num_workers"]))
                grouped[key].append({metric: float(row[metric]) for metric in METRICS})

    header = f"{'dataset':>7} {'cfg':>9} {'n':>3}"
    for metric in METRICS:
        header += f" {metric + '(mean±std)':>22}"
    print(header)
    print("-" * len(header))

    for (dataset, ranks, workers), runs in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        n = len(runs)
        row = f"{dataset:>7} {ranks}r×{workers:<3} {n:>3}"

        for metric in METRICS:
            values = [run[metric] for run in runs]
            mean = statistics.mean(values)
            std = statistics.stdev(values) if n > 1 else 0.0
            row += f" {mean:>10.2f}±{std:<10.2f}"

        print(row)


if __name__ == "__main__":
    main()
