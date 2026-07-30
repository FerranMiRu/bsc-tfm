"""Trim the highest and lowest repetition (by sps_wall) of each configuration
in 5reps_clean.csv and average the survivors. Emits a per-configuration summary
and LaTeX-ready rows for the scale-out table."""

import csv
from collections import defaultdict
from statistics import mean

SRC = "5reps_clean.csv"


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def group_key(row):
    return (row["dataset"], int(row["ranks"]), int(row["num_workers"]))


def trim_and_average(reps):
    """Drop the min and max sps_wall rep, average the rest. Falls back to the
    full set when fewer than three reps survive a config."""
    ordered = sorted(reps, key=lambda r: float(r["sps_wall"]))
    dropped_lo = ordered[0]["seed"]
    dropped_hi = ordered[-1]["seed"]

    kept = ordered[1:-1] if len(ordered) >= 3 else ordered
    if len(ordered) < 3:
        dropped_lo = dropped_hi = "none (n<3)"

    summary = {
        "n": len(ordered),
        "n_kept": len(kept),
        "sps": mean(float(r["sps_wall"]) for r in kept),
        "wall_s": mean(float(r["wall_s"]) for r in kept),
        "startup_s": mean(float(r["startup_s"]) for r in kept),
        "dropped_lo": dropped_lo,
        "dropped_hi": dropped_hi,
    }
    return summary


def main():
    groups = defaultdict(list)
    for row in load_rows(SRC):
        groups[group_key(row)].append(row)

    results = {key: trim_and_average(reps) for key, reps in groups.items()}

    print(f"{'dataset':7} {'rk':>2} {'nw':>2} {'n':>2} "
          f"{'sps':>7} {'wall_s':>7} {'startup_s':>9}   dropped lo/hi")
    for key in sorted(results):
        dataset, ranks, nw = key
        s = results[key]
        print(f"{dataset:7} {ranks:>2} {nw:>2} {s['n']:>2} "
              f"{s['sps']:>7.1f} {s['wall_s']:>7.0f} {s['startup_s']:>9.0f}"
              f"   {s['dropped_lo']}/{s['dropped_hi']}")

    print("\n% LaTeX rows: workers, then single sps/wall/startup, merged sps/wall/startup")
    for nw in (8, 12, 20):
        for ranks in (1, 4, 8):
            cells = []
            for dataset in ("single", "merged"):
                s = results.get((dataset, ranks, nw))
                if s is None:
                    cells += ["--", "--", "--"]
                else:
                    cells += [f"{s['sps']:.0f}", f"{s['wall_s']:.0f}", f"{s['startup_s']:.0f}"]
            print(f"{ranks} & {nw} & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
