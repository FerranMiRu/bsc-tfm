"""Per-job training throughput + quality, grouped by loader/infra, with rep agreement.

A category is ``<nodes>r<workers>w`` (e.g. ``4r20w``); a group is ``<loader>/<category>``. Within
each group the reps should agree on steady-state per-batch time, stall rate, samples/s and best
validation loss. The summary row flags a group as agreeing when the coefficient of variation of
mean batch time stays under 10% and of best val under 3%.
"""

from __future__ import annotations

from collections import defaultdict

from analysis.common import TrainingRun, coefficient_of_variation, load_training_runs


MEAN_MS_TOLERANCE = 0.10
BEST_VAL_TOLERANCE = 0.03

HEADER = (
    f"{'job':>10} {'loader':>7} {'cat':>7} {'p50ms':>6} {'mean':>6} "
    f"{'p95':>6} {'stall%':>6} {'sps/gpu':>7} {'wall_h':>6} {'best_val':>8} {'ep':>3}  flag"
)


def _row(run: TrainingRun) -> str:
    wall = f"{run.wall_hours:.2f}" if run.wall_hours is not None else "-"
    best = f"{run.best_val:.4f}" if run.best_val is not None else "-"
    flag = "" if run.flag == "ok" else run.flag
    return (
        f"{run.jid:>10} {run.loader:>7} {run.category:>7} {run.p50_ms:>6.0f} "
        f"{run.mean_ms:>6.0f} {run.p95_ms:>6.0f} {run.stall_pct:>6.1f} "
        f"{run.sps_per_gpu:>7.1f} {wall:>6} {best:>8} {run.epochs:>3}  {flag}"
    )


def _agreement(runs: list[TrainingRun]) -> str:
    timing_reps = [run for run in runs if not run.degraded]
    quality_reps = [run for run in runs if run.complete]

    if len(timing_reps) < 2 and len(quality_reps) < 2:
        return "single healthy rep"

    mean_cv = coefficient_of_variation([run.mean_ms for run in timing_reps])
    val_cv = coefficient_of_variation([run.best_val for run in quality_reps])
    timing_ok = len(timing_reps) < 2 or mean_cv <= MEAN_MS_TOLERANCE
    quality_ok = len(quality_reps) < 2 or val_cv <= BEST_VAL_TOLERANCE
    verdict = "AGREE" if timing_ok and quality_ok else "CHECK"
    return (
        f"{verdict}  (timing {len(timing_reps)} reps cv={mean_cv * 100:.1f}%, "
        f"quality {len(quality_reps)} reps cv={val_cv * 100:.1f}%)"
    )


def render() -> None:
    runs = load_training_runs()
    groups: dict[str, list[TrainingRun]] = defaultdict(list)

    for run in runs:
        groups[run.group].append(run)

    print("=" * 96)
    print("TRAINING — throughput + quality by loader / infra")
    print("=" * 96)
    print(HEADER)

    for group in sorted(groups):
        reps = sorted(groups[group], key=lambda run: run.jid)
        print("-" * 96)

        for run in reps:
            print(_row(run))

        print(f"{'':>10} >> {group:<14} {len(reps)} reps  ->  {_agreement(reps)}")

    _render_quality_matrix(runs)


def _render_quality_matrix(runs: list[TrainingRun]) -> None:
    by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)

    for run in runs:
        if run.best_val is not None and run.complete:
            by_cell[(run.category, run.loader)].append(run.best_val)

    categories = sorted({category for category, _ in by_cell})

    print()
    print("=" * 60)
    print("QUALITY CROSS-CHECK — best val by category (merged vs raw)")
    print("(bit-identical data: the two loaders should match)")
    print("=" * 60)
    print(f"{'category':>10} {'merged':>16} {'raw':>16}")

    for category in categories:
        merged = by_cell.get((category, "merged"), [])
        raw = by_cell.get((category, "raw"), [])
        print(f"{category:>10} {_cell(merged):>16} {_cell(raw):>16}")


def _cell(values: list[float]) -> str:
    if not values:
        return "-"

    rendered = "/".join(f"{value:.3f}" for value in sorted(values))
    return rendered


def main() -> None:
    render()


if __name__ == "__main__":
    main()
