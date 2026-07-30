"""Recurrent-eval accuracy / macro-F1 by loader/infra, with rep agreement.

Each eval is linked to its training job (via the checkpoint path) so it inherits that job's
loader/category. Year 0 is the one-step prediction; Year 13 is the end of the 14-year recurrent
rollout, where error has accumulated. Reps within a group should agree; the merged and raw loaders
feed bit-identical data, so they should land in the same place too.
"""

from __future__ import annotations

from collections import defaultdict

from analysis.common import EvalRun, coefficient_of_variation, load_eval_runs, load_training_runs


LAST_YEAR = 13
HARD_CLASSES = ["Urban", "Water", "Grass/shrubland"]


def _group_lookup() -> dict[str, tuple[str, str, bool]]:
    return {run.jid: (run.loader, run.category, run.complete) for run in load_training_runs()}


def render() -> None:
    lookup = _group_lookup()
    groups: dict[str, list[tuple[EvalRun, bool]]] = defaultdict(list)

    for run in load_eval_runs():
        loader, category, complete = lookup.get(run.train_jid, ("?", f"{run.nodes}r?w", True))
        groups[f"{loader}/{category}"].append((run, complete))

    print("=" * 96)
    print("EVALUATION — recurrent accuracy / macro-F1 by loader / infra")
    print("=" * 96)
    print(
        f"{'eval':>10} {'train':>10} {'y0_acc':>7} {'y0_f1':>7} "
        f"{'y13_acc':>8} {'y13_f1':>7}  " + " ".join(f"{name[:9]:>9}" for name in HARD_CLASSES)
    )

    for group in sorted(groups):
        reps = sorted(groups[group], key=lambda pair: pair[0].jid)
        print("-" * 96)

        for run, complete in reps:
            print(_row(run, complete))

        valid = [run for run, complete in reps if complete]
        print(f"{'':>10} >> {group:<14} {len(valid)}/{len(reps)} valid  ->  {_agreement(valid)}")


def _row(run: EvalRun, complete: bool) -> str:
    classes = " ".join(f"{run.class_accuracy.get(name, 0.0):>9.3f}" for name in HARD_CLASSES)
    flag = "" if complete else "  <- incomplete train"
    y13_acc = run.year_accuracy.get(LAST_YEAR, 0.0)
    y13_f1 = run.year_macro_f1.get(LAST_YEAR, 0.0)
    return (
        f"{run.jid:>10} {run.train_jid:>10} "
        f"{run.year_accuracy.get(0, 0.0):>7.4f} {run.year_macro_f1.get(0, 0.0):>7.4f} "
        f"{y13_acc:>8.4f} {y13_f1:>7.4f}  "
        f"{classes}{flag}"
    )


def _agreement(reps: list[EvalRun]) -> str:
    if len(reps) < 2:
        return "single valid rep"

    year0 = [run.year_accuracy.get(0, 0.0) for run in reps]
    year13 = [run.year_accuracy.get(LAST_YEAR, 0.0) for run in reps]
    spread = max(year0) - min(year0)
    cv = coefficient_of_variation(year13)
    verdict = "AGREE" if spread <= 0.02 and cv <= 0.02 else "CHECK"
    return f"{verdict}  (y0 spread={spread * 100:.2f}pp, y13 cv={cv * 100:.2f}%)"


def main() -> None:
    render()


if __name__ == "__main__":
    main()
