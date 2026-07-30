"""Run every analyzer in one pass: training, sample timing, throughput, evaluation.

Usage: ``python -m analysis.report`` from ``bsc-tfm/experiments/`` (needs Python >= 3.10; stdlib
only). Reads the job logs under ``runs/final-training/``.
"""

from __future__ import annotations

from analysis import evaluation, sample_timing, throughput, training


def main() -> None:
    for section in (training, sample_timing, throughput, evaluation):
        section.render()
        print()


if __name__ == "__main__":
    main()
