"""Verdict for a parity diff CSV produced by tests/test_loader_parity.py.

Per candidate store, reports how its tensors compare element-wise to the multizarr baseline:
overall max abs diff, fraction of (index, tensor) pairs that are bit-exact, any shape/dtype
mismatch, and a per-tensor breakdown. A final verdict per candidate: EXACT (max diff 0), APPROX
(within ATOL, e.g. float-rounding in baked normalization), or MISMATCH.

Usage:
  python tests/analyze_parity.py path/to/parity_<jobid>.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ATOL = 1e-4


def verdict(max_diff: float, any_shape_mismatch: bool) -> str:
    if any_shape_mismatch:
        return "MISMATCH (shape)"
    if max_diff == 0.0:
        return "EXACT"
    if max_diff <= ATOL:
        return f"APPROX (<= {ATOL:g})"
    return "MISMATCH"


def report_candidate(label: str, sub: pd.DataFrame) -> None:
    any_shape_mismatch = (~sub["shape_match"]).any()
    any_dtype_mismatch = (~sub["dtype_match"]).any()
    max_diff = float(sub["max_abs_diff"].max(skipna=True))
    exact_frac = sub["exact_equal"].mean()

    print(f"\n=== {label} ===")
    print(f"  verdict: {verdict(max_diff, any_shape_mismatch)}")
    print(
        f"  comparisons={len(sub)}  exact={sub['exact_equal'].sum()}/{len(sub)} "
        f"({100 * exact_frac:.1f}%)  max_abs_diff={max_diff:.3e}  "
        f"dtype_mismatch={'yes' if any_dtype_mismatch else 'no'}"
    )

    print(f"  {'tensor':<16}{'max_abs_diff':<16}{'exact':<12}{'shape':<10}{'dtype':<8}")
    for tensor in sorted(sub["tensor"].unique()):
        tdf = sub[sub.tensor == tensor]
        t_max = float(tdf["max_abs_diff"].max(skipna=True))
        t_exact = f"{tdf['exact_equal'].sum()}/{len(tdf)}"
        t_shape = "ok" if tdf["shape_match"].all() else "BAD"
        t_dtype = "ok" if tdf["dtype_match"].all() else "BAD"
        print(f"  {tensor:<16}{t_max:<16.3e}{t_exact:<12}{t_shape:<10}{t_dtype:<8}")


def main() -> int:
    csv_path = Path(sys.argv[1])
    df = pd.read_csv(csv_path)

    print(f"--- {csv_path}")
    print(
        f"rows={len(df)}  candidates={sorted(df.candidate.unique())}  "
        f"samples={df['index'].nunique()}  tensors={sorted(df.tensor.unique())}"
    )

    for label in df["candidate"].unique():
        report_candidate(label, df[df.candidate == label])

    return 0


if __name__ == "__main__":
    sys.exit(main())
