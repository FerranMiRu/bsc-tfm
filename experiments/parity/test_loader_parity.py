"""Loader parity probe: MultiZarrDataset (canonical baseline) vs every consolidated store.

The original per-modality multizarr layout is the ground truth. For each random patch this
compares each candidate single/merged store element-wise against the baseline and dumps one row
per (index, tensor, candidate) to CSV. Verdict is computed in analyze_parity.py.

The baseline reads the 5 separate CONCERTO stores (StoreParams, HILDA var `hilda_labels`); the
candidates read a single bundled / variable-merged store (lowercased vars, HILDA `lulc_states`).
Both are driven at the same time_range and land-index file, so sample ordering is identical and an
element-wise comparison is meaningful.
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml

from ai4land.utils.config_utils import DataParams
from ai4land.utils.datasets import MergedZarrDataset, MultiZarrDataset, SingleZarrDataset


TENSOR_KEYS = ("x_continuous", "x_kg", "x_static", "x_hilda_prior", "hilda_target")

NUM_SAMPLES = 64
SEED = 0
TIME_RANGE = [2001, 2010]

MULTI_CONFIG = "inputs/profiling.yaml"
CANDIDATE_CONFIG = "inputs/debug-single.yaml"

DATA_ROOT = "/gpfs/scratch/ehpc736/data"

CANDIDATES = [
    (
        "single_small",
        SingleZarrDataset,
        f"{DATA_ROOT}/AI4LAND_NON_preprocessed-data-2000-2015.zarr",
        False,
    ),
    (
        "single_big",
        SingleZarrDataset,
        f"{DATA_ROOT}/AI4LAND_NON_preprocessed-data-1901-2015.zarr",
        False,
    ),
    (
        "single_256",
        SingleZarrDataset,
        f"{DATA_ROOT}/AI4LAND_NON_preprocessed-256x256-data-2000-2015.zarr",
        False,
    ),
    (
        "merged_small",
        MergedZarrDataset,
        f"{DATA_ROOT}/AI4LAND_merged-NON_preprocessed-data-2000-2015.zarr",
        False,
    ),
    (
        "merged_big",
        MergedZarrDataset,
        f"{DATA_ROOT}/AI4LAND_merged_NON_preprocessed-data-1901-2015.zarr",
        False,
    ),
    (
        "preprocessed_small",
        SingleZarrDataset,
        f"{DATA_ROOT}/AI4LAND_preprocessed-data-2000-2015.zarr",
        True,
    ),
]

CSV_COLUMNS = [
    "candidate",
    "index",
    "tensor",
    "base_shape",
    "cand_shape",
    "shape_match",
    "dtype_match",
    "n_elems",
    "n_mismatch",
    "max_abs_diff",
    "exact_equal",
]


def base_cfg(config_path: Path, lock_dir: Path, **overrides) -> DataParams:
    raw = yaml.safe_load(config_path.open())
    cfg_data = dict(raw["data"])
    cfg_data["lock_dir"] = lock_dir
    cfg_data["time_range"] = TIME_RANGE
    cfg_data["debug_sample_count"] = None
    cfg_data.update(overrides)
    return DataParams(**cfg_data)


def build_baseline(repo_root: Path, lock_root: Path) -> MultiZarrDataset:
    dp = base_cfg(repo_root / MULTI_CONFIG, lock_root / "multi", preprocessed=False)
    return MultiZarrDataset(cfg=dp, land_index_path=dp.train_land_index_path)


def build_candidate(repo_root, lock_root, label, dataset_class, store_path, preprocessed):
    dp = base_cfg(
        repo_root / CANDIDATE_CONFIG,
        lock_root / label,
        stores=store_path,
        preprocessed=preprocessed,
    )
    return dataset_class(cfg=dp, land_index_path=dp.train_land_index_path)


def print_data_vars(repo_root: Path) -> None:
    print("\n--- data_vars ---")
    multi_raw = yaml.safe_load((repo_root / MULTI_CONFIG).open())
    for name, path in multi_raw["data"]["stores"].items():
        ds = xr.open_zarr(path, chunks={})
        print(f"  multi/{name:<14} {sorted(ds.data_vars)}")
        ds.close()

    for label, _cls, path, _pp in CANDIDATES:
        groups = ("dynamic", "static") if "merged" in label else (None,)
        for group in groups:
            ds = xr.open_zarr(path, group=group, chunks={})
            tag = f"{label}/{group}" if group else label
            if "merged" in label:
                print(
                    f"  {tag:<22} variables={[str(v) for v in ds['data']['variable'].to_numpy()]}"
                )
            else:
                print(f"  {tag:<22} {sorted(ds.data_vars)}")
            ds.close()


def sanity_check(baseline, candidates: dict) -> None:
    print("\n--- sanity (vs baseline multizarr) ---")
    print(
        f"  baseline       len={len(baseline)} "
        f"valid_time_indices={baseline.valid_time_indices.tolist()} "
        f"hw=({baseline.height},{baseline.width})"
    )
    for name, ds in candidates.items():
        match = (
            len(ds) == len(baseline)
            and np.array_equal(ds.valid_time_indices, baseline.valid_time_indices)
            and np.array_equal(ds.land_pixel_coords, baseline.land_pixel_coords)
            and (ds.height, ds.width) == (baseline.height, baseline.width)
        )
        print(f"  {name:<20} len={len(ds)} {'OK' if match else 'MISMATCH'}")


def compare_sample(index, label, base_sample, cand_sample) -> list[dict]:
    rows = []

    for tensor_name, base_tensor, cand_tensor in zip(
        TENSOR_KEYS, base_sample, cand_sample, strict=True
    ):
        base_f = base_tensor.to(torch.float64)
        cand_f = cand_tensor.to(torch.float64)

        shape_match = tuple(base_tensor.shape) == tuple(cand_tensor.shape)
        dtype_match = base_tensor.dtype == cand_tensor.dtype

        if shape_match:
            diff = (base_f - cand_f).abs()
            max_abs_diff = float(diff.max()) if diff.numel() else 0.0
            n_mismatch = int((diff > 0).sum())
            n_elems = diff.numel()
            exact_equal = n_mismatch == 0
        else:
            max_abs_diff = float("nan")
            n_mismatch = -1
            n_elems = base_f.numel()
            exact_equal = False

        rows.append(
            {
                "candidate": label,
                "index": index,
                "tensor": tensor_name,
                "base_shape": str(tuple(base_tensor.shape)),
                "cand_shape": str(tuple(cand_tensor.shape)),
                "shape_match": shape_match,
                "dtype_match": dtype_match,
                "n_elems": n_elems,
                "n_mismatch": n_mismatch,
                "max_abs_diff": max_abs_diff,
                "exact_equal": exact_equal,
            }
        )

    return rows


def main() -> None:
    repo_root = Path(os.environ.get("SLURM_SUBMIT_DIR", Path(__file__).resolve().parent.parent))
    lock_root = Path(tempfile.mkdtemp(prefix="ai4land_parity_lock_"))

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    csv_path = repo_root / "jobs-map" / job_id / f"parity_{job_id}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print_data_vars(repo_root)

    baseline = build_baseline(repo_root, lock_root)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(baseline), size=NUM_SAMPLES).tolist()

    candidates = {}
    for label, dataset_class, store_path, preprocessed in CANDIDATES:
        candidates[label] = build_candidate(
            repo_root, lock_root, label, dataset_class, store_path, preprocessed
        )

    sanity_check(baseline, candidates)

    print(f"\n--- writing diff CSV to {csv_path} ---")
    rows_written = 0

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for index in indices:
            base_sample = baseline[index]

            for label, dataset in candidates.items():
                for row in compare_sample(index, label, base_sample, dataset[index]):
                    writer.writerow(row)
                    rows_written += 1

    print(f"wrote {rows_written} rows")


if __name__ == "__main__":
    main()
