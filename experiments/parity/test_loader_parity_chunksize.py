"""Loader parity probe: 512x512-chunked store vs 256x256-chunked store.

Both use SingleZarrDataset. Verifies the chunk-size move has no data-level
effect on what the dataloader serves.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml

from ai4land.utils.config_utils import DataParams
from ai4land.utils.datasets import SingleZarrDataset


TENSOR_KEYS = ("x_continuous", "x_kg", "x_static", "x_hilda_prior", "hilda_target")

STORE_512 = "/gpfs/scratch/ehpc736/data/AI4LAND_NON_preprocessed-data-2000-2015.zarr"
STORE_256 = "/gpfs/scratch/ehpc736/data/AI4LAND_NON_preprocessed-256x256-data-2000-2015.zarr"

CSV_COLUMNS = [
    "index",
    "tensor",
    "dataset",
    "shape",
    "dtype",
    "min",
    "max",
    "mean",
    "median",
    "std",
    "p25",
    "p75",
    "n_unique",
    "uniq",
]

UNIQ_CAP = 50


def build_single(raw_data: dict, lock_dir: Path, store_path: str, subdir: str) -> SingleZarrDataset:
    cfg_data = dict(raw_data)
    cfg_data["lock_dir"] = lock_dir / subdir
    cfg_data["stores"] = store_path
    cfg_data["preprocessed"] = False
    cfg_data["debug_sample_count"] = None
    dp = DataParams(**cfg_data)
    return SingleZarrDataset(cfg=dp, land_index_path=dp.train_land_index_path)


def print_data_vars(stores: dict[str, str]) -> None:
    print("\n--- data_vars ---")
    for label, path in stores.items():
        ds = xr.open_zarr(path, chunks={})
        print(f"{label}: {sorted(ds.data_vars)}")
        ds.close()


def sanity_check(datasets: dict[str, SingleZarrDataset]) -> None:
    print("\n--- sanity ---")
    ref_name = next(iter(datasets))
    ref = datasets[ref_name]
    print(
        f"reference={ref_name} len={len(ref)} "
        f"valid_time_indices={ref.valid_time_indices.tolist()} "
        f"hw=({ref.height},{ref.width})"
    )

    for name, ds in datasets.items():
        if name == ref_name:
            continue

        match = (
            len(ds) == len(ref)
            and np.array_equal(ds.valid_time_indices, ref.valid_time_indices)
            and np.array_equal(ds.land_pixel_coords, ref.land_pixel_coords)
            and (ds.height, ds.width) == (ref.height, ref.width)
        )
        print(f"  {name}: {'OK' if match else 'MISMATCH'}")


def summary_row(index: int, tensor_name: str, dataset_name: str, tensor: torch.Tensor) -> dict:
    arr = tensor.to(torch.float64).numpy().ravel()
    uniq = np.unique(arr)
    n_unique = len(uniq)

    return {
        "index": index,
        "tensor": tensor_name,
        "dataset": dataset_name,
        "shape": str(tuple(tensor.shape)),
        "dtype": str(tensor.dtype),
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "mean": float(arr.mean()) if arr.size else 0.0,
        "median": float(np.median(arr)) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "p25": float(np.percentile(arr, 25)) if arr.size else 0.0,
        "p75": float(np.percentile(arr, 75)) if arr.size else 0.0,
        "n_unique": n_unique,
        "uniq": uniq.tolist() if n_unique <= UNIQ_CAP else "",
    }


def write_csv(
    datasets: dict[str, SingleZarrDataset],
    indices: list[int],
    csv_path: Path,
) -> int:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for idx in indices:
            for ds_name, ds in datasets.items():
                sample = ds[idx]

                for tensor_idx, tensor_name in enumerate(TENSOR_KEYS):
                    writer.writerow(summary_row(idx, tensor_name, ds_name, sample[tensor_idx]))
                    rows_written += 1

    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="inputs/debug.yaml")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv-out", default=None)
    args = parser.parse_args()

    repo_root = Path(os.environ.get("SLURM_SUBMIT_DIR", Path(__file__).resolve().parent.parent))
    raw = yaml.safe_load((repo_root / args.config).open())
    raw_data = raw["data"]

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    csv_path = (
        Path(args.csv_out)
        if args.csv_out
        else repo_root / "logs" / "tests" / f"parity_chunksize_{job_id}.csv"
    )

    stores_for_listing = {"chunk_512": STORE_512, "chunk_256": STORE_256}
    print_data_vars(stores_for_listing)

    lock_dir = Path(tempfile.mkdtemp(prefix="ai4land_chunksize_parity_"))
    ds_512 = build_single(raw_data, lock_dir, STORE_512, "chunk_512")
    ds_256 = build_single(raw_data, lock_dir, STORE_256, "chunk_256")

    datasets: dict[str, SingleZarrDataset] = {"chunk_512": ds_512, "chunk_256": ds_256}
    sanity_check(datasets)

    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(ds_512), size=args.num_samples).tolist()

    print(f"\n--- writing CSV to {csv_path} ---")
    rows = write_csv(datasets, indices, csv_path)
    print(f"wrote {rows} rows")


if __name__ == "__main__":
    main()
