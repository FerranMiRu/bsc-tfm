"""Parity: RawMergedZarrDataset (raw zarr) vs MergedZarrDataset (xarray) — identical outputs.

Builds both loaders on the final-training merged-big config, pulls the same seeded sample indices
from each (shared `_parse_index` -> same patch/time), and checks every output tensor matches:
exact for the categorical/int tensors, atol=1e-6 (NaN-equal) for the float tensors. CSV stdout:
idx,tensor,max_abs_diff,equal; then a per-tensor max-diff summary and an overall PASS/FAIL verdict.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from ai4land.utils.config_utils import DataParams
from ai4land.utils.datasets import MergedZarrDataset, RawMergedZarrDataset


SUBMIT = Path(os.environ.get("SLURM_SUBMIT_DIR", "."))
CONFIG = SUBMIT / "inputs/final-training.yaml"
N_SAMPLES = 100
SEED = 42
TENSORS = ["x_continuous", "x_kg", "x_static", "x_hilda", "targets"]


def build_dataset(dataset_class):
    raw = yaml.safe_load(CONFIG.open())
    cfg_data = dict(raw["data"])
    cfg_data["lock_dir"] = Path(tempfile.mkdtemp(prefix="ai4land_parity_")) / "lock"
    cfg_data["debug_sample_count"] = None
    dp = DataParams(**cfg_data)
    return dataset_class(cfg=dp, land_index_path=dp.train_land_index_path)


def main() -> None:
    xr_ds = build_dataset(MergedZarrDataset)
    raw_ds = build_dataset(RawMergedZarrDataset)

    torch.manual_seed(SEED)
    indices = torch.randint(0, len(xr_ds), (N_SAMPLES,)).tolist()

    print("idx,tensor,max_abs_diff,equal")
    max_diff = {name: 0.0 for name in TENSORS}
    all_equal = True

    for idx in indices:
        xr_sample = xr_ds[idx]
        raw_sample = raw_ds[idx]

        for name, xr_tensor, raw_tensor in zip(TENSORS, xr_sample, raw_sample, strict=True):
            a = xr_tensor.numpy().astype(np.float64)
            b = raw_tensor.numpy().astype(np.float64)
            diff = float(np.abs(a - b).max()) if a.size else 0.0
            equal = bool(np.array_equal(a, b) or np.allclose(a, b, atol=1e-6, equal_nan=True))

            max_diff[name] = max(max_diff[name], diff)
            all_equal = all_equal and equal

            print(f"{idx},{name},{diff:.3e},{equal}")

    print(f"\n--- per-tensor max |diff| over {N_SAMPLES} samples ---")
    for name in TENSORS:
        print(f"  {name:<14} max_abs_diff={max_diff[name]:.3e}")

    print(f"\nPARITY: {'PASS' if all_equal else 'FAIL'}")


if __name__ == "__main__":
    main()
