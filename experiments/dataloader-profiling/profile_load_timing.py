"""Single-thread load timing across store layouts: single (small/big), merged (small/big), multi.

Loads NUM_SAMPLES patches per arm (no DataLoader, no workers) and times each __getitem__.
Same seeded indices, same time_range, so all arms pull identical (coord, time) samples; the only
difference is the store layout. The merged_small vs merged_big pair tests whether the store-size
penalty (seen for single) also inflates the merged loader — i.e. whether merging still wins at the
full time range. CSV stdout: arm,sample_idx,wall_ms, followed by a per-arm summary (median is the
steady-state cost; first-10 vs last-10 shows cold->warm warming).
"""

from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
import yaml

from ai4land.utils.config_utils import DataParams
from ai4land.utils.datasets import (
    MergedZarrDataset,
    MultiZarrDataset,
    RawMergedZarrDataset,
    SingleZarrDataset,
)


SUBMIT = Path(os.environ.get("SLURM_SUBMIT_DIR", "."))
SINGLE_CONFIG = SUBMIT / "inputs/debug-single.yaml"
MULTI_CONFIG = SUBMIT / "inputs/profiling.yaml"

DATA_ROOT = "/gpfs/scratch/ehpc736/data"
TIME_RANGE = [2001, 2010]
NUM_SAMPLES = 100
SEED = 42

ARMS = [
    (
        "single_small",
        SingleZarrDataset,
        SINGLE_CONFIG,
        f"{DATA_ROOT}/AI4LAND_NON_preprocessed-data-2000-2015.zarr",
    ),
    (
        "single_big",
        SingleZarrDataset,
        SINGLE_CONFIG,
        f"{DATA_ROOT}/AI4LAND_NON_preprocessed-data-1901-2015.zarr",
    ),
    (
        "merged_small",
        MergedZarrDataset,
        SINGLE_CONFIG,
        f"{DATA_ROOT}/AI4LAND_merged-NON_preprocessed-data-2000-2015.zarr",
    ),
    (
        "merged_big",
        MergedZarrDataset,
        SINGLE_CONFIG,
        f"{DATA_ROOT}/AI4LAND_merged_NON_preprocessed-data-1901-2015.zarr",
    ),
    (
        "merged_big_raw",
        RawMergedZarrDataset,
        SINGLE_CONFIG,
        f"{DATA_ROOT}/AI4LAND_merged_NON_preprocessed-data-1901-2015.zarr",
    ),
    ("multi", MultiZarrDataset, MULTI_CONFIG, None),
]


def build_dataset(dataset_class, config_path: Path, store_path: str | None):
    raw = yaml.safe_load(config_path.open())
    cfg_data = dict(raw["data"])
    cfg_data["lock_dir"] = Path(tempfile.mkdtemp(prefix="ai4land_timing_")) / "lock"
    cfg_data["time_range"] = TIME_RANGE
    cfg_data["preprocessed"] = False
    cfg_data["debug_sample_count"] = None

    if store_path is not None:
        cfg_data["stores"] = store_path

    dp = DataParams(**cfg_data)
    return dataset_class(cfg=dp, land_index_path=dp.train_land_index_path)


def measure(arm, dataset_class, config_path, store_path) -> list[float]:
    dataset = build_dataset(dataset_class, config_path, store_path)

    torch.manual_seed(SEED)
    indices = torch.randint(0, len(dataset), (NUM_SAMPLES,)).tolist()

    per_sample_ms = []
    for sample_idx, idx in enumerate(indices):
        start = time.perf_counter_ns()
        dataset[idx]
        wall_ms = (time.perf_counter_ns() - start) / 1e6
        per_sample_ms.append(wall_ms)

        print(f"{arm},{sample_idx},{wall_ms:.3f}")
        sys.stdout.flush()

    return per_sample_ms


def summarize(arm, per_sample_ms) -> None:
    first10 = statistics.mean(per_sample_ms[:10])
    last10 = statistics.mean(per_sample_ms[-10:])
    print(
        f"  {arm:<14} median={statistics.median(per_sample_ms):8.1f} ms  "
        f"mean={statistics.mean(per_sample_ms):8.1f} ms  "
        f"min={min(per_sample_ms):7.1f}  max={max(per_sample_ms):8.1f}  "
        f"cold(first10)={first10:8.1f}  warm(last10)={last10:8.1f}"
    )


def main() -> None:
    print("arm,sample_idx,wall_ms")

    summaries = []
    for arm, dataset_class, config_path, store_path in ARMS:
        summaries.append((arm, measure(arm, dataset_class, config_path, store_path)))

    print("\n--- summary (ms per sample) ---")
    for arm, per_sample_ms in summaries:
        summarize(arm, per_sample_ms)


if __name__ == "__main__":
    main()
