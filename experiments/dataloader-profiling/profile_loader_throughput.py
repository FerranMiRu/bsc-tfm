"""DataLoader steady-state throughput (samples/s) for the merged stores at training settings.

Single-rank loader (no DDP, no model) iterating the merged store with the production DataLoader
config (batch_size=8, prefetch_factor=4, persistent_workers=False, pin_memory=True). Skips warm-up
batches, then measures steady-state samples/s. The acceptance bar is >= 53 samples/s/gpu to keep an
H100 fed during the final training without stalling.

NOTE: single-rank on an exclusive node is OPTIMISTIC vs real 4-rank DDP (4 loaders share GPFS
metadata); Run 35 measured merged DDP scaling ~88-92%, so derate the per-rank number by ~0.88.

Stdout per arm: steady-state sps, per-batch p50/p95/max ms, and stall count (>1 s batches).
"""

from __future__ import annotations

import os
import statistics
import tempfile
import time
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from ai4land.utils.config_utils import DataParams
from ai4land.utils.datasets import MergedZarrDataset, RawMergedZarrDataset


SUBMIT = Path(os.environ.get("SLURM_SUBMIT_DIR", "."))
# Use the EXACT final-training config: store (merged-big), time_range, land-index files, features.
CONFIG = SUBMIT / "inputs/final-training.yaml"

BATCH_SIZE = 8
PREFETCH_FACTOR = 4
WARMUP_BATCHES = 30
MEASURE_BATCHES = 220
TARGET_SPS = 53.0

# (label, dataset_class, num_workers). xarray vs raw-zarr loader, same merged-big store + window.
ARMS = [
    ("merged_w12", MergedZarrDataset, 12),
    ("raw_w12", RawMergedZarrDataset, 12),
    ("merged_w16", MergedZarrDataset, 16),
    ("raw_w16", RawMergedZarrDataset, 16),
    ("merged_w20", MergedZarrDataset, 20),
    ("raw_w20", RawMergedZarrDataset, 20),
]


def build_dataset(dataset_class) -> MergedZarrDataset:
    raw = yaml.safe_load(CONFIG.open())
    cfg_data = dict(raw["data"])
    cfg_data["lock_dir"] = Path(tempfile.mkdtemp(prefix="ai4land_tp_")) / "lock"
    cfg_data["debug_sample_count"] = None
    dp = DataParams(**cfg_data)
    return dataset_class(cfg=dp, land_index_path=dp.train_land_index_path)


def measure(label: str, dataset_class, num_workers: int) -> None:
    dataset = build_dataset(dataset_class)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=False,
        pin_memory=True,
    )

    batch_times = []
    last = time.perf_counter()

    for i, _batch in enumerate(loader):
        now = time.perf_counter()

        if i >= WARMUP_BATCHES:
            batch_times.append((now - last) * 1000.0)

        last = now

        if i >= WARMUP_BATCHES + MEASURE_BATCHES:
            break

    elapsed = sum(batch_times) / 1000.0
    n_samples = len(batch_times) * BATCH_SIZE
    sps = n_samples / elapsed
    verdict = "OK" if sps >= TARGET_SPS else "BELOW TARGET"

    print(
        f"  {label:<18} workers={num_workers:<3} sps={sps:7.1f} "
        f"(target {TARGET_SPS:.0f}, ~{sps * 0.88:6.1f} at 4-rank) [{verdict}]"
    )
    print(
        f"      per-batch ms: p50={statistics.median(batch_times):7.1f} "
        f"p95={sorted(batch_times)[int(len(batch_times) * 0.95)]:8.1f} "
        f"max={max(batch_times):8.1f}  stalls(>1s)={sum(1 for t in batch_times if t > 1000)}"
    )


def main() -> None:
    print(f"--- DataLoader throughput (batch={BATCH_SIZE}, prefetch={PREFETCH_FACTOR}) ---")

    for label, dataset_class, num_workers in ARMS:
        measure(label, dataset_class, num_workers)


if __name__ == "__main__":
    main()
