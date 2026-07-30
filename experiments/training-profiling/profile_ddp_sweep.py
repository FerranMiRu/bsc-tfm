"""Worker scaling sweep under DDP: SingleZarrDataset vs MergedZarrDataset.

Launched via `accelerate launch --multi_gpu` to match production
(`scripts/acc_training.sh`). Each accelerate sub-process reads
`RANK`, `WORLD_SIZE`, `LOCAL_RANK` env vars (set by accelerate from
SLURM_PROCID x GPUS_PER_NODE) and times its own DataLoader iterating
`DistributedSampler(num_replicas=world_size, rank=rank)`.

No collective ops — `DistributedSampler` works with explicit
`num_replicas` / `rank`, so we don't need to call
`init_process_group`. Aggregation is post-processing (sum per-rank
throughputs at each (dataset, num_workers) config).

Sweep: WORKERS_SWEEP x {single, merged}. Per-rank CSV at
`jobs-map/${SLURM_JOB_ID}/ddp_sweep_rank{r}.csv`.
Columns: dataset,num_workers,batch_idx,wall_ms.
"""

import os
import tempfile
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ai4land.utils.config_utils import DataParams
from ai4land.utils.datasets import MergedZarrDataset, SingleZarrDataset


CONFIG = Path(os.environ.get("SLURM_SUBMIT_DIR", ".")) / "inputs/debug.yaml"
MERGED_STORE = "/gpfs/scratch/bsc32/bsc096444/AI4LAND_merged-NON_preprocessed-data-2000-2015.zarr"

NUM_BATCHES = 40
BATCH_SIZE = 8
WORKERS_SWEEP = (1, 4, 8, 12, 20)
PREFETCH_FACTOR = 4
SEED = 42

DATASET_CLASSES = {"single": SingleZarrDataset, "merged": MergedZarrDataset}


def build_dataset(name, raw_data, store_path, lock_subdir):
    lock_dir = Path(tempfile.mkdtemp(prefix="ai4land_ddp_"))
    cfg_data = dict(raw_data)
    cfg_data["lock_dir"] = lock_dir / lock_subdir
    cfg_data["stores"] = store_path
    cfg_data["preprocessed"] = False
    cfg_data["debug_sample_count"] = None
    dp = DataParams(**cfg_data)
    return DATASET_CLASSES[name](cfg=dp, land_index_path=dp.train_land_index_path)


def run_config(dataset, num_workers, world_size, rank):
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED
        )
        sampler.set_epoch(0)
        if num_workers > 0:
            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=False,
                prefetch_factor=PREFETCH_FACTOR,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                sampler=sampler,
                num_workers=0,
                pin_memory=True,
                persistent_workers=False,
            )
    else:
        if num_workers > 0:
            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=False,
                prefetch_factor=PREFETCH_FACTOR,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                persistent_workers=False,
            )

    iterator = iter(loader)
    times = []
    for _ in range(NUM_BATCHES):
        start = time.perf_counter()
        next(iterator)
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def main():
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    torch.manual_seed(SEED + rank)

    raw = yaml.safe_load(CONFIG.open())
    raw_data = raw["data"]

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    submit_dir = os.environ.get("SLURM_SUBMIT_DIR", ".")
    csv_path = Path(submit_dir) / "jobs-map" / job_id / f"ddp_sweep_rank{rank}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w") as f:
        f.write("dataset,num_workers,batch_idx,wall_ms\n")

        for name, store in [("single", raw_data["stores"]), ("merged", MERGED_STORE)]:
            for nw in WORKERS_SWEEP:
                dataset = build_dataset(name, raw_data, store, f"sweep_{name}_w{nw}_r{rank}")
                times = run_config(dataset, nw, world_size, rank)
                for i, t in enumerate(times):
                    f.write(f"{name},{nw},{i},{t:.2f}\n")
                f.flush()
                del dataset


if __name__ == "__main__":
    main()
