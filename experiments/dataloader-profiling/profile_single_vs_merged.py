"""DataLoader throughput: SingleZarrDataset vs MergedZarrDataset.

Mirrors profile_dataloader.py's structure (Phase 1 + Phase 3), no
per-modality timers and no TimedSingleZarrDataset subclass.

Phase 1: NUM_GETITEM single-thread __getitem__ samples per dataset.
Phase 2: num_workers sweep, NUM_BATCHES per (dataset, workers) config,
         each in a fresh spawn subprocess with SIGALRM per-batch watchdog
         and wall-budget force-kill (matches profile_dataloader.py).

Sweep is interleaved across datasets per worker count so time-of-day
GPFS fluctuations affect both equally.

CSV stdout: phase,dataset,num_workers,idx,wall_ms.
Phase 1 rows have num_workers=-1.
"""

import multiprocessing as mp
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ai4land.utils.config_utils import DataParams
from ai4land.utils.datasets import MergedZarrDataset, SingleZarrDataset


CONFIG = Path(os.environ.get("SLURM_SUBMIT_DIR", ".")) / "inputs/debug.yaml"
MERGED_STORE = "/gpfs/scratch/bsc32/bsc096444/AI4LAND_merged-NON_preprocessed-data-2000-2015.zarr"

NUM_GETITEM = 50
NUM_BATCHES = 80
BATCH_SIZE = 8
WORKERS_SWEEP = (0, 1, 4, 12)
PREFETCH_FACTOR = 4
BATCH_TIMEOUT_S = 30
WALL_BUDGET_S = 900
SEED = 42

DATASET_CLASSES = {"single": SingleZarrDataset, "merged": MergedZarrDataset}


def build_dataset(dataset_name, raw_data, store_path, lock_subdir):
    lock_dir = Path(tempfile.mkdtemp(prefix="ai4land_perf_"))
    cfg_data = dict(raw_data)
    cfg_data["lock_dir"] = lock_dir / lock_subdir
    cfg_data["stores"] = store_path
    cfg_data["preprocessed"] = False
    cfg_data["debug_sample_count"] = None
    dp = DataParams(**cfg_data)
    return DATASET_CLASSES[dataset_name](cfg=dp, land_index_path=dp.train_land_index_path)


def phase1_getitem(dataset_name, store_path, raw_data):
    dataset = build_dataset(dataset_name, raw_data, store_path, f"phase1_{dataset_name}")

    torch.manual_seed(SEED)
    indices = torch.randint(0, len(dataset), (NUM_GETITEM,)).tolist()

    for i, idx in enumerate(indices):
        start = time.perf_counter()
        dataset[idx]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(f"phase1,{dataset_name},-1,{i},{elapsed_ms:.2f}")
        sys.stdout.flush()


def _child_loader(dataset_name, store_path, raw_data, num_workers, queue):
    torch.manual_seed(SEED)
    dataset = build_dataset(
        dataset_name, raw_data, store_path, f"sweep_{dataset_name}_w{num_workers}"
    )

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

    def _alarm(_signum, _frame):
        raise TimeoutError

    signal.signal(signal.SIGALRM, _alarm)

    for i in range(NUM_BATCHES):
        signal.alarm(BATCH_TIMEOUT_S)
        start = time.perf_counter()
        try:
            next(iterator)
        except TimeoutError:
            signal.alarm(0)
            queue.put({"times": times, "stopped_at": i})
            return
        signal.alarm(0)
        times.append((time.perf_counter() - start) * 1000.0)

    queue.put({"times": times, "stopped_at": None})


def phase2_sweep_config(dataset_name, store_path, raw_data, num_workers):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    child = ctx.Process(
        target=_child_loader,
        args=(dataset_name, store_path, raw_data, num_workers, queue),
    )
    child.start()
    child.join(timeout=WALL_BUDGET_S)

    if child.is_alive():
        child.terminate()
        child.join(5)
        if child.is_alive():
            child.kill()
        return []

    result = queue.get(timeout=10)
    return result["times"]


def main():
    raw = yaml.safe_load(CONFIG.open())
    raw_data = raw["data"]

    stores = [("single", raw_data["stores"]), ("merged", MERGED_STORE)]

    print("phase,dataset,num_workers,idx,wall_ms")

    for name, store in stores:
        phase1_getitem(name, store, raw_data)

    for num_workers in WORKERS_SWEEP:
        for name, store in stores:
            times = phase2_sweep_config(name, store, raw_data, num_workers)

            for i, t in enumerate(times):
                print(f"phase2,{name},{num_workers},{i},{t:.2f}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
