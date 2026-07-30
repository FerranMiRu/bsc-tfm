"""Count zarr chunk-file opens per __getitem__ for SingleZarrDataset vs MergedZarrDataset.

Monkey-patches DirectoryStore.__getitem__ to log every chunk-data key fetched,
bucketed by variable. Iterates a fixed number of random land-pixel indices in
each dataset (single-process, no DataLoader), resetting the counter between
samples. CSV stdout: dataset,sample_idx,var,n_chunks.
"""

import collections
import os
import re
import sys
import tempfile
from pathlib import Path

import torch
import yaml
import zarr.storage

from ai4land.utils.config_utils import DataParams
from ai4land.utils.datasets import MergedZarrDataset, SingleZarrDataset


CONFIG = Path(os.environ.get("SLURM_SUBMIT_DIR", ".")) / "inputs/debug.yaml"
MERGED_STORE = "/gpfs/scratch/bsc32/bsc096444/AI4LAND_merged-NON_preprocessed-data-2000-2015.zarr"

NUM_SAMPLES = 1000
SEED = 42

DATASET_CLASSES = {"single": SingleZarrDataset, "merged": MergedZarrDataset}

_chunk_re = re.compile(r"^\d+(\.\d+)*$")
_counts = collections.Counter()


def _make_counting(orig_getitem):
    def _wrapped(self, key):
        parts = key.rsplit("/", 1)
        if len(parts) == 2 and _chunk_re.fullmatch(parts[1]):
            _counts[parts[0]] += 1
        return orig_getitem(self, key)

    return _wrapped


for cls_name in ("DirectoryStore", "FSStore", "NestedDirectoryStore"):
    cls = getattr(zarr.storage, cls_name, None)
    if cls is not None:
        cls.__getitem__ = _make_counting(cls.__getitem__)


def build_dataset(dataset_name, raw_data, store_path, lock_subdir):
    lock_dir = Path(tempfile.mkdtemp(prefix="ai4land_perf_"))
    cfg_data = dict(raw_data)
    cfg_data["lock_dir"] = lock_dir / lock_subdir
    cfg_data["stores"] = store_path
    cfg_data["preprocessed"] = False
    cfg_data["debug_sample_count"] = None
    dp = DataParams(**cfg_data)
    return DATASET_CLASSES[dataset_name](cfg=dp, land_index_path=dp.train_land_index_path)


def measure(dataset_name, store_path, raw_data):
    dataset = build_dataset(dataset_name, raw_data, store_path, f"opens_{dataset_name}")

    torch.manual_seed(SEED)
    indices = torch.randint(0, len(dataset), (NUM_SAMPLES,)).tolist()

    for i, idx in enumerate(indices):
        _counts.clear()
        dataset[idx]

        for var, n in _counts.items():
            print(f"{dataset_name},{i},{var},{n}")
        sys.stdout.flush()


def main():
    raw = yaml.safe_load(CONFIG.open())
    raw_data = raw["data"]

    print("dataset,sample_idx,var,n_chunks")

    measure("single", raw_data["stores"], raw_data)
    measure("merged", MERGED_STORE, raw_data)


if __name__ == "__main__":
    main()
