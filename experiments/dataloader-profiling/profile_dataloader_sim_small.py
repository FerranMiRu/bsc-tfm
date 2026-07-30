"""Smaller pools that fit in cache: 31K vs 1,350 inodes. ABA pattern.

Tests whether Run 16's 31x speedup was cache-fit-dominated. If both pools fit
trivially in dentry+page cache, only the per-sample file-count effect remains.

Store A: 23 subdirs x 1,350 files = 31,050 inodes (~2.8 GB total).
Store B: 1 subdir x 1,350 files (each ~2 MB) = ~2.7 GB total.
Sizes sampled from real Run 12 chunk distribution.
Read pattern: A1 -> B -> A2 (different seeds per pass).
"""

import csv
import multiprocessing
import random
import time
from pathlib import Path


BASE_DIR = Path("/gpfs/scratch/bsc32/bsc096444/dataloader_sim_small")
STORE_A = BASE_DIR / "store_A_23dirs"
STORE_B = BASE_DIR / "store_B_1dir"

CHUNK_DIST_CSV = Path(
    "/gpfs/scratch/bsc32/bsc096444/ai4land-tfm/jobs-map/41548701/chunk_sizes.csv"
)
EXCLUDE = {"time", "latitude", "longitude"}

N_SUBDIRS_A = 23
FILES_PER_SUBDIR_A = 1350
FILES_PER_SAMPLE_A = 40

N_FILES_B = 1350
FILES_PER_SAMPLE_B = 2
MERGE_FACTOR = 23

N_WORKERS = 12
SAMPLES_PER_WORKER = 100
SEED = 42


def load_chunk_sizes():
    sizes = []
    with open(CHUNK_DIST_CSV) as f:
        for row in csv.DictReader(f):
            if row["var"] not in EXCLUDE:
                sizes.append(int(row["size_bytes"]))
    return sizes


def create_store_a(chunk_sizes):
    if STORE_A.exists():
        return

    rng = random.Random(SEED)
    STORE_A.mkdir(parents=True)

    for d in range(N_SUBDIRS_A):
        sub = STORE_A / f"sub{d}"
        sub.mkdir()

        for i in range(FILES_PER_SUBDIR_A):
            size = rng.choice(chunk_sizes)
            (sub / f"f{i}").write_bytes(b"\0" * size)


def create_store_b(chunk_sizes):
    if STORE_B.exists():
        return

    rng = random.Random(SEED + 1)
    STORE_B.mkdir(parents=True)
    sub = STORE_B / "sub0"
    sub.mkdir()

    for i in range(N_FILES_B):
        size = sum(rng.choice(chunk_sizes) for _ in range(MERGE_FACTOR))
        (sub / f"f{i}").write_bytes(b"\0" * size)


def worker_a(args):
    worker_id, seed = args
    rng = random.Random(seed)
    results = []

    for sample in range(SAMPLES_PER_WORKER):
        start = time.perf_counter_ns()

        for _ in range(FILES_PER_SAMPLE_A):
            d = rng.randrange(N_SUBDIRS_A)
            i = rng.randrange(FILES_PER_SUBDIR_A)

            with open(STORE_A / f"sub{d}" / f"f{i}", "rb") as fh:
                fh.read()

        ms = (time.perf_counter_ns() - start) / 1e6
        results.append((worker_id, sample, ms))

    return results


def worker_b(args):
    worker_id, seed = args
    rng = random.Random(seed)
    results = []

    for sample in range(SAMPLES_PER_WORKER):
        start = time.perf_counter_ns()

        for _ in range(FILES_PER_SAMPLE_B):
            i = rng.randrange(N_FILES_B)
            with open(STORE_B / "sub0" / f"f{i}", "rb") as fh:
                fh.read()

        ms = (time.perf_counter_ns() - start) / 1e6
        results.append((worker_id, sample, ms))

    return results


def main():
    chunk_sizes = load_chunk_sizes()
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    create_store_a(chunk_sizes)
    create_store_b(chunk_sizes)

    with multiprocessing.Pool(N_WORKERS) as pool:
        a1 = pool.map(worker_a, [(i, SEED + 1000 + i) for i in range(N_WORKERS)])
        b = pool.map(worker_b, [(i, SEED + 2000 + i) for i in range(N_WORKERS)])
        a2 = pool.map(worker_a, [(i, SEED + 3000 + i) for i in range(N_WORKERS)])

    print("store,pass,worker,sample,wall_ms")

    for batch in a1:
        for w, s, ms in batch:
            print(f"A,1,{w},{s},{ms:.2f}")

    for batch in b:
        for w, s, ms in batch:
            print(f"B,1,{w},{s},{ms:.2f}")

    for batch in a2:
        for w, s, ms in batch:
            print(f"A,2,{w},{s},{ms:.2f}")


if __name__ == "__main__":
    main()
