"""Simulate dataloader reads under 12-process concurrency on GPFS.

Store A (current zarr layout):  23 subdirs x 13K files, sizes sampled from
    the real Run 12 chunk-size distribution. Sample = 40 random file reads
    across random subdirs.
Store B (merged-variables):     1 subdir x 15K files, each = byte-sum of 20
    small samples (~2 MB each). Sample = 2 random file reads.
Both ~30 GB total. 12 multiprocessing workers, 100 samples each per store.

Stores are kept across runs (manual cleanup of BASE_DIR).
Caveat: A-then-B order biases B toward colder reads (A's 48K reads evict
B's creation residue). Re-run B alone for a tighter B measurement.
"""

import csv
import multiprocessing
import random
import time
from pathlib import Path


BASE_DIR = Path("/gpfs/scratch/bsc32/bsc096444/dataloader_sim")
STORE_A = BASE_DIR / "store_A_23dirs"
STORE_B = BASE_DIR / "store_B_1dir"

CHUNK_DIST_CSV = Path("/gpfs/scratch/bsc32/bsc096444/ai4land-tfm/jobs-map/41548701/chunk_sizes.csv")
EXCLUDE = {"time", "latitude", "longitude"}

N_SUBDIRS_A = 23
FILES_PER_SUBDIR_A = 13000
FILES_PER_SAMPLE_A = 45

N_FILES_B = 13000
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
        a_results = pool.map(worker_a, [(i, SEED + 1000 + i) for i in range(N_WORKERS)])
        b_results = pool.map(worker_b, [(i, SEED + 2000 + i) for i in range(N_WORKERS)])

    print("store,worker,sample,wall_ms")

    for batch in a_results:
        for w, s, ms in batch:
            print(f"A,{w},{s},{ms:.2f}")

    for batch in b_results:
        for w, s, ms in batch:
            print(f"B,{w},{s},{ms:.2f}")


if __name__ == "__main__":
    main()
