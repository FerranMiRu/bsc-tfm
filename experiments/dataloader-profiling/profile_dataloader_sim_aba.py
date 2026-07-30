"""ABA rerun of Run 16 on the same data dir to test cache/timing reproducibility.

Reuses stores at /gpfs/scratch/bsc32/bsc096444/dataloader_sim/ (created by
Run 16). Read pattern: A1 -> B -> A2. Different seeds per pass so each pass
samples different random files (similar working-set fraction each time).

A1 vs A2 isolates within-job drift (cache warming, GPFS load fluctuation).
A vs B confirms Run 16's ratio under the same cache regime A reports.
"""

import multiprocessing
import random
import time
from pathlib import Path


BASE_DIR = Path("/gpfs/scratch/bsc32/bsc096444/dataloader_sim")
STORE_A = BASE_DIR / "store_A_23dirs"
STORE_B = BASE_DIR / "store_B_1dir"

N_SUBDIRS_A = 23
FILES_PER_SUBDIR_A = 13000
FILES_PER_SAMPLE_A = 40

N_FILES_B = 13000
FILES_PER_SAMPLE_B = 2

N_WORKERS = 12
SAMPLES_PER_WORKER = 100
SEED = 42


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
