"""Count zarr chunk files per array for the single and merged stores (small + big).

Walks each store recursively and counts chunk files (dotless names whose dot-separated parts are
all digits) per array directory. Count only, no stat, so it scales to multi-million-file stores in
a short job. The big-single vs big-merged total is the inode/dentry-cache working set that drives
the store-size effect. CSV stdout: store,array,n_chunk_files, then a per-store total.
"""

import os


DATA_ROOT = "/gpfs/scratch/ehpc736/data"

STORES = [
    f"{DATA_ROOT}/AI4LAND_NON_preprocessed-data-2000-2015.zarr",
    f"{DATA_ROOT}/AI4LAND_NON_preprocessed-data-1901-2015.zarr",
    f"{DATA_ROOT}/AI4LAND_merged-NON_preprocessed-data-2000-2015.zarr",
    f"{DATA_ROOT}/AI4LAND_merged_NON_preprocessed-data-1901-2015.zarr",
]


def is_chunk_file(name: str) -> bool:
    return not name.startswith(".") and all(part.isdigit() for part in name.split("."))


def main() -> None:
    print("store,array,n_chunk_files")

    for store in STORES:
        store_total = 0

        for dirpath, _dirnames, filenames in os.walk(store):
            array_dir = os.path.relpath(dirpath, store)
            n_chunk_files = sum(1 for fname in filenames if is_chunk_file(fname))

            if n_chunk_files:
                print(f"{store},{array_dir},{n_chunk_files}")
                store_total += n_chunk_files

        print(f"{store},TOTAL,{store_total}")


if __name__ == "__main__":
    main()
