"""Dump chunk file sizes for every array in a zarr store as CSV.

Walks the store recursively so it works for both the flat single-zarr layout
(one directory per variable) and the merged layout (nested groups such as
dynamic/data and static/data). Takes all chunk files, not a sample.

Output columns: var,chunk_file,size_bytes  (var = array directory relative to store)
"""

import os

STORE = "/gpfs/scratch/bsc32/bsc096444/AI4LAND_merged-NON_preprocessed-data-2000-2015.zarr"


def is_chunk_file(name):
    return not name.startswith(".") and all(part.isdigit() for part in name.split("."))


print("var,chunk_file,size_bytes")

for dirpath, _dirnames, filenames in os.walk(STORE):
    array_dir = os.path.relpath(dirpath, STORE)

    for fname in filenames:
        if not is_chunk_file(fname):
            continue

        size = os.path.getsize(os.path.join(dirpath, fname))
        print(f"{array_dir},{fname},{size}")
