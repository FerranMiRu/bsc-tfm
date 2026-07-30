"""Layered decomposition test for one LUH2 chunk read.

Times four read layers back-to-back on the same chunk so the deltas localize
where the ~250 ms of non-I/O work per LUH2 call goes:

    (a) raw_read     : open() + read() of one compressed chunk file
    (b) blosc_decode : numcodecs Blosc decode of (a)'s bytes  -> numpy array
    (c) zarr_index   : zarr.open(store)[var][slc]             -> numpy array
    (d) xarray_load  : ds[var].isel(slc).values               -> numpy array

Interpretation (back-to-back on the same chunk means layers (b)-(d) hit the
page cache populated by (a)):

    (a)                 = cold disk + open + read for one chunk
    (b)                 = pure Blosc CPU
    (c) - (b)           = zarr indexing / open / coord resolution + warm I/O
    (d) - (c)           = xarray + dask graph overhead

The on-disk variable name is auto-discovered (first 3-dim float32 variable
found) unless `VAR` env var is set. Override store via `STORE`, sample count
via `N_SAMPLES`, seed via `SEED`.
"""

from __future__ import annotations

import os
import statistics
import time

import numpy as np
import xarray as xr
import zarr

STORE = os.environ.get(
    "STORE",
    "/gpfs/scratch/ehpc736/data/AI4LAND_NON_preprocessed-data-2000-2015.zarr",
)
VAR = os.environ.get("VAR", "")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "50"))
SEED = int(os.environ.get("SEED", "42"))


def discover_3d_float32_var(store_path: str) -> str:
    """Return the on-disk name of the first 3-dim float32 variable in the store."""
    dataset = xr.open_zarr(store_path, consolidated=False, chunks={})

    for name, data_array in dataset.data_vars.items():
        if data_array.ndim != 3 or data_array.dtype != np.float32:
            continue

        for entry in os.listdir(store_path):
            if entry.lower() == name.lower() and os.path.isdir(f"{store_path}/{entry}"):
                return entry

    raise RuntimeError(f"No 3-dim float32 variable found in {store_path}")


def parse_chunk_index(fname: str) -> tuple[int, int, int]:
    parts = fname.split(".")
    return int(parts[0]), int(parts[1]), int(parts[2])


def pick_random_chunk_files(chunk_dir: str, n: int, rng: np.random.Generator) -> list[str]:
    candidates: list[str] = []

    for entry in os.listdir(chunk_dir):
        parts = entry.split(".")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            candidates.append(entry)

    if len(candidates) < n:
        raise RuntimeError(f"Only {len(candidates)} chunks in {chunk_dir}; need {n}")

    return rng.choice(candidates, size=n, replace=False).tolist()


def fmt_stats(label: str, values: list[float]) -> str:
    mean = statistics.mean(values)
    median = statistics.median(values)
    sorted_values = sorted(values)
    p95_index = min(len(sorted_values) - 1, int(len(sorted_values) * 0.95))
    p95 = sorted_values[p95_index]
    return (
        f"  {label:14s} | mean = {mean:8.2f} ms | median = {median:8.2f} ms | "
        f"p95 = {p95:8.2f} ms"
    )


def main() -> None:
    print(f"Store:     {STORE}")
    print(f"N_SAMPLES: {N_SAMPLES}")
    print(f"SEED:      {SEED}")

    var_on_disk = VAR if VAR else discover_3d_float32_var(STORE)
    print(f"Variable:  {var_on_disk}")

    zarr_array = zarr.open(f"{STORE}/{var_on_disk}", mode="r")
    print(f"Shape:        {zarr_array.shape}")
    print(f"Chunks:       {zarr_array.chunks}")
    print(f"Compressor:   {zarr_array.compressor}")
    print(f"Dtype:        {zarr_array.dtype}")

    _, chunk_size_y, chunk_size_x = zarr_array.chunks
    compressor = zarr_array.compressor

    dataset = xr.open_zarr(STORE, consolidated=False, chunks={})
    xr_var = dataset[var_on_disk]
    dim_t, dim_y, dim_x = xr_var.dims
    print(f"xarray dims:  {xr_var.dims}")

    chunk_dir = f"{STORE}/{var_on_disk}"
    rng = np.random.default_rng(SEED)
    chunk_files = pick_random_chunk_files(chunk_dir, N_SAMPLES, rng)
    print(f"Picked {len(chunk_files)} random chunk files")
    print()

    raw_read_ms: list[float] = []
    blosc_decode_ms: list[float] = []
    zarr_index_ms: list[float] = []
    xarray_load_ms: list[float] = []
    chunk_bytes: list[int] = []

    for chunk_fname in chunk_files:
        chunk_path = f"{chunk_dir}/{chunk_fname}"
        t_index, y_chunk, x_chunk = parse_chunk_index(chunk_fname)
        y_start = y_chunk * chunk_size_y
        x_start = x_chunk * chunk_size_x
        y_slice = slice(y_start, y_start + chunk_size_y)
        x_slice = slice(x_start, x_start + chunk_size_x)

        start = time.perf_counter()
        with open(chunk_path, "rb") as file_handle:
            raw_bytes = file_handle.read()
        raw_read_ms.append((time.perf_counter() - start) * 1000.0)
        chunk_bytes.append(len(raw_bytes))

        start = time.perf_counter()
        _decoded = compressor.decode(raw_bytes)
        blosc_decode_ms.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        _zarr_chunk = zarr_array[t_index, y_slice, x_slice]
        zarr_index_ms.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        _xarray_chunk = xr_var.isel({dim_t: t_index, dim_y: y_slice, dim_x: x_slice}).values
        xarray_load_ms.append((time.perf_counter() - start) * 1000.0)

    print(f"--- per-chunk layer timings (n={N_SAMPLES}) ---")
    print(fmt_stats("raw_read",     raw_read_ms))
    print(fmt_stats("blosc_decode", blosc_decode_ms))
    print(fmt_stats("zarr_index",   zarr_index_ms))
    print(fmt_stats("xarray_load",  xarray_load_ms))

    mean_a = statistics.mean(raw_read_ms)
    mean_b = statistics.mean(blosc_decode_ms)
    mean_c = statistics.mean(zarr_index_ms)
    mean_d = statistics.mean(xarray_load_ms)
    mean_bytes = statistics.mean(chunk_bytes)

    print()
    print("--- deltas (means) ---")
    print(f"  (a) raw disk I/O:                  {mean_a:8.2f} ms")
    print(f"  (b) pure decompress:               {mean_b:8.2f} ms")
    print(f"  (c) zarr full read (warm I/O):     {mean_c:8.2f} ms")
    print(f"  (c) - (b)  zarr index + warm I/O:  {mean_c - mean_b:+8.2f} ms")
    print(f"  (d) - (c)  xarray + dask overhead: {mean_d - mean_c:+8.2f} ms")
    print()
    print("--- context ---")
    print(f"  Mean compressed chunk size: {mean_bytes:.0f} bytes")
    print("  Estimated cost of 14 LUH2 vars per call (extrapolated):")
    print(f"    if xarray path:     {14 * mean_d:8.2f} ms")
    print(f"    if zarr-only path:  {14 * mean_c:8.2f} ms")
    print(f"    if decompress only: {14 * mean_b:8.2f} ms")
    print("  (Reference: current _get_luh2_data ~ 314 ms baseline.)")


if __name__ == "__main__":
    main()
