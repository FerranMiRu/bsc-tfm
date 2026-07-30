# `RawMergedZarrDataset` — a no-xarray reader for the merged store

This documents the changes made to `src/ai4land/utils/datasets.py` to add a dataset that reads
patches **without xarray**, and exactly how it works.

## Why

Profiling (results.md Run 42) isolated the per-sample bottleneck as **xarray's `open_zarr` graph +
indexing machinery**, not bytes, decompression, or GPFS I/O. A warm, identical-slice
`DataArray.isel(...).to_numpy()` runs several times slower than the same read through the raw `zarr`
API, because xarray (re)builds and culls a dask task graph that carries the array's full chunk
structure on every access. `chunks=None` does not fix this — it drops dask but keeps xarray's
indexing layer, which is itself the dominant cost. The only way to remove the overhead is to bypass
xarray entirely on the hot path and index the underlying `zarr.Array` directly.

`RawMergedZarrDataset` does exactly that, while producing **bit-identical** output to
`MergedZarrDataset` (verified by `tests/test_loader_parity_raw.py`: max abs diff `0.0` across all
five output tensors over 100 shared samples).

## What changed in `datasets.py`

### 1. Two read hooks extracted on `MergedZarrDataset` (behavior-preserving)

The xarray reads inside `MergedZarrDataset` were factored into two small override points. Nothing
about the base class's behavior changed — the expressions were only moved into named methods so a
subclass can swap the I/O without touching any processing:

- **`_read_static(lat_slice, lon_slice) -> np.ndarray`** — returns the full static stack
  `(variable, lat, lon)` for a patch. Previously inline in `_process_static`.
- **`_read_dynamic_window(ts_sorted, lat_slice, lon_slice) -> np.ndarray`** — returns
  `(variable, time, lat, lon)` for a set of window-relative time indices. Previously inline in
  `_read_dynamic`.

Everything else — normalization, `fillna`, the HILDA code remap, variable stacking/casting, the
`(coord, time) -> patch` index math in `_parse_index`, and `__getitem__` — stays in the base class
and is **inherited unchanged**. This is what guarantees parity: only the bytes' delivery path
differs between the two datasets.

### 2. `RawMergedZarrDataset(MergedZarrDataset)`

A subclass that overrides four methods. The design splits the work into a cheap **one-time init**
(still xarray) and a hot **per-sample read** (raw zarr).

#### Init still uses xarray — on purpose

`_init_dataset` and `_init_time` call `super()` first, so the one-time setup keeps using xarray for
the things xarray is good at and which only happen **once** (so their cost is irrelevant to
throughput):

- **CF-decoding the time axis.** The store encodes time as `days since 1901-01-01` (CF convention).
  `xr.open_zarr` decodes that to real `datetime64` years; reimplementing CF decode by hand would be
  error-prone. We read the decoded years straight off the xarray object.
- **Variable order.** `var_idx` (variable name → axis-0 index) is built by the base class from the
  xarray `variable` coordinate. The raw `zarr.Array` shares the exact same physical axis order, so
  the inherited `var_idx` indexes the raw array correctly with no re-derivation.
- **Window validation.** The base `_init_time` checks that the requested `time_range` (expanded for
  autoregression) is fully present and slices the dynamic array to that window.

After `super()._init_dataset()`, the subclass additionally opens **raw** handles to the two data
arrays and records the store's full year axis:

```python
root = zarr.open_group(str(self.cfg.stores), mode="r")
self._raw = {group: root[group]["data"] for group in ("dynamic", "static")}
self._full_years = self.dataset["dynamic"]["time"].dt.year.to_numpy()
```

`zarr.open_group(..., mode="r")` is lazy — it reads metadata only, no chunk data.

#### The time-offset bookkeeping

The base `_init_time` **slices** the xarray `dynamic` array to the training window, so within the
base class a time index `t` is *window-relative* (0 = first year of the window). The raw
`zarr.Array`, by contrast, still spans the **full** store (1901–2015). So the subclass computes the
absolute offset of the window's first year once:

```python
def _init_time(self):
    super()._init_time()
    start_year = int(self.times[0])               # first year of the (expanded) window
    self._time_start = int(np.where(self._full_years == start_year)[0][0])
```

#### The per-sample reads (the hot path)

```python
def _read_static(self, lat_slice, lon_slice):
    return self._raw["static"][:, lat_slice, lon_slice]

def _read_dynamic_window(self, ts_sorted, lat_slice, lon_slice):
    absolute_ts = [t + self._time_start for t in ts_sorted]
    return self._raw["dynamic"].oindex[:, absolute_ts, lat_slice, lon_slice]
```

- Static is a plain basic-slice read → numpy `(variable, lat, lon)`.
- Dynamic uses zarr **orthogonal indexing** (`.oindex`) because the time axis is a *list* of indices
  (the union of timesteps the sample needs) while lat/lon are contiguous slices. `oindex` selects
  each axis independently, matching xarray's `isel(time=[...], lat=slice, lon=slice)` exactly →
  numpy `(variable, time, lat, lon)`. The window-relative indices are shifted by `_time_start` into
  absolute store positions.

No xarray, no dask graph: just a chunk fetch + Blosc decode into numpy, which the inherited
processing then consumes identically.

### Why the raw read is bit-identical

The merged store's `dynamic/data` is `float32` with `fill_value=NaN` and **no** `scale_factor`,
`add_offset`, or `_FillValue` in its attributes. So xarray's `to_numpy()` applies no mask/scale
transform — it returns the same bytes the raw `zarr.Array` returns. (`static/data` is `float64`,
likewise plain; the inherited code casts it to `float32` in both paths.) With identical bytes and
identical downstream processing, the two datasets return identical tensors.

## How to use it

Set the loader via the existing knob (wired in `config_utils.DataParams.dataset_type` and the
`base_trainer._build_dataset` factory):

```yaml
data:
  dataset_type: "merged_raw"   # was: "merged"
  stores: "/gpfs/scratch/ehpc736/data/AI4LAND_merged_NON_preprocessed-data-1901-2015.zarr"
```

Everything else (store path, features, time range, land index) is unchanged — `merged_raw` reads the
same merged store as `merged`, just without xarray on the per-sample path.

## Caveats / scope

- Only the **merged** two-group layout (`dynamic/data`, `static/data`) is supported; the raw reader
  relies on that stacked structure. Single/multi stores keep their xarray loaders.
- **Inference mode** is inherited from `MergedZarrDataset` and untouched by the override (it uses the
  same `_read_*` hooks), but has not been separately exercised by the parity test, which runs in
  training mode.
- The performance benefit is being benchmarked cleanly (non-contended) before any number is treated
  as fact; correctness (parity) is already established.
