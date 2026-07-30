"""Merge zarr store variables into 2 stacked arrays: 1 dynamic + 1 static.

Source: the bundled singlezarr at SOURCE_STORE.
Target: new zarr store at TARGET_STORE with 2 sub-groups, each a single
DataArray that stacks the variables along a new 'variable' dim. Chunked
so each chunk file contains all variables for one timestep / spatial chunk.

Categorical variables (kg_class, LULC_states) are stored as float32 (the
common dtype across all dynamic vars). The dataloader casts them back to
uint8 in memory. Slight chunk-size penalty (~3-5%) buys 1 fewer file open
per sample.
"""

import shutil

import xarray as xr


SOURCE_STORE = "/gpfs/scratch/ehpc736/data/AI4LAND_NON_preprocessed-data-2000-2015.zarr"
TARGET_STORE = "/gpfs/scratch/bsc32/bsc096444/AI4LAND_merged-NON_preprocessed-data-2000-2015.zarr"

shutil.rmtree(TARGET_STORE, ignore_errors=True)

DYNAMIC_VARS = [
    "c3ann",
    "c3nfx",
    "c3per",
    "c4ann",
    "c4per",
    "pastr",
    "primf",
    "primn",
    "range",
    "secdf",
    "secdn",
    "secma",
    "secmb",
    "urban",
    "number_of_people",
    "kg_class",
    "LULC_states",
]
STATIC_VARS = [
    "ASPECT",
    "elevation",
    "SLOPE",
    "weighted_ORGANIC",
    "weighted_PCT_CLAY",
    "weighted_PCT_SAND",
]

ds = xr.open_zarr(SOURCE_STORE, consolidated=False)

groups = [
    ("dynamic", DYNAMIC_VARS, True, "w"),
    ("static", STATIC_VARS, False, "a"),
]

for group_name, var_list, has_time, mode in groups:
    arr = ds[var_list].to_array(dim="variable")

    chunks = {"variable": len(var_list), "latitude": 512, "longitude": 512}
    if has_time:
        chunks["time"] = 1

    arr.chunk(chunks).to_dataset(name="data").to_zarr(TARGET_STORE, mode=mode, group=group_name)
    print(f"wrote {group_name}: {len(var_list)} vars")
