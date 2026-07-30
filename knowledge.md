# Dataloader Knowledge Base

Lean, current facts about the AI4Land dataloader and the system it runs on. Per-run measurement
detail lives in `results.md`; superseded hypotheses are dropped, not archived here.

---

## The stack

PyTorch `DataLoader` → `xarray.open_zarr` (zarr v2, dask-backed at `chunks={}`) →
Blosc(lz4, clevel=5, shuffle) → GPFS at `/gpfs/scratch/ehpc736/data/`. MN5 `acc_ehpc` partition.
The fast path is **`MergedZarrDataset`** (all variables stacked into one `dynamic` + one `static`
group); `SingleZarrDataset` and `MultiZarrDataset` are slower legacy layouts.

## Workload (per sample)

- 256² patch. Modalities: 14 LUH2 (× 2 timesteps), population (× 2), kg_class (× 2), 6 static
  (× 1), hilda_prior (× 1), hilda_target (× 2). ~9.5 MB tensor payload.
- Sample tuple: `(x_continuous, x_kg, x_static, x_hilda, hilda_target)`.

## Training defaults

| Knob                 | Value           | Where                       |
| -------------------- | --------------- | --------------------------- |
| `num_workers`        | 12              | config                      |
| `batch_size`         | 8               | config                      |
| `accumulation_steps` | 4               | config (effective batch 32) |
| `prefetch_factor`    | 4               | hardcoded `base_trainer.py` |
| `persistent_workers` | False           | hardcoded `base_trainer.py` |
| `pin_memory`         | True            | hardcoded                   |
| CPUs per task        | 20 (80 / 4 GPU) | sbatch script               |

## Per-sample cost (single-thread, `[2001,2010]`, median ms — Run 42)

| store                                 |     ms | ratio                           |
| ------------------------------------- | -----: | ------------------------------- |
| single_small (66 G, 2000–2015)        |   1329 | 1.0×                            |
| single_big (435 G, 1901–2015)         |   5513 | 4.15× single_small              |
| merged_small (59 G)                   |    174 | —                               |
| merged_big (387 G)                    |    286 | 1.64× merged_small              |
| **merged_big_raw (387 G, no xarray)** | **38** | **7.2× faster than merged_big** |
| multizarr (~680 G, 5 stores)          |   7970 | 6.0× single_small               |

Merged vs single: **7.6× faster small, 19.3× faster big** — the merge benefit grows with size.
The raw-zarr loader (`RawMergedZarrDataset` / `dataset_type: merged_raw`) removes the remaining xarray
per-sample overhead: clean median **37.9 ms** (Run 45, 2 reps; reproduced 37–38 ms across 4 probes) =
**~7× faster than xarray merged_big** and faster than even xarray merged_small. Bit-identical output
(parity 0.0).

## Proven facts

- **The dominant cost is xarray/dask graph overhead, not bytes / decompress / I/O.** Per chunk:
  cold raw read ~20 ms, Blosc decode 0.3 ms (negligible), but `xarray.isel().to_numpy()` adds
  ~16 ms of cache-invariant, GIL-bound graph work per chunk.
- **The store-size penalty is dask graph construction + culling, scaling with an array's total
  chunk count (= time extent) — RESOLVED (Run 42), refuting the earlier inode/dentry cache-miss
  hypothesis.** Matched-window control (single_small vs single_big): miss rate identical (0.994),
  cold-open cost identical (25.25 ms), chunk-read I/O ratio 0.99; the gap is a warm, identical-slice
  `.isel().to_numpy()` running **6.87×** slower on single_big (115 time-chunks) than single_small
  (16), with I/O ≈ 0. The graph carries the store's **full** time-extent even when training a
  sub-window. Corroborated by xarray issues #6036 (`chunks=None` avoids the build) and #9111
  (`open_zarr` ~3× slower than raw `zarr.open`).
- **Merging (Lever 3) is the delivered fast path.** Stacking 17 dynamic + 6 static vars cuts per-
  sample chunk opens ~93 → ~9 and store file count 16.9× (single_big 5.0 M → merged_big 296 K).
  Real 4-rank DDP end-to-end **3.8–9.2×** over single (Run 35); merged is worker-insensitive
  (NW=8 saturates the GPU feed), single is worker-hungry (NW=20). DDP scaling 88–92% (merged) vs
  67–70% (single); no 2-node regression.
- **Merging wins on chunk-count/layout, not just the graph — a graph fix alone would NOT make single
  match merged.** Proof: `single_small` is already **7.6×** slower than `merged_small` where the graph
  is negligible (16 time-chunks), so that gap is the ~93-vs-9 chunk reads/sample (layout). A raw
  single loader would still read ~93 chunks/sample vs merged's ~9, so it would stay well behind
  merged_raw. Merging (chunk-count) and the raw loader (xarray removal) are orthogonal wins —
  `merged_raw` stacks both.
- **All consolidated stores are mutually equivalent and data-faithful (Run 41):**
  single == merged == preprocessed to float32 ε; baked preprocessing == read-time normalization.
  They differ from the legacy multizarr **only by loader recipe**, not data: population `log1p` vs
  `log(x+1e-8)`+`nan→0` (→ `ln(1e-8) = −18.42` at zero/ocean cells); KG ocean fill `0` vs `31`;
  static read-time z-score/cos vs pre-baked `static_cube_processed`; `secma`/`secmb` `_full_time`
  vs `_full_time_train`.
- **`kg_class`** has no NaNs on land pixels in the bundled store; ocean cells filled `0` in the loader.
- **HILDA labels** remap raw codes `[11,22,33,40–45,55,66,77]` → dense `[1..7]` (8 classes incl. 0).
- **All data variables are lowercased on `open_zarr`**; config feature names (`lulc_states`,
  `aspect`, …) match regardless of stored case (`LULC_states`, `ASPECT`, …).

## Throughput vs the GPU feed (target: ≥ 53 samples/s/GPU)

Single-rank probe (Run 42, merged_big, `[1960,2000]`): 12 workers = 37 sps, 20 workers = 57 sps.

Real 4-rank DDP training loop (Run 43, merged_big, `[1960,2000]`, batch=8):

| workers | p50 (GPU floor) |  effective mean |     p95 | stalls > 1 s | cadence  |
| ------: | --------------: | --------------: | ------: | -----------: | -------- |
|      12 | 154 ms/51.9 sps | 272 ms/29.4 sps | 1306 ms |         7.7% | every 12 |
|      20 | 157 ms/51.0 sps | 200 ms/40.0 sps |  215 ms |         1.8% | every 20 |

- The **fast-batch floor (~155 ms = 51 sps/GPU) is GPU-compute bound** (batch=8), so 53 sps/GPU is
  **not reachable by adding workers** — only a bigger batch or faster step would lift it.
- The periodic stall is **structural at every-`num_workers` batches** (the whole worker pool refills
  together). More workers give more outstanding I/O to hide it (stalls 7.7%→1.8%, p95 6× tighter), but
  do not remove it. 20 workers raise the effective feed 29.4→40.0 sps/GPU.
- **Raw-zarr loader (`merged_raw`) makes the loop GPU-bound (Run 45, clean).** Clean multi-worker
  throughput (sps, mean of 2 reps): w12 merged 42 → raw **80** (1.90×), w16 53 → **115** (2.18×),
  w20 59 → **145** (2.46×) — the raw advantage _grows_ with workers. In real DDP training, raw sits at
  p50 ≈ 153 ms ≈ **51 sps/GPU (the compute floor) with ~0% stalls** — the stall tax is gone. The single-
  thread win (~7×) compresses to ~2–2.5× end-to-end because many workers overlap the per-sample cost.
- End-to-end: merged-big 12w finishes 10 epochs in **2.34 h vs ~55 h** legacy multizarr (~23×);
  **`merged_raw` 12w does it in 1.27 h** (~1.8× over merged, GPU-bound) and **beats merged 20w**
  (raw 12w 51 sps / 0 stalls > merged 20w 40 sps) — fewer workers needed. Quality is unchanged across
  loaders (bit-identical data → equivalent model: best val ~0.58, eval Year0 acc ~0.87–0.90).

## Fix options for the xarray graph (ranked)

Two distinct costs inflate per-sample time: (a) **xarray graph+indexing** (the per-sample overhead,
~85% of merged_big's cost), and (b) **GPFS op-count** = chunks read/sample (the layout penalty, ~93
single vs ~9 merged). Merging neutralises (b); the raw loader neutralises (a).

1. **Bypass xarray** (raw `zarr` API) = Lever 1 — **SHIPPED + clean-benchmarked (Run 45):**
   `RawMergedZarrDataset` (`dataset_type: merged_raw`) reads patches via `zarr_array.oindex[...]`,
   output **bit-identical** to xarray (parity max_abs_diff 0). **~7× single-thread, ~1.9–2.5× multi-
   worker** (grows with workers), and makes training **GPU-bound (~0% stalls, ~51 sps/GPU, 1.27 h
   vs 2.34 h)** with an unchanged model. This is the recommended loader. (`chunks=None` gave no gain
   because it KEEPS xarray's indexing; the gain needs xarray fully removed.)
2. **Re-layout the store by time shard** (per-decade groups, NOT a trim — keep all years): opening the
   needed window touches few chunks → tiny graph, without dropping data. Structural rebuild.
3. **Pre-shard to a flat patch dataset** (WebDataset-style): turn random N-chunk gather into
   sequential shard reads (1 open/shard). Earthmover-style cloud dataloaders report 15–17× this way.
   Heavyweight; only if 1–2 fall short.
4. **Zarr v3 sharding** — decouples chunk size from file count (small chunks, few files), but does NOT
   shrink the logical chunk count the graph enumerates → low value for _our_ bottleneck. GPU-native
   decode (nvCOMP/DALI) is irrelevant: our Blosc decode is already 0.3 ms.

## Disproven / dead ends

- **Chunk-size change away from 512²** doesn't help random 256²-patch sampling (256² → more chunks
  → more graph; 1024²/2048² → decode-bound). Keep 512².
- **Pre-baking preprocessing** buys ~0 (prep was ~8% of cost; confirmed by parity).
- **`ThreadPoolExecutor` over dask `.load()` calls**: no end-to-end gain; threads=2 regressed
  (Blosc-in-threadpool, numcodecs #239). Bypass dask instead.
- **`ProcessSynchronizer`**: 0 perf effect (read-only path; zarr v2 `_chunk_getitem` never acquires
  it). Creation kept defensively; toggle removed.
- **Inode/dentry cache-miss rate** as the store-size cause: refuted (Run 42 — see Proven facts).
- **NVMe staging**: discarded — production store > 213 GB won't fit on the 480 GB local NVMe.
- **`chunks=None` (drop dask, keep xarray) for throughput**: no multi-worker gain (32.6 vs 37.9 sps
  at 12w). NOT necessarily because we're GPFS-bound — it keeps xarray's _indexing_ overhead. The real
  test is removing xarray entirely (raw zarr, `merged_raw`); clean benchmark pending. Knob removed.

## Datasets on disk

Bundled single (one zarr array per var):

| Path                                             | Size  | Years     | Chunk files |
| ------------------------------------------------ | ----- | --------- | ----------- |
| `…/AI4LAND_NON_preprocessed-data-2000-2015.zarr` | 66 G  | 2000–2015 | 710,571     |
| `…/AI4LAND_NON_preprocessed-data-1901-2015.zarr` | 435 G | 1901–2015 | 5,012,319   |

Merged (variables stacked, Lever 3 — fast path):

| Path                                                    | Size  | Years     | Chunk files |
| ------------------------------------------------------- | ----- | --------- | ----------- |
| `…/AI4LAND_merged-NON_preprocessed-data-2000-2015.zarr` | 59 G  | 2000–2015 | 43,459      |
| `…/AI4LAND_merged_NON_preprocessed-data-1901-2015.zarr` | 387 G | 1901–2015 | 296,503     |

Geometry: grid `[*, 18000, 36000]`, spatial chunks `512²` (= 2556 spatial chunks), time chunked
1/yr; Blosc(lz4, clevel=5). Single = 40,896 files/3-D var × 17 + 6 static × 2556. Merged folds the
17 dynamic vars into one `variable`-dim chunk (`dynamic/data` 40,896 + `static/data` 2556), same
bytes. Legacy multizarr (5 separate per-modality CONCERTO stores) totals ~680 G / 6.73 M files —
worst layout for graph + metadata pressure; no longer used.

## Node hardware (MN5 ACC partition)

2× Intel Xeon 8460Y+ (80 cores), 4× NVIDIA H100 64 GB, 512 GB DDR5, 480 GB NVMe, 4× NDR200 IB.
**20 CPUs per GPU** when all 4 GPUs are used.

## Filesystem context

GPFS: shared, high per-op latency, real metadata-server pressure. Single-stream read ~556 MB/s,
12-stream aggregate ~2.5 GB/s — **not** bandwidth-bound (the xarray graph dominates). Each DDP rank
gets its own `DataLoader`: 4 ranks × 12 workers = 48 worker processes per node hitting GPFS.

## Library versions

| Library     | Pin                    | Note                   |
| ----------- | ---------------------- | ---------------------- |
| `zarr`      | `>=2.18.3,<3.0.0`      | v2 only — no async I/O |
| `xarray`    | `>=2025.6.1,<2026.0.0` |                        |
| `dask`      | `>=2024.9.0,<2025.0.0` |                        |
| `numcodecs` | `>=0.12.1,<0.16`       |                        |

## Settled by Run 45 (clean, full matrix)

- **Worker scaling, GPFS variance, 16-rank scaling are all measured.** `merged_raw` 1r and 4r both
  hold ~48–51 sps/GPU (the compute floor) at every worker count 12/16/20 — i.e. **per-GPU throughput
  is flat across 1→16 ranks** (good DDP scaling) and raw needs few workers (12w already saturates).
  Within-loader timing reps agree to cv ≤ ~3% (tighter than the historical ~10%); the only outliers
  were two 12w reps that overlapped a disk-pressure window. **Quality holds at 4× effective batch /
  same LR** (4r best val ~0.57–0.58, same band as 1r) — no LR rescale needed for the equivalence.

## Open questions / next

- **The ~51 sps/GPU floor is GPU-compute-bound** (batch=8). With `merged_raw` the loop is now GPU-bound,
  so the _only_ remaining lever for more throughput is a **bigger batch or a faster step**, not the
  dataloader. Heavier store re-layout / pre-sharding (Fix options 2–3) are no longer needed for speed.
- **`persistent_workers=True`** — avoid respawning workers + rebuilding the open_zarr graph every
  epoch (cuts the per-epoch cold re-warm). One-line, free; untested. (`prefetch_factor` ruled out: we
  are sustained loader-bound, so a bigger buffer just drains — it only helps if production ≥ consume.)
