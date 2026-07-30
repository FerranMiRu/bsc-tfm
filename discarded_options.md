# Discarded / Deprioritized Levers

Levers considered but excluded from the active 4-test focus (see
`knowledge.md` "Active levers"). Kept for future reference. One-line
reasons only — see `results.md` for the underlying measurements when
relevant.

**Convention:** rows prefixed with `!` are levers the user marked as
"might do if time allows" — not fully dead, just deprioritized vs the
active 4. Treat them as candidates to revisit, not dismissed.

| Lever                                                                           | One-line reason for exclusion                                                                                                                                                                              |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pre-baking preprocessing (`AI4LAND_preprocessed-...zarr`)                       | Run 6 measured ~0 gain vs non-preprocessed; prep was only 8% of cost and the LUH2 "prep" that remains is `to_stacked_array + transpose + reshape`, not normalize/fillna.                                   |
| **Lever A** — `ThreadPoolExecutor` wrapping xarray `.isel().load()`             | Run 8 ruled it out (threads=2 regressed +23%). GIL-bound xarray layer prevented thread overlap. **Superseded by the active "bypass xarray + raw zarr in threads" lever.**                                  |
| **Lever B** — `xr.open_zarr(..., chunks=...)` + `.compute(scheduler='threads')` | Same idea as Lever A; same GIL problem. Superseded.                                                                                                                                                        |
| !**Lever C** — Rechunk LUH2 to `(1, 256, 256)`, keep 14 separate arrays         | Store was already requested but expected gain is small now that we know cost is metadata + xarray overhead, not bytes (profile_layers: 5× more bytes per chunk → only +6% read time). Subsumed by Lever D. |
| **Lever E** — Cache LUH2 cube in worker RAM at start                            | Dead end: doesn't scale to >213 GB production store.                                                                                                                                                       |
| !**Lever G** — `persistent_workers=True`                                        | Caused problems in past runs. User excluded.                                                                                                                                                               |
| !**Lever H** — `prefetch_factor` 4 → 8                                          | Marginal; only helps if queue drain is the bottleneck, which Run 4 ruled out.                                                                                                                              |
| !**Lever I** — `batch_size: 32, accumulation_steps: 1`                          | Trivial; amortizes per-batch fixed overhead. Not worth attention until loader is fast.                                                                                                                     |
| **Lever J** — Switch to MDS / WebDataset / TileDB / Petastorm                   | High effort, unknown gain. Not worth pursuing while simpler levers remain.                                                                                                                                 |
| **Lever K** — Defer loss `.item()` sync to log boundaries                       | Only matters once the loader is fast. Trivial when needed.                                                                                                                                                 |
| **Lever L** — `kg_class` stored as `uint8` (4× compression)                     | Tiny absolute win (~0.5% of cost). Worth doing only if requesting a new store anyway.                                                                                                                      |
| Zarr `ProcessSynchronizer` toggle                                               | 0 effect (Runs 1, 2, 7). Removed from codebase.                                                                                                                                                            |
