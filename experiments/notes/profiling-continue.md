# Next chat: act on profiling findings

Profiling infra is in place (committed in `45e5ec3` on branch
`79-implement-nsys-traces-to-code`). This file is the handoff for the
follow-up work focused on **using the profiling output to optimize the
dataloader**.

## Current state

- `inputs/profiling.yaml` + `scripts/launch_profiling.sh` produce a
  5-batch nsys trace under `logs/profiling/nsys_<jobid>.nsys-rep`.
- NVTX ranges wrap `dataloader / forward / loss / backward / optimizer_step`
  in `UNetTrainer.train_epoch`. `ProfilingEarlyStop` exits cleanly after
  the capture window.
- Detailed reference: `docs/profiling.md` (gitignored, local-only).

## Already-found bottleneck — Zarr `ProcessSynchronizer`

`src/ai4land/utils/combined_loader.py` opened every Zarr store with
`zarr.sync.ProcessSynchronizer`, which uses `fcntl` file locks per
chunk. On GPFS this funnels every chunk read through the metadata
servers and serializes the 12 DataLoader workers.

**Fix is uncommitted in the working tree**: synchronizer removed from
the training (read-only) path, kept on the inference `hilda_pred` path
(which is writable). Verified empirically:

| Metric                  | Before (39787545) | After (40044179) |
|-------------------------|-------------------|------------------|
| 5-batch wall            | 130.6 s           | 82.4 s           |
| Max single GPFS `read`  | 57.7 s            | **1.25 s** (−46×)|
| Fast steps (~100 ms)    | 2 of 5            | **4 of 5**       |

### First task in next chat

Commit the synchronizer removal as the second of the two-commit pair.
Format (per project convention — no body, no Co-Authored-By):

```
fix: drop zarr ProcessSynchronizer from training path

#79
```

Optional cleanup to bundle in this same commit (or a follow-up) — the
`lock_dir` config field and `AI4LAND_LOCK_DIR` env var are now dead on
the training path. Decide whether to keep them for the inference write
path or drop entirely.

## Remaining bottleneck — GPFS cold-cache stalls

After the lock fix, **one step out of five still stalls** (~80 s in the
last run). Per-read max is now 1.25 s, so the stall is many slow reads
in sequence — a worker hitting cold GPFS chunks. This is filesystem
variance, not a code bug.

Options to investigate, roughly in order of effort:

1. **`prefetch_factor` (DataLoader)** — currently default (2). Try 8.
   One-line YAML change in `inputs/profiling.yaml` (and add to
   `TrainingParams` if not exposed). Cheapest experiment.
2. **`OMP_NUM_THREADS=1`** in workers — `~12 × 40` threads contending
   for 20 cores currently. Set in `scripts/launch_profiling.sh` and
   `scripts/acc_training.sh`.
3. **Zarr chunk shape vs `patch_size`** — verify each Zarr store's
   `chunks` is ≥ 256×256 so a single patch doesn't span multiple chunk
   files. If misaligned, re-chunk the stores.
4. **Stage data to `/dev/shm` or local NVMe** before training — fully
   eliminates GPFS variance. Larger change (preprocessing step, disk
   budget) but the only reliable fix for the cold-cache tail.

After each experiment, re-run `sbatch scripts/launch_profiling.sh`
from MN5 and compare `osrt_sum` max `read` and the per-`step_*` times
in `nvtx_sum`.

## Useful commands

```bash
# Push, submit, pull
/hpc-push
/hpc-submit profiling   # → scripts/launch_profiling.sh
/hpc-pull

# Re-stat an existing trace without re-profiling
nsys stats logs/profiling/nsys_<jobid>.nsys-rep
```

## Reference jobs

- `39787545` — baseline with synchronizer (130.6 s, max read 57.7 s)
- `40044179` — synchronizer removed (82.4 s, max read 1.25 s)
- `40043804` — smoke test confirming imports + UNet pass after the change
