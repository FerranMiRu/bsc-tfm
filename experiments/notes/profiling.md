# Profiling U-Net Training

This document describes the **NVIDIA Nsight Systems (nsys) + NVTX** profiling
setup for the U-Net training loop on MN5. It explains what gets traced, how
the capture window is gated, how to launch a run, and how to read the result.

## Why nsys + NVTX

A previous `torch.profiler` run identified the dataloader as the dominant
cost (~74% of wall time). To get a kernel-level GPU timeline — including
async overlap between data loading, host-to-device copies, and compute —
we use **Nsight Systems**. NVTX ranges (built into PyTorch via
`torch.cuda.nvtx`) annotate the timeline so we can attribute time to
phases of the training step.

No new dependency: `torch.cuda.nvtx` ships with PyTorch, and `nsys` is
provided by the `cuda/12.8` module on MN5.

## Files involved

| File                                   | Role                                                                                               |
| -------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `src/ai4land/training/train.py`        | NVTX ranges around dataloader / forward / loss / backward / optimizer_step + capture-window gating |
| `src/ai4land/training/base_trainer.py` | `ProfilingEarlyStop` exception so the run exits cleanly after the capture window                   |
| `src/ai4land/utils/config_utils.py`    | `ProfilingParams` (Pydantic) — `enable`, `active_batches`                                          |
| `inputs/profiling.yaml`                | Hydra config — derived from `debug.yaml` with profiling enabled                                    |
| `scripts/launch_profiling.sh`          | SLURM launcher that wraps `accelerate launch` with `nsys profile`                                  |

## Capture-window mechanism

A profiling run looks like this:

```
nsys is attached for the entire process lifetime
  ├── warmup batches (10)         ← nsys is attached but NOT recording
  ├── cudaProfilerStart()         ← nsys begins recording
  ├── 5 active batches            ← nsys timeline contains only these
  ├── cudaProfilerStop()          ← nsys flushes and stops recording
  └── ProfilingEarlyStop          ← Python exits before validation
```

Two PyTorch APIs cooperate to make this work:

- **`torch.cuda.cudart().cudaProfilerStart()` / `cudaProfilerStop()`** —
  PyTorch's public hook into the CUDA Profiler API. These are no-ops
  unless a profiler is attached.
- **`nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop`** —
  tells nsys to honor those calls: don't record anything until the app
  calls `cudaProfilerStart()`, and stop on `cudaProfilerStop()`.

The result: nsys is attached for ~14 minutes (nsys overhead, model init,
warmup) but the `.nsys-rep` only contains the 5 batches we care about.
This keeps the trace file small and the timeline focused.

Why warmup: the first ~10 batches let the DataLoader prefetch queue,
GPFS page cache, and any one-time CUDA allocations reach steady state.
Without warmup, the trace is dominated by setup costs that don't reflect
real training.

`ProfilingEarlyStop` is raised right after `cudaProfilerStop()`. It's
caught in `BaseTrainer.run()` and short-circuits validation/checkpointing
so the job exits in seconds rather than running the full epoch. Defined in
`src/ai4land/training/base_trainer.py`.

## NVTX ranges

Inside `UNetTrainer.train_epoch`, every step is annotated with nested ranges:

```
step_{i}
├── dataloader      next(loader_iter) — host-side stall when prefetch is empty
├── forward         model(...) — host returns immediately; GPU work is queued
├── loss            CE loss assembly
├── backward        accelerator.backward(loss)
└── optimizer_step  optimizer.step() (gated by gradient accumulation)
```

All ranges use the `torch.cuda.nvtx.range(...)` context manager (Python
`with` statement) — no manual push/pop pairs.

Marks (instantaneous events, no duration) anchor the window edges:

- `epoch_{N}_start`
- `profiling_window_start` / `profiling_window_stop`

### Important: NVTX ranges are host-side markers

A `with nvtx.range("forward"):` bar in the timeline measures **how long
the Python call took to return**, not how long the GPU spent on those
kernels. PyTorch is asynchronous: `model(x)` returns as soon as the
kernels are _queued_. The actual GPU compute happens later.

This is fine — the **CUDA HW row** in the nsys timeline shows real kernel
timestamps independently. Use NVTX bars to _find_ where in time each
phase happens; use the CUDA HW row beneath them to read true GPU time.

A consequence: when you see a "gap" between `backward` and `optimizer_step`
in the NVTX row, that's usually the host hitting an implicit sync (e.g.
`loss.detach().item()`, gradient clipping, or accelerate's accumulation
boundary) and waiting for queued GPU work to drain.

## Configuration

`inputs/profiling.yaml` enables profiling and sizes the loop:

```yaml
training:
  train_steps_per_epoch: 15 # 10 warmup + 5 active
  profiling:
    enable: true
    active_batches: 5 # window width
```

The Pydantic model in `src/ai4land/utils/config_utils.py`:

```python
class ProfilingParams(BaseModel):
    enable: bool = False
    active_batches: int = Field(default=5, ge=1)
```

`ProfilingParams` defaults to disabled, so production runs (`unified.yaml`)
and debug runs (`debug.yaml`) inherit `enable: false` automatically — no
profiling overhead unless explicitly requested.

Warmup count is derived as `train_steps_per_epoch - active_batches`.
Adjust both in YAML to change the split.

## Launching a run

From the repo root on MN5:

```bash
sbatch scripts/launch_profiling.sh
```

Defaults: 1 GPU (single-rank trace is much easier to read than DDP),
30-min wall limit, `acc_debug` queue, `ehpc536` allocation. The launcher
loads `module load cuda/12.8` (provides `nsys`) and wraps
`uv run accelerate launch ... train.py --config-name profiling` with the
nsys command below.

### nsys flags (`scripts/launch_profiling.sh`)

```
nsys profile
  -o ${NSYS_OUT}                    # output path (no extension; .nsys-rep is appended)
  -t cuda,nvtx,osrt,cudnn,cublas    # what to trace
  --capture-range=cudaProfilerApi   # gate recording on cudaProfilerStart()
  --capture-range-end=stop          # stop recording (don't kill app) on cudaProfilerStop()
  --sample=process-tree             # CPU samples across main + workers
  --cpuctxsw=process-tree           # OS scheduling across the tree
  --trace-fork-before-exec=true     # follow forked DataLoader workers
  --cuda-memory-usage=true          # track CUDA allocations
  --gpu-metrics-devices=all         # SM Active, occupancy, clocks, DRAM, PCIe BW
  --gpu-metrics-frequency=10000     # 10 kHz sampling
  --stats=true                      # emit nvtx_sum / cuda_gpu_kern_sum / etc. to stdout
  --force-overwrite=true
```

Why these and not others:

- **`process-tree` scope on `--sample` and `--cpuctxsw`**: PyTorch's
  DataLoader workers are forked Python processes. Without this, you only
  see the main process — the workers (where the bottleneck lives) are
  invisible.
- **`--trace-fork-before-exec=true`**: needed to attach nsys to forked
  children before they `exec` (which Python workers don't, but the flag
  still ensures the fork point is captured cleanly).
- **`--gpu-metrics-devices=all` not `cuda-visible`**: nsys hints at
  `cuda-visible` when `CUDA_VISIBLE_DEVICES` is set, but `all` gives the
  same result on a single-GPU run and matches what the training-profiling
  workshop uses.
- **No `--backtrace=dwarf`**: high overhead, low value for Python
  workloads (most stacks are interpreted frames anyway).

For multi-rank profiling (future work), use per-rank output:
`-o ${NSYS_OUT_DIR}/profile_node%q{SLURM_NODEID}` — a comment in the
launcher already shows this.

## Outputs

After a successful run:

```
logs/profiling/<jobid>.out        # SLURM stdout (includes nsys --stats=true reports)
logs/profiling/<jobid>.err        # SLURM stderr
logs/profiling/nsys_<jobid>.nsys-rep   # binary trace — open in Nsight Systems GUI
logs/profiling/nsys_<jobid>.sqlite     # same data as SQL — query with nsys recipe
```

The `.out` file ends with a host-side summary line:

```
[profiling] 5 active batches (host-side, approx): wall=130.646s,
dataloader stall=129.888s (99.4% of wall). For exact GPU timing see nsys stats output.
```

This is approximate (no syncs around the perf_counter calls — see
"Async caveat" below). The `.nsys-rep` is the source of truth for exact
timing.

## Reading the trace

### CLI: `nsys stats`

The eight reports from `--stats=true` are inline in the `.out` file. The
most useful ones for diagnosing dataloader issues:

- **`nvtx_sum`** — time spent in each NVTX range. Outliers in
  `step_{i}` reveal which batches stalled.
- **`osrt_sum`** — OS runtime calls. **`sem_wait`** dominance means
  workers are blocked (idle, waiting), not CPU-bound. Look at
  **max `read`** — large values are filesystem stalls.
- **`cuda_gpu_kern_sum`** — per-kernel GPU time. Top entries reveal
  which model layers cost the most when the GPU _is_ working.
- **`cuda_api_sum`** — CUDA API call costs. High `cudaStreamSynchronize`
  total means the host is frequently blocked waiting on the GPU.

Re-run later without re-profiling:
`nsys stats logs/profiling/nsys_<jobid>.nsys-rep`.

### GUI: Nsight Systems

Open `nsys_<jobid>.nsys-rep` in `nsys-ui` (locally — pull with
`/hpc-pull` first).

**Process layout you'll see:**

```
[main_pid]  python3       ← main process; has CUDA HW row, ~12 threads
[..]        python3 (×N)  ← N DataLoader workers; no CUDA, ~40+ threads each
[..]        gpu_stats     ← nsys's own GPU metrics sampler
```

Only the main process touches the GPU, so all CUDA HW activity is under
its row. The forked workers (one per `num_workers`) appear as siblings.

**Top-down navigation:**

1. **NVTX row (main process)** — locate `step_0..step_{active-1}`.
   Outliers are immediately visible: a step with a multi-second
   `dataloader` bar = prefetch queue starvation.
2. **CUDA HW row** — directly under NVTX. Shows real kernel execution.
   Compare its activity windows against the NVTX bars to see GPU
   utilization vs. host-side markers.
3. **GPU Metrics row (top)** — `SM Active`, `SM Warp Occupancy`, `DRAM
Bandwidth`, `PCIe Bandwidth`. Confirms whether stalls are GPU-idle
   (yes, in our case) and whether compute kernels saturate the H100.
4. **Worker process threads** — for any long `dataloader` stall, expand
   one worker process and find its `pt_data_worker_*` threads. The OS
   Runtime Libraries row shows whether the worker is in `sem_wait`
   (blocked), `read` (waiting on filesystem), or doing CPU work.

**Finding the longest read:** View → Events View → filter `Name = read`,
sort by Duration descending. Click the top row to jump the timeline to
it. Right-click → "Show in Backtrace view" for the call stack.

## Async caveat — why no `cuda.synchronize()`

We deliberately do **not** call `torch.cuda.synchronize()` between
phases inside the loop. Reasons:

- nsys captures real kernel timestamps from CUDA itself; it doesn't need
  host-side syncs to record GPU activity correctly.
- Per-phase syncs serialize the kernel queue and destroy the very
  async overlap we're trying to measure (e.g., backward kernels of step
  N overlapping with the optimizer of step N-1).

The trade-off: the host-side `time.perf_counter` totals printed in the
log are approximate (a few ms of slop where the host is ahead of the
GPU). For exact wall-clock per phase, read it from `nsys stats` /
Nsight Systems UI, not from the log.

This matches the approach in
[`training-profiling-workshop/exercises/exercise_1_DDP/`](../training-profiling-workshop/exercises/exercise_1_DDP).

## Out of scope (current setup)

- **Multi-GPU profiling**: single-rank only. NCCL ranges and per-rank
  comparisons can be added later by switching to `--num_processes 4` and
  per-rank output paths.
- **TerraTrainer / `train_adapter.py`**: only `UNetTrainer` is
  instrumented.
- **Validation-loop profiling**: `ProfilingEarlyStop` exits before
  validation runs.
