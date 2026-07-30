- define size of xarrays
- utils.py \_separate_hilda_ts() guardar en memoria i no en RAM
- zarr_dataset lines 502-522 es muy lento
- keep the **get_item** clean and only pass tensors
- change numpy to cupy, numba

claude --resume "add-nsys-nvtx-profiling-unet"

---

# Prior profiling finding (2026-04-22, job 39570247)

torch.profiler run, 15 batches, 1 epoch, DDP, 12 workers + prefetch_factor=2:

| Phase       | Time   | % of Total |
| ----------- | ------ | ---------- |
| Dataloader  | 180.7s | 74%        |
| Backward    | 13.2s  | 5%         |
| Forward     | 1.4s   | 1%         |
| Loss        | 0.1s   | 0%         |
| Unaccounted | 50.4s  | 20%        |

The dataloader dominates wall-clock — workers can't keep the prefetch queue
full. The "unaccounted" 20% is likely DDP all-reduce (NCCL). This is what
the new nsys/NVTX instrumentation (`inputs/profiling.yaml`,
`scripts/launch_profiling.sh`) is meant to dig into.
