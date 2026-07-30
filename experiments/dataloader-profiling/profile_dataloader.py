"""Standalone dataloader profiler — three phases.

1. __getitem__ + per-modality timing (load + preprocess merged).
2. __getitem__ + load vs preprocess split per modality.
3. DataLoader sweep over num_workers, each in a fresh subprocess with a
   SIGALRM per-batch watchdog and a hard wall budget force-kill.
"""

import logging
import multiprocessing as mp
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
import numpy as np
import torch
import xarray as xr
from omegaconf import DictConfig, OmegaConf

from ai4land.utils.config_utils import UnifiedConfig
from ai4land.utils.datasets import MultiZarrDataset, SingleZarrDataset


MODALITY_METHODS = (
    "_get_luh2_data",
    "_get_population_data",
    "_get_kg_data",
    "_get_static_data",
    "_get_hilda_prior",
    "_get_hilda_target",
)

MODALITIES = ("luh2", "population", "kg", "static", "hilda_prior", "hilda_target")

HILDA_REMAP_KEYS = (11, 22, 33, 40, 41, 42, 43, 44, 45, 55, 66, 77)
HILDA_REMAP_VALUES = (1, 2, 3, 4, 4, 4, 4, 4, 4, 5, 6, 7)

WALL_BUDGET_S = 900


def fmt_bytes(num_bytes: float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:6.2f} {unit}"
        value /= 1024
    return f"{value:6.2f} TB"


def sample_bytes(sample: tuple) -> int:
    return sum(t.element_size() * t.nelement() for t in sample if isinstance(t, torch.Tensor))


def report_times(label: str, times_s: np.ndarray) -> None:
    logging.info(
        f"--- {label} | n={len(times_s)} | "
        f"mean={times_s.mean() * 1000:7.1f} ms | "
        f"median={np.median(times_s) * 1000:7.1f} ms | "
        f"p95={np.percentile(times_s, 95) * 1000:7.1f} ms | "
        f"max={times_s.max() * 1000:7.1f} ms | "
        f"sum={times_s.sum():7.2f} s"
    )


def build_dataset(cfg: UnifiedConfig, land_index_path: str) -> SingleZarrDataset | MultiZarrDataset:
    if isinstance(cfg.data.stores, Path):
        return SingleZarrDataset(cfg.data, land_index_path=land_index_path)
    return MultiZarrDataset(cfg.data, land_index_path=land_index_path)


class TimedSingleZarrDataset(SingleZarrDataset):
    """Splits each `_get_*_data` into a raw-zarr-read phase and a preprocess phase.

    Load = isel() + materialization to memory (`.load()` for Datasets, `.to_numpy()`
    for single DataArrays). Preprocess = everything after: fillna, normalize,
    stack, transpose, cast, remap, clip.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.times_load: dict[str, list[float]] = {m: [] for m in MODALITIES}
        self.times_prep: dict[str, list[float]] = {m: [] for m in MODALITIES}

    def _patch_slice(self, y0: int, x0: int) -> dict:
        size = self.cfg.patch_size
        return {"latitude": slice(y0, y0 + size), "longitude": slice(x0, x0 + size)}

    def _get_luh2_data(self, t_idx, k, y0, x0):
        t0 = time.perf_counter()
        arr = self.dataset["main"][self.features_to_use["luh2"]].isel(
            time=t_idx + k + self.timesteps_dynamic, **self._patch_slice(y0, x0)
        )
        if self.cfg.luh2_load_threads > 1:
            var_names = list(arr.data_vars)
            with ThreadPoolExecutor(max_workers=self.cfg.luh2_load_threads) as executor:
                loaded = list(executor.map(lambda name: arr[name].load(), var_names))
            arr = xr.Dataset({name: data for name, data in zip(var_names, loaded, strict=True)})
        else:
            arr = arr.load()
        t1 = time.perf_counter()

        if not self.cfg.preprocessed:
            arr = arr.fillna(0)
            for secmx in ("secma", "secmb"):
                if secmx in self.features_to_use["luh2"]:
                    mean = self.norm_stats["luh2"][secmx]["mean" + self.luh2_norm_stats]
                    std = self.norm_stats["luh2"][secmx]["std" + self.luh2_norm_stats]
                    arr[secmx] = (arr[secmx] - mean) / std

        arr = arr.to_stacked_array(
            new_dim="variables", sample_dims=("time", "latitude", "longitude")
        ).to_numpy()
        arr = arr.transpose((3, 0, 1, 2)).reshape(-1, self.cfg.patch_size, self.cfg.patch_size)
        t2 = time.perf_counter()

        self.times_load["luh2"].append(t1 - t0)
        self.times_prep["luh2"].append(t2 - t1)
        return arr

    def _get_population_data(self, t_idx, k, y0, x0):
        t0 = time.perf_counter()
        arr = (
            self.dataset["main"][self.features_to_use["population"][0]]
            .isel(time=t_idx + k + self.timesteps_dynamic, **self._patch_slice(y0, x0))
            .to_numpy()
        )
        t1 = time.perf_counter()

        if not self.cfg.preprocessed:
            arr = np.log1p(arr)
        arr = arr.astype(np.float32)
        t2 = time.perf_counter()

        self.times_load["population"].append(t1 - t0)
        self.times_prep["population"].append(t2 - t1)
        return arr

    def _get_kg_data(self, t_idx, k, y0, x0):
        t0 = time.perf_counter()
        arr = (
            self.dataset["main"][self.features_to_use["koppen_geiger"][0]]
            .isel(time=t_idx + k + self.timesteps_dynamic, **self._patch_slice(y0, x0))
            .to_numpy()
        )
        t1 = time.perf_counter()

        arr = arr.astype(np.uint8)
        t2 = time.perf_counter()

        self.times_load["kg"].append(t1 - t0)
        self.times_prep["kg"].append(t2 - t1)
        return arr

    def _get_static_data(self, y0, x0):
        t0 = time.perf_counter()
        arr = (
            self.dataset["main"][self.features_to_use["static"]]
            .isel(**self._patch_slice(y0, x0))
            .load()
        )
        t1 = time.perf_counter()

        if not self.cfg.preprocessed:
            for var in self.features_to_use["static"]:
                if var not in self.static_norm_stats:
                    continue
                if "weighted" in var.lower():
                    arr[var] = arr[var].fillna(0)
                if var == "aspect":
                    arr[var] = np.cos(np.deg2rad(arr[var]))
                else:
                    mean = self.static_norm_stats[var]["mean"]
                    std = self.static_norm_stats[var]["std"]
                    arr[var] = (arr[var] - mean) / std

        arr = arr.to_stacked_array(
            new_dim="variables", sample_dims=("latitude", "longitude")
        ).to_numpy()
        arr = arr.transpose((2, 0, 1)).astype(np.float32)
        t2 = time.perf_counter()

        self.times_load["static"].append(t1 - t0)
        self.times_prep["static"].append(t2 - t1)
        return arr

    def _get_hilda_prior(self, t_idx, k, y0, x0):
        if not self.cfg.autoregression.enable or len(self.timesteps_hilda) == 0:
            return np.array([], dtype=np.float32)

        hilda_var = self.features_to_use["hilda"][0]

        t0 = time.perf_counter()
        arr = (
            self.dataset["main"][hilda_var]
            .isel(time=t_idx + k + self.timesteps_hilda, **self._patch_slice(y0, x0))
            .to_numpy()
            .astype(np.float32)
        )
        t1 = time.perf_counter()

        if not self.cfg.preprocessed:
            np.nan_to_num(arr, copy=False, nan=0)
            remap = np.zeros(256, dtype=np.uint8)
            remap[list(HILDA_REMAP_KEYS)] = list(HILDA_REMAP_VALUES)
            arr = remap[arr.astype(np.intp)].astype(np.float32)
        np.clip(arr, 0, self.cfg.n_hilda_classes - 1, out=arr)
        t2 = time.perf_counter()

        self.times_load["hilda_prior"].append(t1 - t0)
        self.times_prep["hilda_prior"].append(t2 - t1)
        return arr

    def _get_hilda_target(self, t_idx, y0, x0):
        t0 = time.perf_counter()
        arr = (
            self.dataset["main"][self.features_to_use["hilda"][0]]
            .isel(time=t_idx + self.timesteps_target, **self._patch_slice(y0, x0))
            .to_numpy()
        )
        t1 = time.perf_counter()

        if not self.cfg.preprocessed:
            np.nan_to_num(arr, copy=False, nan=0)
            remap = np.zeros(256, dtype=np.uint8)
            remap[list(HILDA_REMAP_KEYS)] = list(HILDA_REMAP_VALUES)
            arr = remap[arr.astype(np.intp)]
        np.clip(arr, 0, self.cfg.n_hilda_classes - 1, out=arr)
        t2 = time.perf_counter()

        self.times_load["hilda_target"].append(t1 - t0)
        self.times_prep["hilda_target"].append(t2 - t1)
        return arr


def _sample_coords(
    dataset: SingleZarrDataset, rng: np.random.Generator
) -> tuple[int, int, int, int]:
    valid_time_map_idx = int(rng.integers(0, dataset.n_valid_times))
    t_idx = int(dataset.valid_time_indices[valid_time_map_idx])

    coord_idx = int(rng.integers(0, len(dataset.land_pixel_coords)))
    coords = dataset.land_pixel_coords[coord_idx]
    center_y, center_x = int(coords[0]), int(coords[1])

    patch = dataset.cfg.patch_size
    y0 = max(0, min(dataset.height - patch, center_y - patch // 2))
    x0 = max(0, min(dataset.width - patch, center_x - patch // 2))
    k = int(dataset.timesteps_target[0])

    return t_idx, k, y0, x0


def phase_cold_warm(
    dataset: SingleZarrDataset, num_coldwarm: int, rng: np.random.Generator
) -> tuple[list[float], list[float]]:
    """Repeated same-coordinate LUH2 fetches. First call = cold (GPFS hit +
    decompress + xarray). Second call = warm (page cache + decompress +
    xarray). Gap = I/O cost contribution."""
    cold_times = []
    warm_times = []

    for _ in range(num_coldwarm):
        t_idx, k, y0, x0 = _sample_coords(dataset, rng)

        t0 = time.perf_counter()
        dataset._get_luh2_data(t_idx, k, y0, x0)
        cold_times.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        dataset._get_luh2_data(t_idx, k, y0, x0)
        warm_times.append((time.perf_counter() - t0) * 1000.0)

    return cold_times, warm_times


def report_cold_warm(cold_times: list[float], warm_times: list[float]) -> None:
    cold_arr = np.array(cold_times)
    warm_arr = np.array(warm_times)
    gap_mean = cold_arr.mean() - warm_arr.mean()
    gap_pct = 100.0 * gap_mean / cold_arr.mean() if cold_arr.mean() > 0 else 0.0

    logging.info(f"--- cold/warm LUH2 | n={len(cold_times)} pairs ---")
    logging.info(
        f"  cold  | mean = {cold_arr.mean():8.1f} ms | median = {np.median(cold_arr):8.1f} ms"
    )
    logging.info(
        f"  warm  | mean = {warm_arr.mean():8.1f} ms | median = {np.median(warm_arr):8.1f} ms"
    )
    logging.info(f"  delta | mean = {gap_mean:8.1f} ms ({gap_pct:5.1f}% of cold)")


def install_modality_timers(dataset) -> dict[str, list[float]]:
    timings: dict[str, list[float]] = {name: [] for name in MODALITY_METHODS}

    def wrap(name: str) -> None:
        original = getattr(dataset, name)

        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = original(*args, **kwargs)
            timings[name].append(time.perf_counter() - start)
            return result

        setattr(dataset, name, wrapper)

    for name in MODALITY_METHODS:
        wrap(name)

    return timings


def profile_getitem_merged(dataset, indices: np.ndarray) -> None:
    logging.info("=" * 80)
    logging.info(f"PHASE 1 — __getitem__ + per-modality merged ({len(indices)} samples)")
    logging.info("=" * 80)

    modality_times = install_modality_timers(dataset)
    times = np.zeros(len(indices))

    for k, idx in enumerate(indices):
        start = time.perf_counter()
        sample = dataset[int(idx)]
        elapsed = time.perf_counter() - start
        times[k] = elapsed

        if k == 0:
            shapes = " ".join(str(tuple(t.shape)) for t in sample if isinstance(t, torch.Tensor))
            logging.info(f"shapes: {shapes} | bytes/sample: {fmt_bytes(sample_bytes(sample))}")

        logging.info(f"  sample {k + 1:3d}/{len(indices)}  {elapsed * 1000:7.1f} ms")

    report_times("getitem", times)
    total = times.sum()

    logging.info("--- per-modality (load + preprocess merged) ---")
    for name in MODALITY_METHODS:
        durations = np.array(modality_times[name])
        logging.info(
            f"  {name:22s}: calls={durations.size:4d} | "
            f"mean={durations.mean() * 1000:7.1f} ms | "
            f"median={np.median(durations) * 1000:7.1f} ms | "
            f"max={durations.max() * 1000:7.1f} ms | "
            f"total={durations.sum():6.2f} s ({100 * durations.sum() / total:4.1f}%)"
        )


def profile_getitem_split(dataset: TimedSingleZarrDataset, indices: np.ndarray) -> None:
    logging.info("=" * 80)
    logging.info(f"PHASE 2 — __getitem__ + load vs preprocess split ({len(indices)} samples)")
    logging.info("=" * 80)

    times = np.zeros(len(indices))

    for k, idx in enumerate(indices):
        start = time.perf_counter()
        dataset[int(idx)]
        times[k] = time.perf_counter() - start
        logging.info(f"  sample {k + 1:3d}/{len(indices)}  {times[k] * 1000:7.1f} ms")

    report_times("getitem", times)
    total = times.sum()

    logging.info("--- per-modality (load | preprocess split) ---")
    for name in MODALITIES:
        load = np.array(dataset.times_load[name])
        prep = np.array(dataset.times_prep[name])
        load_share = 100 * load.sum() / total if total > 0 else 0.0
        prep_share = 100 * prep.sum() / total if total > 0 else 0.0
        logging.info(
            f"  {name:14s}: calls={load.size:4d} | "
            f"load mean={load.mean() * 1000:7.1f} ms "
            f"total={load.sum():6.2f} s ({load_share:4.1f}%) | "
            f"prep mean={prep.mean() * 1000:7.1f} ms "
            f"total={prep.sum():6.2f} s ({prep_share:4.1f}%)"
        )


def _child_dataloader_test(
    cfg_dict: dict,
    land_index_path: str,
    num_workers: int,
    num_batches: int,
    batch_size: int,
    batch_timeout_s: int,
    result_queue: mp.Queue,
) -> None:
    from pathlib import Path as _Path

    from torch.utils.data import DataLoader as _DataLoader

    from ai4land.utils.config_utils import UnifiedConfig as _UnifiedConfig
    from ai4land.utils.datasets import (
        MultiZarrDataset as _MultiZarrDataset,
    )
    from ai4land.utils.datasets import (
        SingleZarrDataset as _SingleZarrDataset,
    )

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - [w={num_workers}] - %(message)s",
        force=True,
    )

    cfg = _UnifiedConfig(**cfg_dict)
    dataset_cls = _SingleZarrDataset if isinstance(cfg.data.stores, _Path) else _MultiZarrDataset
    dataset = dataset_cls(cfg.data, land_index_path=land_index_path)

    if num_workers > 0:
        loader = _DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=4,
        )
    else:
        loader = _DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )

    iterator = iter(loader)
    times: list[float] = []

    def _alarm(_signum, _frame):
        raise TimeoutError

    signal.signal(signal.SIGALRM, _alarm)

    for i in range(num_batches):
        signal.alarm(batch_timeout_s)
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except TimeoutError:
            signal.alarm(0)
            logging.warning(f"batch {i + 1} exceeded {batch_timeout_s}s, aborting")
            result_queue.put({"times": times, "stopped_at": i + 1})
            return

        signal.alarm(0)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        logging.info(
            f"  batch {i + 1:3d}/{num_batches}  {elapsed * 1000:8.1f} ms  "
            f"{fmt_bytes(sample_bytes(batch))}"
        )

    result_queue.put({"times": times, "stopped_at": None})


def _run_worker_config(
    cfg_dict: dict,
    land_index_path: str,
    num_workers: int,
    num_batches: int,
    batch_size: int,
    batch_timeout_s: int,
) -> None:
    logging.info("-" * 80)
    logging.info(f"num_workers = {num_workers}")
    logging.info("-" * 80)

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    child = ctx.Process(
        target=_child_dataloader_test,
        args=(
            cfg_dict,
            land_index_path,
            num_workers,
            num_batches,
            batch_size,
            batch_timeout_s,
            result_queue,
        ),
    )

    child.start()
    child.join(timeout=WALL_BUDGET_S)

    if child.is_alive():
        logging.warning(
            f"num_workers={num_workers}: wall budget {WALL_BUDGET_S}s exceeded, killing"
        )
        child.terminate()
        child.join(10)
        if child.is_alive():
            child.kill()
            child.join()
        return

    result = result_queue.get(timeout=5)
    times = np.array(result["times"])

    if result["stopped_at"]:
        logging.warning(f"num_workers={num_workers}: stopped at batch {result['stopped_at']}")

    if times.size == 0:
        return

    if times.size > 1:
        logging.info(f"cold batch: {times[0] * 1000:.1f} ms (includes worker startup)")
        report_times("warm batches", times[1:])
        wall = times[1:].sum()
        if wall > 0:
            logging.info(
                f"--- throughput | {batch_size * (times.size - 1) / wall:6.1f} samples/s | "
                f"{(times.size - 1) / wall:6.2f} batches/s"
            )
    else:
        logging.info(f"single batch: {times[0] * 1000:.1f} ms")


def profile_dataloader_sweep(
    cfg_dict: dict,
    land_index_path: str,
    cfg: UnifiedConfig,
    num_batches: int,
    sweep: tuple[int, ...],
    batch_timeout_s: int,
) -> None:
    logging.info("=" * 80)
    logging.info(
        f"PHASE 3 — DataLoader sweep over num_workers={list(sweep)} "
        f"(batch_size={cfg.training.batch_size}, prefetch_factor=4 when workers>0, "
        f"pin_memory=True, persistent_workers=False) — {num_batches} batches per config, "
        f"per-batch timeout={batch_timeout_s}s, wall budget={WALL_BUDGET_S}s"
    )
    logging.info("=" * 80)

    for num_workers in sweep:
        _run_worker_config(
            cfg_dict=cfg_dict,
            land_index_path=land_index_path,
            num_workers=num_workers,
            num_batches=num_batches,
            batch_size=cfg.training.batch_size,
            batch_timeout_s=batch_timeout_s,
        )


def setup_logging() -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


_INPUTS_DIR = str(Path(__file__).parents[3] / "inputs")


@hydra.main(config_path=_INPUTS_DIR, config_name="profiling-singlezarr", version_base="1.1")
def main(cfg: DictConfig) -> None:
    setup_logging()

    profile_cfg = cfg.get("profile") or {}
    num_getitem = int(profile_cfg.get("num_getitem", 50))
    num_batches = int(profile_cfg.get("num_batches", 80))
    split = str(profile_cfg.get("split", "train"))
    batch_timeout_s = int(profile_cfg.get("batch_timeout_s", 30))
    sweep = tuple(int(x) for x in profile_cfg.get("workers_sweep", [0, 1, 4, 12]))
    num_coldwarm = int(profile_cfg.get("num_coldwarm", 50))

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    unified = UnifiedConfig(**cfg_dict)

    land_index_path = str(
        unified.data.train_land_index_path if split == "train" else unified.data.val_land_index_path
    )

    logging.info(f"Split: {split} | land_index={land_index_path}")
    logging.info(f"Stores: {unified.data.stores}")
    logging.info(
        f"Sweep: num_workers={list(sweep)} | num_batches={num_batches} | "
        f"batch_timeout={batch_timeout_s}s"
    )

    rng = np.random.default_rng(42)

    dataset = build_dataset(unified, land_index_path)
    logging.info(f"Dataset: {type(dataset).__name__} | length={len(dataset):,}")
    indices = rng.integers(0, len(dataset), size=num_getitem)
    profile_getitem_merged(dataset, indices)
    del dataset

    if isinstance(unified.data.stores, Path):
        timed = TimedSingleZarrDataset(unified.data, land_index_path=land_index_path)
        indices = rng.integers(0, len(timed), size=num_getitem)
        profile_getitem_split(timed, indices)
        del timed
    else:
        logging.info(
            "PHASE 2 skipped — load/preprocess split only implemented for SingleZarrDataset"
        )

    if isinstance(unified.data.stores, Path) and num_coldwarm > 0:
        logging.info("=" * 80)
        logging.info(f"PHASE 2b — cold/warm LUH2 ({num_coldwarm} pairs)")
        logging.info("=" * 80)
        cw_dataset = SingleZarrDataset(unified.data, land_index_path=land_index_path)
        cw_rng = np.random.default_rng(43)
        cold_times, warm_times = phase_cold_warm(cw_dataset, num_coldwarm, cw_rng)
        report_cold_warm(cold_times, warm_times)
        del cw_dataset

    profile_dataloader_sweep(
        cfg_dict=cfg_dict,
        land_index_path=land_index_path,
        cfg=unified,
        num_batches=num_batches,
        sweep=sweep,
        batch_timeout_s=batch_timeout_s,
    )

    logging.info("=" * 80)
    logging.info("Profiling complete.")


if __name__ == "__main__":
    main()
