from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from .paths import DATASET_DIR
from .registries import get_equation_generator

COORD_AXES = ("t", "x", "y", "z")


@dataclass(frozen=True)
class DatasetBundle:
    """In-memory view of one cached/generated experiment dataset."""

    path: Path
    raw: np.ndarray
    clean: np.ndarray
    grid_info: Dict[str, Any]
    true_equation: Any
    coords: Dict[str, np.ndarray]
    field_names: List[str]
    periodic_axes: Dict[str, bool]


def to_numpy(obj: Any) -> np.ndarray:
    """Convert torch tensors or array-like objects to NumPy arrays."""
    return obj.detach().cpu().numpy() if hasattr(obj, "detach") else np.asarray(obj)


def dataset_path(task_name: str, noise_level: float, dataset_dir: Union[str, Path] = DATASET_DIR) -> Path:
    """Return the canonical cached dataset path for a task/noise pair."""
    return Path(dataset_dir) / f"{task_name}_noise{float(noise_level)}.npz"


def _as_compare_list(value: Any) -> List[str]:
    """Return a normalized list used only for cache-configuration checks."""
    if value is None:
        return []
    if isinstance(value, str):
        import re

        return [p.strip() for p in re.split(r"[,;]+", value) if p.strip()]
    try:
        return [str(p).strip() for p in value if str(p).strip()]
    except TypeError:
        text = str(value).strip()
        return [text] if text else []


def _soc_eis_cache_matches_requested_kwargs(preprocess: Dict[str, Any], requested: Dict[str, Any], logger: Any) -> bool:
    """Reject SoC-EIS caches produced by a different engineering protocol.

    The canonical cache filename only depends on task/noise.  Without this check,
    a previous all-train or battery-holdout dataset can be reused accidentally for
    the fixed Test_1 -> Test_2 validation.
    """
    if not requested:
        return True

    scalar_keys = [
        "split_mode",
        "frequency_band",
        "freq_preprocess_mode",
        "n_logfreq_grid",
        "n_rc_elements",
        "normalization_mode",
        "protocol",
        "protocol_role",
        "ohmic_alignment",
        "freq_weight_mode",
        "soc_smooth_window",
        "exclude_soc100",
    ]
    aliases = {
        "freq_preprocess_mode": "frequency_preprocess_mode",
        "n_logfreq_grid": "n_logfreq_grid",
    }
    for key in scalar_keys:
        if key not in requested:
            continue
        meta_key = aliases.get(key, key)
        rv = requested.get(key)
        mv = preprocess.get(meta_key)
        if isinstance(rv, bool):
            same = bool(mv) == bool(rv)
        elif isinstance(rv, int):
            try:
                same = int(mv) == int(rv)
            except Exception:
                same = False
        elif isinstance(rv, float):
            try:
                same = abs(float(mv) - float(rv)) <= 1e-12
            except Exception:
                same = False
        else:
            same = str(mv).lower() == str(rv).lower()
        if not same:
            logger.warning(f"Cached SoC-EIS dataset protocol mismatch for {key}: cached={mv!r}, requested={rv!r}; regenerating.")
            return False

    list_keys = ["selected_battery_ids", "selected_cycle_ids", "train_cycle_ids", "val_cycle_ids", "test_cycle_ids"]
    for key in list_keys:
        if key not in requested:
            continue
        rv = sorted(_as_compare_list(requested.get(key)))
        mv = sorted(_as_compare_list(preprocess.get(key)))
        if rv != mv:
            logger.warning(f"Cached SoC-EIS dataset protocol mismatch for {key}: cached={mv!r}, requested={rv!r}; regenerating.")
            return False

    # For the repeat-cycle validation, the full cached dataset must contain both
    # train and held-out test surfaces even though WiSED receives only train rows.
    if str(requested.get("split_mode", preprocess.get("split_mode", ""))).lower() in {"fixed_cycle_transfer", "cross_cycle", "test1_to_test2", "repeat_holdout"}:
        splits = preprocess.get("splits") or {}
        # split metadata lives at grid_info["splits"], so callers patch it into
        # preprocess as "_splits_for_cache_check" below.
        splits = preprocess.get("_splits_for_cache_check", splits)
        if not splits.get("train") or not splits.get("test"):
            logger.warning("Cached SoC-EIS fixed-cycle-transfer dataset lacks train/test surfaces; regenerating.")
            return False
    return True




def _am_bench_cache_matches_requested_kwargs(preprocess: Dict[str, Any], requested: Dict[str, Any], logger: Any) -> bool:
    """Reject AM Bench caches produced from a different thermography protocol."""
    if not requested:
        return True
    scalar_keys = [
        "test_name", "phase", "nx", "ny", "nt", "frame_stride", "segment_stride",
        "max_segments", "source_proxy", "source_proxy_mode", "expose_source_channel", "source_field_name", "split_mode", "train_fraction", "active_threshold_c",
        "spatial_smooth_sigma", "spatial_window_mode", "spatial_window_y_px", "spatial_window_x_px",
        "protocol", "scan_speed_mm_s", "laser_power_w", "pad_line_count", "min_required_surfaces", "min_required_test_surfaces",
    ]
    for key in scalar_keys:
        if key not in requested:
            continue
        rv = requested.get(key)
        mv = preprocess.get(key)
        if isinstance(rv, bool):
            same = bool(mv) == bool(rv)
        elif isinstance(rv, int):
            try:
                same = int(mv) == int(rv)
            except Exception:
                same = False
        elif isinstance(rv, float):
            try:
                same = abs(float(mv) - float(rv)) <= 1e-12
            except Exception:
                same = False
        else:
            same = str(mv).lower() == str(rv).lower()
        if not same:
            logger.warning(f"Cached AM-Bench dataset protocol mismatch for {key}: cached={mv!r}, requested={rv!r}; regenerating.")
            return False
    return True

def cached_dataset_satisfies_task_requirements(task_name: str, saved: Any, logger: Any, requested_soc_eis_kwargs: Dict[str, Any] | None = None) -> bool:
    """Reject stale cached datasets whose channel layout or protocol cannot support a task."""
    requested_kwargs = dict(requested_soc_eis_kwargs or {})
    if task_name.startswith("soc_eis_impedance_manifold"):
        try:
            data = np.asarray(saved["data"])
            grid_info = saved["grid_info"].item()
            field_names = list(grid_info.get("field_names", []))
            preprocess = dict(grid_info.get("preprocess", {}))
            preprocess["_splits_for_cache_check"] = dict(grid_info.get("splits", {}))
        except Exception as exc:
            logger.warning(f"{task_name} cache metadata is unreadable ({exc}); regenerating.")
            return False
        ok_channels = data.ndim == 4 and int(data.shape[-1]) == 2 and int(data.shape[1]) >= 3 and int(data.shape[2]) >= 8
        ok_fields = field_names[:2] == ["u", "v"]
        ok_version = str(preprocess.get("preprocess_version", "")) == "soc_eis_impedance_manifold"
        ok_contract = not bool(preprocess.get("manual_derivative_terminals_exported", True)) and not bool(preprocess.get("soc_as_scalar_feature", True))
        if not (ok_channels and ok_fields and ok_version and ok_contract):
            logger.warning("Cached SoC-EIS dataset is stale or not in raw manifold-PDE format; regenerating.")
            return False
        if not _soc_eis_cache_matches_requested_kwargs(preprocess, requested_soc_eis_kwargs or {}, logger):
            return False
        return True

    if task_name.startswith("am_bench_thermal"):
        try:
            data = np.asarray(saved["data"])
            grid_info = saved["grid_info"].item()
            field_names = list(grid_info.get("field_names", []))
            preprocess = dict(grid_info.get("preprocess", {}))
        except Exception as exc:
            logger.warning(f"{task_name} cache metadata is unreadable ({exc}); regenerating.")
            return False
        ok_shape = data.ndim == 5 and int(data.shape[-1]) >= 1 and int(data.shape[1]) >= 5 and int(data.shape[2]) >= 8 and int(data.shape[3]) >= 8
        ok_fields = bool(field_names) and field_names[0] == "u" and len(field_names) == int(data.shape[-1])
        ok_version = str(preprocess.get("preprocess_version", "")).startswith("am_bench_thermal_v")
        if not (ok_shape and ok_fields and ok_version):
            logger.warning("Cached AM-Bench dataset is stale or not in a supported AM-Bench thermal format; regenerating.")
            return False
        if not _am_bench_cache_matches_requested_kwargs(preprocess, requested_soc_eis_kwargs or {}, logger):
            return False
        return True

    if task_name.startswith("battery_thermal_fault"):
        try:
            data = np.asarray(saved["data"])
            grid_info = saved["grid_info"].item()
            field_names = list(grid_info.get("field_names", []))
            preprocess = dict(grid_info.get("preprocess", {}))
        except Exception as exc:
            logger.warning(f"{task_name} cache metadata is unreadable ({exc}); regenerating.")
            return False
        ok_shape = data.ndim == 4 and int(data.shape[-1]) == 3 and int(data.shape[1]) >= 32 and int(data.shape[2]) >= 3
        ok_fields = field_names == ["u", "v", "w"]
        ok_version = str(preprocess.get("preprocess_version", "")).startswith("battery_thermal_fault_open_ode_v")
        ok_open = not bool(preprocess.get("candidate_library_used", True))
        if not (ok_shape and ok_fields and ok_version and ok_open):
            logger.warning("Cached battery thermal fault dataset is stale or not open-discovery compatible; regenerating.")
            return False
        return True

    if task_name.startswith("chua_circuit"):
        try:
            data = np.asarray(saved["data"])
            grid_info = saved["grid_info"].item()
            field_names = list(grid_info.get("field_names", []))
            preprocess = dict(grid_info.get("preprocess", {}))
        except Exception as exc:
            logger.warning(f"{task_name} cache metadata is unreadable ({exc}); regenerating.")
            return False
        ok_shape = data.ndim == 4 and int(data.shape[-1]) == 2 and int(data.shape[1]) >= 32 and int(data.shape[2]) >= 3
        ok_fields = field_names == ["u", "v"]
        ok_version = str(preprocess.get("preprocess_version", "")).startswith("chua_circuit_open_ode_v")
        if not (ok_shape and ok_fields and ok_version):
            logger.warning("Cached Chua circuit dataset is stale or not open-ODE compatible; regenerating.")
            return False
        return True

    if task_name.startswith("chua_mode"):
        try:
            data = np.asarray(saved["data"])
            grid_info = saved["grid_info"].item()
            field_names = list(grid_info.get("field_names", []))
            targets = list(grid_info.get("discovery_targets", []))
            preprocess = dict(grid_info.get("preprocess", {}))
        except Exception as exc:
            logger.warning(f"{task_name} cache metadata is unreadable ({exc}); regenerating.")
            return False
        ok_shape = data.ndim == 4 and int(data.shape[-1]) == 2 and int(data.shape[1]) >= 17 and int(data.shape[2]) >= 3
        ok_fields = field_names == ["u", "v"] and targets == ["u"]
        ok_version = str(preprocess.get("preprocess_version", "")).startswith("chua_modewise_dy1_v")
        requested_mode = None
        if "mode_id" in requested_kwargs:
            try:
                requested_mode = int(requested_kwargs.get("mode_id"))
            except Exception:
                requested_mode = None
        task_mode = None
        import re
        m = re.search(r"chua_mode(\d+)_dy1", str(task_name))
        if m:
            task_mode = int(m.group(1))
        cached_mode = preprocess.get("mode_id")
        ok_mode = True
        try:
            cached_mode_int = int(cached_mode)
            if task_mode is not None:
                ok_mode = cached_mode_int == task_mode
            if requested_mode is not None:
                ok_mode = ok_mode and cached_mode_int == requested_mode
        except Exception:
            ok_mode = False
        if not (ok_shape and ok_fields and ok_version and ok_mode):
            logger.warning("Cached Chua mode-wise dataset is stale or not compatible; regenerating.")
            return False
        return True

    if task_name.startswith("health_status"):
        try:
            data = np.asarray(saved["data"])
            grid_info = saved["grid_info"].item()
            field_names = list(grid_info.get("field_names", []))
            targets = list(grid_info.get("discovery_targets", []))
            preprocess = dict(grid_info.get("preprocess", {}))
        except Exception as exc:
            logger.warning(f"{task_name} cache metadata is unreadable ({exc}); regenerating.")
            return False
        ok_shape = data.ndim == 4 and int(data.shape[-1]) == 2 and int(data.shape[1]) >= 8 and int(data.shape[2]) >= 3
        ok_fields = field_names == ["u", "v"] and targets == ["u"]
        ok_version = str(preprocess.get("preprocess_version", "")).startswith("health_status_ode_v")
        if not (ok_shape and ok_fields and ok_version):
            logger.warning("Cached Health_status dataset is stale or not compatible; regenerating.")
            return False
        return True

    if task_name.startswith("noaa_sst_core_field"):
        try:
            data = np.asarray(saved["data"])
            grid_info = saved["grid_info"].item()
            field_names = list(grid_info.get("field_names", []))
            preprocess = dict(grid_info.get("preprocess", {}))
        except Exception as exc:
            logger.warning(f"NOAA SST core-field cache metadata is unreadable ({exc}); regenerating.")
            return False
        ok_shape = data.ndim == 5 and int(data.shape[-1]) == 1 and int(data.shape[1]) >= 9 and int(data.shape[2]) >= 8 and int(data.shape[3]) >= 8
        ok_fields = field_names == ["u"]
        ok_version = str(preprocess.get("preprocess_version", "")).startswith("noaa_sst_core_field_v")
        ok_contract = not bool(preprocess.get("manual_derivative_terminals_exported", True))
        if not (ok_shape and ok_fields and ok_version and ok_contract):
            logger.warning("Cached NOAA SST dataset is stale or not native-field compatible; regenerating.")
            return False
        for key in ("field_transform", "normalization", "variable", "n_days", "nx", "ny"):
            if key in requested_kwargs and str(preprocess.get(key, "")).lower() != str(requested_kwargs.get(key, "")).lower():
                logger.warning(f"Cached NOAA SST core-field protocol mismatch for {key}: cached={preprocess.get(key)!r}, requested={requested_kwargs.get(key)!r}; regenerating.")
                return False
        return True

    if task_name.startswith("sib_diffusion_raw_field"):
        try:
            data = np.asarray(saved["data"])
            grid_info = saved["grid_info"].item()
            field_names = list(grid_info.get("field_names", []))
            targets = list(grid_info.get("discovery_targets", []))
            preprocess = dict(grid_info.get("preprocess", {}))
        except Exception as exc:
            logger.warning(f"SIB diffusion core-field cache metadata is unreadable ({exc}); regenerating.")
            return False
        ok_shape = data.ndim == 4 and int(data.shape[-1]) in {1, 2} and int(data.shape[1]) >= 9 and int(data.shape[2]) >= 21
        ok_fields = field_names[:1] == ["u"] and targets == ["u"]
        ok_version = str(preprocess.get("preprocess_version", "")).startswith("sib_diffusion_open_field_v")
        ok_contract = not bool(preprocess.get("manual_derivative_terminals_exported", True))
        if not (ok_shape and ok_fields and ok_version and ok_contract):
            logger.warning("Cached SIB diffusion dataset is stale or not native-field compatible; regenerating.")
            return False
        for key in (
            "nx",
            "tau_max",
            "save_every",
            "dependent_transform",
            "geometry_channel",
            "geometry_channel_scale",
            "clip_during_simulation",
            "crop_to_physical_window",
            "physical_u_min",
            "physical_u_max",
        ):
            if key in requested_kwargs and str(preprocess.get(key, "")).lower() != str(requested_kwargs.get(key, "")).lower():
                logger.warning(f"Cached SIB core-field protocol mismatch for {key}: cached={preprocess.get(key)!r}, requested={requested_kwargs.get(key)!r}; regenerating.")
                return False
        return True

    return True

def load_or_generate_dataset(
    task_name: str,
    cfg: Dict[str, Any],
    logger: Any,
    *,
    dataset_dir: Union[str, Path] = DATASET_DIR,
) -> Tuple[Path, Any]:
    """Load a cached dataset, or generate it with the configured seed."""
    noise_level = float(cfg.get("noise_level", 0.0))
    n_required = int(cfg.get("n_train_samples", 8))
    path = dataset_path(task_name, noise_level, dataset_dir)

    saved = None
    use_cache = False
    force_regenerate = bool(cfg.get("force_regenerate_dataset", False))
    if path.exists() and not force_regenerate:
        try:
            saved = np.load(path, allow_pickle=True)
            cache_required = n_required
            use_cache = int(saved["data"].shape[0]) >= cache_required
            if use_cache:
                requested_kwargs = {}
                if str(task_name).startswith("soc_eis"):
                    requested_kwargs = dict(cfg.get("soc_eis_generator_kwargs", {}))
                elif str(task_name).startswith("am_bench_thermal"):
                    requested_kwargs = dict(cfg.get("am_bench_generator_kwargs", {}))
                elif str(task_name).startswith("battery_thermal_fault"):
                    requested_kwargs = dict(cfg.get("battery_thermal_fault_generator_kwargs", {}))
                elif str(task_name).startswith("chua_circuit"):
                    requested_kwargs = dict(cfg.get("chua_generator_kwargs", {}))
                elif str(task_name).startswith("chua_mode"):
                    requested_kwargs = dict(cfg.get("chua_modewise_generator_kwargs", {}))
                elif str(task_name).startswith("health_status"):
                    requested_kwargs = dict(cfg.get("health_status_generator_kwargs", {}))
                elif str(task_name).startswith("noaa_sst_core_field"):
                    requested_kwargs = dict(cfg.get("noaa_sst_core_field_generator_kwargs", {}))
                elif str(task_name).startswith("sib_diffusion_raw_field"):
                    requested_kwargs = dict(cfg.get("sib_diffusion_generator_kwargs", {}))
                use_cache = cached_dataset_satisfies_task_requirements(task_name, saved, logger, requested_kwargs)
            if use_cache:
                print(f"Using cached dataset -> {path}")
                logger.info("Using cached dataset.")
        except Exception as exc:
            logger.warning(f"Failed to read cached dataset ({exc}); regenerating.")
            saved = None

    if not use_cache:
        logger.info(
            f"Generating dataset: task={task_name} | noise={noise_level} | "
            f"required_samples={n_required}"
        )
        generator = get_equation_generator(task_name)
        generator_kwargs = {}
        if str(task_name).startswith("soc_eis"):
            generator_kwargs.update(dict(cfg.get("soc_eis_generator_kwargs", {})))
        elif str(task_name).startswith("am_bench_thermal"):
            generator_kwargs.update(dict(cfg.get("am_bench_generator_kwargs", {})))
        elif str(task_name).startswith("battery_thermal_fault"):
            generator_kwargs.update(dict(cfg.get("battery_thermal_fault_generator_kwargs", {})))
        elif str(task_name).startswith("chua_circuit"):
            generator_kwargs.update(dict(cfg.get("chua_generator_kwargs", {})))
        elif str(task_name).startswith("chua_mode"):
            generator_kwargs.update(dict(cfg.get("chua_modewise_generator_kwargs", {})))
        elif str(task_name).startswith("health_status"):
            generator_kwargs.update(dict(cfg.get("health_status_generator_kwargs", {})))
        elif str(task_name).startswith("noaa_sst_core_field"):
            generator_kwargs.update(dict(cfg.get("noaa_sst_core_field_generator_kwargs", {})))
        elif str(task_name).startswith("sib_diffusion_raw_field"):
            generator_kwargs.update(dict(cfg.get("sib_diffusion_generator_kwargs", {})))
            generator_kwargs.setdefault("dataset_name", str(task_name))
        generator(
            save_dir=str(dataset_dir),
            noise_level=noise_level,
            batch_size=max(32, n_required),
            seed=int(cfg.get("seed", 42)),
            **generator_kwargs,
        )
        saved = np.load(path, allow_pickle=True)

    return path, saved


def build_coords(grid_info: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract ordered coordinate arrays from dataset metadata."""
    coords: Dict[str, np.ndarray] = {}
    for axis in COORD_AXES:
        key = f"{axis}_coords"
        if key in grid_info:
            coords[axis] = to_numpy(grid_info[key])

    if "t" not in coords or "x" not in coords:
        raise ValueError("Dataset must contain at least t_coords and x_coords.")
    return coords


def extract_dataset_bundle(saved: Any, n_required: int, path: Union[str, Path]) -> DatasetBundle:
    """Extract the training subset and metadata from a .npz dataset.

    Standard PDE benchmarks use the first ``n_required`` trajectories.  The
    SoC-EIS spectral engineering case instead stores the author's exact
    train/test split in metadata; discovery must use only the author-compatible
    training indices so that the held-out test rows match the paper code.
    """
    grid_info = saved["grid_info"].item()
    data = np.asarray(saved["data"])
    data_clean = np.asarray(saved["data_clean"])
    preprocess = dict(grid_info.get("preprocess", {}))
    pre_version = str(preprocess.get("preprocess_version", ""))
    if pre_version in {"soc_eis_impedance_manifold", "sib_diffusion_raw_field"}:
        idx = np.asarray(grid_info.get("discovery_indices", grid_info.get("splits", {}).get("train", [])), dtype=int)
        if idx.size == 0:
            idx = np.arange(min(int(n_required), data.shape[0]), dtype=int)
        raw = data[idx]
        clean = data_clean[idx]
    else:
        raw = data[:n_required]
        clean = data_clean[:n_required]
    true_equation = saved["true_eq"].item() if hasattr(saved["true_eq"], "item") else str(saved["true_eq"])
    coords = build_coords(grid_info)
    field_names = list(grid_info.get("field_names", ["u"]))
    periodic_axes = dict(grid_info.get("periodic_axes", {"x": True}))
    return DatasetBundle(
        path=Path(path),
        raw=raw,
        clean=clean,
        grid_info=grid_info,
        true_equation=true_equation,
        coords=coords,
        field_names=field_names,
        periodic_axes=periodic_axes,
    )


def build_training_tensors(
    raw_array: np.ndarray,
    clean_array: np.ndarray,
    coords: Dict[str, np.ndarray],
    field_names: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return tensors with an explicit final channel dimension."""
    expected_without_channel = len(coords) + 1
    expected_with_channel = len(coords) + 2

    if raw_array.ndim == expected_without_channel:
        data_tensor = torch.as_tensor(raw_array, dtype=torch.float32).unsqueeze(-1)
        clean_tensor = torch.as_tensor(clean_array, dtype=torch.float32).unsqueeze(-1)
    elif raw_array.ndim == expected_with_channel:
        data_tensor = torch.as_tensor(raw_array, dtype=torch.float32)
        clean_tensor = torch.as_tensor(clean_array, dtype=torch.float32)
    else:
        raise ValueError(
            f"Unexpected data shape {raw_array.shape}. Expected ndim "
            f"{expected_without_channel} or {expected_with_channel} for coords={list(coords)}."
        )

    n_channels = int(data_tensor.shape[-1])
    if len(field_names) != n_channels:
        raise ValueError(f"Field/channel mismatch: field_names={field_names}, channels={n_channels}.")
    return data_tensor, clean_tensor


def log_dataset_summary(logger: Any, task_name: str, bundle: DatasetBundle) -> None:
    """Write a compact, reproducibility-oriented dataset summary to the log."""
    preprocess = dict(bundle.grid_info.get("preprocess", {})) if isinstance(bundle.grid_info, dict) else {}
    splits = dict(bundle.grid_info.get("splits", {})) if isinstance(bundle.grid_info, dict) else {}
    full_n = int(preprocess.get("n_surfaces", len(bundle.grid_info.get("field_meta", [])) if isinstance(bundle.grid_info, dict) else bundle.raw.shape[0]))
    logger.log_kv(
        "DATASET SUMMARY",
        {
            "task": task_name,
            "training_raw_shape": tuple(bundle.raw.shape),
            "training_clean_shape": tuple(bundle.clean.shape),
            "full_cached_n_surfaces": full_n,
            "full_cached_split_sizes": {k: len(v) for k, v in splits.items()},
            "full_cached_splits": splits,
            "discovery_indices": bundle.grid_info.get("discovery_indices", []) if isinstance(bundle.grid_info, dict) else [],
            "fields": bundle.field_names,
            "coords": {axis: int(len(values)) for axis, values in bundle.coords.items()},
            "periodic_axes": bundle.periodic_axes,
        },
    )
