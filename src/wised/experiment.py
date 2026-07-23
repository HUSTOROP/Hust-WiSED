from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .data_io import (
    build_training_tensors,
    extract_dataset_bundle,
    load_or_generate_dataset,
    log_dataset_summary,
)
from .paths import LOGS_DIR, RESULTS_DIR, ensure_project_root_on_path, task_output_dir
from .reproducibility import seed_everything

ensure_project_root_on_path()

from .experiment_config import build_run_config, case_dir_name
from run_common import (
    discover_multifield_joint,
    make_context_list,
    normalize_target_fields,
    save_multifield_summary,
)
from utils.experiment_logger import WiSEDLogger


try:
    from engineering_validation.soc_eis_impedance_manifold.postprocess import postprocess_soc_eis_impedance_manifold_results
except Exception:
    postprocess_soc_eis_impedance_manifold_results = None

try:
    from engineering_validation.am_bench.postprocess import postprocess_am_bench_thermal_results
except Exception:
    postprocess_am_bench_thermal_results = None

try:
    from engineering_validation.chua.postprocess import postprocess_chua_results
except Exception:
    postprocess_chua_results = None

try:
    from engineering_validation.chua_modewise.postprocess_modewise import postprocess_chua_modewise_results
except Exception:
    postprocess_chua_modewise_results = None

try:
    from engineering_validation.health_status.postprocess import postprocess_health_status_results
except Exception:
    postprocess_health_status_results = None

try:
    from engineering_validation.noaa_sst_core_field_pde.postprocess import postprocess_noaa_sst_core_results
except Exception:
    postprocess_noaa_sst_core_results = None

try:
    from engineering_validation.sib_diffusion_raw_field.analyze_sib_diffusion import (
        postprocess_sib_diffusion_results,
    )
except Exception:
    postprocess_sib_diffusion_results = None


def resolve_run_config(
    equation: str,
    *,
    device: Optional[str],
    seed: Optional[int],
    noise_level: Optional[float],
    config_overrides: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the centralized equation/noise/seed configuration."""
    return build_run_config(
        equation,
        noise_level=noise_level,
        seed=seed,
        device=device,
        overrides=config_overrides,
    )


def default_targets_for_equation(
    equation: str,
    cfg: Dict[str, Any],
    field_names: List[str],
    explicit_targets: Optional[List[str]],
    logger: WiSEDLogger,
    grid_info: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Resolve target fields.

    For engineering datasets, field_names are all channels available to the
    unrestricted RHS library, while discovery_targets are the state equations to
    discover.  This preserves u/v/w internally without treating auxiliary v/w as
    targets.
    """
    targets = explicit_targets
    grid_info = grid_info or {}
    if targets is None and grid_info.get("discovery_targets"):
        targets = list(grid_info.get("discovery_targets", []))
        logger.info(f"Using dataset discovery targets: {targets}. Pass --targets to override.")
    if targets is None and "default_targets" in cfg:
        targets = list(cfg["default_targets"])
        logger.info(f"Using equation default discovery targets: {targets}. Pass --targets to override.")
    _ = equation
    return normalize_target_fields(targets, field_names)


def run_discovery_experiment(
    equation: str,
    *,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    noise_level: Optional[float] = None,
    targets: Optional[List[str]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    results_root: Union[str, Path] = RESULTS_DIR,
    logs_root: Union[str, Path] = LOGS_DIR,
) -> Dict[str, Dict[str, Any]]:
    """Run the full data, weak-form discovery and postprocessing pipeline."""
    cfg = resolve_run_config(
        equation,
        device=device,
        seed=seed,
        noise_level=noise_level,
        config_overrides=config_overrides,
    )
    task_name = str(equation)
    case_name = case_dir_name(noise_level=float(cfg["noise_level"]), seed=int(cfg["seed"]))
    # Keep the scientific task name independent from the on-disk artifact name.
    # Engineering cases with long descriptive names can choose a short artifact
    # prefix/directory to avoid Windows MAX_PATH failures while still resolving
    # the original task-specific dataset and postprocess logic by task_name.
    artifact_prefix = str(cfg.get("artifact_prefix") or task_name)
    case_file_tag = f"{artifact_prefix}_{case_name}"
    output_task_dir_name = str(cfg.get("output_task_dir_name") or task_name)
    result_dir = task_output_dir(task_output_dir(results_root, output_task_dir_name), case_name)
    log_dir = task_output_dir(task_output_dir(logs_root, output_task_dir_name), case_name)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    seed_everything(int(cfg["seed"]))
    logger = WiSEDLogger(str(log_dir), case_file_tag)
    logger.log_config(cfg)

    data_path, saved = load_or_generate_dataset(task_name, cfg, logger)
    n_required = int(cfg.get("n_train_samples", 8))
    bundle = extract_dataset_bundle(saved, n_required, data_path)
    log_dataset_summary(logger, task_name, bundle)

    data_tensor, _ = build_training_tensors(
        bundle.raw,
        bundle.clean,
        bundle.coords,
        bundle.field_names,
    )

    ctx_list = make_context_list(
        bundle.raw,
        bundle.coords,
        bundle.periodic_axes,
        bundle.field_names,
        cfg,
        cache_prefix=task_name,
    )
    target_fields = default_targets_for_equation(task_name, cfg, bundle.field_names, targets, logger, bundle.grid_info)
    logger.log_kv("TARGETS", {"requested_targets": target_fields})

    discovered, train_elapsed = discover_multifield_joint(
        task_prefix=case_file_tag,
        target_fields=target_fields,
        cfg=cfg,
        data_tensor=data_tensor,
        ctx_list=ctx_list,
        result_root=str(result_dir),
        log_root=str(log_dir),
        true_eq=bundle.true_equation,
        logger=logger,
    )

    summary_filename = f"{case_file_tag}_summary.json"
    save_multifield_summary(
        discovered,
        bundle.true_equation,
        str(result_dir),
        summary_filename,
        logger,
    )

    if task_name.startswith("soc_eis_impedance_manifold") and postprocess_soc_eis_impedance_manifold_results is not None:
        try:
            post_path = postprocess_soc_eis_impedance_manifold_results(
                task_name=task_name,
                dataset_path=str(bundle.path),
                summary_path=str(Path(result_dir) / summary_filename),
                result_dir=str(result_dir),
            )
            logger.info(f"SoC-EIS manifold PDE postprocess saved: {Path(post_path).name}")
        except Exception as exc:
            logger.warning(f"SoC-EIS manifold PDE postprocess failed: {exc}")

    if task_name.startswith("am_bench_thermal") and postprocess_am_bench_thermal_results is not None:
        try:
            post_path = postprocess_am_bench_thermal_results(
                task_name=task_name,
                dataset_path=str(bundle.path),
                summary_path=str(Path(result_dir) / summary_filename),
                result_dir=str(result_dir),
            )
            logger.info(f"AM Bench thermography postprocess saved: {Path(post_path).name}")
        except Exception as exc:
            logger.warning(f"AM Bench thermography postprocess failed: {exc}")

    if task_name.startswith("chua_circuit") and postprocess_chua_results is not None:
        try:
            post_path = postprocess_chua_results(
                task_name=task_name,
                dataset_path=str(bundle.path),
                summary_path=str(Path(result_dir) / summary_filename),
                result_dir=str(result_dir),
            )
            logger.info(f"Chua circuit postprocess saved: {Path(post_path).name}")
        except Exception as exc:
            logger.warning(f"Chua circuit postprocess failed: {exc}")

    if task_name.startswith("chua_mode") and postprocess_chua_modewise_results is not None:
        try:
            post_path = postprocess_chua_modewise_results(
                task_name=task_name,
                dataset_path=str(bundle.path),
                summary_path=str(Path(result_dir) / summary_filename),
                result_dir=str(result_dir),
            )
            logger.info(f"Chua mode-wise postprocess saved: {Path(post_path).name}")
        except Exception as exc:
            logger.warning(f"Chua mode-wise postprocess failed: {exc}")

    if task_name.startswith("health_status") and postprocess_health_status_results is not None:
        try:
            post_path = postprocess_health_status_results(
                task_name=task_name,
                dataset_path=str(bundle.path),
                summary_path=str(Path(result_dir) / summary_filename),
                result_dir=str(result_dir),
            )
            logger.info(f"Health_status postprocess saved: {Path(post_path).name}")
        except Exception as exc:
            logger.warning(f"Health_status postprocess failed: {exc}")

    if task_name.startswith("noaa_sst_core_field_pde") and postprocess_noaa_sst_core_results is not None:
        try:
            post_path = postprocess_noaa_sst_core_results(
                task_name=task_name,
                dataset_path=str(bundle.path),
                summary_path=str(Path(result_dir) / summary_filename),
                result_dir=str(result_dir),
            )
            logger.info(f"NOAA SST core-field postprocess saved: {Path(post_path).name}")
        except Exception as exc:
            logger.warning(f"NOAA SST core-field postprocess failed: {exc}")

    if task_name.startswith("sib_diffusion_raw_field") and postprocess_sib_diffusion_results is not None:
        try:
            post_path = postprocess_sib_diffusion_results(
                task_name=task_name,
                dataset_path=str(bundle.path),
                summary_path=str(Path(result_dir) / summary_filename),
                result_dir=str(result_dir),
            )
            logger.info(f"SIB raw-concentration postprocess saved: {Path(post_path).name}")
        except Exception as exc:
            logger.warning(f"SIB raw-concentration postprocess failed: {exc}")

    logger.final_summary(
        {
            "task": task_name,
            "targets": ", ".join(target_fields),
            "fields": ", ".join(bundle.field_names),
            "n_train_samples": int(n_required),
            "summary_file": summary_filename,
        },
        title="RUN SUMMARY",
        elapsed_sec=train_elapsed,
        elapsed_label="Training elapsed",
    )
    return discovered
