from __future__ import annotations

"""Thin interface between the SOC-EIS adapter and the WiSED runner."""

from pathlib import Path
from typing import Any, Dict, Sequence

from src.wised.experiment import run_discovery_experiment


def build_wised_overrides(
    *,
    data_dir: str | Path,
    test_battery: str,
    val_battery: str | None,
    frequency_band: str,
    n_logfreq_grid: int,
    freq_preprocess_mode: str,
    n_rc_elements: int,
    soc_smooth_window: int,
    soc_smooth_polyorder: int,
    exclude_soc100: bool,
    force_regenerate: bool,
    n_train_samples: int,
    weak_window_t: int,
    weak_window_x: int,
    mask_trim_t: int,
    mask_trim_x: int,
    max_eq_len: int,
    n_epochs: int,
    n_eq_samples: int,
    pop_size: int,
    n_offspring: int,
    init_random_trials: int,
    full_refine_topk: int,
    search_context_budget: int,
) -> Dict[str, Any]:
    generator_kwargs = {
        "raw_dir": str(Path(data_dir).resolve()),
        "split_mode": "leave_one_battery_out",
        "test_battery_ids": (str(test_battery),),
        "val_battery_ids": (str(val_battery),) if val_battery else tuple(),
        "frequency_band": str(frequency_band),
        "n_logfreq_grid": int(n_logfreq_grid),
        "exclude_soc100": bool(exclude_soc100),
        "freq_preprocess_mode": str(freq_preprocess_mode),
        "n_rc_elements": int(n_rc_elements),
        "soc_smooth_window": int(soc_smooth_window),
        "soc_smooth_polyorder": int(soc_smooth_polyorder),
        "protocol_role": "autonomous",
    }
    context_budget = int(search_context_budget)
    overrides: Dict[str, Any] = {
        "artifact_prefix": "soc_eis_b11",
        "output_task_dir_name": "soc_eis_b11",
        "force_regenerate_dataset": bool(force_regenerate),
        "n_train_samples": int(n_train_samples),
        "max_eq_len": int(max_eq_len),
        "n_epochs": int(n_epochs),
        "n_eq_samples": int(n_eq_samples),
        "pop_size": int(pop_size),
        "n_offspring": int(n_offspring),
        "init_random_trials": int(init_random_trials),
        "full_refine_topk": int(full_refine_topk),
        "full_refine_topk_max": max(int(full_refine_topk), 16),
        "search_context_budget": context_budget,
        "refine_context_budget": context_budget,
        "final_refit_context_budget": context_budget,
        "weak_window": {"t": int(weak_window_t), "x": int(weak_window_x)},
        "mask_trim": {"t": int(mask_trim_t), "x": int(mask_trim_x)},
        "operator_mode": "pde",
        "scoring_form": "weak",
        "forbidden_rhs_symbols": "t",
        "soc_eis_generator_kwargs": generator_kwargs,
    }
    return overrides


def discover_with_wised(
    *,
    seed: int,
    device: str,
    results_root: str | Path,
    logs_root: str | Path,
    overrides: Dict[str, Any],
    targets: Sequence[str] = ("u", "v"),
) -> None:
    """Run the frozen WiSED experiment entry point.

    No derivative arrays or handcrafted feature tables are accepted by this
    interface.
    """

    run_discovery_experiment(
        "soc_eis_impedance_manifold",
        device=str(device),
        seed=int(seed),
        noise_level=0.0,
        targets=list(targets),
        config_overrides=overrides,
        results_root=Path(results_root),
        logs_root=Path(logs_root),
    )
