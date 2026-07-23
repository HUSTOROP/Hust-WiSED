"""Minimal public task profiles for the core WiSED discovery backbone.

Only task-scale choices live here: profile, strategy, weak-form geometry, search
budget, and model size.  Numeric defaults, structure-family weights, and
small-data execution choices are owned by their implementation modules so they
cannot become task-level hyperparameters.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

ConfigDict = Dict[str, Any]


def _clean(d: Mapping[str, Any]) -> ConfigDict:
    return {k: v for k, v in dict(d).items() if v is not None}


def _merge(*parts: Mapping[str, Any]) -> ConfigDict:
    out: ConfigDict = {}
    for part in parts:
        out.update(_clean(part))
    return out


_NUMERIC_BASE: ConfigDict = {
    "device": "cuda",
    # Final export always refits the selected structure on all available weak-form
    # contexts.  This is now part of the algorithm rather than a runtime branch.
    "final_refit_context_budget": 0,
    "final_refit_scaled_threshold": 1.0e-4,
    "final_refit_ridge": 1.0e-10,
    "final_refit_l2": 0.0,
    # Final coefficient refit is accepted by an MDL-style score, not by MSE alone.
    # This prevents tiny residual improvements from adding visible parasitic terms.
    "final_refit_accept_mode": "mdl",
    "final_refit_score_tol": 1.0e-10,
    "final_refit_mdl_weight": 0.25,
    "small_coeff_prune_tol": 1.0e-4,
    "final_rerank_prune_tol": 1.0e-4,
}

# Public semantic strategies. Detailed spectral/multi-window/structure weights
# are internal defaults in wised_framework.py.
_STRATEGIES: Dict[str, ConfigDict] = {
    "core": {
        "robust_structure_eval": False,
        "structure_prior": "pde_simple",
    },
    "noisy": {
        "robust_structure_eval": True,
        "structure_prior": "robust_pde",
    },
    "small": {
        "robust_structure_eval": False,
        "structure_prior": "pde_simple",
        "budget_profile": "small",
    },
}

_PROFILES: Dict[str, ConfigDict] = {
    "1d": {
        "strategy": "core",
        "d_h": 64, "d_z": 32, "n_scales": 3, "d_gru": 128, "n_gru_layers": 2,
        "max_eq_len": 34,
        "mask_trim": {"t": 3},
        "weak_window": {"t": 15, "x": 41},
        "const_scaled_threshold": 1.0e-4,
        "n_epochs": 40,
        "n_train_samples": 16,
        "n_eq_samples": 160,
        "pop_size": 180,
        "n_offspring": 48,
        "evo_interval": 4,
        "init_random_trials": 900,
        "init_max_ops": 8,
        "full_refine_topk": 40,
        "full_refine_topk_max": 96,
        "population_reuse_k": 8,
    },
    "2d": {
        "strategy": "core",
        "d_h": 64, "d_z": 32, "n_scales": 3, "d_gru": 128, "n_gru_layers": 2,
        "max_eq_len": 42,
        "mask_trim": {"t": 3},
        "weak_window": {"t": 11, "x": 7, "y": 7},
        "const_scaled_threshold": 1.0e-4,
        "n_epochs": 40,
        "n_train_samples": 12,
        "n_eq_samples": 256,
        "pop_size": 240,
        "n_offspring": 128,
        "evo_interval": 3,
        "init_random_trials": 1600,
        "init_max_ops": 9,
        "full_refine_topk": 64,
        "full_refine_topk_max": 128,
        "population_reuse_k": 16,
        "search_subsample": {"t": 2},
        "search_context_budget": 2,
        "refine_context_budget": 3,
    },
    "small": {
        "strategy": "small",
        "d_h": 16, "d_z": 8, "n_scales": 1, "d_gru": 32, "n_gru_layers": 2,
        "max_eq_len": 28,
        "mask_trim": {"t": 3},
        "weak_window": {"t": 9, "x": 3},
        "const_scaled_threshold": 1.0e-4,
        "const_l2": 1.0e-7,
        "n_epochs": 16,
        "n_train_samples": 1,
        "n_eq_samples": 56,
        "pop_size": 56,
        "n_offspring": 10,
        "evo_interval": 4,
        "init_random_trials": 140,
        "init_max_ops": 6,
        "full_refine_topk": 6,
        "full_refine_topk_max": 10,
        "population_reuse_k": 0,
        "search_subsample": {"t": 2},
        "search_context_budget": 1,
        "refine_context_budget": 1,
        "default_targets": ["u"],
    },
}


def make_config(
    profile: str = "1d",
    *,
    strategy: Optional[str] = None,
    noise_level: float = 0.01,
    seed: int = 42,
    device: Optional[str] = None,
    default_targets: Optional[list[str]] = None,
    **overrides: Any,
) -> ConfigDict:
    """Return a legacy-compatible flat config from a compact profile.

    Public knobs: ``profile``, ``strategy``, ``weak_window``, ``mask_trim``,
    ``max_eq_len``, and search-budget fields such as ``n_eq_samples``,
    ``init_random_trials`` and ``full_refine_topk``.  Structure-family weights
    and training schedules are intentionally not public task hyperparameters.
    """
    if profile not in _PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {sorted(_PROFILES)}")

    base = deepcopy(_PROFILES[profile])
    strategy_name = strategy or str(base.pop("strategy"))
    if strategy_name not in _STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy_name!r}; choose from {sorted(_STRATEGIES)}")

    cfg = _merge(
        {"noise_level": float(noise_level), "seed": int(seed)},
        _NUMERIC_BASE,
        _STRATEGIES[strategy_name],
        base,
    )
    if device is not None:
        cfg["device"] = str(device)
    if default_targets is not None:
        cfg["default_targets"] = list(default_targets)
    cfg.update(_clean(overrides))
    return cfg


def build_discovery_config(**kwargs: Any) -> ConfigDict:
    """Backward-compatible alias."""
    return make_config(**kwargs)
