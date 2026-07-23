from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

from models.experiment_profiles import ConfigDict, make_config

DEFAULT_NOISE_LEVELS: Tuple[float, ...] = (0.0, 0.01, 0.05, 0.10)
DEFAULT_SEEDS: Tuple[int, ...] = (42,)


@dataclass(frozen=True)
class ExperimentSpec:
    equation: str
    profile: str
    selected: bool = True
    strategy: Optional[str] = None
    noise_levels: Tuple[float, ...] = DEFAULT_NOISE_LEVELS
    seeds: Tuple[int, ...] = DEFAULT_SEEDS
    default_targets: Optional[Tuple[str, ...]] = None
    config_overrides: Mapping[str, Any] = field(default_factory=dict)


EXPERIMENT_SPECS: Dict[str, ExperimentSpec] = {
    "burgers_1d": ExperimentSpec(
        equation="burgers_1d",
        profile="1d",
        config_overrides={
            "max_eq_len": 30,
            "n_eq_samples": 128,
            "pop_size": 160,
            "n_offspring": 40,
            "init_random_trials": 768,
            "full_refine_topk": 36,
            "population_reuse_k": 6,
        },
    ),
    "kdv_1d": ExperimentSpec(
        equation="kdv_1d",
        profile="1d",
        config_overrides={"max_eq_len": 34, "weak_window": {"t": 21, "x": 51}, "evo_interval": 3},
    ),
    "burgers_2d": ExperimentSpec(
        equation="burgers_2d",
        profile="2d",
        config_overrides={"max_eq_len": 38, "n_train_samples": 16},
    ),
    "fhn_2d": ExperimentSpec(
        equation="fhn_2d",
        profile="2d",
        strategy="noisy",
        config_overrides={
            "max_eq_len": 46,
            "n_eq_samples": 224,
            "full_refine_topk": 72,
            "full_refine_topk_max": 96,
            "population_reuse_k": 20,
            "init_random_trials": 2600,
            "const_scaled_threshold": 1.0e-4,
            "refine_context_budget": 4,
        },
    ),
    "soc_eis_impedance_manifold": ExperimentSpec(
        equation="soc_eis_impedance_manifold",
        profile="small",
        selected=False,
        noise_levels=(0.0,),
        seeds=(0,),
        default_targets=("u", "v"),
        config_overrides={
            "max_eq_len": 34,
            "n_train_samples": 24,
            "operator_mode": "pde",
            "scoring_form": "weak",
            "forbidden_rhs_symbols": "t",
            "soc_eis_generator_kwargs": {"exclude_soc100": True, "frequency_band": "full"},
        },
    ),
    "sib_diffusion_raw_field": ExperimentSpec(
        equation="sib_diffusion_raw_field",
        profile="1d",
        selected=False,
        noise_levels=(0.03,),
        seeds=(0,),
        default_targets=("u",),
        config_overrides={
            "artifact_prefix": "sib",
            "output_task_dir_name": "sib_diffusion",
            "operator_mode": "pde",
            "scoring_form": "weak",
            "sib_diffusion_generator_kwargs": {"dataset_name": "sib_diffusion_raw_field"},
        },
    ),
}


def available_equations(*, selected_only: bool = False) -> Tuple[str, ...]:
    return tuple(
        equation for equation, spec in EXPERIMENT_SPECS.items() if not selected_only or bool(spec.selected)
    )


def get_experiment_spec(equation: str) -> ExperimentSpec:
    try:
        return EXPERIMENT_SPECS[str(equation)]
    except KeyError as exc:
        raise ValueError(f"Unknown equation {equation!r}. Choose from {list(EXPERIMENT_SPECS)}.") from exc


def parse_override(raw: str) -> Tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Override {raw!r} must use KEY=VALUE syntax.")
    key, value_text = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Override {raw!r} has an empty key.")
    try:
        value = ast.literal_eval(value_text)
    except Exception:
        value = value_text
    return key, value


def parse_overrides(raw_items: Optional[Sequence[str]]) -> Dict[str, Any]:
    return dict(parse_override(raw) for raw in (raw_items or []))


def build_run_config(
    equation: str,
    *,
    noise_level: Optional[float] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> ConfigDict:
    spec = get_experiment_spec(equation)
    cfg = make_config(
        spec.profile,
        strategy=spec.strategy,
        noise_level=float(spec.noise_levels[0] if noise_level is None else noise_level),
        seed=int(spec.seeds[0] if seed is None else seed),
        device=device,
        default_targets=list(spec.default_targets) if spec.default_targets is not None else None,
        **dict(spec.config_overrides),
    )
    cfg["equation_name"] = str(equation)
    cfg.update(dict(overrides or {}))
    if device is not None:
        cfg["device"] = str(device)
    if noise_level is not None:
        cfg["noise_level"] = float(noise_level)
    if seed is not None:
        cfg["seed"] = int(seed)
    cfg.setdefault("small_coeff_prune_tol", 1.0e-4)
    cfg.setdefault("final_refit_accept_mode", "mdl")
    cfg.setdefault("final_refit_score_tol", 1.0e-10)
    cfg.setdefault("final_refit_mdl_weight", float(cfg.get("mdl_penalty_weight", 0.25)))
    cfg.setdefault("final_rerank_prune_tol", float(cfg["small_coeff_prune_tol"]))
    cfg.setdefault("final_refit_scaled_threshold", float(cfg["small_coeff_prune_tol"]))
    cfg.setdefault("const_scaled_threshold", float(cfg["small_coeff_prune_tol"]))
    cfg.setdefault("early_stop_enabled", True)
    cfg.setdefault("early_stop_patience", 8)
    return cfg


def format_float_tag(value: float) -> str:
    return f"{float(value):g}"


def case_dir_name(*, noise_level: float, seed: int) -> str:
    return f"noise_{format_float_tag(noise_level)}__seed_{int(seed)}"


def iter_experiment_cases(
    equations: Optional[Iterable[str]] = None,
    *,
    selected_only: bool = True,
    noise_levels: Optional[Sequence[float]] = None,
    seeds: Optional[Sequence[int]] = None,
) -> Iterator[Tuple[str, float, int]]:
    names = tuple(equations) if equations is not None else available_equations(selected_only=selected_only)
    for equation in names:
        spec = get_experiment_spec(equation)
        for noise in tuple(float(v) for v in (noise_levels or spec.noise_levels)):
            for current_seed in tuple(int(v) for v in (seeds or spec.seeds)):
                yield equation, noise, current_seed
