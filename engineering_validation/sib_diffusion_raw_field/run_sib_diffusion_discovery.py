from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.wised.experiment import run_discovery_experiment


TASK_NAME = "sib_diffusion_raw_field"


def best_single_condition_overrides(force_regenerate_dataset: bool) -> dict:
    return {
        "artifact_prefix": "sib",
        "output_task_dir_name": "sib_diffusion",
        "force_regenerate_dataset": bool(force_regenerate_dataset),
        "d_h": 32,
        "d_z": 16,
        "n_scales": 1,
        "d_gru": 64,
        "max_eq_len": 42,
        "n_epochs": 30,
        "n_eq_samples": 512,
        "pop_size": 240,
        "n_offspring": 80,
        "init_random_trials": 2400,
        "init_max_ops": 8,
        "weak_window": {"t": 41, "x": 31},
        "mask_trim": {"t": 4, "x": 8},
        "full_refine_topk": 96,
        "full_refine_topk_max": 128,
        "n_train_samples": 14,
        "search_context_budget": 8,
        "refine_context_budget": 8,
        "final_refit_context_budget": 8,
        "operator_mode": "pde",
        "scoring_form": "weak",
        "allow_coordinate_terminals": True,
        "derivative_scale_audit": True,
        "mdl_penalty_weight": 0.005,
        "final_refit_mdl_weight": 0.005,
        "const_scaled_threshold": 1.0e-4,
        "final_refit_scaled_threshold": 1.0e-4,
        "const_physical_prune_tol": 0.0,
        "final_rerank_prune_tol": 0.0,
        "small_coeff_prune_tol": 0.0,
        "zero_order_field_penalty_weight": 0.0,
        "max_zero_order_field_terms": None,
        "forbidden_rhs_symbols": None,
        "struct_guard_max_polynomial_degree": 8,
        "struct_guard_reject_derivative_powers": False,
        "struct_guard_reject_multi_derivative_products": False,
        "derivative_product_penalty_weight": 0.0,
        "derivative_power_penalty_weight": 0.0,
        "generic_affine_bootstrap_frac": 0.25,
        "evo_primitive_completion_frac": 0.20,
        "evo_random_tree_immigrant_frac": 0.35,
        "final_refit_accept_if_improves": True,
        "final_refit_accept_mode": "mdl",
        "sib_diffusion_generator_kwargs": {
            "phases": ("charge", "discharge"),
            "deltas": (0.25,),
            "initial_profiles": (
                "smooth_quadratic",
                "surface_layer",
                "sin1",
                "sin2",
                "center_bump",
                "surface_bump",
                "mid_bump",
            ),
            "nx": 81,
            "tau_max": 1.0,
            "dtau": 1.0e-4,
            "save_every": 100,
            "dependent_transform": "concentration",
            "geometry_channel": False,
            "geometry_channel_scale": 2.0,
            "field_denoise_mode": "svd_gaussian",
            "field_denoise_rank": 4,
            "field_smooth_sigma_t": 4.0,
            "field_smooth_sigma_x": 3.5,
            "clip_during_simulation": True,
            "crop_to_physical_window": False,
            "physical_u_min": 0.0,
            "physical_u_max": 1.0,
            "dataset_name": TASK_NAME,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SIB raw-concentration engineering validation with the selected single-condition WiSED setup."
    )
    parser.add_argument("--device", default="cuda", help="WiSED device, e.g. cuda or cpu.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-level", type=float, default=0.03)
    parser.add_argument("--use-cached-dataset", action="store_true", help="Reuse an existing cached dataset if it satisfies metadata checks.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results_root = Path(args.output_dir) if args.output_dir else ROOT / "outputs"
    logs_root = Path(args.log_dir) if args.log_dir else ROOT / "outputs" / "logs"
    run_discovery_experiment(
        TASK_NAME,
        device=args.device,
        seed=int(args.seed),
        noise_level=float(args.noise_level),
        targets=["u"],
        config_overrides=best_single_condition_overrides(not bool(args.use_cached_dataset)),
        results_root=results_root,
        logs_root=logs_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
