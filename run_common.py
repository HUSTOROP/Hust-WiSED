from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from models.weak_form_evaluator import DataContext, MaskConfig, compute_fitness, evaluate_scoring_tensors
from models.symbolic_search import optimize_constants
from models.symbol_vocabulary import count_constants, is_valid_sequence, SYM2IDX
from utils.structure_cache import get_structure_key_manager
from models.wised_framework import WiSEDFramework, WiSEDTrainer
from utils.equation_simplifier import polish_discovered_equation
from utils.equation_formatter import compile_equation
from utils.experiment_logger import WiSEDLogger
from utils.evaluation_metrics import evaluate_discovered_equation, save_metrics_txt
from src.wised.reproducibility import seed_everything


def infer_padding_mode(periodic_axes: Dict[str, bool]) -> str:
    """Infer convolution padding from spatial periodicity."""
    spatial_periodic = [bool(v) for k, v in periodic_axes.items() if k != "t"]
    return "circular" if spatial_periodic and all(spatial_periodic) else "zeros"


def infer_spatial_ndim(periodic_axes: Dict[str, bool]) -> int:
    """Infer the number of spatial dimensions from the axis metadata."""
    return int(max(1, len([axis for axis in periodic_axes if axis != "t"])))


def to_np(obj: Any) -> np.ndarray:
    """Convert torch tensors or array-like inputs to NumPy arrays."""
    return obj.detach().cpu().numpy() if hasattr(obj, "detach") else np.asarray(obj)


def build_mask_cfg(cfg: Dict[str, Any]) -> MaskConfig:
    """Build weak-form and trimming mask configuration from a flat config."""
    return MaskConfig(
        trim=cfg.get("mask_trim", {}),
        weak_window=cfg.get("weak_window", {}),
        weak_degree=int(cfg.get("weak_degree", 6)),
    )


def make_context_list(
    u_array: np.ndarray,
    coords: Dict[str, np.ndarray],
    periodic_axes: Dict[str, bool],
    field_names: Optional[List[str]],
    cfg: Dict[str, Any],
    cache_prefix: str,
) -> List[DataContext]:
    """Create one reusable weak-form data context per trajectory."""
    mask_cfg = build_mask_cfg(cfg)
    ctx_device = str(cfg.get("device", "cpu"))
    normalize_mse = bool(cfg.get("normalize_mse", True))
    return [
        DataContext(
            u_array[i],
            coords,
            device=ctx_device,
            periodic_axes=periodic_axes,
            mask_cfg=mask_cfg,
            normalize_mse=normalize_mse,
            cache_tag=f"{cache_prefix}_{i}",
            field_names=field_names,
            scoring_form=str(cfg.get("scoring_form", "weak")),
            allow_coordinate_terminals=bool(cfg.get("allow_coordinate_terminals", False)),
            derivative_scale_audit=bool(cfg.get("derivative_scale_audit", False)),
        )
        for i in range(int(u_array.shape[0]))
    ]


def create_model(
    cfg: Dict[str, Any],
    periodic_axes: Dict[str, bool],
    in_channels: int,
    spatial_ndim: Optional[int] = None,
) -> WiSEDFramework:
    if spatial_ndim is None:
        spatial_ndim = infer_spatial_ndim(periodic_axes)

    return WiSEDFramework(
        in_channels=in_channels,
        d_h=int(cfg["d_h"]),
        d_z=int(cfg["d_z"]),
        n_scales=int(cfg["n_scales"]),
        d_gru=int(cfg["d_gru"]),
        max_eq_len=int(cfg["max_eq_len"]),
        n_gru_layers=int(cfg["n_gru_layers"]),
        spatial_ndim=int(spatial_ndim),
        spatial_padding_mode=infer_padding_mode(periodic_axes),
    )


def normalize_target_fields(targets: Optional[List[str]], field_names: List[str]) -> List[str]:
    """Normalize user target names to canonical du_t/dv_t/dw_t names.

    Accepted inputs are plain field names such as u/v/w or canonical names such
    as du_t/dv_t/dw_t.
    """
    valid_targets = {f"d{name}_t" for name in field_names}

    if not targets:
        requested = [f"d{name}_t" for name in field_names]
    else:
        requested = []
        for raw_target in targets:
            target = str(raw_target).strip()
            if target in field_names:
                requested.append(f"d{target}_t")
            elif target in valid_targets:
                requested.append(target)
            else:
                raise ValueError(
                    f"Invalid target field '{target}'. "
                    f"Only plain field names {field_names} or canonical names {sorted(valid_targets)} are supported."
                )

    normalized: List[str] = []
    seen = set()
    for target in requested:
        if target not in seen:
            normalized.append(target)
            seen.add(target)
    return normalized

def summarize_context(ctx: DataContext) -> Dict[str, Any]:
    """Return compact context metadata for logs."""
    summary = {
        "fields": list(getattr(ctx, "fields", [])),
        "axes": list(getattr(ctx, "axes_order", [])),
        "periodic_axes": dict(getattr(ctx, "periodic_axes", {})),
        "device": str(getattr(ctx, "device", "cpu")),
    }
    if bool(getattr(ctx, "derivative_scale_audit_enabled", False)):
        try:
            summary["derivative_scale_audit"] = ctx.derivative_scale_audit()
        except Exception as exc:
            summary["derivative_scale_audit_error"] = str(exc)
    return summary


def _is_target_time_derivative(target_field: str) -> bool:
    field = str(target_field)
    return field.startswith("d") and field.endswith("_t")


def _field_from_target_field(target_field: str) -> str:
    """Convert a canonical target such as du_t back to its field name."""
    field = str(target_field)
    return field[1:-2] if _is_target_time_derivative(field) else field


def _true_equation_for_target(true_eq: Any, target_field: str) -> str:
    field = _field_from_target_field(target_field)
    return str(true_eq.get(field, "")) if isinstance(true_eq, dict) else str(true_eq)


def _write_target_summary(
    logger: WiSEDLogger,
    target_field: str,
    best_eq: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    logger.log_kv(
        f"TARGET SUMMARY - {target_field}",
        {
            "target": target_field,
            "target_derivative_mse": float(best_eq.get("target_derivative_mse", float("nan"))),
            "target_derivative_rel_l2": float(best_eq.get("target_derivative_rel_l2", float("nan"))),
            "discovered": metrics.get("discovered_readable", ""),
            "true_equation": metrics.get("true_equation", ""),
        },
    )


def _lhs_from_target_field(target_field: str) -> str:
    return f"{_field_from_target_field(target_field)}_t" if _is_target_time_derivative(target_field) else "u_t"


def _require_reference_context(ctx_list: List[DataContext]) -> DataContext:
    if not ctx_list:
        raise ValueError("ctx_list cannot be empty.")
    return ctx_list[0]


def _build_trainer(
    *,
    cfg: Dict[str, Any],
    ctx_list: List[DataContext],
    data_tensor: torch.Tensor,
    log_dir: str,
    logger: WiSEDLogger,
) -> WiSEDTrainer:
    """Build the shared trainer used by single- and multi-target workflows."""
    reference_ctx = _require_reference_context(ctx_list)
    model = create_model(
        cfg,
        reference_ctx.periodic_axes,
        in_channels=int(data_tensor.shape[-1]),
    )
    return WiSEDTrainer(
        model=model,
        ctx=ctx_list,
        config=cfg,
        device=str(cfg["device"]),
        log_dir=log_dir,
        logger=logger,
        enable_population_init=True,
    )


def _split_visible_rhs_terms(eq_str: str) -> List[str]:
    """Split a printed equation RHS into top-level additive terms."""
    text = str(eq_str or "")
    rhs = text.split("=", 1)[1] if "=" in text else text
    rhs = rhs.strip()
    if not rhs or rhs == "0":
        return []
    terms: List[str] = []
    start = 0
    depth = 0
    for i, ch in enumerate(rhs):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0 and i > 0 and ch in "+-":
            prev = rhs[i - 1]
            # Do not split scientific notation, e.g. 1.0e-4.
            if prev in "eE":
                continue
            term = rhs[start:i].strip()
            if term:
                terms.append(term)
            start = i
    tail = rhs[start:].strip()
    if tail:
        terms.append(tail)
    return [t for t in terms if t and t not in {"+", "-"}]


def _visible_equation_complexity(eq_str: str) -> Tuple[int, float]:
    """Lightweight display-level complexity used only for final refit acceptance."""
    terms = _split_visible_rhs_terms(eq_str)
    if not terms:
        return 0, 0.0
    complexity = 0.0
    for term in terms:
        t = term.strip()
        # Base cost per visible additive term.
        complexity += 1.0
        # Multiplicative/nonlinear operators and derivatives make a term harder to describe.
        complexity += 0.35 * t.count("*")
        complexity += 0.70 * (t.count("^") + t.count("**"))
        complexity += 0.50 * (t.count("_{") + t.count("_x") + t.count("_y") + t.count("_z"))
        for fn in ("sin", "cos", "exp", "log"):
            complexity += 1.0 * t.count(fn + "(")
    return len(terms), float(max(complexity, len(terms)))


def _visible_mdl_score(mse: float, eq_str: str, *, n_eff: float, vocab_size: int, mdl_weight: float) -> Tuple[float, Dict[str, Any]]:
    """MDL-style final display score: log residual plus visible description length."""
    if not np.isfinite(mse):
        mse = 1e10
    eff_mse = max(float(mse), 1e-15)
    n_eff = float(max(float(n_eff or 50.0), 1.0))
    vocab_size = int(max(int(vocab_size or len(SYM2IDX) or 20), 2))
    n_terms, visible_complexity = _visible_equation_complexity(eq_str)
    struct_bits = visible_complexity * np.log(vocab_size)
    param_bits = 0.5 * float(n_terms) * np.log(max(n_eff, 2.0))
    mdl_penalty = float(mdl_weight) * (struct_bits + param_bits) / n_eff
    score = float(np.log10(eff_mse) + 15.0 + mdl_penalty)
    return score, {
        "terms": int(n_terms),
        "visible_complexity": float(visible_complexity),
        "mdl_penalty": float(mdl_penalty),
    }


def _simplify_seq_with_consts(seq: List[int], consts: Any, prune_tol: float) -> List[int]:
    """Apply the same constant-aware algebraic pruning used by final rerank."""
    seq0 = list(map(int, seq or []))
    if not seq0 or not is_valid_sequence(seq0):
        return seq0
    try:
        arr = np.asarray(consts, dtype=np.float64).reshape(-1)
    except Exception:
        return seq0
    if arr.size != count_constants(seq0):
        return seq0
    try:
        km = get_structure_key_manager()
        simp = km.normalizer.algebraic_simplify(seq0, consts=arr, prune_tol=float(prune_tol))
        simp = list(map(int, simp))
        if simp and is_valid_sequence(simp):
            return simp
    except Exception:
        pass
    return seq0


def _canonical_seq_key(seq: List[int]) -> str:
    try:
        return "_".join(map(str, get_structure_key_manager().normalize(tuple(map(int, seq or [])))))
    except Exception:
        return "_".join(map(str, seq or []))

def refit_final_constants(
    best_eq: Dict[str, Any],
    ctx_list: List[DataContext],
    target_field: str,
    cfg: Dict[str, Any],
    logger: WiSEDLogger,
) -> Dict[str, Any]:
    """Re-fit constants for the selected structure with MDL-aware acceptance.

    The final exported equation should not blindly accept a coefficient refit
    just because the weak residual MSE decreases by an arbitrarily small amount.
    This routine therefore compares the old and refitted equations using a
    display-level MDL score: log residual + visible equation description length.
    A refit that introduces extra visible terms is accepted only when the MSE
    decrease is large enough to pay for the added complexity.
    """
    seq = list(map(int, best_eq.get("seq", []) or []))
    if not seq:
        return best_eq
    try:
        prune_tol = float(cfg.get("small_coeff_prune_tol", 1e-4))
        context_budget = int(cfg.get("final_refit_context_budget", 0) or len(ctx_list))
        context_budget = max(1, min(context_budget, len(ctx_list)))
        lhs = _lhs_from_target_field(target_field)

        old_mse = float(best_eq.get("mse", best_eq.get("residual_mse", 1e10)))
        old_consts = np.asarray(best_eq.get("consts", np.array([])), dtype=np.float64).reshape(-1)
        old_seq = _simplify_seq_with_consts(seq, old_consts, prune_tol=prune_tol)
        old_eq_raw = compile_equation(old_seq if count_constants(old_seq) == old_consts.size else seq, old_consts, lhs=lhs)
        old_eq = polish_discovered_equation(old_eq_raw, prune_tol=prune_tol)

        consts, fit = optimize_constants(
            seq,
            ctx_list,
            target_field=target_field,
            n_init=int(cfg.get("final_refit_n_init", 3)),
            init_consts=best_eq.get("consts", None),
            const_context_budget=context_budget,
            const_scaled_threshold=float(cfg.get("final_refit_scaled_threshold", cfg.get("small_coeff_prune_tol", 1e-4))),
            const_prune_tol=float(cfg.get("small_coeff_prune_tol", 1e-4)),
            const_l2=float(cfg.get("final_refit_l2", cfg.get("const_l2", 0.0))),
            const_ridge=float(cfg.get("final_refit_ridge", 1e-10)),
            const_clamp=float(cfg.get("const_clamp", 1e4)),
            mdl_penalty_weight=float(cfg.get("mdl_penalty_weight", 0.25)),
            enable_nonlinear_lbfgs=True,
            const_nonlinear_n_init=int(cfg.get("final_refit_nonlinear_n_init", 2)),
            const_lbfgs_max_iter=int(cfg.get("final_refit_lbfgs_max_iter", 40)),
        )
        if not fit.get("pred_valid", False):
            return best_eq

        final_seq = list(seq)
        final_consts = np.asarray(consts, dtype=np.float64).reshape(-1)
        final_fit = dict(fit)

        # Apply the 1e-4 small-coefficient pruning structurally, not only at the
        # string-output layer.  If pruning changes the structure, re-fit the
        # remaining constants once so the final stored sequence and constants match.
        pruned_seq = _simplify_seq_with_consts(final_seq, final_consts, prune_tol=prune_tol)
        if pruned_seq and _canonical_seq_key(pruned_seq) != _canonical_seq_key(final_seq):
            try:
                p_consts, p_fit = optimize_constants(
                    pruned_seq,
                    ctx_list,
                    target_field=target_field,
                    n_init=max(1, int(cfg.get("final_refit_pruned_n_init", 1))),
                    init_consts=None,
                    const_context_budget=context_budget,
                    const_scaled_threshold=float(cfg.get("final_refit_scaled_threshold", cfg.get("small_coeff_prune_tol", 1e-4))),
                    const_prune_tol=float(cfg.get("small_coeff_prune_tol", 1e-4)),
                    const_l2=float(cfg.get("final_refit_l2", cfg.get("const_l2", 0.0))),
                    const_ridge=float(cfg.get("final_refit_ridge", 1e-10)),
                    const_clamp=float(cfg.get("const_clamp", 1e4)),
                    mdl_penalty_weight=float(cfg.get("mdl_penalty_weight", 0.25)),
                    enable_nonlinear_lbfgs=True,
                    const_nonlinear_n_init=max(1, int(cfg.get("final_refit_nonlinear_n_init", 2))),
                    const_lbfgs_max_iter=int(cfg.get("final_refit_lbfgs_max_iter", 40)),
                )
                if isinstance(p_fit, dict) and bool(p_fit.get("pred_valid", False)):
                    final_seq = list(pruned_seq)
                    final_consts = np.asarray(p_consts, dtype=np.float64).reshape(-1)
                    final_fit = dict(p_fit)
            except Exception:
                pass

        new_mse = float(final_fit.get("residual_mse", best_eq.get("mse", 1e10)))
        new_eq_raw = compile_equation(final_seq, final_consts, lhs=lhs)
        new_eq = polish_discovered_equation(new_eq_raw, prune_tol=prune_tol)

        n_eff = float(final_fit.get("n_eff", best_eq.get("n_eff", 50.0)))
        vocab_size = int(final_fit.get("vocab_size", len(SYM2IDX)))
        mdl_w = float(cfg.get("final_refit_mdl_weight", cfg.get("mdl_penalty_weight", 0.25)))
        old_score, old_desc = _visible_mdl_score(old_mse, old_eq, n_eff=n_eff, vocab_size=vocab_size, mdl_weight=mdl_w)
        new_score, new_desc = _visible_mdl_score(new_mse, new_eq, n_eff=n_eff, vocab_size=vocab_size, mdl_weight=mdl_w)
        score_tol = float(cfg.get("final_refit_score_tol", 1e-10))
        rel_improve = (old_mse - new_mse) / max(abs(old_mse), 1e-12) if np.isfinite(old_mse) and np.isfinite(new_mse) else float("nan")

        mode = str(cfg.get("final_refit_accept_mode", cfg.get("final_refit_selection_mode", "mdl"))).strip().lower()
        accept = True
        reason = "accepted"
        if mode in {"always", "force"}:
            accept = True
            reason = "final_refit_accept_mode=always"
        elif mode in {"mse", "improve", "improves"}:
            min_rel = float(cfg.get("final_refit_min_relative_improvement", 0.0))
            accept = bool(np.isfinite(old_mse) and np.isfinite(new_mse) and new_mse <= old_mse and rel_improve >= min_rel)
            reason = "MSE decreased enough" if accept else "MSE decrease was insufficient"
        else:
            # Default: MDL / fitness-style acceptance.  This follows the same
            # idea as compose_fitness(): residual reduction must pay for the
            # equation description length.
            accept = bool(np.isfinite(old_score) and np.isfinite(new_score) and new_score <= old_score - score_tol)
            if accept:
                reason = "visible MDL score improved"
            else:
                if new_desc["terms"] > old_desc["terms"]:
                    reason = "MSE gain did not justify added visible terms under MDL"
                elif new_desc["visible_complexity"] > old_desc["visible_complexity"]:
                    reason = "MSE gain did not justify added visible complexity under MDL"
                else:
                    reason = "visible MDL score did not improve"

        if not accept:
            logger.log_kv(
                f"FINAL COEFFICIENT REFIT - {target_field}",
                {
                    "old_mse": old_mse,
                    "new_mse": new_mse,
                    "relative_mse_improvement": rel_improve,
                    "old_mdl_score": old_score,
                    "new_mdl_score": new_score,
                    "accepted": False,
                    "reason": reason,
                    "selection_mode": mode or "mdl",
                    "context_budget": int(context_budget),
                    "old_visible_terms": int(old_desc["terms"]),
                    "new_visible_terms": int(new_desc["terms"]),
                    "old_visible_complexity": float(old_desc["visible_complexity"]),
                    "new_visible_complexity": float(new_desc["visible_complexity"]),
                    "old_equation": old_eq,
                    "new_equation": new_eq,
                },
            )
            return best_eq

        updated = dict(best_eq)
        updated["seq"] = list(final_seq)
        updated["consts"] = np.asarray(final_consts, dtype=np.float64)
        updated["fitness"] = float(final_fit.get("fitness", best_eq.get("fitness", 1e10)))
        updated["mse"] = new_mse
        updated["residual_mse"] = float(final_fit.get("residual_mse", updated["mse"]))
        updated["raw_mse"] = float(final_fit.get("raw_mse", best_eq.get("raw_mse", np.nan)))
        updated["n_eff"] = float(final_fit.get("n_eff", n_eff))
        updated["final_refit"] = True
        updated["final_refit_context_budget"] = int(context_budget)
        updated["final_refit_accept_mode"] = mode or "mdl"
        updated["final_refit_old_mdl_score"] = float(old_score)
        updated["final_refit_new_mdl_score"] = float(new_score)
        updated["final_refit_relative_mse_improvement"] = float(rel_improve)
        logger.log_kv(
            f"FINAL COEFFICIENT REFIT - {target_field}",
            {
                "old_mse": old_mse,
                "new_mse": updated["mse"],
                "relative_mse_improvement": rel_improve,
                "old_mdl_score": old_score,
                "new_mdl_score": new_score,
                "accepted": True,
                "reason": reason,
                "selection_mode": mode or "mdl",
                "context_budget": int(context_budget),
                "old_visible_terms": int(old_desc["terms"]),
                "new_visible_terms": int(new_desc["terms"]),
                "old_visible_complexity": float(old_desc["visible_complexity"]),
                "new_visible_complexity": float(new_desc["visible_complexity"]),
                "old_equation": old_eq,
                "new_equation": new_eq,
                "consts": [float(x) for x in np.asarray(final_consts).reshape(-1)],
            },
        )
        return updated
    except Exception as exc:
        logger.warning(f"final coefficient refit failed for {target_field}: {exc}")
    return best_eq

def evaluate_equation_on_contexts(
    best_eq: Dict[str, Any],
    ctx_list: List[DataContext],
    target_field: str,
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate the exported equation on a supplied context list without selection."""
    seq = list(map(int, best_eq.get("seq", []) or []))
    if not seq or not ctx_list:
        return {}
    vals = []
    raw_vals = []
    for ctx in ctx_list:
        try:
            fit = compute_fitness(
                seq,
                ctx,
                best_eq.get("consts", []),
                target_field=target_field,
                mdl_penalty_weight=float(cfg.get("mdl_penalty_weight", 0.25)),
            )
            if fit.get("pred_valid", False):
                vals.append(float(fit.get("residual_mse", np.nan)))
                raw_vals.append(float(fit.get("raw_mse", np.nan)))
        except Exception:
            continue
    out: Dict[str, float] = {}
    finite_vals = [v for v in vals if np.isfinite(v)]
    finite_raw = [v for v in raw_vals if np.isfinite(v)]
    if finite_vals:
        out["context_residual_mse"] = float(np.mean(finite_vals))
    if finite_raw:
        out["context_raw_mse"] = float(np.mean(finite_raw))
    return out



def compute_target_prediction_errors(
    best_eq: Dict[str, Any],
    ctx: DataContext,
    target_field: str,
    scoring_form: str = "weak",
) -> Dict[str, float]:
    """Compute only the exported target-derivative prediction errors."""
    seq = list(map(int, best_eq.get("seq", []) or []))
    if not seq:
        return {}
    try:
        lhs_tensor, rhs = evaluate_scoring_tensors(
            ctx,
            seq,
            best_eq.get("consts"),
            target_field=target_field,
            scoring_form=scoring_form,
        )
        if rhs is None or lhs_tensor is None or lhs_tensor.shape != rhs.shape:
            return {}
        if not bool(torch.isfinite(rhs).all().item()) or not bool(torch.isfinite(lhs_tensor).all().item()):
            return {}
        err = (lhs_tensor - rhs).detach().cpu().numpy().astype(np.float64)
        lhs = lhs_tensor.detach().cpu().numpy().astype(np.float64)
        mse = float(np.mean(err ** 2))
        rel_l2 = float(np.linalg.norm(err.ravel()) / (np.linalg.norm(lhs.ravel()) + 1.0e-12))
        return {"target_derivative_mse": mse, "target_derivative_rel_l2": rel_l2}
    except Exception:
        return {}

def finalize_and_plot(
    trainer: WiSEDTrainer,
    history: Dict[str, List[Any]],
    best_eq: Dict[str, Any],
    result_dir: str,
    equation_name: str,
    true_eq: Any,
    target_field: str,
    ctx: DataContext,
    logger: WiSEDLogger,
    cfg: Dict[str, Any],
    *,
    ctx_list_for_refit: Optional[List[DataContext]] = None,
    emit_summary: bool = False,
    summary_elapsed_sec: Optional[float] = None,
    summary_elapsed_label: str = "Training elapsed",
) -> Dict[str, Any]:
    """Finalize one target equation: metrics, plots and result files."""
    os.makedirs(result_dir, exist_ok=True)

    best_eq = refit_final_constants(best_eq, ctx_list_for_refit or [ctx], target_field, cfg, logger)

    lhs = _lhs_from_target_field(target_field)
    if best_eq.get("seq"):
        raw_equation = compile_equation(best_eq.get("seq", []), best_eq.get("consts", []), lhs=lhs)
    else:
        raw_equation = best_eq.get("readable") or best_eq.get("str") or f"{lhs} = 0"

    polished_equation = polish_discovered_equation(raw_equation, prune_tol=float(cfg.get("small_coeff_prune_tol", 1e-4)))
    trainer.best_equations[str(target_field)] = {**best_eq, "raw_readable": raw_equation, "readable": polished_equation}
    trainer._activate_target(str(target_field))
    best_eq = trainer.best_equation

    structural_metrics = evaluate_discovered_equation(
        best_eq.get("seq", []),
        best_eq.get("consts", []),
        equation_name,
        float(best_eq.get("mse", 1e10)),
        float(best_eq.get("fitness", 1e10)),
    )
    prediction_errors = compute_target_prediction_errors(
        best_eq,
        ctx,
        target_field,
        scoring_form=str(cfg.get("scoring_form", getattr(ctx, "scoring_form", "weak"))),
    )
    for key, value in prediction_errors.items():
        best_eq[key] = value
    metrics = {
        "target": target_field,
        "true_equation": _true_equation_for_target(true_eq, target_field),
        "discovered_readable": polished_equation,
        "target_derivative_mse": float(best_eq.get("target_derivative_mse", float("nan"))),
        "target_derivative_rel_l2": float(best_eq.get("target_derivative_rel_l2", float("nan"))),
        "structural_sim": structural_metrics.get("structural_sim", 0.0),
        "operator_recall": structural_metrics.get("operator_recall", 0.0),
    }
    save_metrics_txt(metrics, os.path.join(result_dir, f"{equation_name}_metrics.txt"))

    try:
        from utils.plotting import (
            plot_fitness_evolution,
            plot_population_fitness,
            plot_training_curves,
        )

        os.makedirs(result_dir, exist_ok=True)
        plot_training_curves(history, result_dir, equation_name)
        plot_fitness_evolution(history, result_dir, equation_name)
        if trainer.population.population:
            plot_population_fitness(
                trainer.population.population,
                result_dir,
                equation_name,
                epoch=int(cfg["n_epochs"]),
            )
    except Exception as exc:
        logger.warning(f"plotting training curves failed for {target_field}: {exc}")

    _write_target_summary(logger, target_field, best_eq, metrics)
    trainer.save_results(result_dir=result_dir, equation_name=equation_name, target_field=target_field)

    if emit_summary:
        logger.final_summary(
            {
                "target": target_field,
                "target_derivative_mse": float(best_eq.get("target_derivative_mse", float("nan"))),
                "target_derivative_rel_l2": float(best_eq.get("target_derivative_rel_l2", float("nan"))),
                "discovered": polished_equation,
            },
            elapsed_sec=summary_elapsed_sec,
            elapsed_label=summary_elapsed_label,
        )

    return {
        "best_eq": best_eq,
        "metrics": metrics,
        "polished_equation": polished_equation,
    }


def discover_multifield_target(
    *,
    task_prefix: str,
    target_field: str,
    cfg: Dict[str, Any],
    data_tensor: torch.Tensor,
    ctx_list: List[DataContext],
    result_root: str,
    log_root: str,
    true_eq: Any,
) -> Dict[str, Any]:
    """Run discovery for a single target field."""
    field = _field_from_target_field(target_field)
    equation_name = f"{task_prefix}_{field}"
    result_dir = os.path.join(result_root, field)
    log_dir = os.path.join(log_root, field)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger = WiSEDLogger(log_dir, equation_name)
    reference_ctx = _require_reference_context(ctx_list)
    logger.log_kv("DATA CONTEXT", summarize_context(reference_ctx))

    trainer = _build_trainer(
        cfg=cfg,
        ctx_list=ctx_list,
        data_tensor=data_tensor,
        log_dir=log_dir,
        logger=logger,
    )
    logger.mark_training_start()
    train_start = time.time()
    history, best_eq = trainer.train(
        data_tensor=data_tensor,
        n_epochs=int(cfg["n_epochs"]),
        target_field=target_field,
        verbose=True,
    )
    train_elapsed = logger.mark_training_end(time.time() - train_start)
    # Persist the training runtime immediately, before any post-training
    # finalization step can fail or be interrupted. The final RUN SUMMARY still
    # repeats the same value for backwards compatibility.
    logger.info(f"  {'Training elapsed':24s}: {train_elapsed:.1f}s")

    result = finalize_and_plot(
        trainer,
        history,
        best_eq,
        result_dir,
        equation_name,
        true_eq,
        target_field,
        reference_ctx,
        logger,
        cfg,
        ctx_list_for_refit=ctx_list,
        emit_summary=True,
        summary_elapsed_sec=train_elapsed,
        summary_elapsed_label="Training elapsed",
    )
    return {"field": field, **result, "result_dir": result_dir, "train_elapsed_sec": train_elapsed}


def save_multifield_summary(
    discovered: Dict[str, Dict[str, Any]],
    true_eq: Any,
    result_dir: str,
    summary_filename: str,
    logger: WiSEDLogger,
) -> Dict[str, Dict[str, Any]]:
    """Save the multi-field discovery summary as JSON."""
    summary = {
        field: {
            "true_equation": true_eq.get(field, "") if isinstance(true_eq, dict) else str(true_eq),
            "raw_discovered": item["best_eq"].get("raw_readable", item["best_eq"].get("readable", "")),
            "discovered": item["polished_equation"],
        }
        for field, item in discovered.items()
    }

    os.makedirs(result_dir, exist_ok=True)
    summary_path = os.path.join(result_dir, summary_filename)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.section("MULTI-FIELD SUMMARY")
    logger.info(f"Saved summary: {os.path.basename(summary_path)}")
    for field, item in summary.items():
        logger.info(f"[{field}] eq={item['discovered']}")
    return summary


def discover_multifield_joint(
    *,
    task_prefix: str,
    target_fields: List[str],
    cfg: Dict[str, Any],
    data_tensor: torch.Tensor,
    ctx_list: List[DataContext],
    result_root: str,
    log_root: str,
    true_eq: Any,
    logger: WiSEDLogger,
) -> Tuple[Dict[str, Dict[str, Any]], float]:
    """Jointly discover time-evolution equations for several fields."""
    reference_ctx = _require_reference_context(ctx_list)
    logger.log_kv(
        "JOINT TRAINING SETUP",
        {
            "targets": list(target_fields),
            **summarize_context(reference_ctx),
        },
    )

    trainer = _build_trainer(
        cfg=cfg,
        ctx_list=ctx_list,
        data_tensor=data_tensor,
        log_dir=log_root,
        logger=logger,
    )

    logger.mark_training_start()
    train_start = time.time()
    history, best_eqs = trainer.train(
        data_tensor=data_tensor,
        n_epochs=int(cfg["n_epochs"]),
        target_fields=target_fields,
        verbose=True,
    )
    train_elapsed = logger.mark_training_end(time.time() - train_start)
    # Persist the training runtime immediately, before any post-training
    # finalization step can fail or be interrupted. The final RUN SUMMARY still
    # repeats the same value for backwards compatibility.
    logger.info(f"  {'Training elapsed':24s}: {train_elapsed:.1f}s")

    discovered: Dict[str, Dict[str, Any]] = {}
    for target_field in target_fields:
        field = _field_from_target_field(target_field)
        equation_name = f"{task_prefix}_{field}"
        result_dir = os.path.join(result_root, field)
        os.makedirs(result_dir, exist_ok=True)

        trainer.best_equations[str(target_field)] = dict(
            best_eqs.get(target_field, trainer.best_equations.get(target_field, trainer._empty_best_equation()))
        )
        trainer._activate_target(target_field)
        best_eq = trainer.best_equation

        result = finalize_and_plot(
            trainer,
            history,
            best_eq,
            result_dir,
            equation_name,
            true_eq,
            target_field,
            reference_ctx,
            logger,
            cfg,
            ctx_list_for_refit=ctx_list,
            emit_summary=False,
        )
        discovered[field] = {"field": field, **result, "result_dir": result_dir}

    return discovered, train_elapsed
