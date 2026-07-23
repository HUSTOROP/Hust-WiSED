from __future__ import annotations
import torch, random
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional, Any
from models.symbol_vocabulary import SYM2IDX, IDX2SYM, is_valid_sequence, count_constants
from models.weak_form_evaluator import DataContext, WeakFormEvaluator, compute_fitness, compose_fitness, _Compiler, evaluate_scoring_tensors, get_valid_axis_idxs_for_D, get_valid_terminal_idxs, normalize_target_field_name
from models.operator_policy import OperatorPolicy, build_operator_policy, symbols_to_indices

# ==============================================================================
# 1. Operator and cache initialization.
# ==============================================================================
_BINARY, _UNARY, _DERIV = [], [], []
_ACTIVE_OPERATOR_POLICY: OperatorPolicy = build_operator_policy(epoch=0, spatial_ndim=1)

def get_active_operator_policy() -> OperatorPolicy:
    return _ACTIVE_OPERATOR_POLICY

def set_operator_policy(policy: OperatorPolicy) -> OperatorPolicy:
    global _ACTIVE_OPERATOR_POLICY, _BINARY, _UNARY, _DERIV
    _ACTIVE_OPERATOR_POLICY = policy
    _BINARY = symbols_to_indices(policy.binary_symbols)
    _UNARY = symbols_to_indices(policy.unary_symbols)
    _DERIV = symbols_to_indices(policy.deriv_symbols)
    return _ACTIVE_OPERATOR_POLICY

def refresh_operator_groups(
    epoch: int = 0,
    spatial_ndim: int = 1,
    policy: Optional[OperatorPolicy] = None,
) -> OperatorPolicy:
    if policy is None:
        policy = build_operator_policy(epoch=epoch, spatial_ndim=spatial_ndim)
    return set_operator_policy(policy)

def sync_operator_groups_with_vocab() -> OperatorPolicy:
    # Remap token IDs after a vocabulary update without resetting the active policy.
    return set_operator_policy(_ACTIVE_OPERATOR_POLICY)

refresh_operator_groups(epoch=0, spatial_ndim=1)

_OPT_RESULT_CACHE, _WARM_START_CACHE = {}, {}
_SIMPLIFY_CACHE: Dict[Tuple[Any, ...], List[int]] = {}
_OPT_CACHE_STATS = {"queries": 0, "hits": 0, "misses": 0, "saved_evals": 0, "warm_starts": 0}

def _cached_algebraic_simplify(km, seq, consts=None, prune_tol: float = 5e-5, const_mask=None) -> List[int]:
    """Fast wrapper around normalizer.algebraic_simplify.

    The normalizer is intentionally accuracy-oriented.  During evolution the same
    token trees are simplified many times, so caching by token tuple plus a coarse
    constant/pruning signature removes repeated parsing/canonicalization without
    changing the discovered equation space.
    """
    seq_key = tuple(int(t) for t in seq)
    if consts is None:
        const_key = None
    else:
        arr = np.asarray(consts, dtype=np.float64).reshape(-1)
        # only the zero/nonzero pruning decision matters for algebraic structure
        const_key = tuple((np.abs(arr) >= float(prune_tol)).astype(np.int8).tolist())
    mask_key = None if const_mask is None else tuple(np.asarray(const_mask, dtype=bool).reshape(-1).tolist())
    key = (seq_key, const_key, float(prune_tol), mask_key)
    cached = _SIMPLIFY_CACHE.get(key)
    if cached is not None:
        return list(cached)

    # Accuracy-first mode: cache simplification results, but do not replace full
    # algebraic simplification with a purely token-level shortcut.  The normalizer
    # itself may perform conservative fast pruning first, then falls back to the
    # complete canonical/SymPy path when available.
    try:
        out = km.normalizer.algebraic_simplify(
            seq, consts=consts, prune_tol=prune_tol, const_mask=const_mask
        )
    except Exception:
        out = list(seq)
    out = list(map(int, out)) if out is not None else list(seq)
    if len(_SIMPLIFY_CACHE) > 50000:
        _SIMPLIFY_CACHE.clear()
    _SIMPLIFY_CACHE[key] = out
    return list(out)

def clear_optimizer_caches(*args, reset_stats=False, **kwargs):
    _OPT_RESULT_CACHE.clear()
    _WARM_START_CACHE.clear()
    _SIMPLIFY_CACHE.clear()
    if reset_stats:
        for k in _OPT_CACHE_STATS:
            _OPT_CACHE_STATS[k] = 0

def get_optimizer_cache_stats(*args, **kwargs):
    total = max(1, _OPT_CACHE_STATS["queries"])
    _OPT_CACHE_STATS["hit_rate"] = _OPT_CACHE_STATS["hits"] / total
    return dict(_OPT_CACHE_STATS)

# ==============================================================================
# 2. Scale-aware safeguarded constant optimization.
# ==============================================================================
def _is_affine_symbolic(root_node) -> bool:
    """Conservatively detect whether an expression is affine in its constants."""
    if root_node is None:
        return False

    def walk(n):
        if n.op == "const":
            return True, True
        if not n.children:
            return True, False

        child_results = [walk(c) for c in n.children]
        all_affine = all(res[0] for res in child_results)
        any_const = any(res[1] for res in child_results)
        if not all_affine:
            return False, any_const

        # Constants inside nonlinear operators are no longer affine.
        if n.op in ["sin", "cos", "exp", "log", "^", "/", "sq", "cube"] and any_const:
            return False, True

        # Multiplication is affine only if at most one child depends on constants.
        if n.op == "*":
            if sum(1 for res in child_results if res[1]) > 1:
                return False, True

        return True, any_const

    return walk(root_node)[0]


def _ctxs_as_list(ctx) -> List[DataContext]:
    return list(ctx) if isinstance(ctx, (list, tuple)) else [ctx]


def _make_multictx_template_key(token_seq, ctxs: List[DataContext], target_field: str) -> str:
    """Cache key for constants optimized over the whole context bundle."""
    try:
        from utils.structure_cache import get_structure_key_manager
        km = get_structure_key_manager()
        struct_key = km.template_key(token_seq).key
    except Exception:
        struct_key = str(list(map(int, token_seq)))

    ctx_sig = "|".join(
        str(getattr(cctx, "cache_id", "") or f"ctx{i}")
        for i, cctx in enumerate(ctxs)
    )
    return f"{ctx_sig}::{target_field}::{struct_key}"


def _safe_float(x: Any, default: float = 1e18) -> float:
    try:
        if hasattr(x, "item"):
            x = x.item()
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _mean_base_fitness_stats(
    token_seq,
    ctxs: List[DataContext],
    consts,
    target_field: str,
    mdl_penalty_weight: float = 0.25,
    structural_risk_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate a fixed constant vector on all contexts and average base stats."""
    rows: List[Dict[str, Any]] = []
    risk_kwargs = dict(structural_risk_kwargs or {})

    for cctx in ctxs:
        try:
            fit = compute_fitness(
                token_seq,
                cctx,
                consts,
                lambda_complexity=0.0,
                target_field=target_field,
                mdl_penalty_weight=float(mdl_penalty_weight),
                **risk_kwargs,
            )
            if not isinstance(fit, dict) or not bool(fit.get("pred_valid", False)):
                continue
            mse = _safe_float(fit.get("residual_mse", 1e18))
            if mse >= 1e18:
                continue
            rows.append(fit)
        except Exception:
            continue

    if not rows:
        return {"fitness": 1e10, "residual_mse": 1e10, "pred_valid": False}

    def mean_of(key: str, default: float = 0.0) -> float:
        vals = [_safe_float(r.get(key, default), default=default) for r in rows]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float(default)

    residual_vals = [_safe_float(r.get("residual_mse", 1e18)) for r in rows]
    out = {
        "residual_mse": mean_of("residual_mse", 1e10),
        "raw_mse": mean_of("raw_mse", 1e10),
        "complexity": mean_of("complexity", 1.0),
        "max_deriv_depth": int(max(int(r.get("max_deriv_depth", 0)) for r in rows)),
        "num_constants": int(max(int(r.get("num_constants", 0)) for r in rows)),
        "diverge_penalty": any(bool(r.get("diverge_penalty", False)) for r in rows),
        "pred_valid": True,
        "n_eff": mean_of("n_eff", 100.0),
        "vocab_size": int(max(int(r.get("vocab_size", len(SYM2IDX))) for r in rows)),
        "mdl_penalty_weight": float(mdl_penalty_weight),
        "derivative_product_count": mean_of("derivative_product_count", 0.0),
        "derivative_power_count": mean_of("derivative_power_count", 0.0),
        "explicit_high_order_deriv_count": mean_of("explicit_high_order_deriv_count", 0.0),
        "raw_derivative_count": mean_of("raw_derivative_count", 0.0),
        "composite_derivative_count": mean_of("composite_derivative_count", 0.0),
        "nested_composite_derivative_count": mean_of("nested_composite_derivative_count", 0.0),
        "spectral_high_fraction": mean_of("spectral_high_fraction", 0.0),
        "spectral_mid_fraction": mean_of("spectral_mid_fraction", 0.0),
        "spectral_high_mse": mean_of("spectral_high_mse", 0.0),
        "spectral_mid_mse": mean_of("spectral_mid_mse", 0.0),
        "multi_window_mean_mse": mean_of("multi_window_mean_mse", 0.0),
        "multi_window_logvar": mean_of("multi_window_logvar", 0.0),
        "multi_window_highfreq_proxy": mean_of("multi_window_highfreq_proxy", 0.0),
        "derivative_product_penalty_weight": float(risk_kwargs.get("derivative_product_penalty_weight", 1.20)),
        "derivative_power_penalty_weight": float(risk_kwargs.get("derivative_power_penalty_weight", 1.10)),
        "explicit_high_order_deriv_penalty_weight": float(risk_kwargs.get("explicit_high_order_deriv_penalty_weight", 1.60)),
        "composite_derivative_penalty_weight": float(risk_kwargs.get("composite_derivative_penalty_weight", 1.50)),
        "nested_composite_derivative_penalty_weight": float(risk_kwargs.get("nested_composite_derivative_penalty_weight", 2.00)),
        "spectral_residual_enable": bool(risk_kwargs.get("spectral_residual_enable", False)),
        "spectral_residual_weight": float(risk_kwargs.get("spectral_residual_weight", 0.0)),
        "spectral_mid_residual_weight": float(risk_kwargs.get("spectral_mid_residual_weight", 0.0)),
        "spectral_high_fraction_weight": float(risk_kwargs.get("spectral_high_fraction_weight", 0.0)),
        "spectral_high_threshold": float(risk_kwargs.get("spectral_high_threshold", 0.45)),
        "spectral_mid_threshold": float(risk_kwargs.get("spectral_mid_threshold", 0.20)),
        "spectral_residual_subsample": int(max(1, risk_kwargs.get("spectral_residual_subsample", 1))),
        "multi_window_enable": bool(risk_kwargs.get("multi_window_enable", False)),
        "multi_window_residual_weight": float(risk_kwargs.get("multi_window_residual_weight", 0.0)),
        "multi_window_variance_weight": float(risk_kwargs.get("multi_window_variance_weight", 0.0)),
        "multi_window_highfreq_weight": float(risk_kwargs.get("multi_window_highfreq_weight", 0.0)),
        "multi_window_subsample": int(max(1, risk_kwargs.get("multi_window_subsample", 1))),
        "multi_window_spatial_only": bool(risk_kwargs.get("multi_window_spatial_only", True)),
        "contrastive_reward_spectral_weight": float(risk_kwargs.get("contrastive_reward_spectral_weight", 0.0)),
        "contrastive_reward_multi_window_weight": float(risk_kwargs.get("contrastive_reward_multi_window_weight", 0.0)),
        # Diagnostic fields; downstream code can ignore them.
        "ctx_count": len(rows),
        "ctx_residual_mse_std": float(np.std(residual_vals)),
    }
    return out


def _build_affine_design_across_contexts(
    token_seq,
    ctxs: List[DataContext],
    target_field: str,
    n_consts: int,
    *,
    normalize_each_context: bool = True,
    scoring_form: str = "weak",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build one global least-squares problem over all contexts.

    This intentionally follows the original discovery path: every weak-form
    point contributes to the affine design after context-level normalization.
    No residual-amplitude WLS is applied here, because it biases coefficient
    recovery toward high-amplitude regions and can make the correct symbolic
    structure lose to overfit structures.
    """
    Phi_blocks: List[torch.Tensor] = []
    y_blocks: List[torch.Tensor] = []

    for cctx in ctxs:
        try:
            c_base = np.zeros(n_consts, dtype=np.float64)
            lhs, rhs0 = evaluate_scoring_tensors(
                cctx,
                token_seq,
                c_base,
                target_field=target_field,
                scoring_form=scoring_form,
            )
            if lhs is None:
                continue
            if rhs0 is None or not torch.all(torch.isfinite(rhs0)):
                continue

            y = (lhs - rhs0).flatten()
            Phi_cols: List[torch.Tensor] = []

            ok = True
            for j in range(n_consts):
                c_test = np.zeros(n_consts, dtype=np.float64)
                c_test[j] = 1.0
                _, rhs_j = evaluate_scoring_tensors(
                    cctx,
                    token_seq,
                    c_test,
                    target_field=target_field,
                    scoring_form=scoring_form,
                )
                if rhs_j is None or not torch.all(torch.isfinite(rhs_j)):
                    ok = False
                    break
                Phi_cols.append((rhs_j - rhs0).flatten())

            if not ok or len(Phi_cols) != n_consts:
                continue

            Phi = torch.stack(Phi_cols, dim=1)
            finite = torch.isfinite(y) & torch.all(torch.isfinite(Phi), dim=1)
            if int(finite.sum().item()) < max(4, n_consts + 1):
                continue

            y = y[finite]
            Phi = Phi[finite]

            if normalize_each_context:
                denom = torch.sqrt(torch.var(lhs.flatten()) + 1e-12)
                y = y / denom
                Phi = Phi / denom

            # Equal context weight independent of the number of weak-form points.
            weight = 1.0 / max(1.0, float(y.numel()) ** 0.5)
            y_blocks.append(y * weight)
            Phi_blocks.append(Phi * weight)

        except Exception:
            continue

    if not Phi_blocks:
        return None, None

    try:
        Phi_all = torch.cat(Phi_blocks, dim=0)
        y_all = torch.cat(y_blocks, dim=0)
    except Exception:
        return None, None

    if not torch.all(torch.isfinite(Phi_all)) or not torch.all(torch.isfinite(y_all)):
        return None, None

    col_max = torch.max(torch.abs(Phi_all), dim=0)[0]
    if torch.any(col_max < 1e-12):
        return None, None

    return Phi_all, y_all



def _solve_global_ridge_least_squares(
    Phi: torch.Tensor,
    y: torch.Tensor,
    *,
    ridge: float = 1e-8,
    clamp: float = 1e4,
    scaled_threshold: float = 1e-3,
) -> np.ndarray:
    n_consts = int(Phi.shape[1])

    # Scale each context before ridge solving so constant-offset columns do not
    # dominate the conditioning.
    y_std = torch.std(y)
    y_mean = torch.abs(torch.mean(y))
    # Use the mean scale when the target is nearly constant.
    y_scale = torch.where(y_std > 1e-8, y_std, y_mean + 1e-12)

    Phi_std = torch.std(Phi, dim=0)
    Phi_mean = torch.abs(torch.mean(Phi, dim=0))
    # Constant columns have tiny variance, so scale them by their mean value.
    Phi_scale = torch.where(Phi_std > 1e-8, Phi_std, Phi_mean + 1e-12)

    # Map to a dimensionless least-squares problem.
    y_scaled = y / y_scale
    Phi_scaled = Phi / Phi_scale

    # Build the normal equations.
    A = Phi_scaled.T @ Phi_scaled
    b = Phi_scaled.T @ y_scaled

    try:
        cond_num = torch.linalg.cond(A).item()
    except Exception:
        cond_num = float('inf')

    if cond_num > 1e10:
        raise ValueError("Ill-conditioned matrix")

    # STRidge-style refit on the active set.
    A_reg = A + float(ridge) * torch.eye(n_consts, dtype=A.dtype, device=A.device)

    try:
        c_scaled = torch.linalg.solve(A_reg, b)
    except Exception:
        c_scaled = torch.linalg.lstsq(A_reg, b.unsqueeze(1)).solution.squeeze(1)

    scaled_threshold = float(max(0.0, scaled_threshold))
    active_mask = torch.ones_like(c_scaled, dtype=torch.bool)
    if scaled_threshold > 0.0:
        active_mask = torch.abs(c_scaled) >= scaled_threshold

    if scaled_threshold > 0.0 and not torch.all(active_mask) and torch.any(active_mask):
        A_sub = A[active_mask][:, active_mask]
        b_sub = b[active_mask]
        A_sub_reg = A_sub + float(ridge) * torch.eye(A_sub.shape[0], dtype=A.dtype, device=A.device)
        try:
            c_sub = torch.linalg.solve(A_sub_reg, b_sub)
            c_scaled_new = torch.zeros_like(c_scaled)
            c_scaled_new[active_mask] = c_sub
            c_scaled = c_scaled_new
        except Exception:
            pass
    elif scaled_threshold > 0.0 and not torch.any(active_mask):
        c_scaled = torch.zeros_like(c_scaled)

    # =========================================================================
    # Convert constants back to physical units with the current y_scale and Phi_scale.
    # =========================================================================
    c = c_scaled * (y_scale / Phi_scale)
    c = torch.clamp(c, -float(clamp), float(clamp))
    return c.detach().cpu().numpy().astype(np.float64)


def _nonlinear_objective_across_contexts(
    token_seq,
    ctxs: List[DataContext],
    c_tensor: torch.Tensor,
    target_field: str,
    l2: float = 5e-3,
    scoring_form: str = "weak",
) -> torch.Tensor:
    """Differentiable mean normalized residual over all contexts."""
    losses: List[torch.Tensor] = []

    for cctx in ctxs:
        try:
            c_for_ctx = c_tensor.to(dtype=torch.float64, device=cctx.device)
            lhs, rhs = evaluate_scoring_tensors(
                cctx,
                token_seq,
                c_for_ctx,
                target_field=target_field,
                scoring_form=scoring_form,
            )
            if lhs is None:
                continue
            rhs = torch.clamp(rhs, -1e10, 1e10)
            if rhs is None or not torch.all(torch.isfinite(rhs)):
                continue

            residual = lhs - rhs
            denom = torch.var(lhs) + 1e-12
            losses.append(torch.mean(residual ** 2) / denom)
        except Exception:
            continue

    if not losses:
        return torch.sum(c_tensor * 0.0) + torch.tensor(1e10, dtype=torch.float64, device=c_tensor.device)

    return torch.stack(losses).mean() + float(l2) * torch.sum(c_tensor ** 2)


def _initial_constant_guesses(n_consts: int, init_consts=None, n_init: int = 2) -> List[np.ndarray]:
    guesses: List[np.ndarray] = []

    if init_consts is not None:
        try:
            arr = np.asarray(init_consts, dtype=np.float64).reshape(-1)
            if arr.size == n_consts:
                guesses.append(arr.copy())
        except Exception:
            pass

    guesses.extend([
        np.zeros(n_consts, dtype=np.float64),
        np.ones(n_consts, dtype=np.float64),
        -np.ones(n_consts, dtype=np.float64),
        np.random.randn(n_consts).astype(np.float64) * 0.1,
    ])

    for _ in range(max(0, int(n_init))):
        guesses.append(np.random.randn(n_consts).astype(np.float64) * 0.5)

    out: List[np.ndarray] = []
    seen = set()
    for g in guesses:
        key = tuple(np.round(g, decimals=10).tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append(g)
    return out


def optimize_constants(token_seq, ctx, target_field="du_t", n_init=2, init_consts=None, **kwargs):
    """Optimize one shared constant vector for all contexts.

    This replaces the old ctxs[0]-centric objective with:

        c* = argmin_c mean_i Loss(ctx_i, c)

    It directly reduces constant-induced amplification of spurious correlations,
    without adding post-hoc validation penalties.
    """
    target_field = normalize_target_field_name(target_field)

    ctxs_all = _ctxs_as_list(ctx)
    if not ctxs_all:
        return np.zeros(count_constants(token_seq), dtype=np.float64), {
            "fitness": 1e10,
            "residual_mse": 1e10,
            "pred_valid": False,
        }

    # Accuracy-first context policy.  Coarse search can still use a small
    # context budget, but full refinement may pass a larger budget.  This is
    # important for weakly coupled reaction systems: ranking by one trajectory
    # can promote accidental correlations over the structurally correct terms.
    const_ctx_index = int(kwargs.get("const_ctx_index", 0))
    const_ctx_index = max(0, min(const_ctx_index, len(ctxs_all) - 1))
    ctx_budget = int(kwargs.get("const_context_budget", 1))
    ctx_budget = max(1, min(ctx_budget, len(ctxs_all)))
    if ctx_budget == 1:
        ctxs = [ctxs_all[const_ctx_index]]
    else:
        ordered = list(ctxs_all[const_ctx_index:]) + list(ctxs_all[:const_ctx_index])
        ctxs = ordered[:ctx_budget]

    n_consts = count_constants(token_seq)
    structural_risk_kwargs = {
        "derivative_product_penalty_weight": float(kwargs.get("derivative_product_penalty_weight", 1.20)),
        "derivative_power_penalty_weight": float(kwargs.get("derivative_power_penalty_weight", 1.10)),
        "explicit_high_order_deriv_penalty_weight": float(kwargs.get("explicit_high_order_deriv_penalty_weight", 0.80)),
        "composite_derivative_penalty_weight": float(kwargs.get("composite_derivative_penalty_weight", 1.50)),
        "nested_composite_derivative_penalty_weight": float(kwargs.get("nested_composite_derivative_penalty_weight", 2.00)),
        "spectral_residual_enable": bool(kwargs.get("spectral_residual_enable", False)),
        "spectral_residual_weight": float(kwargs.get("spectral_residual_weight", 0.0)),
        "spectral_mid_residual_weight": float(kwargs.get("spectral_mid_residual_weight", 0.0)),
        "spectral_high_fraction_weight": float(kwargs.get("spectral_high_fraction_weight", 0.0)),
        "spectral_high_threshold": float(kwargs.get("spectral_high_threshold", 0.45)),
        "spectral_mid_threshold": float(kwargs.get("spectral_mid_threshold", 0.20)),
        "spectral_residual_subsample": int(max(1, kwargs.get("spectral_residual_subsample", 1))),
        "multi_window_enable": bool(kwargs.get("multi_window_enable", False)),
        "multi_window_kernels": kwargs.get("multi_window_kernels", (3, 5)),
        "multi_window_residual_weight": float(kwargs.get("multi_window_residual_weight", 0.0)),
        "multi_window_variance_weight": float(kwargs.get("multi_window_variance_weight", 0.0)),
        "multi_window_highfreq_weight": float(kwargs.get("multi_window_highfreq_weight", 0.0)),
        "multi_window_subsample": int(max(1, kwargs.get("multi_window_subsample", 1))),
        "multi_window_spatial_only": bool(kwargs.get("multi_window_spatial_only", True)),
        "contrastive_reward_spectral_weight": float(kwargs.get("contrastive_reward_spectral_weight", 0.0)),
        "contrastive_reward_multi_window_weight": float(kwargs.get("contrastive_reward_multi_window_weight", 0.0)),
        "operator_mode": str(kwargs.get("operator_mode", "pde")),
        "forbidden_rhs_symbols": kwargs.get("forbidden_rhs_symbols", None),
        "zero_order_field_penalty_weight": float(kwargs.get("zero_order_field_penalty_weight", 0.0)),
        "max_zero_order_field_terms": kwargs.get("max_zero_order_field_terms", None),
    }

    # Generic structural guard before any expensive constant optimization.  Keep
    # this before the no-constant path so templates such as u*lap(u) cannot skip
    # diffusion-profile checks.
    ok_struct, struct_report = _passes_structural_guard(list(map(int, token_seq)), **kwargs)
    if not ok_struct:
        base_stats = {
            "fitness": 1e10,
            "residual_mse": 1e10,
            "pred_valid": False,
            "struct_guard_reject": True,
            "struct_guard_reason": str(struct_report.get("reason", "structural_guard")),
            "complexity": float(len(token_seq)),
        }
        return np.zeros(n_consts, dtype=np.float64), compose_fitness(base_stats)

    if n_consts <= 0:
        base = _mean_base_fitness_stats(token_seq, ctxs, [], target_field, float(kwargs.get("mdl_penalty_weight", 0.25)), structural_risk_kwargs)
        return np.array([], dtype=np.float64), compose_fitness(base)

    try:
        root_node = _Compiler(list(map(int, token_seq))).parse()
    except Exception:
        root_node = None

    best_c = np.zeros(n_consts, dtype=np.float64)
    affine_solved = False
    is_affine_expr = bool(_is_affine_symbolic(root_node))

    # The same symbolic template can be scored with different optimization
    # budgets.  In particular, coarse search may disable non-affine LBFGS while
    # elite full-refine enables it.  Include the optimization mode and key numeric
    # knobs in the cache key so a coarse quick score cannot mask a later elite
    # nonlinear optimization.
    opt_mode = "affine" if is_affine_expr else ("nonaff_lbfgs" if bool(kwargs.get("enable_nonlinear_lbfgs", True)) else "nonaff_fast")
    def _sig_value(v):
        if isinstance(v, bool):
            return "1" if v else "0"
        if isinstance(v, (list, tuple)):
            return ",".join(_sig_value(x) for x in v)
        try:
            return f"{float(v):.6g}"
        except Exception:
            return str(v)
    risk_sig = ";".join(f"{k}={_sig_value(structural_risk_kwargs[k])}" for k in sorted(structural_risk_kwargs))
    tpl_key = (
        _make_multictx_template_key(token_seq, ctxs, target_field)
        + f"::mode={opt_mode}"
        + f"::ridge={float(kwargs.get('const_ridge', 1e-8)):.3g}"
        + f"::thr={float(kwargs.get('const_scaled_threshold', 1e-4)):.3g}"
        + f"::prune={float(kwargs.get('const_prune_tol', kwargs.get('const_scaled_threshold', 1e-4))):.3g}"
        + f"::mdl={float(kwargs.get('mdl_penalty_weight', 0.25)):.3g}"
        + f"::score={str(kwargs.get('scoring_form', 'weak'))}"
        + f"::risk={risk_sig}"
    )

    _OPT_CACHE_STATS["queries"] += 1
    if tpl_key in _OPT_RESULT_CACHE:
        _OPT_CACHE_STATS["hits"] += 1
        _OPT_CACHE_STATS["saved_evals"] += 1
        best_c, cached_base = _OPT_RESULT_CACHE[tpl_key]
        return best_c.copy(), compose_fitness(cached_base)
    _OPT_CACHE_STATS["misses"] += 1

    # Fast global solve for affine structures.
    if is_affine_expr:
        try:
            Phi, y = _build_affine_design_across_contexts(
                token_seq,
                ctxs,
                target_field,
                n_consts,
                normalize_each_context=True,
                scoring_form=str(kwargs.get("scoring_form", "weak")),
            )
            if Phi is not None and y is not None:
                best_c = _solve_global_ridge_least_squares(
                    Phi,
                    y,
                    ridge=float(kwargs.get("const_ridge", 1e-8)),
                    clamp=float(kwargs.get("const_clamp", 1e4)),
                    scaled_threshold=float(kwargs.get("const_scaled_threshold", 1e-4)),
                )
                affine_solved = True
        except ValueError as e:
            if str(e) == "Ill-conditioned matrix":
                base_stats = {"fitness": 1e10, "residual_mse": 1e10, "pred_valid": False}
                return np.zeros(n_consts, dtype=np.float64), compose_fitness(base_stats)
            affine_solved = False
        except Exception:
            affine_solved = False

    # Non-affine fallback, or affine solve failure.
    if not affine_solved:
        # Accuracy-first mode: non-affine constants remain fully optimizable by
        # default.  This preserves discovery capacity for structures such as
        # sin(c*u), exp(c*u), or u^c.  Users can explicitly disable this in a
        # speed-first ablation, but it is not the default.
        enable_nonlinear_lbfgs = bool(kwargs.get("enable_nonlinear_lbfgs", True))

        if (not is_affine_expr) and (not enable_nonlinear_lbfgs):
            best_fit_val = 1e18
            guesses = _initial_constant_guesses(n_consts, init_consts=init_consts, n_init=0)[:3]
            for c_init in guesses:
                try:
                    base_try = _mean_base_fitness_stats(
                        token_seq,
                        ctxs,
                        c_init,
                        target_field,
                        float(kwargs.get("mdl_penalty_weight", 0.25)),
                        structural_risk_kwargs,
                    )
                    fit_try = compose_fitness(base_try)
                    val = _safe_float(fit_try.get("fitness", 1e18), default=1e18)
                    if val < best_fit_val:
                        best_fit_val = val
                        best_c = np.asarray(c_init, dtype=np.float64).copy()
                except Exception:
                    continue
        else:
            best_val = 1e18
            device = getattr(ctxs[0], "device", torch.device("cpu"))

            eff_n_init = int(kwargs.get("const_nonlinear_n_init", n_init))
            eff_max_iter = int(kwargs.get("const_lbfgs_max_iter", 20))
            for c_init in _initial_constant_guesses(n_consts, init_consts=init_consts, n_init=eff_n_init):
                try:
                    c_tensor = torch.tensor(c_init, dtype=torch.float64, device=device, requires_grad=True)
                    optimizer = torch.optim.LBFGS(
                        [c_tensor],
                        max_iter=eff_max_iter,
                        history_size=int(kwargs.get("const_lbfgs_history", 10)),
                        line_search_fn="strong_wolfe",
                    )

                    def closure():
                        optimizer.zero_grad()
                        loss = _nonlinear_objective_across_contexts(
                            token_seq,
                            ctxs,
                            c_tensor,
                            target_field,
                            l2=float(kwargs.get("const_l2", 5e-3)),
                            scoring_form=str(kwargs.get("scoring_form", "weak")),
                        )
                        if torch.isfinite(loss):
                            loss.backward()
                        return loss

                    optimizer.step(closure)

                    with torch.no_grad():
                        final_loss = _nonlinear_objective_across_contexts(
                            token_seq,
                            ctxs,
                            c_tensor,
                            target_field,
                            l2=float(kwargs.get("const_l2", 5e-3)),
                            scoring_form=str(kwargs.get("scoring_form", "weak")),
                        )
                        val = _safe_float(final_loss, default=1e18)
                        if val < best_val:
                            best_val = val
                            best_c = c_tensor.detach().cpu().numpy().astype(np.float64)
                except Exception:
                    continue

    const_clamp = float(kwargs.get("const_clamp", 1e4))
    best_c = np.nan_to_num(np.asarray(best_c, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    best_c = np.clip(best_c, -const_clamp, const_clamp)

    # Global raw-coefficient pruning: once coefficients are mapped back to their
    # physical scale, any term with |coef| < const_prune_tol is treated as zero
    # for fitness evaluation, logging, archive/rerank, and final export.
    const_prune_tol = float(kwargs.get("const_prune_tol", kwargs.get("const_scaled_threshold", 1e-4)) or 0.0)
    if const_prune_tol > 0.0 and best_c.size > 0:
        best_c[np.abs(best_c) < const_prune_tol] = 0.0

    base = _mean_base_fitness_stats(token_seq, ctxs, best_c, target_field, float(kwargs.get("mdl_penalty_weight", 0.25)), structural_risk_kwargs)
    final_fit = compose_fitness(base)
    _OPT_RESULT_CACHE[tpl_key] = (best_c.copy(), base)

    return best_c, final_fit




# ==============================================================================
# 2b. Task-agnostic structural guardrails for identifiability
# ==============================================================================
def _structural_pathology_report(seq: List[int]) -> Dict[str, Any]:
    """Return syntax-level pathologies that hurt generic identifiability.

    This deliberately does not encode any target PDE.  It only rejects/flags
    expression families that are almost always non-identifiable under finite
    trajectory data: arbitrary exponentiation, excessive polynomial degree,
    powers of derivatives, and products containing multiple derivative factors.
    Field-times-derivative terms are allowed because generic advection-like
    equations may need them when no macro is used.
    """
    report = {
        "valid": False,
        "has_power_operator": False,
        "max_polynomial_degree": 0,
        "derivative_power_count": 0,
        "multi_derivative_product_count": 0,
        "field_derivative_product_count": 0,
        "token_len": int(len(seq or [])),
        "reason": "",
    }
    try:
        if not is_valid_sequence(list(seq)):
            report["reason"] = "invalid_prefix"
            return report
        root = _Compiler(list(map(int, seq))).parse()
    except Exception as exc:
        report["reason"] = f"parse_error:{exc}"
        return report

    fields = {"u", "v", "w"}

    def walk(n) -> Tuple[int, bool, bool]:
        """Return (field_degree, contains_derivative_like, contains_plain_field)."""
        op = str(getattr(n, "op", ""))
        if op in fields:
            report["max_polynomial_degree"] = max(report["max_polynomial_degree"], 1)
            return 1, False, True
        if op == "const" or op in {"t", "x", "y", "z"}:
            return 0, False, False
        if op in {"D", "lap", "adv"}:
            child = n.children[0] if getattr(n, "children", None) else None
            deg, _has_d, has_f = walk(child) if child is not None else (0, False, False)
            deg = max(1, deg)
            report["max_polynomial_degree"] = max(report["max_polynomial_degree"], deg)
            return deg, True, has_f
        if op in {"neg", "sin", "cos", "exp", "log"}:
            deg, has_d, has_f = walk(n.children[0]) if getattr(n, "children", None) else (0, False, False)
            return deg, has_d, has_f
        if op == "sq":
            deg, has_d, has_f = walk(n.children[0]) if getattr(n, "children", None) else (0, False, False)
            if has_d:
                report["derivative_power_count"] += 1
            deg = 2 * max(1, deg) if has_f or has_d else 0
            report["max_polynomial_degree"] = max(report["max_polynomial_degree"], deg)
            return deg, has_d, has_f
        if op == "cube":
            deg, has_d, has_f = walk(n.children[0]) if getattr(n, "children", None) else (0, False, False)
            if has_d:
                report["derivative_power_count"] += 1
            deg = 3 * max(1, deg) if has_f or has_d else 0
            report["max_polynomial_degree"] = max(report["max_polynomial_degree"], deg)
            return deg, has_d, has_f
        if op == "^":
            report["has_power_operator"] = True
            left = n.children[0] if getattr(n, "children", None) else None
            deg, has_d, has_f = walk(left) if left is not None else (0, False, False)
            if has_d:
                report["derivative_power_count"] += 1
            deg = max(5, deg)
            report["max_polynomial_degree"] = max(report["max_polynomial_degree"], deg)
            return deg, has_d, has_f
        if op == "*":
            l = n.children[0] if getattr(n, "children", None) else None
            r = n.children[1] if getattr(n, "children", None) and len(n.children) > 1 else None
            ldeg, ld, lf = walk(l) if l is not None else (0, False, False)
            rdeg, rd, rf = walk(r) if r is not None else (0, False, False)
            if ld and rd:
                report["multi_derivative_product_count"] += 1
            elif (ld and rf) or (rd and lf):
                report["field_derivative_product_count"] += 1
            deg = ldeg + rdeg
            report["max_polynomial_degree"] = max(report["max_polynomial_degree"], deg)
            return deg, bool(ld or rd), bool(lf or rf)
        if op == "/":
            parts = [walk(c) for c in getattr(n, "children", [])]
            deg = max([x[0] for x in parts] or [0])
            has_d = any(x[1] for x in parts)
            has_f = any(x[2] for x in parts)
            report["max_polynomial_degree"] = max(report["max_polynomial_degree"], deg)
            return deg, has_d, has_f
        if op == "+":
            parts = [walk(c) for c in getattr(n, "children", [])]
            deg = max([x[0] for x in parts] or [0])
            has_d = any(x[1] for x in parts)
            has_f = any(x[2] for x in parts)
            report["max_polynomial_degree"] = max(report["max_polynomial_degree"], deg)
            return deg, has_d, has_f
        parts = [walk(c) for c in getattr(n, "children", [])]
        return max([x[0] for x in parts] or [0]), any(x[1] for x in parts), any(x[2] for x in parts)

    walk(root)
    report["valid"] = True
    return report


def _passes_structural_guard(seq: List[int], **kwargs) -> Tuple[bool, Dict[str, Any]]:
    """Generic hard guard used before expensive scoring.

    The defaults are conservative and task-agnostic.  They are not a candidate
    library and do not encode Burgers/FHN/KdV terms; they only keep the search
    from spending budget on high-power/derivative-product surrogates that are
    structurally non-identifiable on finite data.
    """
    if not bool(kwargs.get("struct_guard_enable", True)):
        return True, {"valid": True, "reason": "disabled"}
    rep = _structural_pathology_report(seq)
    forbidden = _parse_forbidden_symbols(kwargs.get("forbidden_rhs_symbols", None))
    if forbidden:
        syms = [IDX2SYM.get(int(t), "") for t in list(map(int, seq))]
        hit = sorted({s for s in syms if s in forbidden})
        if hit:
            rep["reason"] = "forbidden_rhs_symbol"
            rep["forbidden_rhs_symbols_hit"] = ",".join(hit)
            return False, rep
    if str(kwargs.get("operator_mode", "") or "").strip().lower() in {"diffusion", "diffusion_only", "parabolic_diffusion"}:
        ok, reason = _passes_diffusion_affine_guard(seq)
        if not ok:
            rep["reason"] = reason
            return False, rep
    max_zero_order = kwargs.get("max_zero_order_field_terms", None)
    if max_zero_order is not None:
        try:
            zero_count = _zero_order_field_count(seq)
            rep["zero_order_field_count"] = float(zero_count)
            if float(zero_count) > float(max_zero_order):
                rep["reason"] = "too_many_zero_order_field_terms"
                return False, rep
        except Exception:
            pass
    if not rep.get("valid", False):
        return False, rep
    if int(rep.get("token_len", 0)) > int(kwargs.get("struct_guard_max_tokens", 96)):
        rep["reason"] = "too_many_tokens"
        return False, rep
    if bool(kwargs.get("struct_guard_reject_power_operator", True)) and rep.get("has_power_operator", False):
        rep["reason"] = "arbitrary_power_operator"
        return False, rep
    if int(rep.get("max_polynomial_degree", 0)) > int(kwargs.get("struct_guard_max_polynomial_degree", 3)):
        rep["reason"] = "excessive_polynomial_degree"
        return False, rep
    if bool(kwargs.get("struct_guard_reject_derivative_powers", True)) and int(rep.get("derivative_power_count", 0)) > 0:
        rep["reason"] = "derivative_power"
        return False, rep
    if bool(kwargs.get("struct_guard_reject_multi_derivative_products", True)) and int(rep.get("multi_derivative_product_count", 0)) > 0:
        rep["reason"] = "multi_derivative_product"
        return False, rep
    return True, rep


def _passes_diffusion_affine_guard(seq: List[int]) -> Tuple[bool, str]:
    """Restrict diffusion-profile searches to affine sums of lap(field).

    This guard is opt-in through ``operator_mode=diffusion``.  It keeps the SIB
    raw-field validation focused on discovering transport operators from fields
    instead of letting weak-form search absorb boundary/IC correlations through
    nonlinear products such as ``u*lap(u)`` or ``lap(u)*lap(u)``.
    """
    try:
        root = _Compiler(list(map(int, seq))).parse()
    except Exception:
        return False, "diffusion_guard_parse_failed"

    native_fields = {"u", "v", "w"}

    def is_lap_field(n) -> bool:
        children = list(getattr(n, "children", []) or [])
        if str(getattr(n, "op", "")) != "lap" or len(children) != 1:
            return False
        return str(getattr(children[0], "op", "")) in native_fields

    def is_const(n) -> bool:
        return str(getattr(n, "op", "")) == "const"

    def is_atom(n) -> bool:
        op = str(getattr(n, "op", ""))
        children = list(getattr(n, "children", []) or [])
        if is_const(n) or is_lap_field(n):
            return True
        if op == "neg" and len(children) == 1:
            return is_atom(children[0])
        if op == "*" and len(children) == 2:
            left, right = children
            return (is_const(left) and is_lap_field(right)) or (is_const(right) and is_lap_field(left))
        return False

    def is_expr(n) -> bool:
        op = str(getattr(n, "op", ""))
        children = list(getattr(n, "children", []) or [])
        if is_atom(n):
            return True
        if op == "+" and len(children) == 2:
            return is_expr(children[0]) and is_expr(children[1])
        if op == "neg" and len(children) == 1:
            return is_expr(children[0])
        return False

    return (True, "ok") if is_expr(root) else (False, "diffusion_non_affine_rhs")


def _parse_forbidden_symbols(value: Any) -> set:
    if value is None:
        return set()
    if isinstance(value, str):
        return {x.strip() for x in value.replace(";", ",").split(",") if x.strip()}
    if isinstance(value, (list, tuple, set)):
        return {str(x).strip() for x in value if str(x).strip()}
    return set()


def _zero_order_field_count(seq: List[int]) -> float:
    """Count native field terminals not directly used as derivative operands.

    The guard is generic and opt-in.  It is useful for diffusion-only field-PDE
    validation where zero-order reaction-like state terms can mimic boundary or
    initial-condition correlations.
    """
    try:
        root = _Compiler(list(map(int, seq))).parse()
    except Exception:
        return 0.0
    native_fields = {"u", "v", "w"}
    derivative_parents = {"D", "lap", "adv"}
    count = 0.0

    def walk(node, parent_op: str = "") -> None:
        nonlocal count
        op = str(getattr(node, "op", ""))
        if op in native_fields and parent_op not in derivative_parents:
            count += 1.0
        for child in list(getattr(node, "children", []) or []):
            walk(child, op)

    walk(root)
    return float(count)


# ==============================================================================
# 3. AST-based evolutionary operators.
# ==============================================================================
def _expr_spans(seq: List[int]) -> List[Tuple[int, int]]:
    try: root = _Compiler(list(map(int, seq))).parse()
    except: return [(0, len(seq))]
    spans = []

    def walk(n, start_idx):
        end_idx = start_idx + 1
        if n.op == "D": end_idx += 1
        for c in n.children:
            end_idx = walk(c, end_idx)
        spans.append((start_idx, end_idx))
        return end_idx

    walk(root, 0)
    return spans

def _field_terminal_idxs(valid_terminals: List[int]) -> List[int]:
    return [int(i) for i in valid_terminals if IDX2SYM.get(int(i), "") in {"u", "v", "w"}]

def _coordinate_terminal_idxs(valid_terminals: List[int], valid_axes: List[int]) -> List[int]:
    """Return spatial coordinate leaves that are also legal derivative axes."""
    axes = {int(a) for a in valid_axes}
    return [
        int(i)
        for i in valid_terminals
        if int(i) in axes and IDX2SYM.get(int(i), "") in {"x", "y", "z"}
    ]

def _available_op_idx(sym: str) -> Optional[int]:
    idx = SYM2IDX.get(sym)
    return int(idx) if idx is not None else None

def _contains_const(seq: List[int]) -> bool:
    const_idx = _available_op_idx("const")
    return bool(const_idx is not None and int(const_idx) in [int(x) for x in seq])

def _make_sum_of_terms(terms: List[List[int]]) -> List[int]:
    """Build a left-associated prefix sum from already-valid expression terms.

    This helper does not encode any task-specific PDE.  It only represents the
    general additive form used by many discoverable equations, so stochastic
    search can assemble multi-term RHS expressions without waiting for a deep
    random tree to accidentally contain all terms.
    """
    terms = [list(map(int, t)) for t in terms if t]
    if not terms:
        const_idx = _available_op_idx("const")
        return [const_idx] if const_idx is not None else []
    if len(terms) == 1:
        return list(terms[0])
    add_idx = _available_op_idx("+")
    if add_idx is None:
        return list(terms[0])
    out = list(terms[0])
    for term in terms[1:]:
        out = [add_idx] + out + list(term)
    return out

def _generic_atom_term(
    rng: random.Random,
    valid_terminals: List[int],
    valid_axes: List[int],
    *,
    allow_derivative: bool = True,
    allow_products: bool = True,
) -> List[int]:
    """Sample one compact atom from the current grammar/vocabulary.

    The atom pool is generated from available fields and operators only.  It is
    not a fixed candidate library and it is not conditioned on the task name.
    Low-degree fields, lap(field), adv(field), and low-order products are useful
    because they are generic PDE building blocks and they keep constant fitting
    affine whenever wrapped by one scalar coefficient.
    """
    fields = _field_terminal_idxs(valid_terminals)
    if not fields:
        return [rng.choice(valid_terminals)]

    modes = ["field", "sq", "cube"]
    weights = [0.25, 0.12, 0.18]
    if _available_op_idx("lap") in _UNARY:
        modes.append("lap"); weights.append(0.20)
    if _available_op_idx("adv") in _UNARY:
        modes.append("adv"); weights.append(0.08)
    if allow_products and _available_op_idx("*") in _BINARY:
        modes.extend(["prod", "prod_sq"]); weights.extend([0.10, 0.05])
    if allow_derivative and _DERIV and valid_axes:
        # Keep explicit derivatives possible but rare; lap/adv macros cover most
        # low-order physics while explicit D remains needed for KdV-like terms.
        modes.append("deriv"); weights.append(0.02)
        coord_terms = _coordinate_terminal_idxs(valid_terminals, valid_axes)
        if coord_terms and _available_op_idx("/") in _BINARY:
            modes.append("coord_deriv_ratio"); weights.append(0.08)
        if coord_terms and allow_products and _available_op_idx("*") in _BINARY:
            modes.append("coord_deriv_product"); weights.append(0.04)

    mode = rng.choices(modes, weights=weights, k=1)[0]
    f = rng.choice(fields)

    if mode == "field":
        return [f]
    if mode == "sq" and _available_op_idx("sq") is not None:
        return [_available_op_idx("sq"), f]
    if mode == "cube" and _available_op_idx("cube") is not None:
        return [_available_op_idx("cube"), f]
    if mode == "lap" and _available_op_idx("lap") is not None:
        return [_available_op_idx("lap"), f]
    if mode == "adv" and _available_op_idx("adv") is not None:
        return [_available_op_idx("adv"), f]
    if mode == "prod" and _available_op_idx("*") is not None:
        return [_available_op_idx("*"), f, rng.choice(fields)]
    if mode == "prod_sq" and _available_op_idx("*") is not None and _available_op_idx("sq") is not None:
        return [_available_op_idx("*"), f, _available_op_idx("sq"), rng.choice(fields)]
    if mode == "deriv" and _DERIV and valid_axes:
        return [rng.choice(_DERIV), rng.choice(valid_axes), f]
    if mode == "coord_deriv_ratio" and _DERIV and valid_axes and _available_op_idx("/") is not None:
        coord_terms = _coordinate_terminal_idxs(valid_terminals, valid_axes)
        if coord_terms:
            coord = rng.choice(coord_terms)
            return [_available_op_idx("/"), _DERIV[0], coord, f, coord]
    if mode == "coord_deriv_product" and _DERIV and valid_axes and _available_op_idx("*") is not None:
        coord_terms = _coordinate_terminal_idxs(valid_terminals, valid_axes)
        if coord_terms:
            coord = rng.choice(coord_terms)
            return [_available_op_idx("*"), coord, _DERIV[0], coord, f]
    return [f]

def _generate_scalar_weighted_term(
    rng: random.Random,
    valid_terminals: List[int],
    valid_axes: List[int],
    *,
    p_standalone_const: float = 0.08,
) -> List[int]:
    """Generate c*atom or a standalone c with the current grammar.

    Standalone constants are required for source/bias terms but are sampled with
    modest probability so advection/diffusion benchmarks are not flooded by
    offsets.
    """
    const_idx = _available_op_idx("const")
    mul_idx = _available_op_idx("*")
    if const_idx is None:
        return _generic_atom_term(rng, valid_terminals, valid_axes)
    if rng.random() < float(p_standalone_const) or mul_idx is None:
        return [const_idx]
    atom = _generic_atom_term(rng, valid_terminals, valid_axes)
    return [mul_idx, const_idx] + atom

def _generate_affine_term_sum(
    rng: random.Random,
    valid_terminals: List[int],
    valid_axes: List[int],
    *,
    min_terms: int = 2,
    max_terms: int = 5,
) -> List[int]:
    """Sample a stochastic additive expression from grammar atoms.

    This is a grammar exploration operator, not a hand-coded candidate template:
    the selected fields, operators and term count are random and task-agnostic.
    """
    n_terms = rng.randint(int(min_terms), int(max_terms))
    terms: List[List[int]] = []
    seen = set()
    for _ in range(max(n_terms * 4, n_terms)):
        term = _generate_scalar_weighted_term(rng, valid_terminals, valid_axes)
        key = tuple(term)
        if key in seen:
            continue
        terms.append(term)
        seen.add(key)
        if len(terms) >= n_terms:
            break
    return _make_sum_of_terms(terms)

def _generate_random_tree(depth: int, max_depth: int, rng: random.Random, valid_terminals: List[int], valid_axes: List[int]) -> List[int]:
    if depth >= max_depth or not valid_terminals:
        return [rng.choice(valid_terminals)]

    choices = ["term", "unary", "binary", "deriv"]
    weights = [0.2 + 0.2*depth, 0.2, 0.3, 0.3] if depth < 2 else [0.6, 0.1, 0.15, 0.15]
    choice = rng.choices(choices, weights=weights, k=1)[0]

    if choice == "term":
        return [rng.choice(valid_terminals)]
    elif choice == "deriv" and _DERIV and valid_axes:
        return [rng.choice(_DERIV), rng.choice(valid_axes)] + _generate_random_tree(depth+1, max_depth, rng, valid_terminals, valid_axes)
    elif choice == "unary" and _UNARY:
        op = rng.choice(_UNARY)
        sym = IDX2SYM.get(int(op), "")

        # Wrap new physical macro operators in a linear constant so Ridge can rescale them.
        if sym in {"adv", "lap"}:
            fields = _field_terminal_idxs(valid_terminals)
            field_tok = rng.choice(fields) if fields else rng.choice(valid_terminals)
            mul_tok = SYM2IDX.get("*")
            const_tok = SYM2IDX.get("const")
            if mul_tok is not None and const_tok is not None:
                return [mul_tok, const_tok, op, field_tok]  # Generate [* const lap u].
            return [op, field_tok]

        return [op] + _generate_random_tree(depth+1, max_depth, rng, valid_terminals, valid_axes)

    elif choice == "binary" and _BINARY:
        left = _generate_random_tree(depth+1, max_depth, rng, valid_terminals, valid_axes)
        right = _generate_random_tree(depth+1, max_depth, rng, valid_terminals, valid_axes)
        return [rng.choice(_BINARY)] + left + right

    return [rng.choice(valid_terminals)]



def _strip_scalar_weight(term: List[int]) -> List[int]:
    """Return the structural atom inside a scalar-weighted term when possible."""
    mul_idx = SYM2IDX.get("*")
    const_idx = SYM2IDX.get("const")
    term = list(map(int, term))
    if len(term) >= 3 and mul_idx is not None and const_idx is not None and term[0] == mul_idx and term[1] == const_idx:
        return list(term[2:])
    return list(term)


def _wrap_scalar_weight(term: List[int]) -> List[int]:
    """Wrap a valid atom as c*atom unless it is already scalar-weighted or a const."""
    mul_idx = SYM2IDX.get("*")
    const_idx = SYM2IDX.get("const")
    term = list(map(int, term))
    if const_idx is None or not term:
        return term
    if term == [const_idx]:
        return term
    if len(term) >= 3 and mul_idx is not None and term[0] == mul_idx and term[1] == const_idx:
        return term
    if mul_idx is None:
        return term
    return [mul_idx, const_idx] + term


def _split_additive_terms(seq: List[int]) -> List[List[int]]:
    """Split a prefix expression into top-level additive terms.

    This is deliberately syntax-only and task-agnostic.  It enables term-level
    crossover/drop/replace moves, which preserve PDE building blocks better than
    arbitrary subtree swaps while keeping the same admissible grammar.
    """
    add_idx = SYM2IDX.get("+")
    seq = list(map(int, seq))
    if not seq or not is_valid_sequence(seq) or add_idx is None:
        return [seq]

    def parse_at(pos: int) -> Tuple[List[List[int]], int]:
        tok = seq[pos]
        if tok == add_idx:
            left_terms, p1 = parse_at(pos + 1)
            right_terms, p2 = parse_at(p1)
            return left_terms + right_terms, p2
        spans = _expr_spans(seq[pos:])
        if not spans:
            return [seq[pos:]], len(seq)
        end = pos + spans[0][1]
        return [seq[pos:end]], end

    try:
        terms, end = parse_at(0)
        if end == len(seq) and terms:
            return terms
    except Exception:
        pass
    return [seq]


def _term_level_crossover(a: List[int], b: List[int], rng: random.Random, max_terms: int = 7) -> List[int]:
    """Crossover by exchanging whole additive terms instead of arbitrary subtrees."""
    if not is_valid_sequence(a) or not is_valid_sequence(b):
        return list(a)
    ta = _split_additive_terms(a)
    tb = _split_additive_terms(b)
    if len(ta) <= 1 and len(tb) <= 1:
        return crossover(a, b, rng)
    pool: List[List[int]] = []
    seen = set()
    for term in ta + tb:
        key = tuple(term)
        if key not in seen:
            pool.append(term)
            seen.add(key)
    if not pool:
        return list(a)
    n_min = max(1, min(len(ta), len(tb), len(pool)))
    n_max = max(n_min, min(int(max_terms), len(pool), max(len(ta), len(tb), 2) + 1))
    n_terms = rng.randint(n_min, n_max)
    chosen = rng.sample(pool, k=min(n_terms, len(pool)))
    out = _make_sum_of_terms(chosen)
    return out if is_valid_sequence(out) else list(a)


def _drop_term_mutation(seq: List[int], rng: random.Random) -> List[int]:
    terms = _split_additive_terms(seq)
    if len(terms) <= 1:
        return list(seq)
    keep = [t for i, t in enumerate(terms) if i != rng.randrange(len(terms))]
    out = _make_sum_of_terms(keep)
    return out if is_valid_sequence(out) else list(seq)


def _replace_term_mutation(seq: List[int], rng: random.Random, valid_terminals: List[int], valid_axes: List[int]) -> List[int]:
    terms = _split_additive_terms(seq)
    if not terms:
        return list(seq)
    idx = rng.randrange(len(terms))
    terms[idx] = _generate_scalar_weighted_term(rng, valid_terminals, valid_axes)
    out = _make_sum_of_terms(terms)
    return out if is_valid_sequence(out) else list(seq)


def _append_specific_term(seq: List[int], term: List[int]) -> List[int]:
    add_idx = SYM2IDX.get("+")
    if add_idx is None or not is_valid_sequence(seq):
        return list(seq)
    term = _wrap_scalar_weight(term)
    out = [add_idx] + list(seq) + list(term)
    return out if is_valid_sequence(out) else list(seq)




def _primitive_completion_atoms(valid_terminals: List[int], valid_axes: List[int], *, include_derivatives: bool = False) -> List[List[int]]:
    """Generate a small task-agnostic primitive-neighborhood basis from the active vocab.

    This is not a library of candidate equations.  It only exposes the one-term
    neighbors already present in the grammar so evolution can test missing simple
    building blocks before resorting to high-degree residual patches.
    """
    fields = _field_terminal_idxs(valid_terminals)
    coord_terms = _coordinate_terminal_idxs(valid_terminals, valid_axes)
    atoms: List[List[int]] = []
    for f in fields:
        atoms.append([f])
        if _available_op_idx("sq") is not None:
            atoms.append([_available_op_idx("sq"), f])
        if _available_op_idx("cube") is not None:
            atoms.append([_available_op_idx("cube"), f])
        if _available_op_idx("lap") is not None:
            atoms.append([_available_op_idx("lap"), f])
        if _available_op_idx("adv") is not None:
            atoms.append([_available_op_idx("adv"), f])
        if include_derivatives and _DERIV and valid_axes:
            for ax in valid_axes:
                atoms.append([_DERIV[0], ax, f])
        if _DERIV and coord_terms:
            for ax in coord_terms:
                if _available_op_idx("/") is not None:
                    atoms.append([_available_op_idx("/"), _DERIV[0], ax, f, ax])
                if _available_op_idx("*") is not None:
                    atoms.append([_available_op_idx("*"), ax, _DERIV[0], ax, f])
    const_idx = _available_op_idx("const")
    if const_idx is not None:
        atoms.append([const_idx])
    out: List[List[int]] = []
    seen = set()
    for a in atoms:
        if not a or not is_valid_sequence(a):
            continue
        key = tuple(a)
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


def _build_residual_guided_terms(
    parent_rows: List[Dict[str, Any]],
    ctxs: List[DataContext],
    target_field: str,
    valid_terminals: List[int],
    valid_axes: List[int],
    rng: random.Random,
    *,
    basis_trials: int = 96,
    topk: int = 16,
) -> List[List[int]]:
    """Suggest generic grammar atoms correlated with the current residual.

    This is not reference-equation seeding or task-specific templating: the basis atoms
    are sampled from the current grammar, then ranked only by correlation with
    the residual of already-generated parent equations.  The returned atoms are
    used as optional AddTerm mutations and are still re-fitted/evaluated by the
    normal numerical pipeline.
    """
    if not parent_rows or not ctxs:
        return []
    try:
        cctx = ctxs[0]
        lhs = getattr(cctx, "weak_lhs_cache", {}).get(str(target_field))
        if lhs is None:
            return []
    except Exception:
        return []

    # Average residual from a few elite/diverse parents.
    residuals = []
    for row in parent_rows[: max(1, min(4, len(parent_rows)))]:
        try:
            seq = list(map(int, row.get("seq", [])))
            consts = np.asarray(row.get("consts", np.array([])), dtype=np.float64).reshape(-1)
            if not seq or not is_valid_sequence(seq) or consts.size != count_constants(seq):
                continue
            rhs = WeakFormEvaluator(cctx).evaluate_weak(seq, consts)
            if torch.is_tensor(rhs) and rhs.shape == lhs.shape and torch.isfinite(rhs).all():
                residuals.append((lhs - rhs).detach())
        except Exception:
            continue
    if not residuals:
        return []
    residual = torch.stack(residuals, dim=0).mean(dim=0)
    r = residual.reshape(-1).to(dtype=torch.float64)
    r = r - torch.mean(r)
    rnorm = torch.linalg.norm(r).item()
    if not np.isfinite(rnorm) or rnorm < 1e-14:
        return []

    # Sample a broad set of compact atoms from the grammar, but anchor the set
    # with current-vocabulary primitive one-term neighbors.  This is not a
    # candidate-equation library: atoms are generic local moves and must still
    # be selected by residual correlation and full refit.
    atoms: List[List[int]] = []
    seen = set()
    for atom0 in _primitive_completion_atoms(valid_terminals, valid_axes, include_derivatives=False):
        atom = _strip_scalar_weight(atom0)
        if not atom or not is_valid_sequence(atom):
            continue
        key = tuple(atom)
        if key not in seen:
            seen.add(key)
            atoms.append(atom)

    for _ in range(int(max(1, basis_trials))):
        atom = _strip_scalar_weight(_generate_scalar_weighted_term(rng, valid_terminals, valid_axes, p_standalone_const=0.10))
        if not atom or not is_valid_sequence(atom):
            continue
        ok_atom, _ = _passes_structural_guard(_wrap_scalar_weight(atom), struct_guard_enable=True)
        if not ok_atom:
            continue
        key = tuple(atom)
        if key in seen:
            continue
        seen.add(key)
        atoms.append(atom)

    scored: List[Tuple[float, List[int]]] = []
    ev = WeakFormEvaluator(cctx)
    for atom in atoms:
        try:
            term = _wrap_scalar_weight(atom)
            n_consts = count_constants(term)
            consts = np.ones(n_consts, dtype=np.float64)
            phi = ev.evaluate_weak(term, consts)
            if not torch.is_tensor(phi) or phi.shape != lhs.shape or not torch.isfinite(phi).all():
                continue
            p = phi.reshape(-1).to(dtype=torch.float64)
            p = p - torch.mean(p)
            pnorm = torch.linalg.norm(p).item()
            if not np.isfinite(pnorm) or pnorm < 1e-14:
                continue
            corr = abs(float(torch.dot(r, p).item()) / (rnorm * pnorm + 1e-30))
            # Mildly prefer compact atoms when correlation ties.
            score = corr - 0.002 * len(atom)
            scored.append((float(score), atom))
        except Exception:
            continue
    scored.sort(key=lambda x: x[0], reverse=True)
    return [atom for score, atom in scored[: int(max(1, topk))] if np.isfinite(score)]

def crossover(a: List[int], b: List[int], rng: random.Random) -> List[int]:
    if not is_valid_sequence(a) or not is_valid_sequence(b): return list(a)
    sa = _expr_spans(a)
    sb = _expr_spans(b)
    if not sa or not sb: return list(a)

    span_a = rng.choice(sa)
    span_b = rng.choice(sb)
    out = list(a[:span_a[0]]) + list(b[span_b[0]:span_b[1]]) + list(a[span_a[1]:])
    return out if is_valid_sequence(out) else list(a)

def mutate_sequence(seq: List[int], rng: random.Random, valid_terminals: List[int], valid_axes: List[int], max_depth: int = 4) -> List[int]:
    if not is_valid_sequence(seq): return seq
    spans = _expr_spans(seq)
    if not spans: return seq

    target_span = rng.choice(spans)
    new_subtree = _generate_random_tree(0, max_depth, rng, valid_terminals, valid_axes)
    out = list(seq[:target_span[0]]) + new_subtree + list(seq[target_span[1]:])
    return out if is_valid_sequence(out) else list(seq)

def additive_mutation(seq: List[int], rng: random.Random, valid_terminals: List[int], valid_axes: List[int], max_depth: int = 3) -> List[int]:
    """
    Apply a term-augmentation mutation tailored to PDE discovery. The repair avoids redundant * const * const structures by recognizing subtrees that already carry a leading constant."""
    if not is_valid_sequence(seq):
        return seq

    add_tok = SYM2IDX.get("+")
    mul_tok = SYM2IDX.get("*")
    const_tok = SYM2IDX.get("const")

    if add_tok is None or mul_tok is None or const_tok is None:
        return seq

    # Add a small scalar-weighted grammar atom most of the time; occasionally
    # fall back to a random subtree to preserve exploratory diversity.
    if rng.random() < 0.70:
        new_term = _generate_scalar_weighted_term(rng, valid_terminals, valid_axes)
    else:
        new_term = _generate_random_tree(0, max_depth, rng, valid_terminals, valid_axes)

    # Append directly when the new subtree already begins with * const, such as lap or adv.
    if len(new_term) >= 2 and new_term[0] == mul_tok and new_term[1] == const_tok:
        out = [add_tok] + list(seq) + new_term
    else:
        out = [add_tok] + list(seq) + [mul_tok, const_tok] + new_term

    return out if is_valid_sequence(out) else list(seq)

# ==============================================================================
# 4. Population and evolutionary-control core.
# ==============================================================================
class EvolutionaryPopulation:
    def __init__(self, pop_size=100, rng_seed=42):
        self.pop_size = pop_size
        self.population = []
        self.rng = random.Random(rng_seed)
        self.history_set = set()
        self.key_to_index = {}
        self._canon_seq_cache: Dict[Tuple[int, ...], List[int]] = {}
        self._canon_key_cache: Dict[Tuple[int, ...], str] = {}
        # Coverage-preserving archive.  This does not inject candidates or add
        # terms; it only keeps the best structures that the end-to-end search has
        # already generated, so final reranking can recover candidates that were
        # temporarily outcompeted by trajectory-specific surrogates.
        self.archive_by_key: Dict[str, Dict[str, Any]] = {}
        self.archive_by_signature: Dict[str, List[Dict[str, Any]]] = {}
        self.archive_max_size: int = 4096
        self.archive_keep_per_signature: int = 4
        self.population_keep_per_signature: int = 2

    def get_best(self, k=1):
        return self.population[:k]

    def get_stats(self):
        return {
            "diversity_score": len(set(str(p["seq"]) for p in self.population)) / max(1, len(self.population)),
            "pop_size": len(self.population)
        }

    def _struct_signature(self, seq: List[int]) -> str:
        """Coarse, task-agnostic structural signature for coverage.

        The signature intentionally records *families* of structures rather than
        exact equations.  It keeps advection, diffusion, pressure-gradient,
        reaction, raw-derivative and high-complexity families represented during
        survival/archive selection without giving any of them final-score credit.
        """
        syms = [IDX2SYM.get(int(t), "") for t in seq]
        syset = set(syms)
        fields = sorted([s for s in syset if s in {"u", "v", "w"}])
        has_adv = "adv" in syset
        has_lap = "lap" in syset
        has_const = "const" in syset
        has_poly = any(s in syset for s in {"sq", "cube", "^"})
        has_trans = any(s in syset for s in {"sin", "cos", "exp", "log"})
        has_div = "/" in syset
        d_count = syms.count("D")
        add_count = syms.count("+")
        mul_count = syms.count("*")
        # Approximate raw derivative/pressure-gradient presence from the prefix
        # token stream.  This is a coverage signature only; exact semantics are
        # still handled by the evaluator/normalizer.
        has_raw_deriv = d_count > 0
        has_pressure_grad = False
        for i, tok in enumerate(syms[:-2]):
            if tok == "D" and syms[i + 1] in {"x", "y", "z"} and syms[i + 2] == "w":
                has_pressure_grad = True
                break
        has_deriv_product = has_raw_deriv and mul_count > 0

        # Finer task-agnostic term sketch.  The earlier signature preserved broad
        # families such as "reaction+diffusion", but different additive term
        # combinations inside the same family could still evict each other.  This
        # sketch distinguishes which native fields appear inside key grammar atoms
        # without saying which target equation should prefer which field.
        def _next_field_after(op: str) -> str:
            found = []
            for i, tok in enumerate(syms[:-1]):
                if tok == op and syms[i + 1] in {"u", "v", "w"}:
                    found.append(syms[i + 1])
            return ''.join(sorted(set(found))) or '-'

        lap_fields = _next_field_after("lap")
        adv_fields = _next_field_after("adv")
        cube_fields = _next_field_after("cube")
        sq_fields = _next_field_after("sq")
        rawd_fields = []
        for i, tok in enumerate(syms[:-2]):
            if tok == "D" and syms[i + 1] in {"x", "y", "z"} and syms[i + 2] in {"u", "v", "w"}:
                rawd_fields.append(syms[i + 2])
        rawd_fields_s = ''.join(sorted(set(rawd_fields))) or '-'

        # Advection-like products are identified syntactically as field times a
        # raw derivative.  This is deliberately approximate and used only for
        # coverage, never for final scoring.
        has_adv_like_product = bool(has_deriv_product and any(f in syset for f in {"u", "v", "w"}))

        length_band = min(6, max(0, len(syms) // 8))
        term_band = min(6, add_count)
        deriv_band = min(4, d_count)
        parts = [
            f"F{''.join(fields) or '-'}",
            f"A{int(has_adv)}:{adv_fields}",
            f"L{int(has_lap)}:{lap_fields}",
            f"Pgrad{int(has_pressure_grad)}",
            f"RawD{int(has_raw_deriv)}:{rawd_fields_s}",
            f"AdvLike{int(has_adv_like_product)}",
            f"Dprod{int(has_deriv_product)}",
            f"Sq{sq_fields}",
            f"Cube{cube_fields}",
            f"Poly{int(has_poly)}",
            f"Trans{int(has_trans)}",
            f"Div{int(has_div)}",
            f"Const{int(has_const)}",
            f"T{term_band}",
            f"D{deriv_band}",
            f"Len{length_band}",
        ]
        return "|".join(parts)

    def _archive_update(self, row: Dict[str, Any]) -> None:
        """Update global archive with a generated/evaluated candidate."""
        try:
            seq = self._canonicalize_seq(row.get("seq", []))
            key = self._canonical_key(seq)
            row = dict(row)
            row["seq"] = list(seq)
            row["signature"] = self._struct_signature(seq)
            fit = float(row.get("fitness", 1e18))
        except Exception:
            return
        prev = self.archive_by_key.get(key)
        if prev is not None and fit >= float(prev.get("fitness", 1e18)):
            return
        self.archive_by_key[key] = row
        sig = row.get("signature", "")
        bucket = [r for r in self.archive_by_signature.get(sig, []) if self._canonical_key(r.get("seq", [])) != key]
        bucket.append(row)
        bucket.sort(key=lambda r: float(r.get("fitness", 1e18)))
        self.archive_by_signature[sig] = bucket[: max(1, int(self.archive_keep_per_signature))]
        # Soft cap by dropping the worst archived keys if needed.
        if len(self.archive_by_key) > int(self.archive_max_size):
            keep_keys = set()
            for rows in self.archive_by_signature.values():
                for r in rows:
                    keep_keys.add(self._canonical_key(r.get("seq", [])))
            if len(keep_keys) > self.archive_max_size:
                ranked = sorted(
                    (r for rows in self.archive_by_signature.values() for r in rows),
                    key=lambda r: float(r.get("fitness", 1e18)),
                )[: self.archive_max_size]
                keep_keys = {self._canonical_key(r.get("seq", [])) for r in ranked}
            self.archive_by_key = {k: v for k, v in self.archive_by_key.items() if k in keep_keys}

    def get_archive_candidates(self, topk: int = 256, per_signature: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return a diverse archive candidate list for final reranking."""
        per_sig = int(max(1, per_signature if per_signature is not None else self.archive_keep_per_signature))
        rows: List[Dict[str, Any]] = []
        seen = set()
        for sig, bucket in sorted(self.archive_by_signature.items(), key=lambda kv: kv[0]):
            for row in sorted(bucket, key=lambda r: float(r.get("fitness", 1e18)))[:per_sig]:
                try:
                    key = self._canonical_key(row.get("seq", []))
                except Exception:
                    continue
                if key in seen:
                    continue
                rows.append(dict(row))
                seen.add(key)
        # Fill with globally good archived candidates if the signature pass is
        # too small.
        for row in sorted(self.archive_by_key.values(), key=lambda r: float(r.get("fitness", 1e18))):
            try:
                key = self._canonical_key(row.get("seq", []))
            except Exception:
                continue
            if key in seen:
                continue
            rows.append(dict(row))
            seen.add(key)
            if len(rows) >= int(topk):
                break
        return rows[: int(max(1, topk))]

    def _canonicalize_seq(self, seq: List[int]) -> List[int]:
        """Canonicalize candidate tokens before storage/cache lookup.

        This is called very frequently by initialization, offspring filtering and
        population deduplication.  A bounded per-population cache avoids repeated
        normalizer passes while preserving the same canonical form.
        """
        raw = tuple(int(x) for x in seq)
        cached = self._canon_seq_cache.get(raw)
        if cached is not None:
            return list(cached)
        try:
            from utils.structure_cache import get_structure_key_manager
            km = get_structure_key_manager()
            out = list(map(int, km.normalizer.normalize(list(raw))))
            out = out if is_valid_sequence(out) else list(raw)
        except Exception:
            out = list(raw)
        if len(self._canon_seq_cache) > 50000:
            self._canon_seq_cache.clear()
        self._canon_seq_cache[raw] = list(out)
        return list(out)

    def _canonical_key(self, seq: List[int]) -> str:
        """Return the canonical expression key used for dedup/replacement."""
        canon = tuple(self._canonicalize_seq(seq))
        cached = self._canon_key_cache.get(canon)
        if cached is not None:
            return cached
        try:
            from utils.structure_cache import get_structure_key_manager
            km = get_structure_key_manager()
            key = km.expr_key(list(canon)).key
        except Exception:
            key = "_".join(str(int(x)) for x in canon)
        if len(self._canon_key_cache) > 50000:
            self._canon_key_cache.clear()
        self._canon_key_cache[canon] = str(key)
        return str(key)

    def _deduplicate_population_inplace(self) -> None:
        best_by_key: Dict[str, Dict[str, Any]] = {}

        for ind in self.population:
            seq = self._canonicalize_seq(ind.get("seq", []))
            key = self._canonical_key(seq)
            row = dict(ind)
            row["seq"] = seq

            prev = best_by_key.get(key)
            if prev is None or float(row.get("fitness", 1e18)) < float(prev.get("fitness", 1e18)):
                best_by_key[key] = row

        rows = sorted(best_by_key.values(), key=lambda x: float(x.get("fitness", 1e18)))

        keep_per_signature = int(max(1, getattr(self, "population_keep_per_signature", 2)))
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            sig = self._struct_signature(row["seq"])
            buckets.setdefault(sig, []).append(row)

        kept = []
        seen_keys = set()

        # First pass: retain several candidates from each structural bucket.
        for sig, bucket_rows in buckets.items():
            for row in bucket_rows[:keep_per_signature]:
                key = self._canonical_key(row["seq"])
                if key not in seen_keys:
                    kept.append(row)
                    seen_keys.add(key)
                if len(kept) >= self.pop_size:
                    break
            if len(kept) >= self.pop_size:
                break

        # Second pass: fill remaining positions by fitness.
        for row in rows:
            key = self._canonical_key(row["seq"])
            if key in seen_keys:
                continue
            kept.append(row)
            seen_keys.add(key)
            if len(kept) >= self.pop_size:
                break

        self.population = sorted(kept, key=lambda x: float(x.get("fitness", 1e18)))
        self.key_to_index = {
            self._canonical_key(ind["seq"]): i
            for i, ind in enumerate(self.population)
        }
        self.history_set = set(self.key_to_index.keys())

    def add_evaluated_candidate(self, seq, consts, fit):
        if not fit.get("pred_valid", False):
            return

        seq_raw = list(map(int, seq))
        seq = self._canonicalize_seq(seq_raw)
        consts = np.asarray(consts, dtype=np.float64).reshape(-1)

        # If canonicalization changes constant arity, the old const vector cannot
        # be trusted.  Keep the original semantic sequence in that rare case.
        if int(count_constants(seq)) != int(len(consts)):
            seq = seq_raw
        if int(count_constants(seq)) != int(len(consts)):
            return

        key = self._canonical_key(seq)
        row = {
            "seq": list(seq),
            "consts": consts,
            "fitness": float(fit.get("fitness", 1e10)),
            "mse": float(fit.get("residual_mse", 1e10)),
            "complexity": float(fit.get("complexity", 1.0)),
        }
        try:
            row["signature"] = self._struct_signature(row["seq"])
            self._archive_update(row)
        except Exception:
            pass

        # Do not permanently discard a canonical structure: if a better version
        # appears later, replace the existing population entry.
        if key in self.key_to_index:
            idx = int(self.key_to_index[key])
            if idx < len(self.population) and row["fitness"] < float(self.population[idx].get("fitness", 1e18)):
                self.population[idx] = row
            else:
                return
        else:
            self.population.append(row)

        self._deduplicate_population_inplace()

    def _tournament_select(self, k=3):
        if not self.population: return None
        candidates = self.rng.sample(self.population, min(k, len(self.population)))
        return min(candidates, key=lambda x: x["fitness"])

    def initialize_unbiased(
        self,
        ctx,
        lambda_c,
        target_field,
        n_epochs,
        verbose,
        lambda_dx_chain,
        aggregate_mode,
        init_random_trials,
        init_max_ops,
        init_p_const,
        policy: Optional[OperatorPolicy] = None,
        spatial_ndim: int = 2,
        generic_affine_bootstrap_frac: float = 0.0,
        optimize_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Use the trainer's current operator policy during population init.
        if policy is not None:
            refresh_operator_groups(policy=policy)
        else:
            refresh_operator_groups(epoch=0, spatial_ndim=int(max(1, spatial_ndim)))
        ctxs = ctx if isinstance(ctx, list) else [ctx]
        opt_kwargs = dict(optimize_kwargs or {})
        valid_terminals = get_valid_terminal_idxs(ctxs[0])
        valid_axes = get_valid_axis_idxs_for_D(ctxs[0], forbid_t_on_rhs=True)

        if verbose:
            print(f"  [Evo] Initializing population with {init_random_trials} random trees...")

        evaluated = 0
        from utils.structure_cache import get_structure_key_manager
        km = get_structure_key_manager()

        # Stochastic additive-affine warm start.  This is intentionally generic:
        # it samples from the active grammar and never uses equation/task names or
        # fixed templates.  It improves coverage of multi-term RHS families while
        # leaving unrestricted random-tree initialization intact.
        bootstrap_frac = float(max(0.0, min(1.0, generic_affine_bootstrap_frac)))
        bootstrap_trials = int(max(0, min(int(init_random_trials * bootstrap_frac), self.pop_size)))
        for _boot in range(bootstrap_trials):
            if len(self.population) >= max(1, int(self.pop_size * 0.40)):
                break
            seq = _generate_affine_term_sum(
                self.rng,
                valid_terminals,
                valid_axes,
                min_terms=2,
                max_terms=5,
            )
            try:
                seq = _cached_algebraic_simplify(km, seq)
            except Exception:
                pass
            seq = self._canonicalize_seq(seq)
            if not is_valid_sequence(seq) or self._canonical_key(seq) in self.history_set:
                continue
            consts, fit = optimize_constants(seq, ctx, lambda_c=lambda_c, target_field=target_field, lambda_dx_chain=lambda_dx_chain, **opt_kwargs)
            self.add_evaluated_candidate(seq, consts, fit)
            evaluated += 1

        for _ in range(init_random_trials):
            if len(self.population) >= self.pop_size: break

            seq = _generate_random_tree(0, init_max_ops, self.rng, valid_terminals, valid_axes)
            if self.rng.random() < init_p_const and SYM2IDX.get("const") is not None:
                seq.append(SYM2IDX["const"])

            try:
                seq = _cached_algebraic_simplify(km, seq)
            except Exception:
                pass

            seq = self._canonicalize_seq(seq)
            if not is_valid_sequence(seq) or self._canonical_key(seq) in self.history_set:
                continue

            consts, fit = optimize_constants(seq, ctx, lambda_c=lambda_c, target_field=target_field, lambda_dx_chain=lambda_dx_chain, **opt_kwargs)
            self.add_evaluated_candidate(seq, consts, fit)
            evaluated += 1

            const_prune_tol = float(opt_kwargs.get("const_prune_tol", opt_kwargs.get("const_scaled_threshold", 1e-4)) or 0.0)
            if fit.get("pred_valid", False) and len(consts) > 0 and const_prune_tol > 0.0 and np.any(np.abs(consts) < const_prune_tol):
                try:
                    pruned_seq = _cached_algebraic_simplify(km, seq, consts=consts, prune_tol=const_prune_tol)
                    pruned_seq = self._canonicalize_seq(pruned_seq)
                    if self._canonical_key(pruned_seq) not in self.history_set and is_valid_sequence(pruned_seq):
                        p_consts, p_fit = optimize_constants(pruned_seq, ctx, lambda_c=lambda_c, target_field=target_field, lambda_dx_chain=lambda_dx_chain, **opt_kwargs)
                        self.add_evaluated_candidate(pruned_seq, p_consts, p_fit)
                except Exception:
                    pass

        if verbose:
            print(f"  [Evo] Initialization done. Pop size: {len(self.population)}")

    def _novelty_score(
        self,
        seq: List[int],
        local_keys: Optional[set] = None,
        local_signatures: Optional[Dict[str, int]] = None,
    ) -> float:
        """Score structural novelty before expensive fitness evaluation.

        This is intentionally syntax-level only: it does not privilege any PDE
        template or hand-written term library.  It rewards candidates whose
        canonical key has not been evaluated and whose coarse structural
        signature is rare in the current population / current offspring batch.
        """
        if not is_valid_sequence(seq):
            return -1e18

        try:
            seq = self._canonicalize_seq(seq)
            key = self._canonical_key(seq)
            sig = self._struct_signature(seq)
        except Exception:
            return -1e18

        local_keys = local_keys or set()
        local_signatures = local_signatures or {}

        if key in local_keys:
            return -1e9

        pop_sig_count = 0
        for ind in self.population:
            try:
                if self._struct_signature(ind.get("seq", [])) == sig:
                    pop_sig_count += 1
            except Exception:
                continue

        syms = [IDX2SYM.get(int(t), "") for t in seq]
        nonterminal_kinds = {
            s for s in syms
            if s not in {"const", "u", "v", "w", "x", "y", "z", "t"} and not str(s).startswith("<")
        }

        score = 0.0
        # Unseen canonical structures are useful, but do not hard-exclude seen
        # keys here; the final evaluation loop still filters them.
        score += 3.0 if key not in self.history_set else -3.0
        # Prefer rare broad structural families.
        score += 2.0 / (1.0 + float(pop_sig_count))
        score += 1.0 / (1.0 + float(local_signatures.get(sig, 0)))
        # Mild reward for operator variety; bounded so length alone cannot win.
        score += 0.20 * min(8, len(nonterminal_kinds))
        # Avoid extremely large random trees dominating novelty.
        score -= 0.015 * max(0, len(seq) - 24)
        return float(score)

    def _make_offspring_neighborhood(
        self,
        seq1: List[int],
        seq2: List[int],
        valid_terminals: List[int],
        valid_axes: List[int],
        *,
        n_trials: int = 4,
        keep: int = 1,
        p_cross: float = 0.40,
        p_add: float = 0.30,
        p_mut: float = 0.30,
        p_term_cross: float = 0.20,
        p_drop: float = 0.08,
        p_replace: float = 0.12,
        p_guided: float = 0.00,
        guided_terms: Optional[List[List[int]]] = None,
        p_restart: float = 0.00,
        p_big: float = 0.00,
        add_depth: int = 3,
        mut_depth: int = 8,
    ) -> List[List[int]]:
        """Generate several local variants and keep only the most novel ones.

        The generated variants come from the same grammar-level operators used
        elsewhere: crossover, additive mutation, subtree replacement, and
        optional grammar restart.  No physical basis terms are injected.
        """
        n_trials = int(max(1, n_trials))
        keep = int(max(1, keep))
        candidates: List[List[int]] = []
        local_keys: set = set()
        local_signatures: Dict[str, int] = {}

        guided_terms = list(guided_terms or [])
        # Normalize probability mass defensively.  In addition to the original
        # subtree crossover / additive / mutation moves, include term-level
        # crossover and term drop/replace moves.  These are grammar-preserving
        # operators that better preserve additive PDE building blocks.
        probs = np.asarray([p_restart, p_big, p_cross, p_add, p_mut, p_term_cross, p_drop, p_replace, p_guided], dtype=np.float64)
        probs = np.maximum(probs, 0.0)
        if not guided_terms:
            probs[-1] = 0.0
        if float(probs.sum()) <= 0.0:
            probs = np.asarray([0.0, 0.0, 0.30, 0.25, 0.25, 0.15, 0.03, 0.02, 0.0], dtype=np.float64)
        probs = probs / float(probs.sum())
        cuts = np.cumsum(probs)

        for _ in range(n_trials):
            r = self.rng.random()
            try:
                if r < cuts[0]:
                    # Full grammar restart: jump to a distant structure.
                    child = _generate_random_tree(0, mut_depth, self.rng, valid_terminals, valid_axes)
                elif r < cuts[1]:
                    # Large grammar mutation: intentionally make a non-local move
                    # without injecting any hand-written physical term.
                    mode = self.rng.random()
                    if mode < 0.34:
                        child = _generate_random_tree(0, max(mut_depth + 1, add_depth + 3), self.rng, valid_terminals, valid_axes)
                    elif mode < 0.67:
                        child = additive_mutation(
                            seq1, self.rng, valid_terminals, valid_axes,
                            max_depth=max(add_depth + 2, 5),
                        )
                    else:
                        base = crossover(seq1, seq2, self.rng)
                        child = mutate_sequence(
                            base, self.rng, valid_terminals, valid_axes,
                            max_depth=max(mut_depth + 2, 10),
                        )
                elif r < cuts[2]:
                    child = crossover(seq1, seq2, self.rng)
                elif r < cuts[3]:
                    child = additive_mutation(seq1, self.rng, valid_terminals, valid_axes, max_depth=add_depth)
                elif r < cuts[4]:
                    child = mutate_sequence(seq1, self.rng, valid_terminals, valid_axes, max_depth=mut_depth)
                elif r < cuts[5]:
                    child = _term_level_crossover(seq1, seq2, self.rng)
                elif r < cuts[6]:
                    child = _drop_term_mutation(seq1, self.rng)
                elif r < cuts[7]:
                    child = _replace_term_mutation(seq1, self.rng, valid_terminals, valid_axes)
                else:
                    # Residual-guided AddTerm: still generic and still evaluated
                    # by the same numerical objective.  The residual only biases
                    # which grammar atom to try next.
                    term = self.rng.choice(guided_terms) if guided_terms else _generate_scalar_weighted_term(self.rng, valid_terminals, valid_axes)
                    child = _append_specific_term(seq1, term)
            except Exception:
                continue

            if not is_valid_sequence(child):
                continue

            try:
                child = self._canonicalize_seq(child)
                key = self._canonical_key(child)
                sig = self._struct_signature(child)
            except Exception:
                continue

            if key in local_keys:
                continue

            candidates.append(child)
            local_keys.add(key)
            local_signatures[sig] = int(local_signatures.get(sig, 0)) + 1

        if not candidates:
            return []

        ranked = sorted(
            candidates,
            key=lambda seq: self._novelty_score(seq, local_keys=set(), local_signatures=local_signatures),
            reverse=True,
        )
        return ranked[:keep]

    def _stagnation_exploration_schedule(
        self,
        stagnated: int,
        *,
        neighborhood_trials: int = 4,
        big_mutation_patience: int = 4,
        restart_patience: int = 8,
        max_restart_prob: float = 0.25,
        max_big_mutation_prob: float = 0.35,
        max_add_depth: int = 6,
        max_mut_depth: int = 12,
    ) -> Dict[str, Any]:
        """Return a stagnation-aware exploration schedule.

        This replaces fixed mutation probabilities with a schedule controlled by
        search stagnation.  The admissible expression grammar is unchanged; only
        the probability of non-local grammar moves grows when best fitness stops
        improving.
        """
        stagnated = int(max(0, stagnated))
        big_mutation_patience = int(max(1, big_mutation_patience))
        restart_patience = int(max(big_mutation_patience, restart_patience))
        max_restart_prob = float(max(0.0, min(0.90, max_restart_prob)))
        max_big_mutation_prob = float(max(0.0, min(0.90, max_big_mutation_prob)))
        max_add_depth = int(max(3, max_add_depth))
        max_mut_depth = int(max(4, max_mut_depth))

        if stagnated < big_mutation_patience:
            return {
                "p_restart": 0.00,
                "p_big": 0.00,
                "p_cross": 0.40,
                "p_add": 0.30,
                "p_mut": 0.30,
                "add_depth": 3,
                "mut_depth": 8,
                "neighborhood_trials": int(max(1, neighborhood_trials)),
            }

        # Smoothly grow exploration strength from the first stagnation threshold
        # to the hard restart threshold.
        span = float(max(1, restart_patience - big_mutation_patience))
        progress = min(1.0, max(0.0, (stagnated - big_mutation_patience + 1) / span))

        p_restart = max_restart_prob * max(0.0, (stagnated - restart_patience + 1) / max(1.0, restart_patience))
        p_restart = min(max_restart_prob, p_restart)
        p_big = min(max_big_mutation_prob, 0.08 + progress * (max_big_mutation_prob - 0.08))

        # As stagnation grows, reduce local crossover pressure and move mass to
        # additive / subtree / large mutations.
        p_cross = max(0.10, 0.40 - 0.24 * progress)
        p_add = min(0.42, 0.30 + 0.08 * progress)
        p_mut = min(0.40, 0.30 + 0.06 * progress)

        add_depth = min(max_add_depth, 3 + int(round(3 * progress)))
        mut_depth = min(max_mut_depth, 8 + int(round(4 * progress)))
        trials = int(max(neighborhood_trials, 4 + round(4 * progress)))
        if stagnated >= restart_patience:
            trials = max(trials, 8)

        return {
            "p_restart": float(p_restart),
            "p_big": float(p_big),
            "p_cross": float(p_cross),
            "p_add": float(p_add),
            "p_mut": float(p_mut),
            "add_depth": int(add_depth),
            "mut_depth": int(mut_depth),
            "neighborhood_trials": int(trials),
        }

    def evolve_with_diversity_control(
        self,
        ctx,
        lambda_c: float = 1e-4,
        target_field: str = "du_t",
        epoch: int = 0,
        n_offspring: int = 36,
        lambda_dx_chain: float = 0.0,
        aggregate_mode: str = "mean",
        valid_terminals: Optional[List[int]] = None,
        policy: Optional[OperatorPolicy] = None,
        spatial_ndim: int = 2,
        tournament_k: int = 2,
        random_parent_prob: float = 0.25,
        stagnated: int = 0,
        neighborhood_trials: int = 4,
        neighborhood_keep: int = 1,
        big_mutation_patience: int = 4,
        restart_patience: int = 8,
        max_restart_prob: float = 0.25,
        max_big_mutation_prob: float = 0.35,
        max_add_depth: int = 6,
        max_mut_depth: int = 12,
        term_crossover_prob: float = 0.20,
        drop_term_prob: float = 0.06,
        replace_term_prob: float = 0.10,
        residual_guided_prob: float = 0.12,
        residual_guided_basis_trials: int = 96,
        residual_guided_topk: int = 16,
        random_immigrant_frac: float = 0.10,
        primitive_completion_frac: float = 0.15,
        random_tree_immigrant_frac: float = 0.25,
        optimize_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Evolve population with neighborhood multi-try and novelty filtering.

        `n_offspring` is now interpreted as the number of parent-selection
        events.  Each event samples several grammar-level neighbors, keeps only
        the most novel `neighborhood_keep` children, and evaluates those children.
        """
        if len(self.population) < 2:
            return 0

        if policy is not None:
            refresh_operator_groups(policy=policy)
        else:
            refresh_operator_groups(epoch, spatial_ndim=int(max(1, spatial_ndim)))

        ctxs = ctx if isinstance(ctx, list) else [ctx]
        opt_kwargs = dict(optimize_kwargs or {})
        if not ctxs:
            return 0

        if valid_terminals is None:
            valid_terminals = get_valid_terminal_idxs(ctxs[0])
        valid_terminals = [int(i) for i in valid_terminals]
        valid_axes = get_valid_axis_idxs_for_D(ctxs[0], forbid_t_on_rhs=True)

        stagnated = int(max(0, stagnated))
        tournament_k = int(max(1, tournament_k))
        random_parent_prob = float(max(0.0, min(1.0, random_parent_prob)))
        neighborhood_trials = int(max(1, neighborhood_trials))
        neighborhood_keep = int(max(1, neighborhood_keep))

        # Stagnation-triggered large mutation schedule.  Probabilities are
        # no longer fixed constants: they grow with the number of epochs since
        # the last meaningful best-fitness improvement.
        schedule = self._stagnation_exploration_schedule(
            stagnated,
            neighborhood_trials=neighborhood_trials,
            big_mutation_patience=int(big_mutation_patience),
            restart_patience=int(restart_patience),
            max_restart_prob=float(max_restart_prob),
            max_big_mutation_prob=float(max_big_mutation_prob),
            max_add_depth=int(max_add_depth),
            max_mut_depth=int(max_mut_depth),
        )
        p_restart = float(schedule["p_restart"])
        p_big = float(schedule["p_big"])
        p_cross = float(schedule["p_cross"])
        p_add = float(schedule["p_add"])
        p_mut = float(schedule["p_mut"])
        add_depth = int(schedule["add_depth"])
        mut_depth = int(schedule["mut_depth"])
        neighborhood_trials = int(schedule["neighborhood_trials"])
        p_term_cross = float(max(0.0, term_crossover_prob))
        p_drop = float(max(0.0, drop_term_prob))
        p_replace = float(max(0.0, replace_term_prob))
        p_guided = float(max(0.0, residual_guided_prob))
        random_immigrant_frac = float(max(0.0, min(0.75, random_immigrant_frac)))
        primitive_completion_frac = float(max(0.0, min(0.75, primitive_completion_frac)))
        random_tree_immigrant_frac = float(max(0.0, min(1.0, random_tree_immigrant_frac)))

        # Residual-guided atom suggestions are computed from existing parents and
        # the current residual; no ground-truth structures or task names are used.
        guided_terms: List[List[int]] = []
        if p_guided > 0.0:
            try:
                parents_for_guidance = sorted(self.population, key=lambda r: float(r.get("fitness", 1e18)))[: max(4, tournament_k * 2)]
                guided_terms = _build_residual_guided_terms(
                    parents_for_guidance,
                    ctxs,
                    target_field,
                    valid_terminals,
                    valid_axes,
                    self.rng,
                    basis_trials=int(max(8, residual_guided_basis_trials)),
                    topk=int(max(1, residual_guided_topk)),
                )
            except Exception:
                guided_terms = []

        offspring_seqs: List[List[int]] = []
        seen_offspring_keys: set = set()

        n_events = int(max(1, n_offspring))
        n_immigrants = int(round(n_events * random_immigrant_frac))
        n_completion = int(round(n_events * primitive_completion_frac))

        # Immigrants are mostly grammar-uniform additive expressions rather than
        # deep unconstrained trees.  This improves structural coverage without
        # flooding the population with high-power derivative monsters.  A small
        # fraction remains fully random to preserve open-ended discovery.
        for _ in range(n_immigrants):
            try:
                if self.rng.random() < random_tree_immigrant_frac:
                    child = _generate_random_tree(0, max(3, min(mut_depth, 6)), self.rng, valid_terminals, valid_axes)
                else:
                    child = _generate_affine_term_sum(self.rng, valid_terminals, valid_axes, min_terms=2, max_terms=5)
                child = self._canonicalize_seq(child)
                ok_child, _ = _passes_structural_guard(child, **opt_kwargs)
                key = self._canonical_key(child)
                if ok_child and key not in seen_offspring_keys and key not in self.history_set and is_valid_sequence(child):
                    offspring_seqs.append(child)
                    seen_offspring_keys.add(key)
            except Exception:
                continue

        # Primitive completion neighborhood: for a few strong/diverse parents,
        # append simple one-term neighbors generated from the active grammar.
        # This is a generic local expansion mechanism, not a preset equation bank.
        if n_completion > 0:
            primitive_atoms = _primitive_completion_atoms(valid_terminals, valid_axes, include_derivatives=False)
            parent_pool = sorted(self.population, key=lambda r: (float(r.get("fitness", 1e18)), float(r.get("complexity", 1e18))))[: max(1, min(len(self.population), 8))]
            made = 0
            for parent in parent_pool:
                base_seq = list(parent.get("seq", []))
                atoms = list(primitive_atoms)
                self.rng.shuffle(atoms)
                for atom in atoms:
                    if made >= n_completion:
                        break
                    try:
                        child = _append_specific_term(base_seq, atom)
                        child = self._canonicalize_seq(child)
                        ok_child, _ = _passes_structural_guard(child, **opt_kwargs)
                        key = self._canonical_key(child)
                        if ok_child and key not in seen_offspring_keys and key not in self.history_set and is_valid_sequence(child):
                            offspring_seqs.append(child)
                            seen_offspring_keys.add(key)
                            made += 1
                    except Exception:
                        continue
                if made >= n_completion:
                    break

        for _ in range(max(1, n_events - n_immigrants)):
            p1 = self._tournament_select(k=tournament_k)
            if self.rng.random() < random_parent_prob:
                p2 = self.rng.choice(self.population)
            else:
                p2 = self._tournament_select(k=tournament_k)

            if not p1 or not p2:
                continue

            seq1, seq2 = list(p1.get("seq", [])), list(p2.get("seq", []))
            children = self._make_offspring_neighborhood(
                seq1,
                seq2,
                valid_terminals,
                valid_axes,
                n_trials=neighborhood_trials,
                keep=neighborhood_keep,
                p_cross=p_cross,
                p_add=p_add,
                p_mut=p_mut,
                p_term_cross=p_term_cross,
                p_drop=p_drop,
                p_replace=p_replace,
                p_guided=p_guided,
                guided_terms=guided_terms,
                p_restart=p_restart,
                p_big=p_big,
                add_depth=add_depth,
                mut_depth=mut_depth,
            )

            for child in children:
                try:
                    key = self._canonical_key(child)
                except Exception:
                    continue
                if key in seen_offspring_keys or key in self.history_set:
                    continue
                ok_child, _ = _passes_structural_guard(child, **opt_kwargs)
                if not ok_child:
                    continue
                offspring_seqs.append(child)
                seen_offspring_keys.add(key)

        if not offspring_seqs:
            return 0

        from utils.structure_cache import get_structure_key_manager
        km = get_structure_key_manager()
        evaluated = 0

        # Evaluate the most novel children first.  This keeps the expensive
        # optimize_constants calls focused on fresh structural regions.
        offspring_seqs = sorted(
            offspring_seqs,
            key=lambda seq: self._novelty_score(seq),
            reverse=True,
        )

        for child_seq in offspring_seqs:
            try:
                simplified_seq = _cached_algebraic_simplify(km, child_seq)
            except Exception:
                simplified_seq = child_seq

            simplified_seq = self._canonicalize_seq(simplified_seq)
            if not is_valid_sequence(simplified_seq):
                continue

            key = self._canonical_key(simplified_seq)
            if key in self.history_set:
                continue

            try:
                consts, fit = optimize_constants(
                    simplified_seq,
                    ctx,
                    lambda_c=lambda_c,
                    target_field=target_field,
                    lambda_dx_chain=lambda_dx_chain,
                    aggregate_mode=aggregate_mode,
                    **opt_kwargs,
                )
            except TypeError:
                consts, fit = optimize_constants(
                    simplified_seq,
                    ctx,
                    target_field=target_field,
                    **opt_kwargs,
                )
            self.add_evaluated_candidate(simplified_seq, consts, fit)
            evaluated += 1

            # ====================================================================
            # Zero-triggered pruning.
            # STRidge has already set inactive coefficients to zero; remove only their associated trees.
            if fit.get("pred_valid", False) and len(consts) > 0:
                # Coefficient pruning is threshold-based rather than only
                # exact-zero-based, so small fitted terms are removed throughout
                # the search process.
                const_prune_tol = float(opt_kwargs.get("const_prune_tol", opt_kwargs.get("const_scaled_threshold", 1e-4)) or 0.0)
                if const_prune_tol > 0.0 and np.any(np.abs(consts) < const_prune_tol):
                    try:
                        pruned_seq = _cached_algebraic_simplify(
                            km,
                            simplified_seq,
                            consts=consts,
                            prune_tol=const_prune_tol
                        )
                        pruned_seq = self._canonicalize_seq(pruned_seq)

                        if is_valid_sequence(pruned_seq) and self._canonical_key(pruned_seq) not in self.history_set:
                            # Refit the retained structure with unbiased ordinary least squares.
                            p_consts, p_fit = optimize_constants(
                                pruned_seq, ctx, lambda_c=0.0, target_field=target_field,
                                lambda_dx_chain=lambda_dx_chain, aggregate_mode=aggregate_mode, **opt_kwargs
                            )
                            self.add_evaluated_candidate(pruned_seq, p_consts, p_fit)
                    except Exception:
                        pass
            # ===================================================================

        return evaluated
