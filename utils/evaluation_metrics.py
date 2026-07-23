from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

import numpy as np

from models.symbol_vocabulary import IDX2SYM, SYM2IDX, COMPLEXITY_MAP, is_valid_sequence, sequence_to_str
from utils.structure_cache import get_structure_key_manager

# ============================================================
# Edit Distance
# ============================================================
def token_edit_distance(seq1, seq2) -> int:
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return int(dp[m][n])


# ============================================================
# Structural Similarity
# ============================================================
def structural_similarity(seq1: List[int], seq2: List[int]) -> float:
    """
    閫氳繃鍏ㄥ眬 StructureKeyManager 鑾峰彇瑙勮寖鍖栫殑瀛楃涓诧紝姣旇緝缁撴瀯鐩镐技搴︺€?    """
    if not seq1 and not seq2:
        return 1.0
    if not seq1 or not seq2:
        return 0.0

    try:
        km = get_structure_key_manager()
        norm1 = km.canonical_form(seq1)
        norm2 = km.canonical_form(seq2)
    except Exception:
        # Fallback 淇濇姢鏈哄埗
        norm1 = "_".join(str(int(x)) for x in seq1)
        norm2 = "_".join(str(int(x)) for x in seq2)

    return 1.0 if norm1 == norm2 else 0.0


# ============================================================
# Operator Bag
# ============================================================
def operator_bag(seq) -> Dict[str, int]:
    bag: Dict[str, int] = {}
    for tok in seq:
        sym = IDX2SYM.get(int(tok), "")
        if sym in ("u", "v", "w", "x", "y", "z", "t", "const"):
            continue
        if sym.startswith("<"):
            continue
        bag[sym] = bag.get(sym, 0) + 1
    return bag

def operator_recall(discovered, reference) -> float:
    ref_bag = operator_bag(reference)
    dis_bag = operator_bag(discovered)
    if not ref_bag:
        return 1.0
    total = sum(ref_bag.values())
    matched = sum(min(dis_bag.get(op, 0), cnt) for op, cnt in ref_bag.items())
    return float(matched / max(total, 1))


# ============================================================
# Reference Equations
# ============================================================
def _make_ref(sym_list: Iterable[str]) -> List[int]:
    """Convert a symbolic reference template to token ids.

    The active vocabulary is rebuilt per task. If a reference requires a token
    that is absent from the current vocabulary (for example, v in a single-field
    task), return [] so downstream metrics explicitly report N/A instead of
    raising during result summarization.
    """
    try:
        return [int(SYM2IDX[s]) for s in sym_list]
    except Exception:
        return []


# 1D Burgers: u_t = -u*u_x + c*u_xx
_BURGERS_1D_U = [
    "+",
    "neg", "*", "u", "D", "x", "u",
    "*", "const", "D", "x", "D", "x", "u",
]

# 1D KdV: u_t = -c*u*u_x - u_xxx
_KDV_1D_U = [
    "+",
    "neg", "*", "const", "*", "u", "D", "x", "u",
    "neg", "D", "x", "D", "x", "D", "x", "u",
]

# 2D Burgers, with the restricted macro semantics:
# adv(q) = u*q_x + v*q_y (+ w*q_z if present), q must be a native field.
# lap(q) = q_xx + q_yy (+ q_zz if present), q must be a native field.
_BURGERS_2D_U = [
    "+",
    "neg", "adv", "u",
    "*", "const", "lap", "u",
]
_BURGERS_2D_V = [
    "+",
    "neg", "adv", "v",
    "*", "const", "lap", "v",
]

# FitzHugh-Nagumo 2D generator used in data/fhn_2d.py:
# u_t = c1*lap(u) + c2*u + c3*u^3 + c4*v
# v_t = c1*lap(v) + c2*u + c3*v + c4
# Keep all reaction coefficients as const-bearing terms; otherwise the metric
# reference would incorrectly expect a fixed -1 coefficient for u^3 while the
# generator uses -1/3.
_FHN_2D_U = [
    "+",
        "*", "const", "lap", "u",
        "+",
            "*", "const", "u",
            "+",
                "*", "const", "cube", "u",
                "*", "const", "v",
]
_FHN_2D_V = [
    "+",
        "*", "const", "lap", "v",
        "+",
            "*", "const", "u",
            "+",
                "*", "const", "v",
                "const",
]



EQUATION_REF_SYMBOLS = {
    # Legacy names kept intact.
    "burgers": _BURGERS_1D_U,
    "kdv": _KDV_1D_U,

    # Runtime names emitted by run_common.finalize_and_plot():
    # equation_name = f"{task_prefix}_{field}".
    "burgers_1d": _BURGERS_1D_U,
    "burgers_1d_u": _BURGERS_1D_U,
    "kdv_1d": _KDV_1D_U,
    "kdv_1d_u": _KDV_1D_U,
    "burgers_2d_u": _BURGERS_2D_U,
    "burgers_2d_v": _BURGERS_2D_V,
    "fhn_2d_u": _FHN_2D_U,
    "fhn_2d_v": _FHN_2D_V,

    # Common external aliases.
    "fitzhugh_nagumo_2d_u": _FHN_2D_U,
    "fitzhugh_nagumo_2d_v": _FHN_2D_V,
    "fitzhugh_nagumo_u": _FHN_2D_U,
    "fitzhugh_nagumo_v": _FHN_2D_V,
}


EQUATION_REF_ALIASES = {
    "burgers_u": "burgers_1d_u",
    "kdv_u": "kdv_1d_u",
    "burgers2d_u": "burgers_2d_u",
    "burgers2d_v": "burgers_2d_v",
    "fhn_u": "fhn_2d_u",
    "fhn_v": "fhn_2d_v",
    "fhn2d_u": "fhn_2d_u",
    "fhn2d_v": "fhn_2d_v",
    "fitzhugh_nagumo2d_u": "fhn_2d_u",
    "fitzhugh_nagumo2d_v": "fhn_2d_v",
}


def _normalize_equation_type_name(equation_type: str) -> str:
    key = str(equation_type or "").strip().lower()
    key = key.replace("-", "_").replace(" ", "_")
    while "__" in key:
        key = key.replace("__", "_")
    return key.strip("_")


def resolve_reference_equation_type(equation_type: str) -> str:
    """Resolve user/runtime equation names to a supported reference key."""
    key = _normalize_equation_type_name(equation_type)
    if key in EQUATION_REF_SYMBOLS:
        return key
    if key in EQUATION_REF_ALIASES:
        return EQUATION_REF_ALIASES[key]

    # Flexible fallback: accept names with extra prefixes/suffixes as long as
    # the canonical task + field suffix is visible.
    suffix_rules = (
        ("burgers_2d", "u", "burgers_2d_u"),
        ("burgers_2d", "v", "burgers_2d_v"),
        ("fhn_2d", "u", "fhn_2d_u"),
        ("fhn_2d", "v", "fhn_2d_v"),
        ("fitzhugh_nagumo", "u", "fhn_2d_u"),
        ("fitzhugh_nagumo", "v", "fhn_2d_v"),
        ("burgers_1d", "u", "burgers_1d_u"),
        ("kdv_1d", "u", "kdv_1d_u"),
    )
    for marker, field, canonical in suffix_rules:
        if marker in key and key.endswith(f"_{field}"):
            return canonical
    return key


def _field_from_equation_type(equation_type: str) -> str:
    ref_type = resolve_reference_equation_type(equation_type)
    maybe_field = ref_type.rsplit("_", 1)[-1]
    return maybe_field if maybe_field in {"u", "v", "w"} else "u"


def get_reference_seq(equation_type: str) -> List[int]:
    ref_type = resolve_reference_equation_type(equation_type)
    syms = EQUATION_REF_SYMBOLS.get(ref_type, None)
    if not syms:
        return []
    return _make_ref(syms)


# ============================================================
# Numerical Metrics
# ============================================================
def nmse(pred, true) -> float:
    denom = np.var(true) + 1e-12
    return float(np.mean((pred - true) ** 2) / denom)


def relative_l2(pred, true) -> float:
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))


def coefficient_relative_error(estimated, true) -> float:
    if len(estimated) == 0 or len(true) == 0:
        return float("nan")
    norm_true = np.linalg.norm(true)
    if norm_true < 1e-12:
        return float("nan")
    min_len = min(len(estimated), len(true))
    return float(np.linalg.norm(np.asarray(estimated[:min_len]) - np.asarray(true[:min_len])) / norm_true)


# ============================================================
# Unified Evaluation
# ============================================================
def evaluate_discovered_equation(
    discovered_seq,
    discovered_consts,
    equation_type,
    residual_mse,
    fitness,
    true_consts=None,
):
    try:
        from utils.equation_formatter import compile_equation

        readable = compile_equation(
            discovered_seq,
            discovered_consts,
            lhs=f"{_field_from_equation_type(str(equation_type))}_t",
        )
    except Exception:
        readable = sequence_to_str(discovered_seq)

    ref_type = resolve_reference_equation_type(str(equation_type))
    ref_seq = get_reference_seq(str(equation_type))
    seq_syms = [IDX2SYM.get(int(t), "") for t in discovered_seq]

    metrics = {
        "equation_type": equation_type,
        "reference_equation_type": ref_type if ref_seq else "N/A",
        "discovered_readable": readable,
        "discovered_tokens": sequence_to_str(discovered_seq),
        "reference_tokens": sequence_to_str(ref_seq) if ref_seq else "N/A",
        "residual_mse": float(residual_mse),
        "fitness": float(fitness),
        "is_valid": bool(is_valid_sequence(discovered_seq)),
        "complexity": float(sum(float(COMPLEXITY_MAP.get(s, 1.0)) for s in seq_syms if s not in ["<START>", "<END>", "<PAD>", "<UNK>"])),
        "token_length": int(len(discovered_seq)),
        "exact_match": list(discovered_seq) == list(ref_seq) if ref_seq else False,
        "structural_sim": structural_similarity(discovered_seq, ref_seq) if ref_seq else 0.0,
        "operator_recall": operator_recall(discovered_seq, ref_seq) if ref_seq else 0.0,
        "edit_distance": token_edit_distance(discovered_seq, ref_seq) if ref_seq else -1,
    }

    if true_consts is not None:
        metrics["coeff_rel_error"] = coefficient_relative_error(discovered_consts, true_consts)
    else:
        metrics["coeff_rel_error"] = float("nan")

    return metrics

def print_metrics_table(metrics: Dict[str, object]) -> None:
    print("\n" + "=" * 60)
    print("  EQUATION DISCOVERY METRICS")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:.6f}")
        else:
            print(f"  {key:30s}: {value}")
    print("=" * 60 + "\n")

def save_metrics_txt(metrics: Dict[str, object], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("WiSED - Equation Evaluation Metrics\n")
        f.write("=" * 60 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key:30s}: {value:.8f}\n")
            else:
                f.write(f"{key:30s}: {value}\n")

__all__ = [
    "EQUATION_REF_SYMBOLS",
    "coefficient_relative_error",
    "evaluate_discovered_equation",
    "get_reference_seq",
    "resolve_reference_equation_type",
    "nmse",
    "operator_recall",
    "print_metrics_table",
    "relative_l2",
    "save_metrics_txt",
    "structural_similarity",
    "token_edit_distance",
]

