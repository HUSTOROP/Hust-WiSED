from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

_EPS = 1.0e-12


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _finite_mask(*arrs: np.ndarray) -> np.ndarray:
    mask = np.ones_like(np.asarray(arrs[0], dtype=np.float64), dtype=bool)
    for arr in arrs:
        mask &= np.isfinite(np.asarray(arr, dtype=np.float64))
    return mask


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    m = _finite_mask(aa, bb)
    return float(np.sqrt(np.mean((aa[m] - bb[m]) ** 2))) if np.any(m) else float("nan")


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    m = _finite_mask(aa, bb)
    return float(np.mean(np.abs(aa[m] - bb[m]))) if np.any(m) else float("nan")


def _r2(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    m = _finite_mask(aa, bb)
    if not np.any(m):
        return float("nan")
    y = aa[m]
    pred = bb[m]
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return float(1.0 - ss_res / max(ss_tot, _EPS))


def _standardized_to_ohm(u: np.ndarray, v: np.ndarray, scales: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = float(scales.get("shared_scale_ohm", scales.get("complex_scale_ohm", 1.0)))
    r_mu = float(scales.get("R_mu_ohm", 0.0))
    mx_mu = float(scales.get("minusX_mu_ohm", 0.0))
    r = np.asarray(u, dtype=np.float64) * scale + r_mu
    minus_x = np.asarray(v, dtype=np.float64) * scale + mx_mu
    return r, minus_x, -minus_x


def _rhs_expression(equation: str) -> str:
    text = str(equation or "0")
    return text.split("=", 1)[1].strip() if "=" in text else text.strip()


def _prepare_expr(expr: str) -> str:
    out = str(expr or "0")
    out = re.sub(r"([uvw])_\{([xyz]+)\}", r"\1_\2", out)
    out = out.replace("^", "**")
    out = out.replace("\u2212", "-")
    for fn in ("sin", "cos", "exp", "log", "sqrt", "tanh"):
        out = re.sub(rf"(?<![A-Za-z0-9_.]){fn}\(", f"np.{fn}(", out)
    return out


def _derivatives_1d(field: np.ndarray, x_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    f = np.asarray(field, dtype=np.float64)
    x = np.asarray(x_coords, dtype=np.float64)
    if f.ndim == 1:
        fx = np.gradient(f, x, edge_order=2)
        fxx = np.gradient(fx, x, edge_order=2)
    else:
        fx = np.gradient(f, x, axis=-1, edge_order=2)
        fxx = np.gradient(fx, x, axis=-1, edge_order=2)
    return fx, fxx


def _eval_rhs(equation: str, u: np.ndarray, v: np.ndarray, x_coords: np.ndarray) -> np.ndarray:
    expr = _prepare_expr(_rhs_expression(equation))
    ux, uxx = _derivatives_1d(u, x_coords)
    vx, vxx = _derivatives_1d(v, x_coords)
    local = {
        "np": np,
        "u": np.asarray(u, dtype=np.float64),
        "v": np.asarray(v, dtype=np.float64),
        "u_x": ux,
        "u_xx": uxx,
        "v_x": vx,
        "v_xx": vxx,
        "x": np.asarray(x_coords, dtype=np.float64),
    }
    val = eval(expr, {"__builtins__": {}}, local)
    out = np.asarray(val, dtype=np.float64)
    if out.shape == ():
        out = np.zeros_like(np.asarray(u, dtype=np.float64)) + float(out)
    return out


def _split_metrics(values_true: np.ndarray, values_pred: np.ndarray, splits: Dict[str, List[int]]) -> Dict[str, Dict[str, Any]]:
    all_idx = list(range(int(values_true.shape[0])))
    out: Dict[str, Dict[str, Any]] = {}
    for name, raw_idx in {"train": splits.get("train", []), "val": splits.get("val", []), "test": splits.get("test", []), "all": all_idx}.items():
        idx = np.asarray(raw_idx, dtype=int)
        if idx.size == 0:
            out[name] = {"n_surfaces": 0, "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
            continue
        out[name] = {
            "n_surfaces": int(idx.size),
            "rmse": _rmse(values_true[idx], values_pred[idx]),
            "mae": _mae(values_true[idx], values_pred[idx]),
            "r2": _r2(values_true[idx], values_pred[idx]),
        }
    return out


def _local_rhs_and_one_step(
    clean: np.ndarray,
    t_coords: np.ndarray,
    x_coords: np.ndarray,
    eq_u: str,
    eq_v: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(clean, dtype=np.float64)
    u = arr[..., 0]
    v = arr[..., 1]
    du_numeric = np.gradient(u, t_coords, axis=1, edge_order=2)
    dv_numeric = np.gradient(v, t_coords, axis=1, edge_order=2)
    rhs_u = np.zeros_like(u)
    rhs_v = np.zeros_like(v)
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            rhs_u[i, j] = _eval_rhs(eq_u, u[i, j], v[i, j], x_coords)
            rhs_v[i, j] = _eval_rhs(eq_v, u[i, j], v[i, j], x_coords)

    u_step = np.full_like(u, np.nan)
    v_step = np.full_like(v, np.nan)
    for j in range(u.shape[1] - 1):
        dt = float(t_coords[j + 1] - t_coords[j])
        u_step[:, j + 1] = u[:, j] + dt * rhs_u[:, j]
        v_step[:, j + 1] = v[:, j] + dt * rhs_v[:, j]
    return du_numeric, dv_numeric, rhs_u, rhs_v, u_step, v_step


def _rollout(clean: np.ndarray, t_coords: np.ndarray, x_coords: np.ndarray, eq_u: str, eq_v: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(clean, dtype=np.float64)
    u_true = arr[..., 0]
    v_true = arr[..., 1]
    u_pred = np.full_like(u_true, np.nan)
    v_pred = np.full_like(v_true, np.nan)
    u_pred[:, 0] = u_true[:, 0]
    v_pred[:, 0] = v_true[:, 0]
    for i in range(u_true.shape[0]):
        for j in range(u_true.shape[1] - 1):
            dt = float(t_coords[j + 1] - t_coords[j])
            ru = _eval_rhs(eq_u, u_pred[i, j], v_pred[i, j], x_coords)
            rv = _eval_rhs(eq_v, u_pred[i, j], v_pred[i, j], x_coords)
            u_pred[i, j + 1] = u_pred[i, j] + dt * ru
            v_pred[i, j + 1] = v_pred[i, j] + dt * rv
    return u_pred, v_pred


def _nyquist_metrics(clean: np.ndarray, u_pred: np.ndarray, v_pred: np.ndarray, splits: Dict[str, List[int]], scales: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    u_true = np.asarray(clean[..., 0], dtype=np.float64)
    v_true = np.asarray(clean[..., 1], dtype=np.float64)
    r_true, minus_x_true, x_true = _standardized_to_ohm(u_true, v_true, scales)
    r_pred, minus_x_pred, x_pred = _standardized_to_ohm(u_pred, v_pred, scales)
    all_idx = list(range(int(u_true.shape[0])))
    out: Dict[str, Dict[str, Any]] = {}
    for name, raw_idx in {"train": splits.get("train", []), "val": splits.get("val", []), "test": splits.get("test", []), "all": all_idx}.items():
        idx = np.asarray(raw_idx, dtype=int)
        if idx.size == 0:
            out[name] = {"n_surfaces": 0}
            continue
        nyq_rmse = float(math.sqrt(max(_rmse(r_true[idx], r_pred[idx]) ** 2 + _rmse(x_true[idx], x_pred[idx]) ** 2, 0.0)))
        out[name] = {
            "n_surfaces": int(idx.size),
            "R_RMSE_ohm": _rmse(r_true[idx], r_pred[idx]),
            "minusX_RMSE_ohm": _rmse(minus_x_true[idx], minus_x_pred[idx]),
            "X_RMSE_ohm": _rmse(x_true[idx], x_pred[idx]),
            "Nyquist_RMSE_ohm": nyq_rmse,
            "standardized_u_RMSE": _rmse(u_true[idx], u_pred[idx]),
            "standardized_v_RMSE": _rmse(v_true[idx], v_pred[idx]),
            "standardized_complex_R2": _r2(np.stack([u_true[idx], v_true[idx]], axis=-1), np.stack([u_pred[idx], v_pred[idx]], axis=-1)),
        }
    return out


def _manifold_smoothness(clean: np.ndarray, t_coords: np.ndarray, splits: Dict[str, List[int]]) -> Dict[str, Any]:
    arr = np.asarray(clean, dtype=np.float64)
    curvature = np.gradient(np.gradient(arr, t_coords, axis=1, edge_order=2), t_coords, axis=1, edge_order=2)
    norm = np.sqrt(np.sum(curvature * curvature, axis=-1))
    out: Dict[str, Any] = {}
    all_idx = list(range(int(arr.shape[0])))
    for name, raw_idx in {"train": splits.get("train", []), "val": splits.get("val", []), "test": splits.get("test", []), "all": all_idx}.items():
        idx = np.asarray(raw_idx, dtype=int)
        out[name] = {
            "n_surfaces": int(idx.size),
            "mean_d2Z_dq2_norm": float(np.nanmean(norm[idx])) if idx.size else float("nan"),
            "p95_d2Z_dq2_norm": float(np.nanpercentile(norm[idx], 95)) if idx.size else float("nan"),
        }
    return out


def postprocess_soc_eis_impedance_manifold_results(
    *,
    task_name: str,
    dataset_path: str,
    summary_path: str,
    result_dir: str,
) -> str:
    dataset = np.load(dataset_path, allow_pickle=True)
    grid = dataset["grid_info"].item()
    clean = np.asarray(dataset["data_clean"], dtype=np.float64)
    summary = _load_json(Path(summary_path))
    t_coords = np.asarray(grid["t_coords"], dtype=np.float64)
    x_coords = np.asarray(grid["x_coords"], dtype=np.float64)
    splits = {str(k): [int(i) for i in v] for k, v in dict(grid.get("splits", {})).items()}
    scales = dict(grid.get("scales", {}))

    eq_u = str(summary.get("u", {}).get("discovered") or summary.get("u", {}).get("raw_discovered") or "u_t = 0")
    eq_v = str(summary.get("v", {}).get("discovered") or summary.get("v", {}).get("raw_discovered") or "v_t = 0")

    du_numeric, dv_numeric, rhs_u, rhs_v, u_step, v_step = _local_rhs_and_one_step(clean, t_coords, x_coords, eq_u, eq_v)
    u_roll, v_roll = _rollout(clean, t_coords, x_coords, eq_u, eq_v)

    local = {
        "u_q": _split_metrics(du_numeric, rhs_u, splits),
        "v_q": _split_metrics(dv_numeric, rhs_v, splits),
    }
    one_step = _nyquist_metrics(clean[:, 1:, :, :], u_step[:, 1:, :], v_step[:, 1:, :], splits, scales)
    rollout = _nyquist_metrics(clean, u_roll, v_roll, splits, scales)
    smoothness = _manifold_smoothness(clean, t_coords, splits)

    out = {
        "workflow": "SOC-EIS raw impedance manifold PDE discovery. SoC is q=1-SoC coordinate only; RHS uses u/v and frequency derivatives searched by WiSED.",
        "task_name": task_name,
        "dataset_path": str(dataset_path),
        "discovered_equations": {"u": eq_u, "v": eq_v},
        "input_contract": dict(grid.get("preprocess", {})),
        "field_mapping": dict(grid.get("paper_variable_mapping", {})),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "local_derivative_consistency": local,
        "one_step_impedance_prediction": one_step,
        "rollout_impedance_prediction": rollout,
        "manifold_smoothness": smoothness,
        "interpretation_guardrails": [
            "Good derivative fit means the equation describes local impedance evolution along discharge progress, not direct SoC regression.",
            "Good held-out rollout means the discovered law can generate a cross-battery impedance trajectory from an initial high-SoC spectrum.",
            "Poor rollout with good local residual indicates local closure is unstable or missing unobserved battery-specific state.",
        ],
    }

    result_dir_p = Path(result_dir)
    out_json = result_dir_p / f"{task_name}_postprocess_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    np.savez_compressed(
        result_dir_p / f"{task_name}_predictions.npz",
        rhs_u=rhs_u.astype(np.float32),
        rhs_v=rhs_v.astype(np.float32),
        du_numeric=du_numeric.astype(np.float32),
        dv_numeric=dv_numeric.astype(np.float32),
        u_one_step=u_step.astype(np.float32),
        v_one_step=v_step.astype(np.float32),
        u_rollout=u_roll.astype(np.float32),
        v_rollout=v_roll.astype(np.float32),
        t_coords=t_coords.astype(np.float32),
        x_coords=x_coords.astype(np.float32),
    )

    out_md = result_dir_p / f"{task_name}_postprocess_summary.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# SOC-EIS manifold PDE validation\n\n")
        f.write("## Discovered equations\n\n")
        f.write(f"- `u`: `{eq_u}`\n")
        f.write(f"- `v`: `{eq_v}`\n\n")
        f.write("## Held-out rollout metrics\n\n")
        for split, vals in rollout.items():
            f.write(f"### {split}\n")
            f.write(f"- Nyquist_RMSE_ohm: `{vals.get('Nyquist_RMSE_ohm')}`\n")
            f.write(f"- standardized_complex_R2: `{vals.get('standardized_complex_R2')}`\n")
            f.write(f"- n_surfaces: `{vals.get('n_surfaces')}`\n\n")
        f.write("## Local derivative consistency\n\n")
        for field, by_split in local.items():
            f.write(f"### {field}\n")
            for split, vals in by_split.items():
                f.write(f"- {split}: RMSE `{vals.get('rmse')}`, R2 `{vals.get('r2')}`\n")
            f.write("\n")

    return str(out_json)
