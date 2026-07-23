from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


TRUE_PDE_EQUATION = "u_t = 2.0*u_{x} / x + 1.0*u_{xx}"
ROLLOUT_HORIZON_MULTIPLE = 6.0
PHYSICAL_U_MIN = 0.0
PHYSICAL_U_MAX = 1.0
PHYSICAL_TOL = 1.0e-8


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    err = y_pred - y_true
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
        "n": int(y_true.size),
    }


def _load_equation(summary_path: Path) -> str:
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    entry = summary.get("u", summary)
    equation = entry.get("discovered") or entry.get("raw_discovered")
    if not equation:
        raise ValueError(f"No discovered equation found in {summary_path}")
    return str(equation)


def _safe_rhs_eval(equation: str, *, u: np.ndarray, ux: np.ndarray, uxx: np.ndarray, x: np.ndarray) -> np.ndarray:
    rhs_code = _compile_rhs(equation)
    return _eval_rhs(rhs_code, u=u, ux=ux, uxx=uxx, x=x)


def _compile_rhs(equation: str) -> Any:
    rhs = equation.split("=", 1)[1] if "=" in equation else equation
    expr = rhs
    expr = expr.replace("u_{xx}", "uxx")
    expr = expr.replace("u_{x}", "ux")
    expr = expr.replace("^", "**")
    expr = re.sub(r"\bu\b", "u", expr)
    expr = expr.strip()
    return compile(expr, "<wised_rhs>", "eval")


def _eval_rhs(rhs_code: Any, *, u: np.ndarray, ux: np.ndarray, uxx: np.ndarray, x: np.ndarray) -> np.ndarray:
    x_grid = np.maximum(x[None, None, :], 1.0e-8)
    namespace: dict[str, Any] = {
        "__builtins__": {},
        "np": np,
        "u": u,
        "ux": ux,
        "uxx": uxx,
        "x": x_grid,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "log": np.log,
    }
    return np.asarray(eval(rhs_code, namespace, {}), dtype=np.float64)


def _derivatives(field: np.ndarray, t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ut = np.gradient(field, t, axis=1, edge_order=2)
    ux = np.gradient(field, x, axis=2, edge_order=2)
    uxx = np.gradient(ux, x, axis=2, edge_order=2)
    return ut, ux, uxx


def _surface_fluxes(grid_info: dict[str, Any], n_surfaces: int) -> np.ndarray:
    fluxes = []
    for meta in grid_info.get("field_meta", []):
        phase = str(meta.get("phase", "charge")).lower()
        delta = float(meta.get("delta", 0.0))
        fluxes.append(delta if phase == "charge" else -delta)
    if not fluxes:
        fluxes = [0.0] * n_surfaces
    return np.asarray(fluxes[:n_surfaces], dtype=np.float64)


def _spatial_derivatives_with_flux(u: np.ndarray, x: np.ndarray, flux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = float(x[1] - x[0])
    ux = np.zeros_like(u)
    uxx = np.zeros_like(u)
    ux[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dx)
    uxx[:, 1:-1] = (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / (dx * dx)
    ux[:, 0] = 0.0
    uxx[:, 0] = 2.0 * (u[:, 1] - u[:, 0]) / (dx * dx)
    ghost = u[:, -2] + 2.0 * dx * flux
    ux[:, -1] = flux
    uxx[:, -1] = (ghost - 2.0 * u[:, -1] + u[:, -2]) / (dx * dx)
    return ux, uxx


def _rollout_equation(
    equation: str,
    *,
    initial: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    flux: np.ndarray,
    clip_state: bool = False,
) -> np.ndarray:
    dx = float(x[1] - x[0])
    max_step = 0.20 * dx * dx
    rhs_code = _compile_rhs(equation)
    pred = np.empty((initial.shape[0], len(t), len(x)), dtype=np.float64)
    state = np.asarray(initial, dtype=np.float64).copy()
    if clip_state:
        state = np.clip(state, PHYSICAL_U_MIN, PHYSICAL_U_MAX)
    pred[:, 0, :] = state
    for i in range(len(t) - 1):
        dt_total = float(t[i + 1] - t[i])
        n_sub = max(1, int(np.ceil(dt_total / max_step)))
        dt = dt_total / n_sub
        for _ in range(n_sub):
            ux, uxx = _spatial_derivatives_with_flux(state, x, flux)
            rhs = _eval_rhs(
                rhs_code,
                u=state[:, None, :],
                ux=ux[:, None, :],
                uxx=uxx[:, None, :],
                x=x,
            )[:, 0, :]
            state = state + dt * rhs
            if clip_state:
                state = np.clip(state, PHYSICAL_U_MIN, PHYSICAL_U_MAX)
            if not np.all(np.isfinite(state)):
                raise FloatingPointError(f"Non-finite field rollout for equation: {equation}")
        pred[:, i + 1, :] = state
    return pred


def _rollout_metrics(reference: np.ndarray, pred: np.ndarray, trim_x: int) -> dict[str, float]:
    sl = (slice(None), slice(1, None), slice(trim_x, -trim_x))
    return _metrics(reference[sl], pred[sl])


def _rmse_by_time(reference: np.ndarray, pred: np.ndarray, trim_x: int) -> np.ndarray:
    err = pred[:, :, trim_x:-trim_x] - reference[:, :, trim_x:-trim_x]
    return np.sqrt(np.mean(err * err, axis=(0, 2)))


def _representative_surface_index(field: np.ndarray) -> int:
    spatial_contrast = np.mean(np.std(field, axis=2), axis=1)
    return int(np.argmax(spatial_contrast))


def _future_time_grid(t: np.ndarray, horizon_multiple: float = ROLLOUT_HORIZON_MULTIPLE) -> np.ndarray:
    if len(t) < 2:
        raise ValueError("Need at least two time points for rollout.")
    t0 = float(t[0])
    t_end = float(t[-1])
    window = t_end - t0
    if window <= 0.0:
        raise ValueError("Time coordinates must be increasing.")
    n_intervals = int(round((float(horizon_multiple) - 1.0) * (len(t) - 1)))
    return np.linspace(t_end, float(horizon_multiple) * window + t0, n_intervals + 1)


def _even_snapshot_times(t: np.ndarray, n: int = 5) -> tuple[float, ...]:
    return tuple(float(v) for v in np.linspace(float(t[0]), float(t[-1]), int(n)))


def _physical_cutoff_snapshot_times(t: np.ndarray, *fields: np.ndarray, n: int = 4) -> tuple[float, ...]:
    cutoffs = [cutoff for field in fields if (cutoff := _first_cutoff_tau(t, field)) is not None]
    if not cutoffs:
        return _even_snapshot_times(t, n)
    cutoff_idx = int(np.searchsorted(t, min(cutoffs), side="left"))
    candidate_indices = [
        0,
        max(0, cutoff_idx // 3),
        max(0, (2 * cutoff_idx) // 3),
        min(max(0, cutoff_idx), len(t) - 1),
    ]
    indices: list[int] = []
    for idx in candidate_indices:
        if idx not in indices:
            indices.append(idx)
    idx = indices[-1] if indices else 0
    while len(indices) < int(n) and idx + 1 < len(t):
        idx += 1
        indices.append(idx)
    return tuple(float(t[idx]) for idx in indices[: int(n)])


def _save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def _apply_nature_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
        }
    )


def _save_rollout_rmse(path: Path, t: np.ndarray, curves: dict[str, np.ndarray], *, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    _apply_nature_style()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.7, 2.35), dpi=220)
    for label, values in curves.items():
        if label.lower().startswith("wised"):
            color = "#8a004f"
            linewidth = 1.55
            alpha = 1.0
            zorder = 3
        else:
            color = "0.32"
            linewidth = 1.05
            alpha = 0.72
            zorder = 2
        ax.plot(t, values, label=label, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("field RMSE")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    ax.grid(True, color="0.9", linewidth=0.45)
    fig.tight_layout()
    _save_figure(fig, path)
    plt.close(fig)


def _radial_image(profile: np.ndarray, x: np.ndarray, size: int = 151) -> np.ndarray:
    lin = np.linspace(-1.0, 1.0, int(size))
    xx, yy = np.meshgrid(lin, lin)
    rr = np.sqrt(xx * xx + yy * yy)
    image = np.interp(np.clip(rr, 0.0, 1.0), x, profile)
    image[rr > 1.0] = np.nan
    return image


def _valid_physical_time_mask(field: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    time_axis = values.ndim - 2
    flat = np.moveaxis(values, time_axis, 0).reshape(values.shape[time_axis], -1)
    return (np.nanmin(flat, axis=1) >= PHYSICAL_U_MIN - PHYSICAL_TOL) & (
        np.nanmax(flat, axis=1) <= PHYSICAL_U_MAX + PHYSICAL_TOL
    )


def _first_cutoff_tau(t: np.ndarray, field: np.ndarray) -> float | None:
    valid = _valid_physical_time_mask(field)
    invalid = np.flatnonzero(~valid)
    return float(t[int(invalid[0])]) if invalid.size else None


def _draw_cutoff_disk(ax: Any) -> None:
    disk = _radial_image(np.ones(2), np.asarray([0.0, 1.0]), size=151)
    ax.imshow(disk, cmap="Greys", vmin=0.0, vmax=1.0, alpha=0.22)
    ax.text(0.5, 0.5, "cutoff", transform=ax.transAxes, ha="center", va="center", color="0.25", fontsize=7)


def _save_radial_snapshots(
    path: Path,
    t: np.ndarray,
    x: np.ndarray,
    true_field: np.ndarray,
    pred_field: np.ndarray,
    *,
    title: str,
    display_mode: str = "concentration",
    snapshot_times: tuple[float, ...] | None = None,
    per_time_value_scale: bool = False,
) -> None:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except Exception:
        return
    _apply_nature_style()
    if snapshot_times is None:
        desired = [1.0, 2.0, 4.0, 6.0] if float(t[-1]) > 1.5 else [0.0, 0.2, 0.4, 0.6, 0.8]
    else:
        desired = list(snapshot_times)
    indices = [int(np.argmin(np.abs(t - value))) for value in desired]
    if display_mode == "deviation":
        true_display = true_field - np.mean(true_field, axis=1, keepdims=True)
        pred_display = pred_field - np.mean(pred_field, axis=1, keepdims=True)
        value_label = r"$u-\langle u\rangle_r$"
        value_cmap = "RdBu_r"
        value_abs = float(np.nanmax(np.abs(np.concatenate([true_display, pred_display], axis=0))))
        value_vmin, value_vmax = -value_abs, value_abs
    elif display_mode == "concentration":
        true_display = true_field
        pred_display = pred_field
        value_label = "concentration u"
        value_cmap = "viridis"
        if per_time_value_scale:
            value_vmin = value_vmax = float("nan")
        else:
            value_vmin = float(np.nanmin(np.concatenate([true_display, pred_display], axis=0)))
            value_vmax = float(np.nanmax(np.concatenate([true_display, pred_display], axis=0)))
    else:
        raise ValueError(f"Unknown radial snapshot display_mode: {display_mode}")

    err = pred_display - true_display
    fig_width = max(5.8, 1.25 * len(indices) + 1.0)
    fig, axes = plt.subplots(3, len(indices), figsize=(fig_width, 3.55), dpi=220, constrained_layout=True)
    err_abs = float(np.nanmax(np.abs(err)))
    value_images = []
    for col, idx in enumerate(indices):
        if per_time_value_scale:
            col_values = np.concatenate([true_display[idx], pred_display[idx]])
            col_vmin = float(np.nanmin(col_values))
            col_vmax = float(np.nanmax(col_values))
            if abs(col_vmax - col_vmin) < 1.0e-12:
                col_vmin -= 1.0e-6
                col_vmax += 1.0e-6
        else:
            col_vmin, col_vmax = value_vmin, value_vmax
        im = axes[0, col].imshow(_radial_image(true_display[idx], x), cmap=value_cmap, vmin=col_vmin, vmax=col_vmax)
        value_images.append(im)
        axes[0, col].set_title(rf"$\tau={t[idx]:.3f}$")
        axes[1, col].imshow(_radial_image(pred_display[idx], x), cmap=value_cmap, vmin=col_vmin, vmax=col_vmax)
        axes[2, col].imshow(_radial_image(err[idx], x), cmap="coolwarm", vmin=-err_abs, vmax=err_abs)
        for row in range(3):
            axes[row, col].set_axis_off()
    axes[0, 0].text(-0.10, 0.50, "Ref.", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="center")
    axes[1, 0].text(-0.10, 0.50, "WiSED", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="center")
    axes[2, 0].text(-0.10, 0.50, "Error", transform=axes[2, 0].transAxes, rotation=90, va="center", ha="center")
    if per_time_value_scale:
        for col, im in enumerate(value_images):
            fig.colorbar(
                im,
                ax=[axes[0, col], axes[1, col]],
                fraction=0.040,
                pad=0.010,
                label=value_label if col == len(value_images) - 1 else None,
            )
    else:
        value_sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(value_vmin, value_vmax), cmap=value_cmap)
        value_sm.set_array([])
        fig.colorbar(value_sm, ax=axes[:2, :].ravel().tolist(), fraction=0.025, pad=0.015, label=value_label)
    error_sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-err_abs, err_abs), cmap="coolwarm")
    error_sm.set_array([])
    fig.colorbar(error_sm, ax=axes[2, :].ravel().tolist(), fraction=0.025, pad=0.015, label="WiSED - Ref.")
    fig.suptitle(title, fontsize=8.5)
    _save_figure(fig, path)
    plt.close(fig)


def _save_physical_cutoff_snapshots(
    path: Path,
    t: np.ndarray,
    x: np.ndarray,
    true_field: np.ndarray,
    pred_field: np.ndarray,
    *,
    title: str,
    snapshot_times: tuple[float, ...],
) -> None:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except Exception:
        return
    _apply_nature_style()
    indices = [int(np.argmin(np.abs(t - value))) for value in snapshot_times]
    true_valid = _valid_physical_time_mask(true_field)
    pred_valid = _valid_physical_time_mask(pred_field)
    valid_values = []
    valid_errors = []
    for idx in indices:
        if true_valid[idx]:
            valid_values.append(true_field[idx])
        if pred_valid[idx]:
            valid_values.append(pred_field[idx])
        if true_valid[idx] and pred_valid[idx]:
            valid_errors.append(pred_field[idx] - true_field[idx])
    value_vmax = max(float(np.nanmax(np.concatenate(valid_values))), 1.0e-6) if valid_values else 1.0
    err_abs = max(float(np.nanmax(np.abs(np.concatenate(valid_errors)))), 1.0e-6) if valid_errors else 1.0e-6
    fig, axes = plt.subplots(3, len(indices), figsize=(max(5.8, 1.25 * len(indices) + 1.0), 3.55), dpi=220, constrained_layout=True)
    for col, idx in enumerate(indices):
        axes[0, col].set_title(rf"$\tau={t[idx]:.3f}$")
        if true_valid[idx]:
            axes[0, col].imshow(_radial_image(true_field[idx], x), cmap="viridis", vmin=0.0, vmax=value_vmax)
        else:
            _draw_cutoff_disk(axes[0, col])
        if pred_valid[idx]:
            axes[1, col].imshow(_radial_image(pred_field[idx], x), cmap="viridis", vmin=0.0, vmax=value_vmax)
        else:
            _draw_cutoff_disk(axes[1, col])
        if true_valid[idx] and pred_valid[idx]:
            axes[2, col].imshow(_radial_image(pred_field[idx] - true_field[idx], x), cmap="coolwarm", vmin=-err_abs, vmax=err_abs)
        else:
            _draw_cutoff_disk(axes[2, col])
        for row in range(3):
            axes[row, col].set_axis_off()
    axes[0, 0].text(-0.10, 0.50, "Ref.", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="center")
    axes[1, 0].text(-0.10, 0.50, "WiSED", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="center")
    axes[2, 0].text(-0.10, 0.50, "Error", transform=axes[2, 0].transAxes, rotation=90, va="center", ha="center")
    value_sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0.0, value_vmax), cmap="viridis")
    value_sm.set_array([])
    fig.colorbar(value_sm, ax=axes[:2, :].ravel().tolist(), fraction=0.025, pad=0.015, label="concentration u")
    error_sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-err_abs, err_abs), cmap="coolwarm")
    error_sm.set_array([])
    fig.colorbar(error_sm, ax=axes[2, :].ravel().tolist(), fraction=0.025, pad=0.015, label="WiSED - Ref.")
    fig.suptitle(title, fontsize=8.5)
    _save_figure(fig, path)
    plt.close(fig)


def _save_physical_bounds_plot(path: Path, t: np.ndarray, curves: dict[str, np.ndarray], *, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    _apply_nature_style()
    fig, ax = plt.subplots(figsize=(3.9, 2.45), dpi=220)
    colors = {"Ref.": "0.25", "WiSED": "#8a004f"}
    cutoff_values = []
    visible_values = []
    for label, field in curves.items():
        lower = np.nanmin(field, axis=1)
        upper = np.nanmax(field, axis=1)
        valid = _valid_physical_time_mask(field)
        cutoff = _first_cutoff_tau(t, field)
        if cutoff is not None:
            cutoff_values.append(cutoff)
        lower_plot = lower.copy()
        upper_plot = upper.copy()
        lower_plot[~valid] = np.nan
        upper_plot[~valid] = np.nan
        visible_values.extend(lower_plot[np.isfinite(lower_plot)].tolist())
        visible_values.extend(upper_plot[np.isfinite(upper_plot)].tolist())
        color = colors.get(label, None)
        ax.plot(t, lower_plot, color=color, linewidth=1.35, marker="o", markersize=2.4, label=f"{label} min")
        ax.plot(t, upper_plot, color=color, linewidth=1.1, linestyle="--", marker="o", markersize=2.4, label=f"{label} max")
    if cutoff_values:
        cutoff_start = min(cutoff_values)
        ax.axvspan(cutoff_start, float(t[-1]), color="0.88", zorder=0)
        ax.axvline(cutoff_start, color="0.35", linewidth=0.8)
        ax.text(cutoff_start, 0.97, "cutoff", transform=ax.get_xaxis_transform(), ha="left", va="top", color="0.3")
        ax.set_xlim(float(t[0]) - 0.005, min(float(t[-1]), cutoff_start + 0.04))
    else:
        ax.set_xlim(float(t[0]), float(t[-1]))
    ax.axhline(PHYSICAL_U_MIN, color="0.35", linewidth=0.7)
    if visible_values:
        low = min(min(visible_values), PHYSICAL_U_MIN)
        high = max(max(visible_values), PHYSICAL_U_MIN + 1.0e-6)
        margin = max((high - low) * 0.12, 5.0e-4)
        ax.set_ylim(low - margin, high + margin)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("concentration range")
    ax.set_title(title)
    ax.grid(True, color="0.9", linewidth=0.45)
    ax.legend(frameon=False, loc="best", ncol=2)
    fig.tight_layout()
    _save_figure(fig, path)
    plt.close(fig)


def _coefficient_hint(equation: str) -> dict[str, float | None]:
    compact = equation.replace(" ", "")
    coef_ux_over_x = None
    coef_uxx = None

    match = re.search(r"([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\*u_\{x\}/x", compact, re.IGNORECASE)
    if match:
        coef_ux_over_x = float(match.group(1))
    match = re.search(r"([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\*\(?u_\{xx\}\)?", compact, re.IGNORECASE)
    if match:
        coef_uxx = float(match.group(1))
    return {"coef_u_x_over_x": coef_ux_over_x, "coef_u_xx": coef_uxx}


def _save_scatter(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size > 12000:
        idx = np.linspace(0, y_true.size - 1, 12000).astype(int)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=180)
    ax.scatter(y_true, y_pred, s=4, alpha=0.25, linewidths=0)
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    ax.set_xlabel("reference u_t")
    ax.set_ylabel("discovered RHS")
    ax.set_title("SIB raw concentration PDE")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def postprocess_sib_diffusion_results(
    *,
    task_name: str,
    dataset_path: str,
    summary_path: str,
    result_dir: str,
) -> str:
    dataset = np.load(dataset_path, allow_pickle=True)
    grid_info = dataset["grid_info"].item()
    observed = np.asarray(dataset["data"], dtype=np.float64)[:, :, :, 0]
    field = np.asarray(dataset["data_clean"], dtype=np.float64)[:, :, :, 0]
    t = np.asarray(grid_info["t_coords"], dtype=np.float64)
    x = np.asarray(grid_info["x_coords"], dtype=np.float64)
    equation = _load_equation(Path(summary_path))

    ut, ux, uxx = _derivatives(field, t, x)
    safe_x = np.maximum(x[None, None, :], 1.0e-8)
    true_rhs = uxx + 2.0 * ux / safe_x
    pred_rhs = _safe_rhs_eval(equation, u=field, ux=ux, uxx=uxx, x=x)

    trim_t = 4
    trim_x = 8
    flux = _surface_fluxes(grid_info, field.shape[0])

    # In-window reconstruction: use the same initial condition and boundary
    # forcing to reconstruct the field on the discovery interval 0 <= tau <= T.
    discovered_reconstruction = _rollout_equation(
        equation,
        initial=observed[:, 0, :],
        t=t,
        x=x,
        flux=flux,
        clip_state=True,
    )
    true_reconstruction = _rollout_equation(
        TRUE_PDE_EQUATION,
        initial=observed[:, 0, :],
        t=t,
        x=x,
        flux=flux,
        clip_state=True,
    )
    reconstruction_summary = {
        "discovered_vs_clean": _rollout_metrics(field, discovered_reconstruction, trim_x),
        "true_pde_vs_clean": _rollout_metrics(field, true_reconstruction, trim_x),
        "terminal_discovered_vs_clean": _metrics(field[:, -1, trim_x:-trim_x], discovered_reconstruction[:, -1, trim_x:-trim_x]),
        "terminal_true_pde_vs_clean": _metrics(field[:, -1, trim_x:-trim_x], true_reconstruction[:, -1, trim_x:-trim_x]),
        "initial_condition_source": "denoised noisy observed field at tau=0",
        "boundary_condition_source": "known SIB boundary flux from dataset metadata",
        "time_window": "0 <= tau <= T",
        "state_projection": "hard clip to [0, 1] after each rollout step",
    }

    # Out-of-window rollout: discover on 0 <= tau <= T, then start from the
    # observed T-state and predict T <= tau <= 6T.
    future_t = _future_time_grid(t)
    future_reference = _rollout_equation(
        TRUE_PDE_EQUATION,
        initial=field[:, -1, :],
        t=future_t,
        x=x,
        flux=flux,
        clip_state=True,
    )
    discovered_future = _rollout_equation(
        equation,
        initial=observed[:, -1, :],
        t=future_t,
        x=x,
        flux=flux,
        clip_state=True,
    )
    discovered_future_physical = _rollout_equation(
        equation,
        initial=np.clip(observed[:, -1, :], PHYSICAL_U_MIN, PHYSICAL_U_MAX),
        t=future_t,
        x=x,
        flux=flux,
        clip_state=True,
    )
    true_future = _rollout_equation(
        TRUE_PDE_EQUATION,
        initial=observed[:, -1, :],
        t=future_t,
        x=x,
        flux=flux,
        clip_state=True,
    )
    rollout_summary = {
        "discovered_vs_future_reference": _rollout_metrics(future_reference, discovered_future, trim_x),
        "true_pde_from_observed_T_vs_future_reference": _rollout_metrics(future_reference, true_future, trim_x),
        "terminal_discovered_vs_future_reference": _metrics(
            future_reference[:, -1, trim_x:-trim_x],
            discovered_future[:, -1, trim_x:-trim_x],
        ),
        "terminal_true_pde_from_observed_T_vs_future_reference": _metrics(
            future_reference[:, -1, trim_x:-trim_x],
            true_future[:, -1, trim_x:-trim_x],
        ),
        "initial_condition_source": "denoised noisy observed field at tau=T",
        "future_reference_source": "nominal true PDE rollout initialized from clean tau=T",
        "boundary_condition_source": "known SIB boundary flux from dataset metadata",
        "time_window": "T <= tau <= 6T",
        "state_projection": "hard clip to [0, 1] after each rollout step",
    }
    sl = (slice(None), slice(trim_t, -trim_t), slice(trim_x, -trim_x))
    rep_idx = _representative_surface_index(field)
    report = {
        "task_name": task_name,
        "dataset_path": str(dataset_path),
        "summary_path": str(summary_path),
        "official_discovered_equation": equation,
        "true_equation": "u_t = u_xx + 2*u_x/x",
        "coefficient_hint": _coefficient_hint(equation),
        "metrics": {
            "true_rhs_vs_ut": _metrics(ut[sl], true_rhs[sl]),
            "discovered_rhs_vs_ut": _metrics(ut[sl], pred_rhs[sl]),
            "discovered_rhs_vs_true_rhs": _metrics(true_rhs[sl], pred_rhs[sl]),
        },
        "representative_surface_index": rep_idx,
        "representative_surface_meta": grid_info.get("field_meta", [])[rep_idx],
        "field_reconstruction_0_to_T": reconstruction_summary,
        "rollout_T_to_6T": rollout_summary,
        "physical_cutoff_T_to_6T": {
            "bounds": [PHYSICAL_U_MIN, PHYSICAL_U_MAX],
            "representative_reference_cutoff_tau": _first_cutoff_tau(future_t, future_reference[rep_idx]),
            "representative_wised_projected_initial_cutoff_tau": _first_cutoff_tau(future_t, discovered_future_physical[rep_idx]),
            "representative_wised_observed_initial_cutoff_tau": _first_cutoff_tau(future_t, discovered_future[rep_idx]),
            "all_surface_reference_cutoff_tau": _first_cutoff_tau(future_t, future_reference),
            "all_surface_wised_projected_initial_cutoff_tau": _first_cutoff_tau(future_t, discovered_future_physical),
            "all_surface_wised_observed_initial_cutoff_tau": _first_cutoff_tau(future_t, discovered_future),
            "note": "Rollout states are hard-clipped to [0, 1] after every numerical step in this clipped protocol; cutoff is retained only as a diagnostic and is expected to be absent unless numerical or equation evaluation produces non-finite values.",
        },
        "evaluation_note": (
            "Postprocess evaluates the official WiSED-selected equation only; "
            "it does not rerank candidates. Reconstruction and rollout are "
            "separate downstream evaluations."
        ),
    }

    out_dir = Path(result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "sib_diffusion_equation_assessment.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = out_dir / "sib_diffusion_equation_assessment.md"
    m = report["metrics"]["discovered_rhs_vs_ut"]
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# SIB raw concentration official equation assessment\n\n")
        f.write(f"- Official WiSED equation: `{equation}`\n")
        f.write("- True PDE form: `u_t = u_xx + 2*u_x/x`\n")
        f.write(f"- Discovered RHS vs u_t: RMSE={m['rmse']:.6g}, R2={m['r2']:.6g}, MAE={m['mae']:.6g}\n")
        r = reconstruction_summary["discovered_vs_clean"]
        q = rollout_summary["discovered_vs_future_reference"]
        f.write(f"- In-window field reconstruction, 0 to T: RMSE={r['rmse']:.6g}, R2={r['r2']:.6g}, MAE={r['mae']:.6g}\n")
        f.write(f"- Out-of-window rollout, T to 6T: RMSE={q['rmse']:.6g}, R2={q['r2']:.6g}, MAE={q['mae']:.6g}\n")
        f.write("- Note: this assessment does not replace the official selected equation or rerank candidates.\n")

    _save_scatter(out_dir / "sib_diffusion_rhs_consistency.png", ut[sl], pred_rhs[sl])
    reconstruction_curves = {
        "WiSED equation": _rmse_by_time(field, discovered_reconstruction, trim_x),
        "nominal PDE reference": _rmse_by_time(field, true_reconstruction, trim_x),
    }
    rollout_curves = {
        "WiSED rollout": _rmse_by_time(future_reference, discovered_future, trim_x),
        "nominal PDE from observed T": _rmse_by_time(future_reference, true_future, trim_x),
    }
    _save_rollout_rmse(
        out_dir / "sib_diffusion_reconstruction_rmse.png",
        t,
        reconstruction_curves,
        title=r"Reconstruction error within discovery window",
    )
    _save_rollout_rmse(
        out_dir / "sib_diffusion_future_rollout_rmse.png",
        future_t,
        rollout_curves,
        title=r"Out-of-window rollout prediction, $T\leq \tau \leq 6T$",
    )
    _save_radial_snapshots(
        out_dir / "sib_diffusion_reconstruction_radial_field.png",
        t,
        x,
        field[rep_idx],
        discovered_reconstruction[rep_idx],
        title=r"In-window field reconstruction, $0\leq \tau \leq T$",
        display_mode="concentration",
        snapshot_times=_even_snapshot_times(t),
    )
    rollout_snapshot_times = _physical_cutoff_snapshot_times(
        future_t,
        future_reference[rep_idx],
        discovered_future[rep_idx],
    )
    physical_rollout_snapshot_times = _physical_cutoff_snapshot_times(
        future_t,
        future_reference[rep_idx],
        discovered_future_physical[rep_idx],
    )
    _save_radial_snapshots(
        out_dir / "sib_diffusion_future_rollout_radial_field.png",
        future_t,
        x,
        future_reference[rep_idx],
        discovered_future[rep_idx],
        title=r"Out-of-window absolute concentration, clipped rollout",
        display_mode="concentration",
        snapshot_times=rollout_snapshot_times,
    )
    _save_radial_snapshots(
        out_dir / "sib_diffusion_future_rollout_polarization.png",
        future_t,
        x,
        future_reference[rep_idx],
        discovered_future[rep_idx],
        title=r"Out-of-window radial polarization, clipped rollout",
        display_mode="deviation",
        snapshot_times=rollout_snapshot_times,
    )
    _save_physical_cutoff_snapshots(
        out_dir / "sib_diffusion_future_rollout_cutoff.png",
        future_t,
        x,
        future_reference[rep_idx],
        discovered_future_physical[rep_idx],
        title=r"Out-of-window rollout, clipped to physical range",
        snapshot_times=physical_rollout_snapshot_times,
    )
    _save_physical_bounds_plot(
        out_dir / "sib_diffusion_future_rollout_bounds.png",
        future_t,
        {"Ref.": future_reference[rep_idx], "WiSED": discovered_future_physical[rep_idx]},
        title=r"Physical concentration bounds during rollout",
    )
    csv_path = out_dir / "sib_diffusion_application_metrics.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("task,method,mae,rmse,r2,n\n")
        for name, metric in (
            ("wised_reconstruction", reconstruction_summary["discovered_vs_clean"]),
            ("nominal_pde_reconstruction", reconstruction_summary["true_pde_vs_clean"]),
        ):
            f.write(f"field_reconstruction_0_to_T,{name},{metric['mae']},{metric['rmse']},{metric['r2']},{metric['n']}\n")
        for name, metric in (
            ("wised_rollout", rollout_summary["discovered_vs_future_reference"]),
            ("nominal_pde_from_observed_T", rollout_summary["true_pde_from_observed_T_vs_future_reference"]),
        ):
            f.write(f"rollout_T_to_6T,{name},{metric['mae']},{metric['rmse']},{metric['r2']},{metric['n']}\n")
    return str(json_path)
