from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engineering_validation.sib_diffusion_raw_field.analyze_sib_diffusion import (  # noqa: E402
    PHYSICAL_U_MAX,
    PHYSICAL_U_MIN,
    TRUE_PDE_EQUATION,
    _apply_nature_style,
    _coefficient_hint,
    _compile_rhs,
    _derivatives,
    _eval_rhs,
    _first_cutoff_tau,
    _future_time_grid,
    _load_equation,
    _metrics,
    _physical_cutoff_snapshot_times,
    _radial_image,
    _rmse_by_time,
    _rollout_equation,
    _save_figure,
    _save_physical_bounds_plot,
    _save_physical_cutoff_snapshots,
    _save_radial_snapshots,
    _surface_fluxes,
)


DATASET_PATH = ROOT / "data" / "dataset" / "sib_diffusion_raw_field_noise0.03.npz"
RUN_DIR = ROOT / "outputs" / "sib_diffusion" / "noise_0.03__seed_0"
SUMMARY_PATH = RUN_DIR / "sib_noise_0.03__seed_0_summary.json"
PANEL_DIR = RUN_DIR / "publication_panels"


def _load_case() -> dict:
    dataset = np.load(DATASET_PATH, allow_pickle=True)
    grid_info = dataset["grid_info"].item()
    field = np.asarray(dataset["data_clean"], dtype=np.float64)[:, :, :, 0]
    observed = np.asarray(dataset["data"], dtype=np.float64)[:, :, :, 0]
    t = np.asarray(grid_info["t_coords"], dtype=np.float64)
    x = np.asarray(grid_info["x_coords"], dtype=np.float64)
    equation = _load_equation(SUMMARY_PATH)
    flux = _surface_fluxes(grid_info, field.shape[0])
    return {
        "grid_info": grid_info,
        "field": field,
        "observed": observed,
        "t": t,
        "x": x,
        "equation": equation,
        "flux": flux,
    }


def _representative_surface_index(field: np.ndarray) -> int:
    radial_std = np.std(field, axis=2)
    score = np.mean(radial_std, axis=1)
    return int(np.argmax(score))


def _computed_fields(case: dict) -> dict:
    field = case["field"]
    observed = case["observed"]
    t = case["t"]
    x = case["x"]
    flux = case["flux"]
    equation = case["equation"]

    reconstruction = _rollout_equation(equation, initial=observed[:, 0, :], t=t, x=x, flux=flux, clip_state=True)
    nominal = _rollout_equation(TRUE_PDE_EQUATION, initial=observed[:, 0, :], t=t, x=x, flux=flux, clip_state=True)
    future_t = _future_time_grid(t)
    future_reference = _rollout_equation(TRUE_PDE_EQUATION, initial=field[:, -1, :], t=future_t, x=x, flux=flux, clip_state=True)
    future_wised_physical = _rollout_equation(
        equation,
        initial=np.clip(observed[:, -1, :], PHYSICAL_U_MIN, PHYSICAL_U_MAX),
        t=future_t,
        x=x,
        flux=flux,
        clip_state=True,
    )
    return {
        "reconstruction": reconstruction,
        "nominal": nominal,
        "future_t": future_t,
        "future_reference": future_reference,
        "future_wised_physical": future_wised_physical,
    }


def _plot_panel_a_workflow(path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    _apply_nature_style()
    fig, ax = plt.subplots(figsize=(5.8, 2.25), dpi=220)
    ax.set_axis_off()

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0.0)
    profile = 0.12 + 0.78 * np.exp(-2.7 * np.linspace(0.0, 1.0, 121) ** 2)
    particle = _radial_image(profile, np.linspace(0.0, 1.0, 121), size=180)
    ax.imshow(particle, extent=(0.025, 0.215, 0.31, 0.76), cmap=cmap, vmin=0.0, vmax=1.0, zorder=1)
    ax.annotate("", xy=(0.18, 0.54), xytext=(0.12, 0.54), arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "white"})
    ax.text(0.095, 0.79, "spherical\nparticle", ha="center", va="bottom", fontsize=6.8)
    ax.text(0.165, 0.58, r"$r$", color="white", fontsize=6.8)

    nodes = [
        (0.31, 0.50, "noisy\nfield"),
        (0.50, 0.50, "weak-form\nWiSED"),
        (0.69, 0.50, "recovered\nPDE"),
        (0.88, 0.50, "field use\n+ cutoff"),
    ]
    for x0, y0, label in nodes:
        box = FancyBboxPatch(
            (x0 - 0.070, y0 - 0.075),
            0.14,
            0.15,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=0.7,
            edgecolor="0.35",
            facecolor="0.97",
        )
        ax.add_patch(box)
        ax.text(x0, y0, label, ha="center", va="center", fontsize=6.2)
    for start, end in [(0.215, 0.24), (0.38, 0.43), (0.57, 0.62), (0.76, 0.81)]:
        ax.add_patch(FancyArrowPatch((start, 0.50), (end, 0.50), arrowstyle="-|>", mutation_scale=8, lw=0.8, color="0.25"))

    ax.text(0.50, 0.19, r"target: recover $u_\tau = u_{rr}+2u_r/r$ from clipped raw concentration fields", ha="center", fontsize=6.6)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.12, 0.88)
    _save_figure(fig, path)
    plt.close(fig)


def _plot_panel_b_equation(path: Path, equation: str) -> None:
    import matplotlib.pyplot as plt

    _apply_nature_style()
    coefs = _coefficient_hint(equation)
    labels = [r"$u_{rr}$", r"$u_r/r$"]
    true = np.array([1.0, 2.0])
    found = np.array([coefs["coef_u_xx"] or np.nan, coefs["coef_u_x_over_x"] or np.nan])
    rel_err = np.abs(found - true) / np.maximum(np.abs(true), 1.0e-12) * 100.0

    fig, ax = plt.subplots(figsize=(3.75, 2.85), dpi=220)
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, true, width=width, color="0.72", edgecolor="0.25", linewidth=0.5, label="true")
    ax.bar(x + width / 2, found, width=width, color="#8a004f", edgecolor="0.25", linewidth=0.5, label="WiSED")
    for i, err in enumerate(rel_err):
        ax.text(i + width / 2, found[i] + 0.08, f"{err:.1f}%", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("coefficient")
    ax.set_title("Recovered spherical-diffusion operator")
    ax.legend(frameon=False, loc="upper left")
    fig.subplots_adjust(bottom=0.28)
    found_text = (
        rf"WiSED: $u_\tau={found[0]:.4g}u_{{rr}}+{found[1]:.4g}u_r/r$"
        if np.all(np.isfinite(found))
        else r"WiSED: selected open-form equation"
    )
    fig.text(
        0.17,
        0.04,
        r"true: $u_\tau=u_{rr}+2u_r/r$" + "\n" + found_text,
        ha="left",
        va="bottom",
        fontsize=6.6,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0.0, 0.16, 1.0, 1.0))
    _save_figure(fig, path)
    plt.close(fig)


def _plot_panel_c_rhs(path: Path, case: dict) -> None:
    import matplotlib.pyplot as plt

    _apply_nature_style()
    field = case["field"]
    t = case["t"]
    x = case["x"]
    equation = case["equation"]
    ut, ux, uxx = _derivatives(field, t, x)
    safe_x = np.maximum(x[None, None, :], 1.0e-8)
    true_rhs = uxx + 2.0 * ux / safe_x
    pred_rhs = _eval_rhs(_compile_rhs(equation), u=field, ux=ux, uxx=uxx, x=x)
    sl = (slice(None), slice(4, -4), slice(8, -8))
    metrics = _metrics(true_rhs[sl], pred_rhs[sl])

    y_true = true_rhs[sl].ravel()
    y_pred = pred_rhs[sl].ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size > 18000:
        idx = np.linspace(0, y_true.size - 1, 18000).astype(int)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    axis_lim = 0.85
    axis_ticks = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])

    fig, ax = plt.subplots(figsize=(3.0, 2.75), dpi=220)
    ax.scatter(y_true, y_pred, s=3.0, alpha=0.18, color="#8a004f", linewidths=0)
    ax.plot([-axis_lim, axis_lim], [-axis_lim, axis_lim], color="0.15", linewidth=0.9)
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_xticks(axis_ticks)
    ax.set_yticks(axis_ticks)
    ax.set_xlabel("true PDE RHS")
    ax.set_ylabel("WiSED RHS")
    ax.set_title("Operator-level consistency")
    ax.text(0.05, 0.95, f"RMSE={metrics['rmse']:.2e}\n$R^2$={metrics['r2']:.3f}", transform=ax.transAxes, va="top", fontsize=6.7)
    ax.grid(False)
    fig.tight_layout()
    _save_figure(fig, path)
    plt.close(fig)


def _plot_panel_e_metrics(path: Path, case: dict, fields: dict, rep_idx: int, valid_tau_max: float) -> None:
    import matplotlib.pyplot as plt

    _apply_nature_style()
    field = case["field"]
    t = case["t"]
    reconstruction = fields["reconstruction"]
    nominal = fields["nominal"]
    rmse = _rmse_by_time(field, reconstruction, trim_x=8)
    nominal_rmse = _rmse_by_time(field, nominal, trim_x=8)
    valid_t = t <= float(valid_tau_max) + 1.0e-12
    summary = _metrics(field[:, valid_t, 8:-8], reconstruction[:, valid_t, 8:-8])

    fig, axes = plt.subplots(1, 2, figsize=(5.1, 2.35), dpi=220)
    axes[0].plot(t[valid_t], rmse[valid_t], color="#8a004f", lw=1.35, label="WiSED")
    axes[0].plot(t[valid_t], nominal_rmse[valid_t], color="0.55", lw=1.1, label="nominal PDE")
    axes[0].set_xlabel(r"$\tau$")
    axes[0].set_ylabel("field RMSE")
    axes[0].set_title("Clipped-window reconstruction error")
    axes[0].text(0.05, 0.95, f"RMSE={summary['rmse']:.3f}\n$R^2$={summary['r2']:.3f}", transform=axes[0].transAxes, va="top", fontsize=6.5)
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].grid(True, color="0.9", linewidth=0.45)

    ref_min = np.min(field[rep_idx], axis=1)
    ref_max = np.max(field[rep_idx], axis=1)
    rec_min = np.min(reconstruction[rep_idx], axis=1)
    rec_max = np.max(reconstruction[rep_idx], axis=1)
    axes[1].fill_between(t, PHYSICAL_U_MIN, PHYSICAL_U_MAX, color="0.93", zorder=0)
    axes[1].plot(t[valid_t], ref_min[valid_t], color="0.25", lw=1.0, label="Ref. min")
    axes[1].plot(t[valid_t], ref_max[valid_t], color="0.25", lw=1.0, ls="--", label="Ref. max")
    axes[1].plot(t[valid_t], rec_min[valid_t], color="#8a004f", lw=1.0, label="WiSED min")
    axes[1].plot(t[valid_t], rec_max[valid_t], color="#8a004f", lw=1.0, ls="--", label="WiSED max")
    axes[1].axhline(0.0, color="0.35", lw=0.7)
    axes[1].set_xlabel(r"$\tau$")
    axes[1].set_ylabel("concentration range")
    axes[1].set_title("Physical range check")
    axes[1].set_xlim(float(t[valid_t][0]), float(t[valid_t][-1]))
    axes[1].grid(True, color="0.9", linewidth=0.45)
    axes[1].legend(frameon=False, loc="best", fontsize=5.9, ncol=2)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_figure(fig, path)
    plt.close(fig)


def main() -> int:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    case = _load_case()
    fields = _computed_fields(case)
    rep_idx = _representative_surface_index(case["field"])
    valid_tau_max = float(case["t"][-1])
    snapshot_times = tuple(float(v) for v in np.linspace(float(case["t"][0]), valid_tau_max, 5))

    _plot_panel_a_workflow(PANEL_DIR / "sib_panel_a_workflow.png")
    _plot_panel_b_equation(PANEL_DIR / "sib_panel_b_equation_recovery.png", case["equation"])
    _plot_panel_c_rhs(PANEL_DIR / "sib_panel_c_rhs_consistency.png", case)
    valid_t = case["t"] <= valid_tau_max + 1.0e-12
    _save_radial_snapshots(
        PANEL_DIR / "sib_panel_d_inwindow_radial_reconstruction.png",
        case["t"][valid_t],
        case["x"],
        case["field"][rep_idx][valid_t],
        fields["reconstruction"][rep_idx][valid_t],
        title=rf"Clipped full-window field reconstruction, $0\leq \tau \leq {valid_tau_max:.3f}$",
        display_mode="concentration",
        snapshot_times=snapshot_times,
    )
    _plot_panel_e_metrics(PANEL_DIR / "sib_panel_e_reconstruction_metrics.png", case, fields, rep_idx, valid_tau_max)
    cutoff_snapshot_times = _physical_cutoff_snapshot_times(
        fields["future_t"],
        fields["future_reference"][rep_idx],
        fields["future_wised_physical"][rep_idx],
    )
    _save_physical_cutoff_snapshots(
        PANEL_DIR / "sib_panel_f_outwindow_physical_cutoff.png",
        fields["future_t"],
        case["x"],
        fields["future_reference"][rep_idx],
        fields["future_wised_physical"][rep_idx],
        title=r"Out-of-window rollout, clipped to physical range",
        snapshot_times=cutoff_snapshot_times,
    )
    _save_physical_bounds_plot(
        PANEL_DIR / "sib_panel_g_outwindow_validity_bounds.png",
        fields["future_t"],
        {"Ref.": fields["future_reference"][rep_idx], "WiSED": fields["future_wised_physical"][rep_idx]},
        title=r"Physical concentration bounds during clipped rollout",
    )

    cutoff_ref = _first_cutoff_tau(fields["future_t"], fields["future_reference"][rep_idx])
    cutoff_wised = _first_cutoff_tau(fields["future_t"], fields["future_wised_physical"][rep_idx])
    print(f"publication panels written to: {PANEL_DIR}")
    print(f"representative surface index: {rep_idx}")
    print(f"physical cutoff tau: Ref.={cutoff_ref}, WiSED={cutoff_wised}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
