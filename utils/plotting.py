from __future__ import annotations

import os
import warnings
from typing import Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

warnings.filterwarnings("ignore")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_figure(fig, path: str, dpi: int = 150) -> str:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _finite_pairs(xs: Iterable[float], ys: Iterable[float], upper: float = 1e9) -> Tuple[List[float], List[float]]:
    out_x: List[float] = []
    out_y: List[float] = []
    for x, y in zip(xs, ys):
        if np.isfinite(y) and float(y) < float(upper):
            out_x.append(float(x))
            out_y.append(float(y))
    return out_x, out_y


def plot_training_curves(history: dict, result_dir: str, equation_name: str, save: bool = True) -> str:
    epochs = list(history.get("epoch", []))
    if not epochs:
        return ""

    _ensure_dir(result_dir)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(f"WiSED Training Curves - {str(equation_name).upper()}", fontsize=14, fontweight="bold", y=1.01)

    plots = [
        (axes[0, 0], "kl_loss", "KL Loss", True, "#2196F3"),
        (axes[0, 1], "reinforce_loss", "REINFORCE Loss", False, "#F44336"),
        (axes[0, 2], "struct_loss", "Sparsity Loss", True, "#4CAF50"),
        (axes[0, 3], "total_loss", "Total Loss", False, "#212121"),
        (axes[1, 0], "best_mse", "Best Residual MSE", True, "#9C27B0"),
        (axes[1, 1], "best_fitness", "Best Fitness", True, "#FF9800"),
        (axes[1, 2], "gamma", "Liquid Gate Gamma", False, "#00BCD4"),
        (axes[1, 3], "lr", "Learning Rate", True, "#795548"),
    ]

    for ax, key, label, logy, color in plots:
        xs, ys = _finite_pairs(epochs, history.get(key, []))
        if ys:
            ax.plot(xs, ys, color=color, lw=1.8, label=label)
            if logy and min(ys) > 0:
                ax.set_yscale("log")
            ax.legend(fontsize=8)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(result_dir, f"{equation_name}_training_curves.png")
    if save:
        _save_figure(fig, path)
    else:
        plt.close(fig)
    print(f"  [Viz] Training curves -> {path}")
    return path


def plot_fitness_evolution(history: dict, result_dir: str, equation_name: str) -> str:
    epochs = list(history.get("epoch", []))
    if not epochs:
        return ""

    _ensure_dir(result_dir)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    ep_mse, mse = _finite_pairs(epochs, history.get("best_mse", []))
    ep_fit, fit = _finite_pairs(epochs, history.get("best_fitness", []))

    if mse:
        ax1.plot(ep_mse, mse, color="#9C27B0", lw=2, label="Residual MSE")
        if min(mse) > 0:
            ax1.set_yscale("log")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Residual MSE", color="#9C27B0")
    ax1.tick_params(axis="y", labelcolor="#9C27B0")

    if fit:
        ax2.plot(ep_fit, fit, color="#FF9800", lw=2, linestyle="--", label="Fitness")
        if min(fit) > 0:
            ax2.set_yscale("log")
    ax2.set_ylabel("Fitness", color="#FF9800")
    ax2.tick_params(axis="y", labelcolor="#FF9800")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines1 or lines2:
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    plt.title(f"Fitness Evolution - {str(equation_name).upper()}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(result_dir, f"{equation_name}_fitness_evolution.png")
    _save_figure(fig, path)
    print(f"  [Viz] Fitness evolution -> {path}")
    return path


def plot_population_fitness(population: list, result_dir: str, equation_name: str, epoch: int = -1) -> str:
    fitnesses = [float(p["fitness"]) for p in population if float(p.get("fitness", 1e10)) < 1e9]
    complexities = [float(p["complexity"]) for p in population if float(p.get("fitness", 1e10)) < 1e9]
    if not fitnesses:
        return ""

    _ensure_dir(result_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(fitnesses, bins=30, color="#2196F3", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Fitness", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)
    axes[0].set_title("Fitness Distribution", fontsize=11)
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    vmax = max(float(np.percentile(fitnesses, 90)), min(fitnesses))
    sc = axes[1].scatter(
        complexities,
        fitnesses,
        c=fitnesses,
        cmap="viridis_r",
        s=25,
        alpha=0.7,
        norm=Normalize(vmin=min(fitnesses), vmax=vmax),
    )
    axes[1].set_xlabel("Equation Complexity (tokens)", fontsize=10)
    axes[1].set_ylabel("Fitness", fontsize=10)
    axes[1].set_title("Fitness vs Complexity", fontsize=11)
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    fig.colorbar(sc, ax=axes[1], label="Fitness")

    ep_str = f" (epoch {epoch})" if int(epoch) >= 0 else ""
    fig.suptitle(f"Population - {str(equation_name).upper()}{ep_str}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(result_dir, f"{equation_name}_population_fitness.png")
    _save_figure(fig, path)
    print(f"  [Viz] Population fitness -> {path}")
    return path


__all__ = [
    "plot_fitness_evolution",
    "plot_population_fitness",
    "plot_training_curves",
]
