from __future__ import annotations

import os

import numpy as np
import torch
from scipy.fft import fftfreq, fftn, ifftn

DTYPE_NP = np.float64
DTYPE_TORCH = torch.float32


def _random_spiral_ic(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a noisy two-field initial condition for the FHN system."""
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    u0 = np.exp(-0.1 * (x_grid**2 + y_grid**2))
    v0 = np.exp(-0.1 * ((x_grid - 2.0) ** 2 + y_grid**2))
    u0 += 0.05 * rng.normal(size=x_grid.shape)
    v0 += 0.05 * rng.normal(size=x_grid.shape)
    return u0.astype(DTYPE_NP), v0.astype(DTYPE_NP)


def _fhn_rhs(
    u_hat: np.ndarray,
    v_hat: np.ndarray,
    k_sq: np.ndarray,
    du: float,
    dv: float,
    eps: float,
    alpha: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    lap_u = ifftn(-k_sq * u_hat).real
    lap_v = ifftn(-k_sq * v_hat).real

    u = ifftn(u_hat).real
    v = ifftn(v_hat).real

    rhs_u = du * lap_u + u - (1.0 / 3.0) * u**3 - v
    rhs_v = dv * lap_v + eps * u + eps * alpha - eps * gamma * v
    return fftn(rhs_u), fftn(rhs_v)


def solve_fhn_rk4(
    u0: np.ndarray,
    v0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    du: float,
    dv: float,
    eps: float,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    """Integrate the periodic 2D FitzHugh-Nagumo system with RK4 sub-stepping."""
    nx, ny = len(x), len(y)
    lx = x[-1] - x[0] + (x[1] - x[0])
    ly = y[-1] - y[0] + (y[1] - y[0])

    kx = 2.0 * np.pi * fftfreq(nx, d=lx / nx).astype(DTYPE_NP)
    ky = 2.0 * np.pi * fftfreq(ny, d=ly / ny).astype(DTYPE_NP)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_sq = kx_grid**2 + ky_grid**2

    nt = len(t)
    dt_save = float(t[1] - t[0]) if nt > 1 else 0.0

    # FHN has stiff diffusion and cubic reaction terms. Sub-stepping keeps the
    # saved trajectory tied to the PDE while avoiding RK4 instability between
    # output frames.
    max_k_sq = float(np.max(k_sq))
    max_diffusion = max(float(du), float(dv))
    dt_diff_stable = 2.0 / (max_diffusion * max_k_sq + 1e-12)
    dt_stable = min(dt_diff_stable, 0.05)
    n_inner = max(1, int(np.ceil(dt_save / dt_stable)))
    dt_inner = dt_save / n_inner

    u_hat = fftn(u0.astype(DTYPE_NP))
    v_hat = fftn(v0.astype(DTYPE_NP))

    traj = np.zeros((nt, nx, ny, 2), dtype=DTYPE_NP)
    traj[0, ..., 0] = u0
    traj[0, ..., 1] = v0

    for n in range(nt - 1):
        for _ in range(n_inner):
            k1_u, k1_v = _fhn_rhs(u_hat, v_hat, k_sq, du, dv, eps, alpha, gamma)
            k2_u, k2_v = _fhn_rhs(
                u_hat + 0.5 * dt_inner * k1_u,
                v_hat + 0.5 * dt_inner * k1_v,
                k_sq,
                du,
                dv,
                eps,
                alpha,
                gamma,
            )
            k3_u, k3_v = _fhn_rhs(
                u_hat + 0.5 * dt_inner * k2_u,
                v_hat + 0.5 * dt_inner * k2_v,
                k_sq,
                du,
                dv,
                eps,
                alpha,
                gamma,
            )
            k4_u, k4_v = _fhn_rhs(
                u_hat + dt_inner * k3_u,
                v_hat + dt_inner * k3_v,
                k_sq,
                du,
                dv,
                eps,
                alpha,
                gamma,
            )

            u_hat = u_hat + (dt_inner / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
            v_hat = v_hat + (dt_inner / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        traj[n + 1, ..., 0] = ifftn(u_hat).real
        traj[n + 1, ..., 1] = ifftn(v_hat).real

    return traj


def fitzhugh_nagumo2d_equation(
    save_dir: str = "data/dataset",
    noise_level: float = 0.0,
    batch_size: int = 4,
    nx2d: int = 64,
    ny2d: int = 64,
    nt2d: int = 41,
    t_max2d: float = 20.0,
    x_range: tuple[float, float] = (-10.0, 10.0),
    y_range: tuple[float, float] = (-10.0, 10.0),
    du: float = 1.0,
    dv: float = 0.01,
    eps: float = 0.008,
    alpha: float = 0.7,
    gamma: float = 0.8,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, dict, dict[str, str], np.ndarray, np.ndarray, np.ndarray]:
    """Generate 2D FitzHugh-Nagumo data for two-field PDE discovery."""
    print(f"--- Generating 2D FitzHugh-Nagumo Data (Spirals) | Noise: {noise_level} ---")
    x = np.linspace(x_range[0], x_range[1], nx2d, endpoint=False, dtype=DTYPE_NP)
    y = np.linspace(y_range[0], y_range[1], ny2d, endpoint=False, dtype=DTYPE_NP)
    t = np.linspace(0.0, t_max2d, nt2d, dtype=DTYPE_NP)
    rng = np.random.RandomState(seed)

    trajectories_clean = []
    trajectories_noisy = []
    for _ in range(batch_size):
        u0, v0 = _random_spiral_ic(x, y, rng)
        clean = solve_fhn_rk4(u0, v0, x, y, t, du, dv, eps, alpha, gamma)
        trajectories_clean.append(clean)

        if noise_level > 0:
            sigma = np.std(clean, axis=(0, 1, 2), keepdims=True)
            noisy = clean + rng.normal(size=clean.shape).astype(DTYPE_NP) * sigma * noise_level
            trajectories_noisy.append(noisy)
        else:
            trajectories_noisy.append(clean.copy())

    clean_array = np.stack(trajectories_clean, axis=0)
    noisy_array = np.stack(trajectories_noisy, axis=0)
    data_tensor = torch.tensor(noisy_array, dtype=DTYPE_TORCH)
    clean_tensor = torch.tensor(clean_array, dtype=DTYPE_TORCH)

    if torch.isnan(data_tensor).any() or torch.isinf(data_tensor).any():
        raise ValueError("NaN/Inf in 2D FHN data.")

    true_eq = {
        "u": f"u_t = {du} * (u_xx + u_yy) + u - (1/3)*u^3 - v",
        "v": f"v_t = {dv} * (v_xx + v_yy) + {eps} * u + {eps}*{alpha} - {eps}*{gamma}*v",
    }
    grid_info = {
        "t_coords": torch.tensor(t, dtype=DTYPE_TORCH),
        "x_coords": torch.tensor(x, dtype=DTYPE_TORCH),
        "y_coords": torch.tensor(y, dtype=DTYPE_TORCH),
        "is_uniform": True,
        "equation_type": "fhn_2d",
        "periodic_axes": {"x": True, "y": True},
        "field_names": ["u", "v"],
    }

    os.makedirs(save_dir, exist_ok=True)
    filename = f"fhn_2d_noise{noise_level}.npz"
    np.savez_compressed(
        os.path.join(save_dir, filename),
        data=noisy_array,
        data_clean=clean_array,
        grid_info=grid_info,
        true_eq=true_eq,
    )
    print(f"--- Data Saved to {filename} ---")
    return data_tensor, clean_tensor, grid_info, true_eq, x, y, t

