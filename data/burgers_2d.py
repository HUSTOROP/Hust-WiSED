from __future__ import annotations

import os
import numpy as np
import torch
from scipy.fft import fftfreq, fftn, ifftn

DTYPE_NP = np.float64
DTYPE_TORCH = torch.float32

def _random_periodic_field_2d(x: np.ndarray, y: np.ndarray, rng: np.random.RandomState, n_modes: int = 3) -> np.ndarray:
    X, Y = np.meshgrid(x, y, indexing="ij")
    field = np.zeros_like(X, dtype=DTYPE_NP)
    field += rng.uniform(-0.05, 0.05)
    Lx = x[-1] - x[0] + (x[1] - x[0])
    Ly = y[-1] - y[0] + (y[1] - y[0])
    for kx in range(1, n_modes + 1):
        for ky in range(1, n_modes + 1):
            amp = rng.uniform(0.05, 0.3) / (kx + ky)
            phx = rng.uniform(0, 2 * np.pi)
            phy = rng.uniform(0, 2 * np.pi)
            field += amp * np.sin(2 * np.pi * kx * (X - x[0]) / Lx + phx) * np.cos(2 * np.pi * ky * (Y - y[0]) / Ly + phy)
    field /= (np.max(np.abs(field)) + 1e-8)
    return field.astype(DTYPE_NP)

def _make_dealias_mask(kx_grid: np.ndarray, ky_grid: np.ndarray) -> np.ndarray:
    kx_max = np.max(np.abs(kx_grid)) + 1e-12
    ky_max = np.max(np.abs(ky_grid)) + 1e-12
    return ((np.abs(kx_grid) <= (2.0 / 3.0) * kx_max) & (np.abs(ky_grid) <= (2.0 / 3.0) * ky_max)).astype(DTYPE_NP)

def _burgers2d_rhs(
    u: np.ndarray,
    v: np.ndarray,
    kx_grid: np.ndarray,
    ky_grid: np.ndarray,
    dealias_mask: np.ndarray,
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    u_hat = fftn(u)
    v_hat = fftn(v)
    ux = ifftn(1j * kx_grid * u_hat).real
    uy = ifftn(1j * ky_grid * u_hat).real
    vx = ifftn(1j * kx_grid * v_hat).real
    vy = ifftn(1j * ky_grid * v_hat).real
    lap_u = ifftn(-(kx_grid ** 2 + ky_grid ** 2) * u_hat).real
    lap_v = ifftn(-(kx_grid ** 2 + ky_grid ** 2) * v_hat).real
    adv_u = u * ux + v * uy
    adv_v = u * vx + v * vy
    rhs_u = -adv_u + nu * lap_u
    rhs_v = -adv_v + nu * lap_v
    rhs_u_hat = fftn(rhs_u) * dealias_mask
    rhs_v_hat = fftn(rhs_v) * dealias_mask
    return ifftn(rhs_u_hat).real, ifftn(rhs_v_hat).real

def solve_burgers2d_rk4(u0: np.ndarray, v0: np.ndarray, x: np.ndarray, y: np.ndarray, t: np.ndarray, nu: float) -> np.ndarray:
    nx, ny = len(x), len(y)
    dx = float(x[1] - x[0]) if nx > 1 else 1.0
    dy = float(y[1] - y[0]) if ny > 1 else 1.0
    kx = 2.0 * np.pi * fftfreq(nx, d=dx).astype(DTYPE_NP)
    ky = 2.0 * np.pi * fftfreq(ny, d=dy).astype(DTYPE_NP)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    dealias_mask = _make_dealias_mask(kx_grid, ky_grid)

    nt = len(t)
    dt = float(t[1] - t[0]) if nt > 1 else 0.0
    u = u0.astype(DTYPE_NP).copy()
    v = v0.astype(DTYPE_NP).copy()
    traj = np.zeros((nt, nx, ny, 2), dtype=DTYPE_NP)
    traj[0, ..., 0] = u
    traj[0, ..., 1] = v
    for n in range(nt - 1):
        k1_u, k1_v = _burgers2d_rhs(u, v, kx_grid, ky_grid, dealias_mask, nu)
        k2_u, k2_v = _burgers2d_rhs(u + 0.5 * dt * k1_u, v + 0.5 * dt * k1_v, kx_grid, ky_grid, dealias_mask, nu)
        k3_u, k3_v = _burgers2d_rhs(u + 0.5 * dt * k2_u, v + 0.5 * dt * k2_v, kx_grid, ky_grid, dealias_mask, nu)
        k4_u, k4_v = _burgers2d_rhs(u + dt * k3_u, v + dt * k3_v, kx_grid, ky_grid, dealias_mask, nu)
        u = u + (dt / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
        v = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        traj[n + 1, ..., 0] = u
        traj[n + 1, ..., 1] = v
    return traj

def burgers2d_equation(
    save_dir: str = "data/dataset",
    noise_level: float = 0.0,
    batch_size: int = 36,
    nx2d: int = 48,
    ny2d: int = 48,
    nt2d: int = 51,
    t_max2d: float = 0.5,
    x_range: tuple[float, float] = (-1.0, 1.0),
    y_range: tuple[float, float] = (-1.0, 1.0),
    nu2d: float = 0.01,
    seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, dict, dict[str, str], np.ndarray, np.ndarray, np.ndarray]:
    print(f"--- Generating 2D Burgers Data (Pseudo-spectral RK4) | Noise: {noise_level} ---")
    x = np.linspace(x_range[0], x_range[1], nx2d, endpoint=False, dtype=DTYPE_NP)
    y = np.linspace(y_range[0], y_range[1], ny2d, endpoint=False, dtype=DTYPE_NP)
    t = np.linspace(0.0, t_max2d, nt2d, dtype=DTYPE_NP)
    rng = np.random.RandomState(seed)

    trajectories_clean = []
    trajectories_noisy = []
    for _ in range(batch_size):
        u0 = _random_periodic_field_2d(x, y, rng, n_modes=3)
        v0 = _random_periodic_field_2d(x, y, rng, n_modes=3)
        clean = solve_burgers2d_rk4(u0, v0, x, y, t, nu2d)
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
    data_clean_tensor = torch.tensor(clean_array, dtype=DTYPE_TORCH)
    if torch.isnan(data_tensor).any() or torch.isinf(data_tensor).any():
        raise ValueError("NaN/Inf in 2D Burgers data!")

    true_eq = {
        "u": f"u_t = -((u * u_x) + (v * u_y)) + {nu2d} * (u_xx + u_yy)",
        "v": f"v_t = -((u * v_x) + (v * v_y)) + {nu2d} * (v_xx + v_yy)",
    }
    grid_info = {
        "t_coords": torch.tensor(t, dtype=DTYPE_TORCH),
        "x_coords": torch.tensor(x, dtype=DTYPE_TORCH),
        "y_coords": torch.tensor(y, dtype=DTYPE_TORCH),
        "is_uniform": True,
        "equation_type": "burgers_2d",
        "periodic_axes": {"x": True, "y": True},
        "field_names": ["u", "v"],
    }

    os.makedirs(save_dir, exist_ok=True)
    filename = f"burgers_2d_noise{noise_level}.npz"
    np.savez_compressed(
        os.path.join(save_dir, filename),
        data=noisy_array,
        data_clean=clean_array,
        grid_info=grid_info,
        true_eq=true_eq,
    )
    print(f"--- Data Saved to {filename} ---")
    return data_tensor, data_clean_tensor, grid_info, true_eq, x, y, t
