from __future__ import annotations

import os
import numpy as np
import torch
from scipy.fft import fft, ifft, fftfreq

DTYPE_NP = np.float64
DTYPE_TORCH = torch.float32

def generate_random_ic(x: np.ndarray, rng: np.random.RandomState, n_modes: int = 4) -> np.ndarray:
    u = np.zeros_like(x)
    for k in range(1, n_modes + 1):
        amp = rng.uniform(0.1, 0.5) / k
        phase = rng.uniform(0, 2 * np.pi)
        u += amp * np.cos(k * x + phase)
    u += 0.2 * np.exp(-0.5 * (x / 0.8) ** 2)

    # 关键修改：去掉空间均值，避免 KdV 中 u*u_x 和 u_x 混淆
    u = u - np.mean(u)

    # 再归一化幅值
    u = u / (np.max(np.abs(u)) + 1e-8)

    return u.astype(DTYPE_NP)

def solve_kdv_etdrk4(u0: np.ndarray, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    nx = len(x)
    nt = len(t)
    dt = t[1] - t[0]
    L_domain = x[-1] - x[0] + (x[1] - x[0])
    k = 2 * np.pi * fftfreq(nx, d=L_domain / nx).astype(DTYPE_NP)
    L_op = 1j * k ** 3
    E = np.exp(L_op * dt)
    E2 = np.exp(L_op * dt / 2)
    M = 64
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L_op[:, None] + r[None, :]
    Q = dt * np.mean((np.exp(LR / 2) - 1) / LR, axis=1)
    f1 = dt * np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1)
    f2 = dt * np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1)
    f3 = dt * np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1)
    Q[0] = dt / 2
    f1[0] = dt / 6
    f2[0] = dt / 6
    f3[0] = dt / 6

    def nonlinear_term(u_hat: np.ndarray) -> np.ndarray:
        u = np.real(ifft(u_hat))
        return -3.0j * k * fft(u ** 2)

    u_hat = fft(u0)
    out = np.zeros((nt, nx), dtype=DTYPE_NP)
    out[0] = u0
    for n in range(nt - 1):
        Nu = nonlinear_term(u_hat)
        a = E2 * u_hat + Q * Nu
        Na = nonlinear_term(a)
        b = E2 * u_hat + Q * Na
        Nb = nonlinear_term(b)
        c = E2 * a + Q * (2 * Nb - Nu)
        Nc = nonlinear_term(c)
        u_hat = E * u_hat + f1 * Nu + 2 * f2 * (Na + Nb) + f3 * Nc
        out[n + 1] = np.real(ifft(u_hat))
    return out

def kdv_equation(
    save_dir: str = "data/dataset",
    noise_level: float = 0.0,
    batch_size: int = 32,
    nx: int = 256,
    nt: int = 101,
    t_max: float = 1.0,
    x_range: tuple[float, float] = (-np.pi, np.pi),
    seed: int = 123
) -> tuple[torch.Tensor, torch.Tensor, dict, str, np.ndarray, np.ndarray]:
    print(f"--- Generating 1D KdV Data (ETD-RK4 Spectral) | Noise: {noise_level} ---")
    x = np.linspace(x_range[0], x_range[1], nx, endpoint=False, dtype=DTYPE_NP)
    t = np.linspace(0.0, t_max, nt, dtype=DTYPE_NP)
    rng = np.random.RandomState(seed)

    trajectories_noisy, trajectories_clean = [], []
    for _ in range(batch_size):
        u0 = generate_random_ic(x, rng)
        u_clean = solve_kdv_etdrk4(u0, x, t)
        trajectories_clean.append(u_clean.copy())

        if noise_level > 0:
            u_noisy = u_clean + rng.normal(size=u_clean.shape).astype(DTYPE_NP) * np.std(u_clean) * noise_level
            trajectories_noisy.append(u_noisy)
        else:
            trajectories_noisy.append(u_clean.copy())

    u_noisy_array = np.stack(trajectories_noisy, axis=0)
    u_clean_array = np.stack(trajectories_clean, axis=0)

    data_tensor = torch.tensor(u_noisy_array, dtype=DTYPE_TORCH).unsqueeze(-1)
    data_clean_tensor = torch.tensor(u_clean_array, dtype=DTYPE_TORCH).unsqueeze(-1)

    if torch.isnan(data_tensor).any() or torch.isinf(data_tensor).any():
        raise ValueError("NaN/Inf in KdV data!")

    grid_info = {
        "t_coords": torch.tensor(t, dtype=DTYPE_TORCH),
        "x_coords": torch.tensor(x, dtype=DTYPE_TORCH),
        "is_uniform": True, "equation_type": "kdv_1d", "periodic_axes": {"x": True}, "field_names": ["u"],
    }
    true_eq = "u_t = -6 * u * u_x - u_xxx"

    os.makedirs(save_dir, exist_ok=True)
    filename = f"kdv_1d_noise{noise_level}.npz"
    np.savez_compressed(
        os.path.join(save_dir, filename),
        data=u_noisy_array, data_clean=u_clean_array, grid_info=grid_info, true_eq=true_eq
    )

    print(f"--- Data Saved to {filename} ---")
    return data_tensor, data_clean_tensor, grid_info, true_eq, x, t
