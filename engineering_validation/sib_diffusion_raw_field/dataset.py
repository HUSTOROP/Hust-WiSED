from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


TRUE_EQUATION = "C_tau = C_xx + 2/x*C_x"
PREPROCESS_VERSION = "sib_diffusion_open_field_v4_clipped_full_window"


def _as_tuple(value: Iterable[str] | str) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part) for part in value)


def _initial_profile(x: np.ndarray, name: str, phase: str) -> np.ndarray:
    sign = 1.0 if phase == "charge" else -1.0
    base = 0.55 + 0.03 * sign
    if name == "smooth_quadratic":
        profile = base + 0.12 * sign * (x**2 - 0.35)
    elif name == "surface_layer":
        profile = base + 0.14 * sign * np.exp(-(1.0 - x) / 0.12)
    elif name == "sin1":
        profile = base + 0.10 * sign * np.sin(np.pi * x)
    elif name == "sin2":
        profile = base + 0.08 * sign * np.sin(2.0 * np.pi * x)
    elif name == "center_bump":
        profile = base + 0.12 * sign * np.exp(-(x / 0.25) ** 2)
    elif name == "surface_bump":
        profile = base + 0.12 * sign * np.exp(-((x - 1.0) / 0.18) ** 2)
    elif name == "mid_bump":
        profile = base + 0.10 * sign * np.exp(-((x - 0.55) / 0.20) ** 2)
    else:
        raise ValueError(f"Unknown SIB initial profile: {name}")
    return np.clip(profile, 0.05, 0.95)


def _rhs_spherical_diffusion(c: np.ndarray, x: np.ndarray, flux: float) -> np.ndarray:
    dx = float(x[1] - x[0])
    rhs = np.empty_like(c)
    rhs[0] = 6.0 * (c[1] - c[0]) / (dx * dx)

    cx = (c[2:] - c[:-2]) / (2.0 * dx)
    cxx = (c[2:] - 2.0 * c[1:-1] + c[:-2]) / (dx * dx)
    rhs[1:-1] = cxx + 2.0 * cx / np.maximum(x[1:-1], 1.0e-8)

    ghost = c[-2] + 2.0 * dx * flux
    cx_s = flux
    cxx_s = (ghost - 2.0 * c[-1] + c[-2]) / (dx * dx)
    rhs[-1] = cxx_s + 2.0 * cx_s / x[-1]
    return rhs


def _simulate_surface(
    x: np.ndarray,
    *,
    phase: str,
    delta: float,
    profile_name: str,
    tau_max: float,
    dtau: float,
    save_every: int,
    clip_during_simulation: bool,
) -> tuple[np.ndarray, np.ndarray]:
    dx = float(x[1] - x[0])
    stable_dt = min(float(dtau), 0.20 * dx * dx)
    n_steps = int(np.ceil(float(tau_max) / stable_dt))
    c = _initial_profile(x, profile_name, phase).astype(np.float64)
    flux = float(delta) if phase == "charge" else -float(delta)

    frames = [c.copy()]
    times = [0.0]
    for step in range(1, n_steps + 1):
        rhs = _rhs_spherical_diffusion(c, x, flux)
        c = c + stable_dt * rhs
        if clip_during_simulation:
            c = np.clip(c, 0.0, 1.0)
        if step % int(save_every) == 0 or step == n_steps:
            frames.append(c.copy())
            times.append(min(step * stable_dt, float(tau_max)))
    return np.asarray(times, dtype=np.float64), np.asarray(frames, dtype=np.float64)


def _first_invalid_frame(surface: np.ndarray, u_min: float, u_max: float) -> int | None:
    valid = (np.min(surface, axis=1) >= float(u_min)) & (np.max(surface, axis=1) <= float(u_max))
    invalid = np.flatnonzero(~valid)
    return int(invalid[0]) if invalid.size else None


def _svd_denoise(surface: np.ndarray, rank: int) -> np.ndarray:
    if rank <= 0:
        return surface
    mean = surface.mean(axis=0, keepdims=True)
    centered = surface - mean
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(int(rank), len(s))
    return (u[:, :k] * s[:k]) @ vt[:k, :] + mean


def _gaussian_denoise(surface: np.ndarray, sigma_t: float, sigma_x: float) -> np.ndarray:
    if sigma_t <= 0.0 and sigma_x <= 0.0:
        return surface
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return surface
    return gaussian_filter(surface, sigma=(float(sigma_t), float(sigma_x)), mode="nearest")


def _denoise(noisy: np.ndarray, mode: str, rank: int, sigma_t: float, sigma_x: float) -> np.ndarray:
    if mode == "none":
        return noisy
    out = np.empty_like(noisy)
    for i in range(noisy.shape[0]):
        surface = noisy[i, :, :, 0]
        if mode in {"svd", "svd_gaussian"}:
            surface = _svd_denoise(surface, rank)
        if mode in {"gaussian", "svd_gaussian"}:
            surface = _gaussian_denoise(surface, sigma_t, sigma_x)
        out[i, :, :, 0] = surface
    return out


def make_sib_diffusion_raw_field_dataset(
    *,
    save_dir: str,
    noise_level: float = 0.03,
    batch_size: int = 32,
    seed: int = 0,
    dataset_name: str = "sib_diffusion_raw_field",
    phases: Iterable[str] | str = ("charge", "discharge"),
    deltas: Iterable[float] = (0.25,),
    initial_profiles: Iterable[str] | str = (
        "smooth_quadratic",
        "surface_layer",
        "sin1",
        "sin2",
        "center_bump",
        "surface_bump",
        "mid_bump",
    ),
    nx: int = 81,
    tau_max: float = 1.0,
    dtau: float = 1.0e-4,
    save_every: int = 100,
    dependent_transform: str = "concentration",
    geometry_channel: bool = False,
    geometry_channel_scale: float = 2.0,
    field_denoise_mode: str = "svd_gaussian",
    field_denoise_rank: int = 4,
    field_smooth_sigma_t: float = 4.0,
    field_smooth_sigma_x: float = 3.5,
    clip_during_simulation: bool = True,
    crop_to_physical_window: bool = False,
    physical_u_min: float = 0.0,
    physical_u_max: float = 1.0,
    **_: object,
) -> None:
    if dependent_transform != "concentration":
        raise ValueError("This case intentionally keeps the raw concentration field; use dependent_transform='concentration'.")
    if geometry_channel:
        raise ValueError("This case does not expose geometry or inverse-radius channels to WiSED.")

    rng = np.random.default_rng(int(seed))
    x = np.linspace(0.0, 1.0, int(nx), dtype=np.float64)
    phases_t = _as_tuple(phases)
    profiles_t = _as_tuple(initial_profiles)
    deltas_t = tuple(float(d) for d in deltas)

    clean_surfaces = []
    field_meta = []
    tau_ref = None
    for delta in deltas_t:
        for phase in phases_t:
            for profile_name in profiles_t:
                tau, surface = _simulate_surface(
                    x,
                    phase=phase,
                    delta=delta,
                    profile_name=profile_name,
                    tau_max=tau_max,
                    dtau=dtau,
                    save_every=save_every,
                    clip_during_simulation=bool(clip_during_simulation),
                )
                if tau_ref is None:
                    tau_ref = tau
                elif len(tau) != len(tau_ref) or float(np.max(np.abs(tau - tau_ref))) > 1.0e-12:
                    raise RuntimeError("SIB generated inconsistent time coordinates.")
                clean_surfaces.append(surface[:, :, None])
                field_meta.append({"phase": phase, "delta": delta, "initial_profile": profile_name})

    physical_keep_frames = len(tau_ref)
    first_invalid_times = []
    if bool(crop_to_physical_window):
        for surface in clean_surfaces:
            invalid_idx = _first_invalid_frame(surface[:, :, 0], physical_u_min, physical_u_max)
            if invalid_idx is not None:
                first_invalid_times.append(float(tau_ref[invalid_idx]))
                physical_keep_frames = min(physical_keep_frames, max(1, invalid_idx))
        min_required_frames = 9
        if physical_keep_frames < min_required_frames:
            raise RuntimeError(
                "Physical-window crop left too few time frames. "
                f"keep_frames={physical_keep_frames}, required>={min_required_frames}."
            )
        tau_ref = tau_ref[:physical_keep_frames]
        clean_surfaces = [surface[:physical_keep_frames] for surface in clean_surfaces]

    clean = np.asarray(clean_surfaces, dtype=np.float32)
    noisy = clean.copy()
    if float(noise_level) > 0.0:
        scale = np.std(clean, axis=(1, 2, 3), keepdims=True)
        scale = np.maximum(scale, 1.0e-8)
        noisy = noisy + rng.normal(size=noisy.shape).astype(np.float32) * float(noise_level) * scale
    data = _denoise(
        noisy,
        mode=str(field_denoise_mode),
        rank=int(field_denoise_rank),
        sigma_t=float(field_smooth_sigma_t),
        sigma_x=float(field_smooth_sigma_x),
    ).astype(np.float32)

    n_surfaces = int(data.shape[0])
    n_train = min(int(batch_size), n_surfaces)
    discovery_indices = list(range(n_train))
    grid_info = {
        "t_coords": np.asarray(tau_ref, dtype=np.float64),
        "x_coords": x,
        "field_names": ["u"],
        "discovery_targets": ["u"],
        "periodic_axes": {"x": False},
        "splits": {"train": discovery_indices, "test": []},
        "discovery_indices": discovery_indices,
        "field_meta": field_meta,
        "preprocess": {
            "preprocess_version": PREPROCESS_VERSION,
            "manual_derivative_terminals_exported": False,
            "candidate_library_used": False,
            "clip_during_simulation": bool(clip_during_simulation),
            "crop_to_physical_window": bool(crop_to_physical_window),
            "physical_u_min": float(physical_u_min),
            "physical_u_max": float(physical_u_max),
            "full_tau_max": float(tau_max),
            "physical_window_tau_max": float(tau_ref[-1]),
            "physical_window_n_frames": int(len(tau_ref)),
            "first_invalid_tau_min": min(first_invalid_times) if first_invalid_times else None,
            "dependent_transform": "concentration",
            "geometry_channel": False,
            "geometry_channel_scale": float(geometry_channel_scale),
            "nx": int(nx),
            "tau_max": float(tau_max),
            "dtau": float(dtau),
            "save_every": int(save_every),
            "noise_level": float(noise_level),
            "deltas": list(deltas_t),
            "phases": list(phases_t),
            "initial_profiles": list(profiles_t),
            "field_denoise_mode": str(field_denoise_mode),
            "field_denoise_rank": int(field_denoise_rank),
            "field_smooth_sigma_t": float(field_smooth_sigma_t),
            "field_smooth_sigma_x": float(field_smooth_sigma_x),
            "n_surfaces": n_surfaces,
        },
    }

    save_path = Path(save_dir) / f"{dataset_name}_noise{float(noise_level)}.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        data=data,
        data_clean=clean.astype(np.float32),
        grid_info=grid_info,
        true_eq=TRUE_EQUATION,
    )
    print(f"--- SIB raw concentration field dataset saved to {save_path.name} ---")
    print(f"    data shape: {data.shape} | fields: ['u'] | discovery surfaces: {n_train}/{n_surfaces}")
