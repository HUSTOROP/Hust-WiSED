from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_SELECTED_FREQUENCIES = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_FEATURE_KINDS = ("R", "X", "Zabs", "Zphase")
DEFAULT_FEATURE_MODE = "physics"



def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _float_tag(value: float) -> str:
    return f"{float(value):g}"


def _source_label(path: Path, *, row_index: Optional[int] = None) -> str:
    """Return a non-path source label safe for persisted metadata."""
    label = path.name
    if row_index is not None:
        label = f"{label}::row_{int(row_index)}"
    return label


def _read_csv_auto(path: Path) -> pd.DataFrame:
    """Read a CSV from the public dataset with tolerant delimiter handling."""
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def _canonical_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Map flexible dataset headers to canonical names.

    The public Mendeley release uses unit-qualified headers such as
    ``Frequency(Hz)``, ``R(ohm)``, ``X(ohm)``, ``V(V)``,
    ``T(deg C)`` and ``Range(Ohm)``.  Earlier versions of this
    converter expected the unit-free names from the paper table.  This
    function deliberately strips units, spaces and punctuation so both
    formats are accepted.
    """

    def norm(name: Any) -> str:
        # Keep only letters and digits so, for example:
        #   Frequency(Hz) -> frequencyhz
        #   R(ohm)       -> rohm
        #   T(deg C)     -> tdegc
        return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower().replace("\ufeff", ""))

    normalized: Dict[str, str] = {norm(c): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for name in names:
            key = norm(name)
            if key in normalized:
                return normalized[key]
        return None

    out = {
        "Frequency": pick("Frequency", "Frequency(Hz)", "Freq", "Freq(Hz)", "f", "Hz"),
        "R": pick("R", "R(ohm)", "R(Ohm)", "Real", "Real(Ohm)", "Re", "ReZ", "RealZ", "Real(Z)", "Zreal"),
        "X": pick("X", "X(ohm)", "X(Ohm)", "Imaginary", "Imaginary(Ohm)", "Im", "ImZ", "ImagZ", "Imaginary(Z)", "Zimag"),
        "V": pick("V", "V(V)", "Voltage", "BatteryVoltage", "Battery Voltage"),
        "T": pick("T", "T(deg C)", "T(degC)", "Temperature", "Temp", "Temperature(C)", "Temperature(deg C)"),
        "Range": pick("Range", "Range(Ohm)", "Range(ohm)"),
    }
    required = ["Frequency", "R", "X"]
    missing = [key for key in required if out.get(key) is None]
    if missing:
        raise ValueError(f"Missing required EIS columns {missing}; found columns={list(df.columns)}")
    return {k: v for k, v in out.items() if v is not None}


def _parse_soc_from_path(path: Path) -> float:
    text = " ".join(path.parts[-4:])
    patterns = [
        r"(?:SoC|SOC|soc)[_\-\s]*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*%",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            value = float(m.group(1))
            return value / 100.0 if value > 1.0 else value
    raise ValueError(f"Could not parse SoC value from file path: {path}")


def _parse_battery_id(path: Path) -> str:
    for part in path.parts:
        if re.fullmatch(r"B\d{1,3}", part, flags=re.IGNORECASE):
            return part.upper()
    m = re.search(r"(B\d{1,3})", str(path), flags=re.IGNORECASE)
    return m.group(1).upper() if m else "UNKNOWN_BATTERY"


def _parse_cycle_id(path: Path) -> str:
    for part in path.parts:
        if re.fullmatch(r"Test[_\- ]?\d+", part, flags=re.IGNORECASE):
            return part.replace(" ", "_")
    m = re.search(r"(Test[_\- ]?\d+)", str(path), flags=re.IGNORECASE)
    return m.group(1).replace(" ", "_") if m else "UNKNOWN_TEST"


def _is_eis_csv(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    lower_parts = [p.lower() for p in path.parts]
    if any("capacity" in p for p in lower_parts):
        return False
    if not any("eis" in p or "hioki" in p for p in lower_parts):
        # Some extracted datasets place the files directly under Test_N/Hioki.
        # We still allow them when the file name contains SoC.
        return bool(re.search(r"soc", path.name, flags=re.IGNORECASE))
    return True


def discover_eis_files(raw_dir: Path) -> List[Path]:
    files = sorted(p for p in raw_dir.rglob("*.csv") if _is_eis_csv(p))
    return files


def _find_author_matrix_files(raw_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Find the author's released matrix CSV and frequency CSV.

    The authors' public code reads a single file named like
    ``Data_IFR14500_Batt_1-11_cycle_1_2.csv`` plus ``Frequencies.csv``.
    This helper keeps our engineering-validation split exactly aligned with
    that code when those files are present, while preserving the legacy
    recursive per-spectrum CSV reader as a fallback.
    """
    data_candidates = sorted(raw_dir.rglob("Data_IFR14500_Batt_1-11_cycle_1_2.csv"))
    if not data_candidates:
        data_candidates = sorted(raw_dir.rglob("Data_IFR*.csv"))
    freq_candidates = sorted(raw_dir.rglob("Frequencies.csv"))
    return (data_candidates[0] if data_candidates else None, freq_candidates[0] if freq_candidates else None)


def _load_author_matrix_spectra(raw_dir: Path) -> List[Dict[str, Any]]:
    """Load spectra from the exact matrix format used by the paper authors.

    Rows are spectra; columns are f1-Re, f1-Im, ..., f28-Re, f28-Im, SoC,
    Battery, cycle.  The frequency grid is stored in Frequencies.csv.
    """
    data_path, freq_path = _find_author_matrix_files(raw_dir)
    if data_path is None or freq_path is None:
        return []
    df = pd.read_csv(data_path, sep=",")
    freq_df = pd.read_csv(freq_path, sep=",")
    if "frequencies" not in freq_df.columns:
        # Be permissive about a unit-qualified header.
        freq_col = freq_df.columns[0]
    else:
        freq_col = "frequencies"
    freq = pd.to_numeric(freq_df[freq_col], errors="coerce").to_numpy(dtype=np.float64)
    n_freq = int(freq.size)
    spectra: List[Dict[str, Any]] = []
    for row_idx, row in df.iterrows():
        try:
            R = np.asarray([float(row[f"f{i}-Re"]) for i in range(1, n_freq + 1)], dtype=np.float64)
            X = np.asarray([float(row[f"f{i}-Im"]) for i in range(1, n_freq + 1)], dtype=np.float64)
            soc_raw = float(row["SoC"])
            soc = soc_raw / 100.0 if soc_raw > 1.0 else soc_raw
            battery = str(row.get("Battery", "UNKNOWN_BATTERY"))
            cycle = f"Test_{int(row.get('cycle', 0))}" if np.isfinite(float(row.get("cycle", 0))) else str(row.get("cycle", "UNKNOWN_TEST"))
            valid = np.isfinite(freq) & np.isfinite(R) & np.isfinite(X) & (freq > 0)
            ff, rr, xx = freq[valid], R[valid], X[valid]
            order = np.argsort(ff)
            spectra.append({
                "source_label": _source_label(data_path, row_index=int(row_idx)),
                "freq": ff[order],
                "R": rr[order],
                "X": xx[order],
                "V": float("nan"),
                "T": float("nan"),
                "soc": float(soc),
                "battery_id": battery,
                "cycle_id": cycle,
                "author_row_index": int(row_idx),
            })
        except Exception:
            continue
    return spectra


def _parse_cycle_number(path: Path) -> int:
    """Return the numeric Test_N cycle id used by the author's matrix."""
    for part in path.parts:
        m = re.fullmatch(r"Test[_\- ]?(\d+)", part, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    m = re.search(r"Test[_\- ]?(\d+)", str(path), flags=re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _battery_sort_value(battery_id: str) -> Tuple[int, str]:
    m = re.fullmatch(r"B(\d+)", str(battery_id).upper())
    return (int(m.group(1)), str(battery_id).upper()) if m else (10**9, str(battery_id))


def _load_raw_spectrum_as_author_record(path: Path) -> Dict[str, Any]:
    """Load one original per-spectrum CSV and return an author-matrix-like row.

    This is the in-memory equivalent of building one row of
    Data_IFR14500_Batt_1-11_cycle_1_2.csv.  It deliberately does not write any
    intermediate CSV files; it only preserves the same row semantics used by the
    paper author's train_test_split code: columns f1-Re/f1-Im/... plus SoC,
    Battery, and cycle.
    """
    df = _read_csv_auto(path)
    cols = _canonical_columns(df)
    freq = pd.to_numeric(df[cols["Frequency"]], errors="coerce").to_numpy(dtype=np.float64)
    R = pd.to_numeric(df[cols["R"]], errors="coerce").to_numpy(dtype=np.float64)
    X = pd.to_numeric(df[cols["X"]], errors="coerce").to_numpy(dtype=np.float64)
    V = float("nan")
    T = float("nan")
    if "V" in cols:
        vv = pd.to_numeric(df[cols["V"]], errors="coerce").to_numpy(dtype=np.float64)
        V = float(np.nanmedian(vv)) if np.isfinite(vv).any() else float("nan")
    if "T" in cols:
        tt = pd.to_numeric(df[cols["T"]], errors="coerce").to_numpy(dtype=np.float64)
        T = float(np.nanmedian(tt)) if np.isfinite(tt).any() else float("nan")
    good = np.isfinite(freq) & np.isfinite(R) & np.isfinite(X) & (freq > 0)
    if int(good.sum()) < 2:
        raise ValueError(f"Too few valid EIS rows in {path}")
    freq, R, X = freq[good], R[good], X[good]
    order = np.argsort(freq)
    soc = float(_parse_soc_from_path(path))
    return {
        "source_label": _source_label(path),
        "freq": freq[order],
        "R": R[order],
        "X": X[order],
        "V": V,
        "T": T,
        "soc": soc,
        "battery_id": _parse_battery_id(path),
        "cycle_id": f"Test_{_parse_cycle_number(path)}",
        "cycle_number": int(_parse_cycle_number(path)),
        "source_filename": path.name,
    }


def _frequency_grid_key(freq: np.ndarray) -> Tuple[float, ...]:
    return tuple(np.round(np.log10(np.asarray(freq, dtype=np.float64)), 8).tolist())


def _load_author_matrix_from_raw_in_memory(raw_dir: Path, *, strict_frequency_grid: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Convert original raw EIS CSV folders into author-style records in memory.

    The user still points ``--raw-dir`` to the original extracted B01..B11 files.
    This function performs the same logical conversion as the helper script that
    writes Data_IFR14500_Batt_1-11_cycle_1_2.csv, but it returns spectra directly
    to the engineering-validation pipeline and never creates intermediate files.
    """
    files = discover_eis_files(raw_dir)
    records: List[Dict[str, Any]] = []
    skipped: List[Tuple[str, str]] = []
    for fp in files:
        try:
            records.append(_load_raw_spectrum_as_author_record(fp))
        except Exception as exc:
            skipped.append((_source_label(fp), str(exc)))
    if not records:
        return [], {
            "raw_conversion": "author_matrix_from_raw_in_memory",
            "n_skipped_files": len(skipped),
            "skipped_files_preview": skipped[:10],
        }

    # Match the author matrix row semantics as closely as possible: battery,
    # cycle, then lexicographic file name.  This is the order used by the
    # standalone builder, and train_test_split(random_state=17, stratify=label)
    # is deterministic with respect to this row order.
    records.sort(key=lambda r: (_battery_sort_value(str(r["battery_id"])), int(r.get("cycle_number", 0)), str(r.get("source_filename") or r.get("source_label", ""))))

    # The released dataset uses a common frequency vector.  If some raw files do
    # not match, keep the modal grid to avoid interpolation artifacts unless the
    # caller explicitly wants a hard failure.
    grid_counts: Dict[Tuple[float, ...], int] = {}
    for rec in records:
        key = _frequency_grid_key(rec["freq"])
        grid_counts[key] = grid_counts.get(key, 0) + 1
    modal_key = max(grid_counts, key=grid_counts.get)
    if strict_frequency_grid and len(grid_counts) > 1:
        raise ValueError(f"Original raw EIS files contain {len(grid_counts)} frequency grids; expected one common grid.")
    kept = [rec for rec in records if _frequency_grid_key(rec["freq"]) == modal_key]
    dropped = len(records) - len(kept)

    summary = {
        "raw_conversion": "author_matrix_from_raw_in_memory",
        "n_original_csv_files": len(files),
        "n_loaded_spectra": len(records),
        "n_kept_modal_frequency_grid": len(kept),
        "n_dropped_nonmodal_frequency_grid": int(dropped),
        "n_skipped_files": len(skipped),
        "skipped_files_preview": skipped[:10],
        "batteries": sorted({str(r["battery_id"]) for r in kept}, key=_battery_sort_value),
        "cycles": sorted({int(r.get("cycle_number", 0)) for r in kept}),
        "soc_values_percent": sorted({int(round(float(r["soc"]) * 100.0)) for r in kept}),
    }
    return kept, summary


def _author_soc10_label(soc: float) -> int:
    """Replicate the author's 10-class label mapping.

    Their code stores SoC as integer percentages and maps odd 5%-offset labels
    upward: 5 -> 10, 15 -> 20, ..., 95 -> 100, while 10,20,... remain fixed.
    """
    pct = int(round(float(soc) * 100.0))
    if pct % 2 != 0:
        pct += 5
    return int(max(10, min(100, pct)))


# -----------------------------------------------------------------------------
# 2D impedance-surface dataset for open equation discovery

# -----------------------------------------------------------------------------
SPECTRAL_PREPROCESS_VERSION = "soc_eis_impedance_manifold"


def _unique_sorted_xy(x: np.ndarray, *ys: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Return sorted finite x/y arrays with duplicate x values averaged.

    The raw Hioki files should not contain duplicate frequencies, but this
    guard keeps the converter stable when users pass partially edited CSVs.
    """
    x = np.asarray(x, dtype=np.float64)
    y_arrays = [np.asarray(y, dtype=np.float64) for y in ys]
    good = np.isfinite(x)
    for y in y_arrays:
        good &= np.isfinite(y)
    x = x[good]
    y_arrays = [y[good] for y in y_arrays]
    order = np.argsort(x)
    x = x[order]
    y_arrays = [y[order] for y in y_arrays]
    if x.size == 0:
        return (x, *y_arrays)
    ux, inv = np.unique(x, return_inverse=True)
    if ux.size == x.size:
        return (x, *y_arrays)
    out = [ux]
    for y in y_arrays:
        yy = np.zeros_like(ux, dtype=np.float64)
        cnt = np.zeros_like(ux, dtype=np.float64)
        np.add.at(yy, inv, y)
        np.add.at(cnt, inv, 1.0)
        out.append(yy / np.maximum(cnt, 1.0))
    return tuple(out)


def _voigt_rc_tau_grid(freq_hz: np.ndarray, n_rc_elements: int, tau_margin_decades: float = 0.5) -> np.ndarray:
    """Fixed relaxation-time grid for a Voigt RC approximation.

    The grid spans slightly beyond the measured frequency window.  It is used
    only as a physically admissible smoothing/interpolation basis, not as the
    candidate library for the subsequent equation-discovery step.
    """
    f = np.asarray(freq_hz, dtype=np.float64)
    f = f[np.isfinite(f) & (f > 0.0)]
    if f.size < 2:
        raise ValueError("At least two positive frequencies are required for RC smoothing.")
    w_min = 2.0 * np.pi * float(np.min(f))
    w_max = 2.0 * np.pi * float(np.max(f))
    margin = float(max(0.0, tau_margin_decades))
    tau_min = (10.0 ** (-margin)) / max(w_max, 1.0e-30)
    tau_max = (10.0 ** (margin)) / max(w_min, 1.0e-30)
    return np.logspace(np.log10(tau_min), np.log10(tau_max), int(max(2, n_rc_elements)))


def _voigt_rc_design(omega: np.ndarray, taus: np.ndarray, *, include_series_inductance: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Real/imaginary design matrices for R0 + sum_k Rk/(1+j*w*tau_k) [+ j*wL]."""
    omega = np.asarray(omega, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    z = omega[:, None] * taus[None, :]
    denom = 1.0 + z * z
    re_cols = [np.ones((omega.size, 1), dtype=np.float64), 1.0 / denom]
    im_cols = [np.zeros((omega.size, 1), dtype=np.float64), -z / denom]
    if include_series_inductance:
        # The column is scaled to ohm-sized coefficients for numerical stability.
        w_scale = max(float(np.nanmax(np.abs(omega))), 1.0e-30)
        re_cols.append(np.zeros((omega.size, 1), dtype=np.float64))
        im_cols.append((omega / w_scale)[:, None])
    return np.hstack(re_cols), np.hstack(im_cols)


def _rc_fit_and_evaluate(
    freq_hz: np.ndarray,
    R: np.ndarray,
    X: np.ndarray,
    x_coords: np.ndarray,
    *,
    n_rc_elements: int = 12,
    include_series_inductance: bool = True,
    tau_margin_decades: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Jointly smooth R and X with a K-K-consistent Voigt RC basis.

    This is a preprocessing filter.  It enforces the basic causal coupling of
    the real and imaginary impedance components during resampling to a uniform
    log-frequency grid, while the downstream WiSED search remains unrestricted.
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    x_coords = np.asarray(x_coords, dtype=np.float64)
    logf, R, X = _unique_sorted_xy(np.log10(freq_hz), R, X)
    freq_hz = 10.0 ** logf
    omega = 2.0 * np.pi * freq_hz
    taus = _voigt_rc_tau_grid(freq_hz, n_rc_elements, tau_margin_decades)
    A_re, A_im = _voigt_rc_design(omega, taus, include_series_inductance=bool(include_series_inductance))
    A = np.vstack([A_re, A_im])
    y = np.concatenate([R, X])

    try:
        from scipy.optimize import nnls

        coeff, _ = nnls(A, y)
    except Exception:
        # Fallback keeps the converter usable in minimal environments.  The
        # clipped least-squares solution is less strict than NNLS but preserves
        # the same joint complex basis.
        coeff = np.linalg.lstsq(A, y, rcond=None)[0]
        coeff = np.maximum(coeff, 0.0)

    raw_fit = A @ coeff
    R_fit = raw_fit[: R.size]
    X_fit = raw_fit[R.size :]

    omega_eval = 2.0 * np.pi * (10.0 ** x_coords)
    E_re, E_im = _voigt_rc_design(omega_eval, taus, include_series_inductance=bool(include_series_inductance))
    R_eval = E_re @ coeff
    X_eval = E_im @ coeff

    absZ = np.maximum(np.hypot(R, X), 1.0e-12)
    d_re = (R - R_fit) / absZ
    d_im = (X - X_fit) / absZ
    diag = {
        "mode": "rc_voigt_joint",
        "n_raw_frequency_points": int(freq_hz.size),
        "n_rc_elements": int(max(2, n_rc_elements)),
        "include_series_inductance": bool(include_series_inductance),
        "rc_rel_rms_re_percent": float(100.0 * np.sqrt(np.mean(d_re * d_re))),
        "rc_rel_rms_im_percent": float(100.0 * np.sqrt(np.mean(d_im * d_im))),
        "rc_rel_rms_complex_percent": float(100.0 * np.sqrt(np.mean(d_re * d_re + d_im * d_im))),
        "rc_max_abs_re_mohm": float(1000.0 * np.max(np.abs(R - R_fit))),
        "rc_max_abs_im_mohm": float(1000.0 * np.max(np.abs(X - X_fit))),
        "rc_r0_ohm": float(coeff[0]) if coeff.size else float("nan"),
    }
    if include_series_inductance and coeff.size >= int(max(2, n_rc_elements)) + 2:
        diag["series_inductance_scaled_ohm"] = float(coeff[-1])
    return R_eval.astype(np.float64), X_eval.astype(np.float64), diag


def _interp_or_spline_evaluate(
    freq_hz: np.ndarray,
    R: np.ndarray,
    X: np.ndarray,
    x_coords: np.ndarray,
    *,
    mode: str,
    spline_smoothing: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Synchronized non-RC frequency preprocessing for ablation studies."""
    logf, R, X = _unique_sorted_xy(np.log10(freq_hz), R, X)
    if logf.size < 2:
        raise ValueError("At least two valid frequencies are required for interpolation.")
    mode = str(mode).lower()
    if mode == "pchip":
        try:
            from scipy.interpolate import PchipInterpolator

            R_eval = PchipInterpolator(logf, R, extrapolate=False)(x_coords)
            X_eval = PchipInterpolator(logf, X, extrapolate=False)(x_coords)
        except Exception:
            R_eval = np.interp(x_coords, logf, R)
            X_eval = np.interp(x_coords, logf, X)
    elif mode == "spline":
        try:
            from scipy.interpolate import UnivariateSpline

            # Same smoothing policy for both channels.  The smoothing value is
            # scaled by each channel variance so that R and X are not treated as
            # unrelated preprocessing problems.
            sR = float(max(0.0, spline_smoothing)) * max(float(np.var(R)), 1.0e-18) * float(logf.size)
            sX = float(max(0.0, spline_smoothing)) * max(float(np.var(X)), 1.0e-18) * float(logf.size)
            k = int(min(3, max(1, logf.size - 1)))
            R_eval = UnivariateSpline(logf, R, s=sR, k=k)(x_coords)
            X_eval = UnivariateSpline(logf, X, s=sX, k=k)(x_coords)
        except Exception:
            R_eval = np.interp(x_coords, logf, R)
            X_eval = np.interp(x_coords, logf, X)
    else:
        R_eval = np.interp(x_coords, logf, R)
        X_eval = np.interp(x_coords, logf, X)

    diag = {
        "mode": mode,
        "n_raw_frequency_points": int(logf.size),
        "rc_rel_rms_re_percent": None,
        "rc_rel_rms_im_percent": None,
        "rc_rel_rms_complex_percent": None,
    }
    return np.asarray(R_eval, dtype=np.float64), np.asarray(X_eval, dtype=np.float64), diag


def _evaluate_frequency_spectrum(
    row: Dict[str, Any],
    x_coords: np.ndarray,
    *,
    freq_preprocess_mode: str,
    n_rc_elements: int,
    include_series_inductance: bool,
    spline_smoothing: float,
    frequency_min_hz: Optional[float] = None,
    frequency_max_hz: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Evaluate one complex spectrum on the common log-frequency grid."""
    freq = np.asarray(row["freq"], dtype=np.float64)
    R_raw = np.asarray(row["R"], dtype=np.float64)
    X_raw = np.asarray(row["X"], dtype=np.float64)
    band_mask = np.isfinite(freq) & np.isfinite(R_raw) & np.isfinite(X_raw) & (freq > 0.0)
    if frequency_min_hz is not None:
        band_mask &= freq >= float(frequency_min_hz) * (1.0 - 1.0e-12)
    if frequency_max_hz is not None:
        band_mask &= freq <= float(frequency_max_hz) * (1.0 + 1.0e-12)
    if int(np.count_nonzero(band_mask)) < 4:
        raise ValueError(
            f"Too few frequencies remain for {row.get('battery_id')} {row.get('cycle_id')} "
            f"SoC={row.get('soc')} after applying band [{frequency_min_hz}, {frequency_max_hz}] Hz."
        )
    freq = freq[band_mask]
    R_raw = R_raw[band_mask]
    X_raw = X_raw[band_mask]
    mode = str(freq_preprocess_mode or "rc").lower()
    if mode in {"rc", "rc_voigt", "lin_kk", "kk", "phys"}:
        R_eval, X_eval, diag = _rc_fit_and_evaluate(
            freq,
            R_raw,
            X_raw,
            x_coords,
            n_rc_elements=int(n_rc_elements),
            include_series_inductance=bool(include_series_inductance),
        )
    else:
        R_eval, X_eval, diag = _interp_or_spline_evaluate(
            freq,
            R_raw,
            X_raw,
            x_coords,
            mode=mode,
            spline_smoothing=float(spline_smoothing),
        )
    diag.update({
        "battery_id": str(row.get("battery_id", "")),
        "cycle_id": str(row.get("cycle_id", "")),
        "soc": float(row.get("soc", float("nan"))),
        "source": str(row.get("source_filename") or row.get("source_label", "")),
    })
    return R_eval, -X_eval, diag


def _smooth_along_soc_axis(arr: np.ndarray, *, window: int, polyorder: int) -> np.ndarray:
    """Mild SoC-direction smoothing after SoC=100% has been removed."""
    arr = np.asarray(arr, dtype=np.float64)
    nt = int(arr.shape[0])
    win = int(window)
    if win <= 2 or nt < 5:
        return arr.copy()
    if win % 2 == 0:
        win += 1
    win = min(win, nt if nt % 2 == 1 else nt - 1)
    if win <= int(polyorder) or win < 3:
        return arr.copy()
    try:
        from scipy.signal import savgol_filter

        return np.asarray(savgol_filter(arr, window_length=win, polyorder=int(polyorder), axis=0, mode="interp"), dtype=np.float64)
    except Exception:
        return arr.copy()


def _summarize_spectrum_diagnostics(diags: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compact statistics for preprocessing diagnostics."""
    out: Dict[str, Any] = {"n_spectra": int(len(diags))}
    for key in ("rc_rel_rms_re_percent", "rc_rel_rms_im_percent", "rc_rel_rms_complex_percent", "rc_max_abs_re_mohm", "rc_max_abs_im_mohm"):
        vals = np.asarray([d.get(key, np.nan) for d in diags if d.get(key) is not None], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[key] = {
                "median": float(np.median(vals)),
                "p95": float(np.percentile(vals, 95)),
                "max": float(np.max(vals)),
            }
    if diags:
        worst = sorted(
            [d for d in diags if d.get("rc_rel_rms_complex_percent") is not None],
            key=lambda d: float(d.get("rc_rel_rms_complex_percent", -np.inf)),
            reverse=True,
        )[:8]
        out["worst_rc_fits_preview"] = [
            {
                "battery_id": d.get("battery_id"),
                "cycle_id": d.get("cycle_id"),
                "soc_percent": None if d.get("soc") is None else float(d.get("soc")) * 100.0,
                "complex_rms_percent": d.get("rc_rel_rms_complex_percent"),
                "source": d.get("source"),
            }
            for d in worst
        ]
    return out


def _resolve_raw_dir(raw_dir: Optional[str] = None) -> Path:
    """Resolve the extracted SoC-EIS raw-data directory.

    The engineering validation expects the user's original extracted dataset
    directory, not a zip file.  Valid roots include the full raw root containing
    B01..B11, a Measurements folder, or a single battery folder such as B01.
    """
    if raw_dir is not None and str(raw_dir).strip():
        p = Path(raw_dir).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Raw SoC-EIS dataset directory not found: {p}")
        if not p.is_dir():
            raise NotADirectoryError(
                f"Raw SoC-EIS path must be an extracted directory, not a file: {p}."
            )
        return p

    root = _project_root()
    candidates = [
        root / "engineering_validation" / "soc_eis_impedance_manifold" / "data",
        root / "engineering_validation" / "soc_eis_impedance_manifold" / "data" / "Measurements",
        root / "data" / "soc_eis" / "raw",
        root / "data" / "soc_eis" / "raw" / "Measurements",
    ]
    for p in candidates:
        if p.exists() and p.is_dir() and any(p.rglob("*.csv")):
            return p
    return candidates[0]


def _split_surface_fields(
    field_meta: Sequence[Dict[str, Any]],
    *,
    split_mode: str,
    seed: int,
    val_battery_ids: Optional[Sequence[str]] = None,
    test_battery_ids: Optional[Sequence[str]] = None,
    train_cycle_ids: Optional[Sequence[str]] = None,
    val_cycle_ids: Optional[Sequence[str]] = None,
    test_cycle_ids: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Split complete battery-cycle impedance surfaces without spectrum-level mixing.

    The PDE-like impedance-surface task treats one complete battery-cycle
    ``Z(log f, discharge_progress)`` surface as the atomic sample.  Therefore
    the SoC spectra inside a surface are never shuffled independently.  This is
    deliberately different from the paper author's SVM split, which is a
    spectrum-level stratified random split for supervised classification.

    Recommended modes:
      * battery_holdout: train on earlier battery IDs, validate on one held-out
        battery, and test on the last two battery IDs.
      * leave_one_battery_out: train on all batteries except one explicit
    test_battery_id. This is the retained strict out-of-distribution protocol.
      * cycle_holdout: train on one discharge cycle and test on the other.
      * paper_like: kept only as a CLI-compatible alias for battery_holdout when
        multiple batteries are available; it is not the author's SVM split.
    """
    n = len(field_meta)
    idx = np.arange(n, dtype=int)
    batteries = np.asarray([m.get("battery_id", "UNKNOWN") for m in field_meta], dtype=object)
    cycles = np.asarray([m.get("cycle_id", "UNKNOWN") for m in field_meta], dtype=object)
    mode = str(split_mode or "battery_holdout").lower()

    if mode in {"leave_one_battery_out", "lobo", "battery_lobo"}:
        all_bats = sorted(set(str(b) for b in batteries.tolist()), key=_battery_sort_value)
        test_bats = set(str(b).strip() for b in (test_battery_ids or []) if str(b).strip())
        if not test_bats:
            raise ValueError(
                "leave_one_battery_out requires one explicit test_battery_ids entry. "
                f"Available batteries={all_bats}."
            )
        unknown = sorted(test_bats - set(all_bats), key=_battery_sort_value)
        if unknown:
            raise ValueError(f"Unknown LOBO test batteries={unknown}; available={all_bats}.")
        val_bats = set(str(b).strip() for b in (val_battery_ids or []) if str(b).strip())
        val_bats -= test_bats
        if not val_bats:
            train_candidates = [b for b in all_bats if b not in test_bats]
            if len(train_candidates) >= 2:
                val_bats = {train_candidates[-1]}
        train_mask = (~np.isin(batteries, list(test_bats))) & (~np.isin(batteries, list(val_bats)))
        val_mask = np.isin(batteries, list(val_bats))
        test_mask = np.isin(batteries, list(test_bats))
        train = idx[train_mask]
        test = idx[test_mask]
        if train.size == 0 or test.size == 0:
            raise ValueError(
                "leave_one_battery_out produced an empty train or test split. "
                f"test_battery_ids={sorted(test_bats)}, val_battery_ids={sorted(val_bats)}."
            )
        return {"train": np.sort(train), "val": np.sort(idx[val_mask]), "test": np.sort(test)}

    # Explicit repeated-test transfer: discover on one named cycle, validate/test
    # on another named cycle from the same battery.  This is the first genuinely
    # falsifiable SoC-EIS law test: the discovered equation no longer sees the
    # held-out repeat measurement during symbolic search.
    if mode in {"fixed_cycle_transfer", "cross_cycle", "test1_to_test2", "repeat_holdout"}:
        cyc_norm = np.asarray([_normalize_cycle_id(c) for c in cycles.tolist()], dtype=object)
        train_set = set(_normalize_cycle_id(c) for c in (train_cycle_ids or ["Test_1"]))
        test_set = set(_normalize_cycle_id(c) for c in (test_cycle_ids or ["Test_2"]))
        val_set = set(_normalize_cycle_id(c) for c in (val_cycle_ids or []))
        train = idx[np.isin(cyc_norm, list(train_set))]
        val = idx[np.isin(cyc_norm, list(val_set))] if val_set else np.empty(0, dtype=int)
        test = idx[np.isin(cyc_norm, list(test_set))]
        if train.size == 0 or test.size == 0:
            raise ValueError(
                "fixed_cycle_transfer requires at least one assembled train and test surface. "
                f"Available cycles={sorted(set(cyc_norm.tolist()))}, "
                f"train_cycle_ids={sorted(train_set)}, test_cycle_ids={sorted(test_set)}."
            )
        return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}

    # Minimal-loop and debugging runs: train on every assembled surface.  This
    # is intentionally not a generalization protocol; it is the smallest closed
    # loop used to test whether the discovered local spectral--SoC relation is
    # coherent before cross-cycle or cross-battery transfer is attempted.
    if mode in {"all_train", "single_surface", "minimal", "minimal_loop", "none"}:
        return {"train": np.sort(idx), "val": np.empty(0, dtype=int), "test": np.empty(0, dtype=int)}

    # Main leakage-safe split: no battery appears in more than one split.
    if mode in {"paper_like", "battery_holdout", "battery_holdout_surface", "surface_battery_holdout"} and len(set(batteries.tolist())) >= 3:
        bats = sorted(set(batteries.tolist()), key=_battery_sort_value)
        if len(bats) >= 5:
            test_bats = set(bats[-2:])
            val_bats = {bats[-3]}
        else:
            test_bats = {bats[-1]}
            val_bats = {bats[-2]}
        train_mask = (~np.isin(batteries, list(test_bats))) & (~np.isin(batteries, list(val_bats)))
        val_mask = np.isin(batteries, list(val_bats))
        test_mask = np.isin(batteries, list(test_bats))
        return {"train": np.sort(idx[train_mask]), "val": np.sort(idx[val_mask]), "test": np.sort(idx[test_mask])}

    # Cycle holdout: test a full cycle across all batteries, never individual SoC spectra.
    if mode in {"cycle_holdout", "cycle_holdout_surface"} and len(set(cycles.tolist())) >= 2:
        cyc = sorted(set(cycles.tolist()))
        test_cycle = cyc[-1]
        train_cycles = set(cyc[:-1])
        train_val = idx[np.isin(cycles, list(train_cycles))]
        test = idx[cycles == test_cycle]
        # If there are enough batteries, reserve the largest battery ID from the
        # training cycle(s) as validation; otherwise keep validation empty.
        val = np.empty(0, dtype=int)
        train = train_val.copy()
        if len(set(batteries[train_val].tolist())) >= 3:
            train_bats = sorted(set(batteries[train_val].tolist()), key=_battery_sort_value)
            val_bat = train_bats[-1]
            val = train_val[batteries[train_val] == val_bat]
            train = train_val[batteries[train_val] != val_bat]
        return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}

    # Fallback for single-battery development runs such as B01 only.  This still
    # splits complete surfaces, not SoC spectra.  With two cycles, train one and
    # test the other.
    if len(set(cycles.tolist())) >= 2:
        cyc = sorted(set(cycles.tolist()))
        test_cycle = cyc[-1]
        return {"train": np.sort(idx[cycles != test_cycle]), "val": np.empty(0, dtype=int), "test": np.sort(idx[cycles == test_cycle])}

    # Last-resort complete-surface split.
    rng = np.random.default_rng(int(seed))
    order = idx.copy()
    rng.shuffle(order)
    n_test = max(1, int(round(0.20 * n))) if n > 1 else 0
    test = np.sort(order[:n_test])
    train = np.sort(order[n_test:])
    return {"train": train, "val": np.empty(0, dtype=int), "test": test}

def _assemble_impedance_surfaces(
    spectra: Sequence[Dict[str, Any]],
    *,
    n_logfreq_grid: int,
    exclude_soc100: bool,
    freq_preprocess_mode: str = "rc",
    n_rc_elements: int = 12,
    include_series_inductance: bool = True,
    spline_smoothing: float = 1.0e-3,
    soc_smooth_window: int = 5,
    soc_smooth_polyorder: int = 2,
    selected_battery_ids: Optional[Sequence[str]] = None,
    selected_cycle_ids: Optional[Sequence[str]] = None,
    frequency_min_hz: Optional[float] = None,
    frequency_max_hz: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Build [battery-cycle, discharge_progress, log_frequency, channels] arrays.

    t-coordinate is discharge progress q = 1 - SoC, so positive evolution means
    progressing from high SoC toward low SoC.  x-coordinate is a uniform
    log10(frequency) grid.  Each original complex spectrum is resampled by a
    synchronized R/X preprocessing rule; the default uses a joint Voigt-RC basis
    so that the generated R and X are coupled during frequency smoothing rather
    than treated as independent scalar curves.
    """
    records = list(spectra)
    if selected_battery_ids:
        wanted = {str(b).upper() for b in selected_battery_ids}
        records = [s for s in records if str(s.get("battery_id", "")).upper() in wanted]
    if selected_cycle_ids:
        wanted_cycles = {_normalize_cycle_id(c) for c in selected_cycle_ids}
        records = [s for s in records if _normalize_cycle_id(s.get("cycle_id", "")) in wanted_cycles]
    if exclude_soc100:
        records = [s for s in records if abs(float(s["soc"]) - 1.0) > 1e-12]
    filtered_records: List[Dict[str, Any]] = []
    for s in records:
        freq = np.asarray(s["freq"], dtype=np.float64)
        mask = np.isfinite(freq) & (freq > 0.0)
        if frequency_min_hz is not None:
            mask &= freq >= float(frequency_min_hz) * (1.0 - 1.0e-12)
        if frequency_max_hz is not None:
            mask &= freq <= float(frequency_max_hz) * (1.0 + 1.0e-12)
        if int(np.count_nonzero(mask)) >= 4:
            filtered_records.append(s)
    records = filtered_records
    if not records:
        raise RuntimeError("No usable EIS spectra remain after filtering.")

    def _band_logfreq(s: Dict[str, Any]) -> np.ndarray:
        f = np.asarray(s["freq"], dtype=np.float64)
        m = np.isfinite(f) & (f > 0.0)
        if frequency_min_hz is not None:
            m &= f >= float(frequency_min_hz) * (1.0 - 1.0e-12)
        if frequency_max_hz is not None:
            m &= f <= float(frequency_max_hz) * (1.0 + 1.0e-12)
        return np.log10(f[m])

    all_logf = np.concatenate([_band_logfreq(s) for s in records])
    x_min = float(np.nanmin(all_logf))
    x_max = float(np.nanmax(all_logf))
    n_x = int(max(8, n_logfreq_grid))
    x_coords = np.linspace(x_min, x_max, n_x, dtype=np.float64)

    soc_levels = np.array(sorted({round(float(s["soc"]), 8) for s in records}, reverse=True), dtype=np.float64)
    # Discharge progress increases from high SoC to low SoC.
    t_coords = (1.0 - soc_levels).astype(np.float64)
    order_t = np.argsort(t_coords)
    t_coords = t_coords[order_t]
    soc_levels_for_t = soc_levels[order_t]

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for s in records:
        key = (str(s["battery_id"]), str(s["cycle_id"]))
        grouped.setdefault(key, []).append(s)

    fields_R: List[np.ndarray] = []
    fields_minusX: List[np.ndarray] = []
    field_meta: List[Dict[str, Any]] = []
    dropped_groups: List[Dict[str, Any]] = []
    spectrum_diagnostics: List[Dict[str, Any]] = []

    required_soc = {round(float(s), 8) for s in soc_levels_for_t}
    for (battery_id, cycle_id), rows in sorted(grouped.items(), key=lambda kv: (_battery_sort_value(kv[0][0]), kv[0][1])):
        by_soc = {round(float(r["soc"]), 8): r for r in rows}
        if not required_soc.issubset(set(by_soc)):
            dropped_groups.append({
                "battery_id": battery_id,
                "cycle_id": cycle_id,
                "missing_soc": sorted(required_soc - set(by_soc)),
            })
            continue
        R_grid = np.zeros((len(t_coords), len(x_coords)), dtype=np.float64)
        mX_grid = np.zeros_like(R_grid)
        source_files: List[str] = []
        voltages: List[float] = []
        temps: List[float] = []
        for j, soc_val in enumerate(soc_levels_for_t):
            row = by_soc[round(float(soc_val), 8)]
            R_eval, mX_eval, diag = _evaluate_frequency_spectrum(
                row,
                x_coords,
                freq_preprocess_mode=str(freq_preprocess_mode),
                n_rc_elements=int(n_rc_elements),
                include_series_inductance=bool(include_series_inductance),
                spline_smoothing=float(spline_smoothing),
                frequency_min_hz=frequency_min_hz,
                frequency_max_hz=frequency_max_hz,
            )
            R_grid[j] = R_eval
            mX_grid[j] = mX_eval
            spectrum_diagnostics.append(diag)
            source_files.append(str(row.get("source_filename") or row.get("source_label", "")))
            if row.get("V") is not None and np.isfinite(float(row.get("V"))):
                voltages.append(float(row.get("V")))
            if row.get("T") is not None and np.isfinite(float(row.get("T"))):
                temps.append(float(row.get("T")))

        R_grid = _smooth_along_soc_axis(R_grid, window=int(soc_smooth_window), polyorder=int(soc_smooth_polyorder))
        mX_grid = _smooth_along_soc_axis(mX_grid, window=int(soc_smooth_window), polyorder=int(soc_smooth_polyorder))
        fields_R.append(R_grid)
        fields_minusX.append(mX_grid)
        field_meta.append({
            "battery_id": battery_id,
            "cycle_id": cycle_id,
            "source_files": source_files,
            "mean_voltage_V": float(np.mean(voltages)) if voltages else None,
            "mean_temperature_C": float(np.mean(temps)) if temps else None,
        })

    if not fields_R:
        raise RuntimeError("No complete battery-cycle EIS surfaces could be assembled.")

    R = np.stack(fields_R, axis=0)
    minusX = np.stack(fields_minusX, axis=0)
    summary = {
        "n_input_spectra": int(len(records)),
        "n_battery_cycle_surfaces": int(len(fields_R)),
        "n_dropped_incomplete_groups": int(len(dropped_groups)),
        "dropped_groups_preview": dropped_groups[:8],
        "soc_levels": soc_levels_for_t.astype(float).tolist(),
        "discharge_progress_coords": t_coords.astype(float).tolist(),
        "log10_frequency_min": x_min,
        "log10_frequency_max": x_max,
        "n_logfreq_grid": int(n_x),
        "frequency_preprocess_mode": str(freq_preprocess_mode),
        "n_rc_elements": int(n_rc_elements),
        "include_series_inductance": bool(include_series_inductance),
        "spline_smoothing": float(spline_smoothing),
        "soc_smooth_window": int(soc_smooth_window),
        "soc_smooth_polyorder": int(soc_smooth_polyorder),
        "selected_battery_ids": [str(x) for x in (selected_battery_ids or [])],
        "selected_cycle_ids": [str(x) for x in (selected_cycle_ids or [])],
        "frequency_min_hz": None if frequency_min_hz is None else float(frequency_min_hz),
        "frequency_max_hz": None if frequency_max_hz is None else float(frequency_max_hz),
        "frequency_preprocess_diagnostics": _summarize_spectrum_diagnostics(spectrum_diagnostics),
    }
    return R, minusX, t_coords, x_coords, field_meta, summary


def _as_optional_list(value: Optional[Any]) -> Optional[List[str]]:
    """Normalize CLI/string/list filters to a list of nonempty strings."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in re.split(r"[,;]+", value) if p.strip()]
        return parts or None
    try:
        parts = [str(p).strip() for p in value if str(p).strip()]
        return parts or None
    except TypeError:
        text = str(value).strip()
        return [text] if text else None


def _normalize_cycle_id(value: Any) -> str:
    """Canonicalise cycle identifiers such as '1', 'Test1', 'Test_1'."""
    text = str(value).strip().replace(" ", "_").replace("-", "_")
    if not text:
        return text
    m = re.fullmatch(r"(?:Test_?)?(\d+)", text, flags=re.IGNORECASE)
    if m:
        return f"Test_{int(m.group(1))}"
    m = re.search(r"Test_?(\d+)", text, flags=re.IGNORECASE)
    if m:
        return f"Test_{int(m.group(1))}"
    return text


def _normalize_optional_cycles(value: Optional[Any]) -> Optional[List[str]]:
    parts = _as_optional_list(value)
    return [_normalize_cycle_id(p) for p in parts] if parts else None


def _frequency_band_to_bounds(
    band: Optional[str],
    frequency_min_hz: Optional[float],
    frequency_max_hz: Optional[float],
) -> Tuple[Optional[float], Optional[float], str]:
    """Map a named EIS band to frequency bounds unless explicit bounds are given."""
    name = str(band or "full").strip().lower()
    presets = {
        "full": (None, None),
        "all": (None, None),
        # The boundaries are deliberately simple and transparent.  They are not
        # used as hard electrochemical claims; they only define ablation windows
        # for the minimal-loop validation.
        "low": (0.01, 1.0),
        "diffusion": (0.01, 1.0),
        "mid": (1.0, 100.0),
        "middle": (1.0, 100.0),
        "charge_transfer": (1.0, 100.0),
        "high": (100.0, 1000.0),
        "ohmic": (100.0, 1000.0),
    }
    if name not in presets:
        raise ValueError(f"Unknown frequency_band={band!r}; choose from full, low, mid, high.")
    lo, hi = presets[name]
    if frequency_min_hz is not None:
        lo = float(frequency_min_hz)
    if frequency_max_hz is not None:
        hi = float(frequency_max_hz)
    return lo, hi, name


def _battery_affine_standardize(
    R_all: np.ndarray,
    minusX_all: np.ndarray,
    x_coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """Per battery-cycle low-dimensional impedance calibration.

    For the minimal closed-loop study we remove only a static ohmic shift and a
    static impedance scale per assembled surface.  The same parameters are used
    for every SoC point in that surface, so SoC-dependent amplitude evolution is
    retained instead of being normalized away.
    """
    R_all = np.asarray(R_all, dtype=np.float64)
    minusX_all = np.asarray(minusX_all, dtype=np.float64)
    x_coords = np.asarray(x_coords, dtype=np.float64)
    u = np.zeros_like(R_all, dtype=np.float64)
    v = np.zeros_like(minusX_all, dtype=np.float64)
    per_surface: List[Dict[str, float]] = []
    n_surface = int(R_all.shape[0])
    n_hf = int(min(3, R_all.shape[2]))
    high_freq_idx = np.argsort(x_coords)[-n_hf:]
    for i in range(n_surface):
        R_ohm = float(np.nanmedian(R_all[i, :, high_freq_idx]))
        shifted_R = R_all[i] - R_ohm
        radial = np.sqrt(shifted_R * shifted_R + minusX_all[i] * minusX_all[i])
        scale = float(np.nanpercentile(radial, 95))
        if not np.isfinite(scale) or scale < 1.0e-12:
            scale = float(np.nanstd(np.concatenate([shifted_R.ravel(), minusX_all[i].ravel()])))
        if not np.isfinite(scale) or scale < 1.0e-12:
            scale = 1.0
        u[i] = shifted_R / scale
        v[i] = minusX_all[i] / scale
        per_surface.append({
            "R_mu_ohm": R_ohm,
            "R_scale_ohm": scale,
            "minusX_mu_ohm": 0.0,
            "minusX_scale_ohm": scale,
            "shared_scale_ohm": scale,
            "normalization_mode": "battery_affine",
        })
    return u, v, per_surface

_UNSAFE_CHANNELWISE_NORM_MODES = {"separate", "per_channel", "channelwise"}
_UNSAFE_SURFACE_NORM_MODES = {"battery_affine", "surface_affine", "individual_affine", "ohmic_affine"}
_COMPLEX_NORM_ALIASES = {
    "joint_magnitude": "complex_affine",
    "joint_absz": "complex_affine",
    "shared_magnitude": "complex_affine",
    "shared_percentile": "complex_affine",
    "global_train_affine": "complex_affine",
    "band_train_affine": "band_complex_affine",
}


def _canonical_complex_norm_mode(mode: Optional[str]) -> str:
    """Return the only scientifically supported SoC-EIS normalization modes.

    Re(Z) and -Im(Z) are coupled projections of the same complex impedance.
    Main experiments therefore use a complex-plane affine transform: separate
    train-only centers for the two axes but one shared positive impedance scale.
    Channel-wise scaling and per-surface scaling are intentionally disabled.
    """
    raw = str(mode or "band_complex_affine").strip().lower()
    if raw in _UNSAFE_CHANNELWISE_NORM_MODES:
        raise ValueError(
            "Channel-wise normalization is disabled for SoC-EIS. Re(Z) and -Im(Z) "
            "must not be scaled independently. Use 'band_complex_affine' or "
            "'complex_affine'."
        )
    if raw in _UNSAFE_SURFACE_NORM_MODES:
        raise ValueError(
            "Per-surface/battery affine normalization is disabled for SoC-EIS main "
            "experiments because it can erase physically meaningful battery-to-battery "
            "impedance differences and can leak held-out surface information. Use "
            "'band_complex_affine' or 'complex_affine'."
        )
    return _COMPLEX_NORM_ALIASES.get(raw, raw)


def _complex_affine_standardize(
    R_all: np.ndarray,
    minusX_all: np.ndarray,
    train_idx: np.ndarray,
    *,
    mode_label: str = "band_complex_affine",
    scale_quantile: float = 95.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Train-only complex-plane affine standardization for impedance spectra.

    This is the canonical SoC-EIS preprocessing used by the engineering
    validation: Re(Z) and -Im(Z) are centered independently but scaled by one
    shared impedance magnitude, preserving Nyquist geometry and real/imaginary
    coupling.  Statistics are computed only from discovery/train surfaces.
    """
    R_all = np.asarray(R_all, dtype=np.float64)
    minusX_all = np.asarray(minusX_all, dtype=np.float64)
    train_idx = np.asarray(train_idx, dtype=int)
    if train_idx.size == 0:
        train_idx = np.arange(R_all.shape[0], dtype=int)

    R_train = R_all[train_idx]
    X_train = minusX_all[train_idx]
    R_mu = float(np.nanmedian(R_train))
    X_mu = float(np.nanmedian(X_train))
    dR = np.asarray(R_train - R_mu, dtype=np.float64)
    dX = np.asarray(X_train - X_mu, dtype=np.float64)
    radial = np.sqrt(dR * dR + dX * dX)
    scale = float(np.nanpercentile(radial, float(scale_quantile)))
    if not np.isfinite(scale) or scale < 1.0e-12:
        scale = float(np.nanstd(np.concatenate([dR.ravel(), dX.ravel()])))
    if not np.isfinite(scale) or scale < 1.0e-12:
        scale = 1.0

    u = (R_all - R_mu) / scale
    v = (minusX_all - X_mu) / scale
    scales = {
        "R_mu_ohm": R_mu,
        "minusX_mu_ohm": X_mu,
        "R_scale_ohm": scale,
        "minusX_scale_ohm": scale,
        "shared_scale_ohm": scale,
        "complex_scale_ohm": scale,
        "normalization_mode": str(mode_label),
        "normalization_coupled": True,
        "normalization_center_mode": "train_median",
        "normalization_scale_quantile": float(scale_quantile),
        "channelwise_scale_ratio": 1.0,
    }
    return u, v, scales


def make_soc_eis_impedance_manifold_dataset(
    *,
    save_dir: str = "data/dataset",
    noise_level: float = 0.0,
    seed: int = 42,
    raw_dir: Optional[str] = None,
    split_mode: str = "paper_like",
    exclude_soc100: bool = True,
    n_logfreq_grid: int = 64,
    freq_preprocess_mode: str = "rc",
    n_rc_elements: int = 12,
    include_series_inductance: bool = True,
    spline_smoothing: float = 1.0e-3,
    soc_smooth_window: int = 5,
    soc_smooth_polyorder: int = 2,
    normalization_mode: str = "band_complex_affine",
    max_surfaces: Optional[int] = None,
    selected_battery_ids: Optional[Any] = None,
    selected_cycle_ids: Optional[Any] = None,
    val_battery_ids: Optional[Any] = None,
    test_battery_ids: Optional[Any] = None,
    train_cycle_ids: Optional[Any] = None,
    val_cycle_ids: Optional[Any] = None,
    test_cycle_ids: Optional[Any] = None,
    frequency_band: str = "full",
    frequency_min_hz: Optional[float] = None,
    frequency_max_hz: Optional[float] = None,
    ohmic_alignment: str = "none",
    freq_weight_mode: str = "uniform",
    include_coordinates: Optional[Any] = None,
    protocol_role: str = "",
    **_: Any,
) -> Dict[str, Any]:
    """Convert raw EIS data into a 2D impedance-manifold WiSED dataset.

    Internal coordinates/fields:
      t -> discharge progress q = 1 - SoC, uniform in 0.05 increments when all SoC levels are present;
      x -> uniform log10(frequency_Hz) grid;
      u -> standardized Re(Z);
      v -> standardized -Im(Z).

    Discovery targets are u and v, so du_t/dv_t represent the evolution of the
    complex impedance manifold with respect to discharge progress, not physical
    clock time and not a supervised SoC regression label.
    The default frequency preprocessing jointly fits R and X with a Voigt-RC
    basis before grid resampling, preserving the physical coupling of the
    complex impedance components during smoothing.
    """
    raw_root = _resolve_raw_dir(raw_dir)
    conversion_summary: Dict[str, Any] = {}
    spectra = _load_author_matrix_spectra(raw_root)
    if spectra:
        raw_format = "author_matrix_csv"
        conversion_summary = {"raw_conversion": "prebuilt_author_matrix_csv"}
    else:
        spectra, conversion_summary = _load_author_matrix_from_raw_in_memory(raw_root)
        raw_format = "author_matrix_from_raw_in_memory" if spectra else "recursive_eis_csv"
    if not spectra:
        raise FileNotFoundError(
            "No EIS CSV files were found in the configured raw SoC-EIS directory. Expected either the author's matrix CSVs or the original Bxx/EIS Measurement/Test_N CSV tree."
        )

    selected_battery_ids_list = _as_optional_list(selected_battery_ids)
    selected_cycle_ids_list = _normalize_optional_cycles(selected_cycle_ids)
    val_battery_ids_list = _as_optional_list(val_battery_ids)
    test_battery_ids_list = _as_optional_list(test_battery_ids)
    train_cycle_ids_list = _normalize_optional_cycles(train_cycle_ids)
    val_cycle_ids_list = _normalize_optional_cycles(val_cycle_ids)
    test_cycle_ids_list = _normalize_optional_cycles(test_cycle_ids)
    frequency_min_hz, frequency_max_hz, frequency_band_name = _frequency_band_to_bounds(
        frequency_band,
        frequency_min_hz,
        frequency_max_hz,
    )

    R_all, minusX_all, t_coords, x_coords, field_meta, surface_summary = _assemble_impedance_surfaces(
        spectra,
        n_logfreq_grid=int(n_logfreq_grid),
        exclude_soc100=bool(exclude_soc100),
        freq_preprocess_mode=str(freq_preprocess_mode),
        n_rc_elements=int(n_rc_elements),
        include_series_inductance=bool(include_series_inductance),
        spline_smoothing=float(spline_smoothing),
        soc_smooth_window=int(soc_smooth_window),
        soc_smooth_polyorder=int(soc_smooth_polyorder),
        selected_battery_ids=selected_battery_ids_list,
        selected_cycle_ids=selected_cycle_ids_list,
        frequency_min_hz=frequency_min_hz,
        frequency_max_hz=frequency_max_hz,
    )
    if max_surfaces is not None and int(max_surfaces) > 0 and int(max_surfaces) < R_all.shape[0]:
        rng = np.random.default_rng(int(seed))
        keep = np.sort(rng.choice(np.arange(R_all.shape[0]), size=int(max_surfaces), replace=False))
        R_all = R_all[keep]
        minusX_all = minusX_all[keep]
        field_meta = [field_meta[int(i)] for i in keep]

    splits = _split_surface_fields(
        field_meta,
        split_mode=str(split_mode),
        seed=int(seed),
        val_battery_ids=val_battery_ids_list,
        test_battery_ids=test_battery_ids_list,
        train_cycle_ids=train_cycle_ids_list,
        val_cycle_ids=val_cycle_ids_list,
        test_cycle_ids=test_cycle_ids_list,
    )
    train_idx = np.asarray(splits.get("train", []), dtype=int)
    if train_idx.size == 0:
        train_idx = np.arange(R_all.shape[0], dtype=int)

    alignment_mode = str(ohmic_alignment or "none").lower()
    ohmic_metadata: Dict[str, Any] = {
        "ohmic_alignment": alignment_mode,
        "R_ohmic": [],
        "X_ohmic": [],
        "minusX_ohmic": [],
        "high_tail_indices": [],
    }
    if alignment_mode == "subtract_high_freq_intercept":
        if frequency_band_name not in {"full", "all", "high", "ohmic"}:
            raise ValueError(
                "ohmic_alignment='subtract_high_freq_intercept' requires a full/high-frequency "
                "frequency band. The current band has already been cropped, so using its upper "
                "edge as an ohmic intercept would be physically misleading."
            )
        freq_hz = np.power(10.0, np.asarray(x_coords, dtype=np.float64))
        n_tail = max(1, int(math.ceil(0.2 * max(1, freq_hz.size))))
        high_tail = np.sort(np.argsort(freq_hz)[-n_tail:]).astype(int)
        R_ohmic = np.nanmedian(R_all[:, :, high_tail], axis=2)
        minusX_ohmic = np.nanmedian(minusX_all[:, :, high_tail], axis=2)
        R_all = R_all - R_ohmic[:, :, None]
        minusX_all = minusX_all - minusX_ohmic[:, :, None]
        ohmic_metadata = {
            "ohmic_alignment": alignment_mode,
            "R_ohmic": R_ohmic.astype(float).tolist(),
            "X_ohmic": (-minusX_ohmic).astype(float).tolist(),
            "minusX_ohmic": minusX_ohmic.astype(float).tolist(),
            "high_tail_indices": high_tail.astype(int).tolist(),
        }

    norm_mode_requested = str(normalization_mode or "band_complex_affine").lower()
    norm_mode = _canonical_complex_norm_mode(norm_mode_requested)
    per_surface_scales: Optional[List[Dict[str, float]]] = None
    if norm_mode not in {"complex_affine", "band_complex_affine"}:
        raise ValueError(
            f"Unsupported SoC-EIS normalization_mode={normalization_mode!r}. "
            "Use only 'band_complex_affine' or 'complex_affine'."
        )

    u, v, complex_scales = _complex_affine_standardize(
        R_all,
        minusX_all,
        train_idx,
        mode_label=norm_mode,
    )
    R_mu = float(complex_scales["R_mu_ohm"])
    X_mu = float(complex_scales["minusX_mu_ohm"])
    R_scale = X_scale = float(complex_scales["shared_scale_ohm"])
    data_clean = np.stack([u, v], axis=-1).astype(np.float32)
    if float(noise_level) > 0.0:
        rng = np.random.default_rng(int(seed) + 2029)
        sig = np.nanstd(data_clean[train_idx], axis=(0, 1, 2), keepdims=True)
        data = data_clean + rng.normal(size=data_clean.shape).astype(np.float32) * sig.astype(np.float32) * float(noise_level)
    else:
        data = data_clean.copy()

    grid_info: Dict[str, Any] = {
        "t_coords": t_coords.astype(np.float32),
        "x_coords": x_coords.astype(np.float32),
        "field_names": ["u", "v"],
        "discovery_targets": ["u", "v"],
        "periodic_axes": {"t": False, "x": False},
        "preprocess": {
            "preprocess_version": SPECTRAL_PREPROCESS_VERSION,
            "formulation": "raw_eis_complex_impedance_manifold_discharge_progress_log_frequency",
            "input_contract": "WiSED receives only Z(f,q) field channels and coordinates q=1-SoC, log10(f); no scalar SoC feature or hand-crafted derivative table is exported.",
            "raw_format": raw_format,
            "raw_conversion_summary": conversion_summary,
            "surface_assembly_summary": surface_summary,
            "split_mode": str(split_mode),
            "split_protocol": "complete_battery_cycle_surface_split_no_spectrum_shuffle",
            "split_note": "Equation discovery uses whole battery-cycle impedance surfaces. Individual SoC spectra are never randomly shuffled across train/test.",
            "manual_derivative_terminals_exported": False,
            "soc_as_scalar_feature": False,
            "soc_role": "coordinate_only_pseudo_time_axis",
            "exclude_soc100": bool(exclude_soc100),
            "frequency_preprocess_mode": str(freq_preprocess_mode),
            "frequency_band": frequency_band_name,
            "frequency_min_hz": None if frequency_min_hz is None else float(frequency_min_hz),
            "frequency_max_hz": None if frequency_max_hz is None else float(frequency_max_hz),
            "normalization_mode": norm_mode,
            "protocol": str(_.get("protocol", "")),
            "protocol_role": str(protocol_role or _.get("protocol_role", "")),
            "include_coordinates": list(include_coordinates or []),
            "ohmic_alignment": alignment_mode,
            "freq_weight_mode": str(freq_weight_mode or "uniform"),
            "normalization_note": "Re(Z) and -Im(Z) are treated as coupled components of one complex impedance: independent train-only centers, one shared train-only impedance scale, and no channel-wise or per-surface scaling.",
            "normalization_coupled": True,
            "selected_battery_ids": selected_battery_ids_list or [],
            "selected_cycle_ids": selected_cycle_ids_list or [],
            "val_battery_ids": val_battery_ids_list or [],
            "test_battery_ids": test_battery_ids_list or [],
            "train_cycle_ids": train_cycle_ids_list or [],
            "val_cycle_ids": val_cycle_ids_list or [],
            "test_cycle_ids": test_cycle_ids_list or [],
            "n_surfaces": int(data.shape[0]),
            "n_discharge_points": int(data.shape[1]),
            "n_logfreq_grid": int(data.shape[2]),
        },
        "paper_variable_mapping": {
            "t_internal": "discharge progress q = 1 - SoC; positive direction goes from high SoC to low SoC",
            "x_internal": "uniform log10(frequency_Hz) grid after interpolation from raw EIS frequencies",
            "u": "standardized Re(Z)",
            "v": "standardized -Im(Z)",
            "target du_t": "d standardized Re(Z) / d(discharge progress)",
            "target dv_t": "d standardized [-Im(Z)] / d(discharge progress)",
            "forbidden_interpretation": "SoC is not a right-hand-side feature. It only orders the impedance manifold along q.",
        },
        "scales": {
            "R_mu_ohm": R_mu,
            "R_scale_ohm": R_scale,
            "minusX_mu_ohm": X_mu,
            "minusX_scale_ohm": X_scale,
            "shared_scale_ohm": R_scale,
            "complex_scale_ohm": R_scale,
            "normalization_mode": norm_mode,
            "normalization_coupled": True,
            "normalization_center_mode": "train_median",
            "normalization_scale_quantile": 95.0,
            "channelwise_scale_ratio": 1.0,
            "per_surface": [],
            "ohmic_alignment": alignment_mode,
            "ohmic_metadata": ohmic_metadata,
        },
        "frequency_Hz_uniform": (10.0 ** x_coords).astype(float).tolist(),
        "soc_levels": (1.0 - t_coords).astype(float).tolist(),
        "field_meta": field_meta,
        "source_meta": field_meta,
        "splits": {k: np.asarray(vv, dtype=int).tolist() for k, vv in splits.items()},
        "discovery_indices": np.asarray(splits.get("train", []), dtype=int).tolist(),
    }
    true_eq = {
        "u": "Complex-impedance manifold evolution: d standardized Re(Z) / d(discharge progress) = F(u, v, frequency derivatives). No SoC scalar feature, equivalent-circuit candidate library, or precomputed PDE term table is imposed.",
        "v": "Complex-impedance manifold evolution: d standardized [-Im(Z)] / d(discharge progress) = G(u, v, frequency derivatives). No SoC scalar feature, equivalent-circuit candidate library, or precomputed PDE term table is imposed.",
    }
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"soc_eis_impedance_manifold_noise{float(noise_level)}.npz"
    np.savez_compressed(
        out_path,
        data=data.astype(np.float32),
        data_clean=data_clean.astype(np.float32),
        grid_info=np.array(grid_info, dtype=object),
        true_eq=np.array(true_eq, dtype=object),
        t_coords=t_coords.astype(np.float32),
        x_coords=x_coords.astype(np.float32),
        field_meta_json=np.array(json.dumps(field_meta, ensure_ascii=False), dtype=object),
        splits_json=np.array(json.dumps(grid_info["splits"], ensure_ascii=False), dtype=object),
    )
    return grid_info
