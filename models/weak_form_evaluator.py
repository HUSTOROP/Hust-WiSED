from __future__ import annotations
import hashlib
import numpy as np
import scipy.signal
from typing import Dict, List, Optional, Tuple, Any
from models.symbol_vocabulary import IDX2SYM, SYM2IDX, count_constants, get_default_vocab, is_valid_sequence
from utils.structure_cache import get_structure_key_manager

import math
import torch
import torch.nn.functional as F


def normalize_target_field_name(target_field: Optional[str]) -> str:
    """瑙勮寖鍖栫洰鏍囧瓧娈靛悕锛屽彧淇濈暀 du_t / dv_t / dw_t 褰㈠紡銆?"""
    field = str(target_field or "du_t").strip()
    if field in {"u", "v", "w"}:
        return f"d{field}_t"
    if field in {"du_t", "dv_t", "dw_t"}:
        return field
    raise ValueError(
        f"Invalid target field '{field}'. Only u/v/w or du_t/dv_t/dw_t are supported."
    )


def robust_strong_derivative(u: torch.Tensor, axis: int, L: float, order: int = 1, is_periodic: bool = True) -> torch.Tensor:
    """
    绾噣鐗堟嫙璋辨眰瀵?(铻嶅悎 Hou-Li 楂橀樁鎸囨暟婊ゆ尝鍣?
    褰诲簳閲婃斁 10^-13 鐨勮氨绮惧害锛屽悓鏃朵粠棰戠巼鏍规簮涓婃壖鏉€婵€娉㈠紩鍙戠殑 Gibbs 鎸搩銆?    """
    n = u.shape[axis]
    if n < 3: return torch.zeros_like(u)

    if is_periodic:
        u_double = u.to(torch.float64)
        k = 2.0 * torch.pi * torch.fft.fftfreq(n, d=L / n, device=u.device, dtype=torch.float64)

        # =========================================================
        # 銆愭牴婧愬崌绾с€戯細Hou-Li 36闃惰氨婊ゆ尝鍣?(Spectral Filter)
        # 褰诲簳鏇夸唬绌洪棿鍩熺殑鏈夐檺宸垎锛屽湪棰戝煙瀹岀編闀囧帇婵€娉㈡尟閾?        # =========================================================
        k_max = torch.max(torch.abs(k))
        if k_max > 0:
            # alpha=36, p=36 鏄鐞嗘縺娉㈢殑鏍囧噯瓒呭弬鏁帮紝淇濊瘉浣庨 10^-15 鏃犳崯
            filter_func = torch.exp(-36.0 * (torch.abs(k) / k_max)**36)
        else:
            filter_func = torch.ones_like(k)

        shape = [1] * u.ndim
        shape[axis] = n
        k = k.view(*shape)
        filter_func = filter_func.view(*shape)

        # 灏嗘眰瀵肩畻瀛愪笌婊ゆ尝鍣ㄥ湪棰戝煙鐩镐箻
        multiplier = ((1j * k) ** order) * filter_func
        fft_u = torch.fft.fftn(u_double, dim=(axis,))
        out = torch.fft.ifftn(fft_u * multiplier, dim=(axis,)).real

        return out.to(u.dtype)
    else:
        out = u
        spacing = L / max(1, n - 1)
        for _ in range(order):
            out = torch.gradient(out, spacing=(spacing,), dim=axis)[0]
        return out



class MaskConfig:
    def __init__(
        self,
        trim: Optional[Dict[str, int]] = None,
        weak_window: Optional[Dict[str, int]] = None,
        weak_degree: int = 6,
    ):
        self.trim = dict(trim or {})
        self.weak_window = {str(k): int(v) for k, v in dict(weak_window or {}).items()}
        self.weak_degree = int(weak_degree)

class DataContext:
    def __init__(self, data_array: np.ndarray, coords: Dict[str, np.ndarray], device="cuda",
                 periodic_axes=None, mask_cfg=None, normalize_mse: bool=True,
                 cache_tag: str="", field_names: Optional[List[str]]=None,
                 scoring_form: str = "weak",
                 allow_coordinate_terminals: bool = False,
                 derivative_scale_audit: bool = False):
        self.device = torch.device(device)
        self.allow_coordinate_terminals = bool(allow_coordinate_terminals)
        self.derivative_scale_audit_enabled = bool(derivative_scale_audit)

        self.periodic_axes = dict(periodic_axes or {})

        # 鍙厑璁?uvw 浣滀负鍦哄悕锛涙湭鎻愪緵鏃堕粯璁ゅ崟鍦?u
        raw_fields = list(field_names) if field_names is not None else ["u"]
        valid_field_pool = ["u", "v", "w"]
        self.fields = [f for f in raw_fields if f in valid_field_pool]
        if not self.fields:
            self.fields = ["u"]

        self.coords = {
            str(k): torch.as_tensor(v, dtype=torch.float64, device=self.device)
            for k, v in coords.items()
        }

        # 鍙厑璁?txyz 浣滀负杞达紱淇濇寔鏁版嵁缁欏嚭鐨勯『搴忥紝浣嗗己鍒?t 鍦ㄥ墠
        raw_axes = [ax for ax in self.coords.keys() if ax in ("t", "x", "y", "z")]
        if "t" in raw_axes:
            self.axes_order = ["t"] + [ax for ax in raw_axes if ax != "t"]
        else:
            self.axes_order = raw_axes

        # Unified axis metadata for vocabulary/decoder/evaluator.
        self.axes = tuple(self.axes_order)
        self.spatial_axes = tuple(ax for ax in self.axes_order if ax != "t")
        self.has_v = "v" in self.fields
        self.has_w = "w" in self.fields

        self.steps = {name: float(arr[1] - arr[0]) if len(arr) > 1 else 1.0 for name, arr in self.coords.items()}
        self.domain_lengths = {n: float(arr[-1]-arr[0]+self.steps[n]) if self.periodic_axes.get(n) else float(arr[-1]-arr[0]) for n, arr in self.coords.items()}

        self._cache = {}
        self.trim_map = {str(k): int(v) for k, v in dict(getattr(mask_cfg, "trim", {}) if mask_cfg is not None else {}).items()}
        self.scoring_form = str(scoring_form or "weak").strip().lower()

        # ==============================================================
        # 鍒ゆ柇 data_array 鐨勭淮搴︽槸鍚﹀垰濂界瓑浜庡潗鏍囪酱鐨勬暟閲?(渚嬪 [T, X] 鏈?2 涓酱)
        # 濡傛灉鐩哥瓑锛岃鏄庣己澶变簡鏈€鍚庣殑閫氶亾缁村害 C锛屾垜浠渶瑕佺粰瀹冭ˉ涓?np.newaxis
        # ==============================================================
        if data_array.ndim == len(self.coords):
            data_array = data_array[..., np.newaxis]

        n_channels = int(data_array.shape[-1])
        if n_channels < len(self.fields):
            raise ValueError(
                f"Data channels ({n_channels}) < declared fields ({self.fields})."
            )
        if n_channels > len(self.fields):
            # Allow extra data channels, but keep the declared physical fields.
            data_array = data_array[..., :len(self.fields)]

        # data_array now has an explicit channel dimension, e.g. [T, X, 1].
        for i, fname in enumerate(self.fields):
            # 灏嗘瘡涓墿鐞嗗満鐙珛瀛樺偍骞舵敞鍐屼负绫诲睘鎬?(渚嬪 ctx.u, ctx.v)
            field_tensor = torch.as_tensor(data_array[..., i], dtype=torch.float64, device=self.device)
            setattr(self, fname, field_tensor)
            self._cache[fname] = field_tensor

        # 鍏煎鏃т唬鐮佸 ctx.u 鐨勭‖缂栫爜寮曠敤
        self.u = self._cache.get(self.fields[0])

        for name, grid in zip(self.coords.keys(), torch.meshgrid(*self.coords.values(), indexing='ij')):
            self._cache[name] = grid

        self.use_weak_form = self.scoring_form != "strong"

        # ------------------------------------------------------------------
        # Axis-specific weak-form windows.
        # ------------------------------------------------------------------
        # `MaskConfig.weak_window` may be provided as {"t": 15, "x": 41, ...}.
        # The previous implementation collapsed this dict to the x-window before
        # weak-form filters were built, causing the temporal projection and N_eff
        # to use the wrong physical scale. Build the per-axis map before any
        # weak-form cache is initialised, and keep `self.weak_window` only as a
        # backwards-compatible diagnostic/default value for old call sites.
        self.weak_p_degree = int(getattr(mask_cfg, "weak_degree", 6) if mask_cfg is not None else 6)
        self.weak_window_map = self._build_weak_window_map(mask_cfg)
        self.weak_window = int(
            self.weak_window_map.get(
                "x",
                next(iter(self.weak_window_map.values()), 21),
            )
        )

        self._weak_axis_local_filters: Dict[str, torch.Tensor] = {}
        self._weak_axis_operator_cache: Dict[Tuple[str, int], torch.Tensor] = {}
        self.normalize_mse = normalize_mse
        self.cache_id = cache_tag

        # Resolution-independent physical sample count must use the same per-axis
        # windows as the weak projection itself; otherwise the MDL/BIC penalty is
        # calibrated to a different smoothing scale than the residual.
        self.n_eff_physical = self._compute_physical_neff()
        self._init_weak_form_torch()

    def _sanitize_axis_weak_window(self, ax: str, raw_value: Any) -> int:
        """Return a safe odd weak-form window for a specific axis."""
        try:
            w = int(raw_value)
        except Exception:
            w = 21

        w = max(3, w)
        if w % 2 == 0:
            w += 1

        n = int(len(self.coords.get(str(ax), [])))
        if n >= 3 and w > n:
            w = n if n % 2 == 1 else n - 1
            w = max(3, w)
        return int(w)

    def _build_weak_window_map(self, mask_cfg) -> Dict[str, int]:
        raw = getattr(mask_cfg, "weak_window", {}) if mask_cfg is not None else {}
        out = {}
        for ax in self.axes_order:
            if isinstance(raw, dict) and ax in raw:
                out[ax] = self._sanitize_axis_weak_window(ax, raw[ax])
            else:
                N = int(self.coords[ax].numel())
                w = int(N * 0.05)
                w = w if w % 2 == 1 else w + 1
                out[ax] = min(max(5, w), 9)
        return out

    def _axis_window(self, ax: str) -> int:
        return int(self.weak_window_map.get(str(ax), self.weak_window))

    def _axis_pad_size(self, ax: str) -> int:
        return int((self._axis_window(ax) - 1) // 2)

    def _compute_physical_neff(self):
        n_eff = 1.0
        for ax in self.axes_order:
            c = self.coords[ax]
            if len(c) < 2:
                continue

            L_phys = float(self.domain_lengths.get(ax, float(c[-1] - c[0])))
            dx = float(self.steps.get(ax, float(c[1] - c[0])))
            w_grid = self._axis_window(ax)
            W_phys = max(dx, float(w_grid) * dx)

            # Number of approximately independent non-overlapping windows along this axis.
            n_axis = max(1.0, L_phys / W_phys)
            n_eff *= n_axis

        # 璁惧畾鐗╃悊搴曠嚎锛氬嵆浣跨獥鍙ｅ緢澶э紝涔熻涓鸿嚦灏戞湁 10 涓嫭绔嬭娴嬬偣
        return max(10.0, n_eff)

    def _make_local_psi(self, ax: str) -> torch.Tensor:
        w = self._axis_window(ax)
        c = (w - 1) / 2
        z = (torch.arange(w, device=self.device, dtype=torch.float64) - c) / c
        return (1 - z**2)**self.weak_p_degree

    def _apply_valid_axis_filter(self, arr: torch.Tensor, axis_idx: int, filt: torch.Tensor) -> torch.Tensor:
        filt = filt.to(dtype=arr.dtype, device=arr.device)
        arr_perm = torch.movedim(arr, axis_idx, -1)
        windows = arr_perm.unfold(-1, filt.numel(), 1)
        view_shape = [1] * windows.ndim
        view_shape[-1] = filt.numel()
        out = (windows * filt.view(*view_shape)).sum(dim=-1)
        return torch.movedim(out, -1, axis_idx)

    def _pad_for_weak_form(self, arr: torch.Tensor) -> torch.Tensor:
        padding = []
        for ax in reversed(self.axes_order):
            pad_size = self._axis_pad_size(ax)
            if self.periodic_axes.get(ax, False):
                padding.extend([pad_size, pad_size])
            else:
                padding.extend([0, 0])
        if any(padding):
            return F.pad(arr.unsqueeze(0).unsqueeze(0), padding, mode='circular').squeeze(0).squeeze(0)
        return arr

    def _fftconvolve_valid_torch(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        full_shape = tuple(xs + hs - 1 for xs, hs in zip(x.shape, h.shape))
        y = torch.fft.ifftn(torch.fft.fftn(x, s=full_shape) * torch.fft.fftn(h, s=full_shape)).real
        slices = tuple(slice(hs - 1, xs) for xs, hs in zip(x.shape, h.shape))
        return y[slices]

    def _init_weak_form_torch(self):
        self.vol = np.prod(list(self.steps.values()))
        for ax in self.axes_order:
            self._weak_axis_local_filters[ax] = self._make_local_psi(ax)
            self._weak_axis_operator_cache[(ax, 0)] = self._build_axis_operator_filter(ax, 0)

        self.weak_lhs_cache = {}
        if "t" in self.axes_order:
            # Precompute weak-form LHS for all active fields using du_t/dv_t/... keys.
            for fname in self.fields:
                field_tensor = self._cache[fname]
                lhs = self.project_weak(field_tensor, {"t": 1})
                self.weak_lhs_cache[f"d{fname}_t"] = lhs

    def _build_axis_operator_filter(self, ax: str, order: int) -> torch.Tensor:
        cached = self._weak_axis_operator_cache.get((ax, order))
        if cached is not None:
            return cached

        w = self._axis_window(ax)
        step = self.steps[ax]
        c = (w - 1) / 2

        # 1. 鏍稿績淇锛氳嚜鍙橀噺 z 蹇呴』鎸傝浇 requires_grad=True锛屾墠鑳芥瀯寤烘眰瀵艰绠楀浘
        z = (torch.arange(w, dtype=torch.float64, device=self.device) - c) / c
        z.requires_grad_(True)

        filt_0 = (1 - z**2)**self.weak_p_degree
        filt = filt_0

        for _ in range(order):
            filt = torch.autograd.grad(filt.sum(), z, create_graph=True)[0]

        filt = filt.detach()
        filt_0 = filt_0.detach()
        z_val = z.detach()

        # --- 鏁板鏍稿績锛氱鏁ｅ垎閮ㄧН鍒嗕弗鏍兼牎鍑?---
        if order > 0:
            import math
            # 鏋勯€犲椤瑰紡娴嬭瘯椤?z^k / k!
            test_monomial = (z_val ** order) / math.factorial(order)

            discrete_rhs = torch.sum(filt * ((-1)**order) * test_monomial)
            discrete_lhs = torch.sum(filt_0)

            # 璁＄畻鏍″噯涔樺瓙锛屽交搴曟秷闄ら粠鏇煎拰绂绘暎鍖栫殑鎴柇璇樊
            correction = discrete_lhs / (discrete_rhs + 1e-12)
            filt = filt * correction

        filt = filt / ((c * step)**order)
        filt = filt * ((-1)**order) * step

        self._weak_axis_operator_cache[(ax, order)] = filt
        return filt

    def effective_physical_sample_size(self) -> float:
        """Estimate the number of approximately independent weak-form observations."""
        return float(getattr(self, "n_eff_physical", self._compute_physical_neff()))

    def derivative_scale_audit(self, field: Optional[str] = None) -> Dict[str, Any]:
        """Return scale diagnostics for native field derivatives.

        The audit is deliberately diagnostic only.  It does not rescale tensors
        used by the evaluator, so exported coefficients remain in the physical
        coordinate system.  Noisy engineering runs use this to detect when
        higher-order terms such as u_xx are numerically present but vulnerable to
        coefficient pruning or poor candidate ranking.
        """
        def _rms(t: Optional[torch.Tensor]) -> float:
            if t is None or not torch.is_tensor(t):
                return float("nan")
            finite = torch.isfinite(t)
            if not bool(torch.any(finite).detach().cpu().item()):
                return float("nan")
            vals = t[finite].detach().to(dtype=torch.float64)
            return float(torch.sqrt(torch.mean(vals * vals)).detach().cpu().item())

        fname = str(field or (self.fields[0] if self.fields else "u"))
        out: Dict[str, Any] = {
            "field": fname,
            "scoring_form": self.scoring_form,
            "weak_window": dict(getattr(self, "weak_window_map", {})),
            "mask_trim": dict(getattr(self, "trim_map", {})),
        }
        base = self.get(fname)
        if base is None:
            out["available"] = False
            return out

        out["available"] = True
        out["strong_field_rms"] = _rms(base)
        out["weak_field_rms"] = _rms(self.project_weak(base, {}))
        lhs_key = f"d{fname}_t"
        lhs = getattr(self, "weak_lhs_cache", {}).get(lhs_key)
        out["weak_dt_rms"] = _rms(lhs)

        for ax in [a for a in self.axes_order if a != "t"]:
            idx = self.axes_order.index(ax)
            L = self.domain_lengths.get(ax, 1.0)
            is_periodic = bool(self.periodic_axes.get(ax, False))
            try:
                strong_dx = robust_strong_derivative(base, idx, L, order=1, is_periodic=is_periodic)
                strong_dxx = robust_strong_derivative(base, idx, L, order=2, is_periodic=is_periodic)
            except Exception:
                strong_dx = None
                strong_dxx = None

            weak_dx = self.project_weak_field_derivative(fname, {ax: 1})
            weak_dxx = self.project_weak_field_derivative(fname, {ax: 2})
            out[f"strong_d{ax}_rms"] = _rms(strong_dx)
            out[f"strong_d{ax}{ax}_rms"] = _rms(strong_dxx)
            out[f"weak_d{ax}_rms"] = _rms(weak_dx)
            out[f"weak_d{ax}{ax}_rms"] = _rms(weak_dxx)

            if lhs is not None:
                lhs_rms = max(_rms(lhs), 1.0e-12)
                out[f"weak_d{ax}_to_dt_rms_ratio"] = float(_rms(weak_dx) / lhs_rms)
                out[f"weak_d{ax}{ax}_to_dt_rms_ratio"] = float(_rms(weak_dxx) / lhs_rms)

            if ax in self.coords and strong_dx is not None:
                try:
                    grid = self.get(ax)
                    geom = strong_dx / (torch.abs(grid) + 1.0e-6)
                    weak_geom = self.project_weak(geom, {})
                    out[f"weak_d{ax}_over_{ax}_rms"] = _rms(weak_geom)
                    if lhs is not None:
                        lhs_rms = max(_rms(lhs), 1.0e-12)
                        out[f"weak_d{ax}_over_{ax}_to_dt_rms_ratio"] = float(_rms(weak_geom) / lhs_rms)
                except Exception:
                    pass

        return out



    def project_weak(self, arr: torch.Tensor, derivative_orders: Optional[Dict[str, int]] = None) -> torch.Tensor:
        """澶氱淮杩炵画鍒嗙寮忓急褰㈠紡绉垎锛岀粺涓€浜嗗懆鏈熶笌闈炲懆鏈熻竟鐣岀殑閫昏緫"""
        derivative_orders = derivative_orders or {}

        out = arr.to(dtype=torch.float64, device=self.device)

        for axis_idx, ax in enumerate(self.axes_order):
            order = int(derivative_orders.get(ax, 0))
            try:
                filt = self._build_axis_operator_filter(ax, order)
            except Exception:
                nan_shape = list(out.shape)
                if not bool(self.periodic_axes.get(ax, False)):
                    w = self._axis_window(ax)
                    nan_shape[axis_idx] = max(0, nan_shape[axis_idx] - w + 1)
                return torch.full(nan_shape, float('nan'), dtype=torch.float64, device=self.device)

            pad_size = self._axis_pad_size(ax)

            if bool(self.periodic_axes.get(ax, False)):
                # ====================================================================
                # 鎵嬪姩瀹炵幇 Circular Padding锛岀‘淇濆嵎绉牳鍦ㄨ竟鐣屽姝ｇ‘鐜粫锛?                # 瀹岀編鏀寔浠绘剰缁村害 Tensor锛岄伩寮€ PyTorch F.pad 瀵?(Batch, Channel) 缁村害鐨勭‖鎬ц姹?                # ====================================================================
                if pad_size > 0:
                    # Last pad_size elements along this axis.
                    head = out.narrow(axis_idx, out.shape[axis_idx] - pad_size, pad_size)
                    # First pad_size elements along this axis.
                    tail = out.narrow(axis_idx, 0, pad_size)
                    # 鎷兼帴锛歔灏鹃儴, 鍘熸暟鎹? 澶撮儴]
                    out_padded = torch.cat([head, out, tail], dim=axis_idx)
                else:
                    out_padded = out

                out = self._apply_valid_axis_filter(out_padded, axis_idx, filt)
            else:
                # 濡傛灉涓嶆槸鍛ㄦ湡杈圭晫锛岀洿鎺ユ粦鍔ㄧ獥鍙ｏ紝鐢变簬缂轰箯杈圭晫淇℃伅锛岀墿鐞嗙┖闂翠細鑷姩鏀剁缉 (w-1)
                out = self._apply_valid_axis_filter(out, axis_idx, filt)

        return out

    def apply_weak_form(self, arr: torch.Tensor) -> torch.Tensor:
        # 缁熶竴鍑哄彛锛氬皢浠讳綍寮哄舰寮忕殑鏁版嵁鍦猴紝绾补鏃犲鏁板湴鎶曞奖鍒板急褰㈠紡鐗瑰緛绌洪棿
        return self.project_weak(arr, {})

    def project_weak_field_derivative(
        self,
        field: str,
        derivative_orders: Optional[Dict[str, int]] = None,
    ) -> torch.Tensor:
        """Cache-aware weak projection for native field derivatives.

        This is the preferred path for weak-form PDE discovery.  For example,
        lap(u) in weak form should be evaluated as

            project_weak(u, {"x": 2}) + project_weak(u, {"y": 2})

        instead of first constructing strong-form u_xx / u_yy and then
        projecting those arrays.  The latter route is more sensitive to
        derivative noise and repeated spectral filtering.
        """
        field = str(field)
        orders = {
            str(k): int(v)
            for k, v in dict(derivative_orders or {}).items()
            if int(v) != 0
        }

        ref = self.u if self.u is not None else next(iter(self._cache.values()))
        if field not in self.fields or field not in self._cache:
            return self.project_weak(
                torch.full_like(ref, float("nan"), dtype=torch.float64, device=self.device),
                {},
            )

        if orders:
            suffix = "_".join(f"{ax}{order}" for ax, order in sorted(orders.items()))
        else:
            suffix = "0"
        cache_key = f"weak_{field}_{suffix}"

        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        out = self.project_weak(self._cache[field], orders)
        self._cache[cache_key] = out
        return out

    def get(self, key: str) -> Optional[torch.Tensor]:
        return self._cache.get(key)

class _Node:
    def __init__(self, op, token, children, const_index=-1, axis="", has_const=False):
        self.op, self.token, self.children = op, token, children
        self.const_index, self.axis, self.has_const = const_index, axis, has_const
        child_keys = "_".join([c.key for c in children])
        self.key = f"{op}_{axis}_{const_index}_{child_keys}"

class _Compiler:
    def __init__(self, tokens):
        self.tokens = list(map(int, tokens))
        self.pos = 0; self.c_idx = 0

    def parse(self):
        tok = self.tokens[self.pos]; self.pos += 1; sym = IDX2SYM.get(tok, "")
        if sym == "const":
            n = _Node(sym, tok, [], self.c_idx, has_const=True); self.c_idx += 1; return n
        if sym in ("u", "v", "w") or sym in ["t", "x", "y", "z"]:
            return _Node(sym, tok, [])
        if sym == "D":
            ax_tok = self.tokens[self.pos]; self.pos += 1; child = self.parse()
            return _Node(sym, tok, [child], axis=IDX2SYM.get(ax_tok, ""), has_const=child.has_const)
        if sym in {"neg", "sin", "cos", "exp", "log", "lap", "adv", "sq", "cube"}:
            c = self.parse(); return _Node(sym, tok, [c], has_const=c.has_const)
        if sym in {"+", "*", "/", "^"}:
            l = self.parse(); r = self.parse()
            return _Node(sym, tok, [l, r], has_const=l.has_const or r.has_const)
        return _Node(sym, tok, [])

_COMPILED_CACHE, _SUBTREE_CACHE, _WEAK_TARGET_CACHE = {}, {}, {}
_EVAL_CACHE_STATS = {"queries": 0, "hits": 0, "misses": 0, "saved_evals": 0}

def _parse_cached_tree(token_seq: List[int]) -> _Node:
    """Parse a token sequence once and reuse the immutable syntax tree.

    Constant optimization evaluates the same structure many times with different
    constants.  Re-parsing the prefix sequence in every objective call is pure
    overhead, so the parsed tree is cached by the exact token tuple.  The cache is
    still cleared by clear_evaluator_caches(), and init_vocab_from_context() also
    resets evaluator caches when the active vocabulary changes.
    """
    key = tuple(int(t) for t in token_seq)
    root = _COMPILED_CACHE.get(key)
    if root is None:
        root = _Compiler(list(key)).parse()
        _COMPILED_CACHE[key] = root
    return root

def clear_evaluator_caches(*args, reset_stats=False, clear_compiled=True, clear_subtree=True, **kwargs):
    if clear_compiled:
        _COMPILED_CACHE.clear()
    if clear_subtree:
        _SUBTREE_CACHE.clear()
        _WEAK_TARGET_CACHE.clear()
    if reset_stats:
        for k in _EVAL_CACHE_STATS:
            _EVAL_CACHE_STATS[k] = 0

def get_evaluator_cache_stats(*args, **kwargs):
    total = max(1, _EVAL_CACHE_STATS["queries"])
    _EVAL_CACHE_STATS["hit_rate"] = _EVAL_CACHE_STATS["hits"] / total
    return dict(_EVAL_CACHE_STATS)

class WeakFormEvaluator:
    def __init__(self, ctx: DataContext):
        self.ctx = ctx
        self._nan = torch.full_like(ctx.u, float('nan'), dtype=torch.float64, device=ctx.device)

    # =========================================================================
    # 銆愪慨澶嶉噸鐐?1銆? 灏?evaluate 瀹屽叏闄愬埗鍦?PyTorch Tensor 绌洪棿鍐?    # =========================================================================
    def evaluate(self, token_seq: List[int], constants: Optional[np.ndarray] = None) -> torch.Tensor:
        if not is_valid_sequence(token_seq): return self._nan.clone()
        consts = self._as_tensor_constant(constants, token_seq)

        try: root = _parse_cached_tree(list(map(int, token_seq)))
        except Exception: return self._nan.clone()

        try:
            res = self._eval_node(root, consts, {})
            if res.ndim == 0: res = torch.full_like(self.ctx.u, res.item())
            elif res.shape != self.ctx.u.shape: res = torch.broadcast_to(res, self.ctx.u.shape)
            return torch.clamp(res, -1e10, 1e10)
        except Exception as e:
            return self._nan.clone()

    def _get_deriv_chain(self, node: _Node) -> Tuple[_Node, List[str]]:
        axes = []
        curr = node
        while curr.op == "D":
            axes.append(curr.axis)
            if not curr.children: break
            curr = curr.children[0]
        return curr, axes

    def _as_tensor_constant(self, constants: Optional[np.ndarray], token_seq: List[int]) -> torch.Tensor:
        if constants is not None:
            if isinstance(constants, torch.Tensor):
                return constants.to(dtype=torch.float64, device=self.ctx.device)
            return torch.tensor(constants, dtype=torch.float64, device=self.ctx.device)
        return torch.ones(count_constants(token_seq), dtype=torch.float64, device=self.ctx.device)

    def _is_scalar_tensor(self, val: torch.Tensor) -> bool:
        return isinstance(val, torch.Tensor) and val.ndim == 0

    def _same_expr(self, a: _Node, b: _Node) -> bool:
        return isinstance(a, _Node) and isinstance(b, _Node) and a.key == b.key

    def _is_derivative_free_state_expr(self, node: _Node) -> bool:
        op = node.op
        if op == "D":
            return False
        if op in self.ctx.coords:
            return False
        if op in self.ctx.fields or op == "const":
            return True
        if op == "lap":
            return len(node.children) == 1 and node.children[0].op in self.ctx.fields
        if op == "adv":
            return len(node.children) == 1 and node.children[0].op in self.ctx.fields
        if op in {"neg", "sin", "cos", "exp", "log", "sq", "cube"}:
            return len(node.children) == 1 and self._is_derivative_free_state_expr(node.children[0])
        if op in {"+", "*", "/", "^"}:
            return len(node.children) == 2 and all(self._is_derivative_free_state_expr(c) for c in node.children)
        return False

    def _is_scalar_only_expr(self, node: _Node) -> bool:
        op = node.op
        if op == "const":
            return True
        if op in self.ctx.fields or op in self.ctx.coords or op == "D":
            return False
        if not node.children:
            return False
        return all(self._is_scalar_only_expr(c) for c in node.children)


    def _weak_nan_like(self) -> torch.Tensor:
        return self.ctx.project_weak(self._nan.clone(), {})

    def _eval_lap_field_weak(self, field: str) -> torch.Tensor:
        """
        缁ф壙 1D 鏋舵瀯鐨勭鏉ヤ箣绗旓細瀹屽叏鎶涘純 FFT 寮烘眰瀵硷紝
        鍒╃敤 project_weak 灏嗕簩闃剁┖闂村鏁拌浆绉诲埌娴嬭瘯鍑芥暟涓婏紝瀹炵幇瀹岀編鎶楀櫔銆?        """
        val = self.ctx.get(field)
        if val is None:
            return self._weak_nan_like()

        acc = None
        for ax in [a for a in self.ctx.axes_order if a != "t"]:
            # Weak-form second derivative along this spatial axis.
            term = self.ctx.project_weak(val, {ax: 2})
            acc = term if acc is None else acc + term

        return acc if acc is not None else self._weak_nan_like()

    def _eval_lap_strong(self, node: _Node, consts: torch.Tensor, memo: Dict) -> torch.Tensor:
        """Strong-form pointwise Laplacian for evaluate()/fallback only.

        During weak-form training, lap(field) is handled by _eval_lap_field_weak.
        This strong path remains necessary for pointwise evaluation.
        It only accepts lap(field); lap(complex_expr) is intentionally invalid to
        avoid high-noise second derivatives of nonlinear composite expressions.
        """
        child_node = node.children[0] if node.children else None
        if child_node is None or child_node.op not in self.ctx.fields:
            return self._nan.clone()

        field = child_node.op
        base = self.ctx.get(field)
        if base is None:
            return self._nan.clone()

        acc = None
        for ax in [a for a in self.ctx.axes_order if a != "t"]:
            cache_key = f"{field}_{ax}{ax}"
            cached = self.ctx.get(cache_key)
            if cached is not None:
                term = cached
            else:
                idx = self.ctx.axes_order.index(ax)
                L = self.ctx.domain_lengths.get(ax, 1.0)
                is_periodic = bool(self.ctx.periodic_axes.get(ax, False))
                term = robust_strong_derivative(base, idx, L, order=2, is_periodic=is_periodic)
                self.ctx._cache[cache_key] = term
            acc = term if acc is None else acc + term
        return acc if acc is not None else torch.zeros_like(base)

    def _add_poly_dict(self, left: Dict[int, torch.Tensor], right: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        out: Dict[int, torch.Tensor] = {}
        for deg, coeff in left.items():
            out[deg] = out.get(deg, torch.tensor(0.0, dtype=torch.float64, device=self.ctx.device)) + coeff
        for deg, coeff in right.items():
            out[deg] = out.get(deg, torch.tensor(0.0, dtype=torch.float64, device=self.ctx.device)) + coeff
        return out

    def _mul_poly_dict(self, left: Dict[int, torch.Tensor], right: Dict[int, torch.Tensor], max_degree: int = 6) -> Optional[Dict[int, torch.Tensor]]:
        out: Dict[int, torch.Tensor] = {}
        for dl, cl in left.items():
            for dr, cr in right.items():
                deg = dl + dr
                if deg > max_degree:
                    return None
                out[deg] = out.get(deg, torch.tensor(0.0, dtype=torch.float64, device=self.ctx.device)) + cl * cr
        return out

    def _pow_poly_dict(self, poly: Dict[int, torch.Tensor], exp: int, max_degree: int = 6) -> Optional[Dict[int, torch.Tensor]]:
        if exp < 0 or exp > max_degree:
            return None
        out: Dict[int, torch.Tensor] = {0: torch.tensor(1.0, dtype=torch.float64, device=self.ctx.device)}
        for _ in range(exp):
            out = self._mul_poly_dict(out, poly, max_degree=max_degree)
            if out is None:
                return None
        return out

    def _extract_polynomial_in_base(self, node: _Node, base_node: _Node, consts: torch.Tensor, memo: Dict, max_degree: int = 6) -> Optional[Dict[int, torch.Tensor]]:
        if self._same_expr(node, base_node):
            return {1: torch.tensor(1.0, dtype=torch.float64, device=self.ctx.device)}
        if self._is_scalar_only_expr(node):
            scalar_val = self._eval_node(node, consts, memo)
            if self._is_scalar_tensor(scalar_val):
                return {0: scalar_val}
            return None

        op = node.op
        if op == "neg":
            child_poly = self._extract_polynomial_in_base(node.children[0], base_node, consts, memo, max_degree=max_degree)
            if child_poly is None:
                return None
            return {deg: -coeff for deg, coeff in child_poly.items()}
        if op == "+":
            left_poly = self._extract_polynomial_in_base(node.children[0], base_node, consts, memo, max_degree=max_degree)
            right_poly = self._extract_polynomial_in_base(node.children[1], base_node, consts, memo, max_degree=max_degree)
            if left_poly is None or right_poly is None:
                return None
            return self._add_poly_dict(left_poly, right_poly)
        if op == "*":
            left_poly = self._extract_polynomial_in_base(node.children[0], base_node, consts, memo, max_degree=max_degree)
            right_poly = self._extract_polynomial_in_base(node.children[1], base_node, consts, memo, max_degree=max_degree)
            if left_poly is None or right_poly is None:
                return None
            return self._mul_poly_dict(left_poly, right_poly, max_degree=max_degree)
        if op == "sq":
            child_poly = self._extract_polynomial_in_base(node.children[0], base_node, consts, memo, max_degree=max_degree)
            if child_poly is None:
                return None
            return self._pow_poly_dict(child_poly, 2, max_degree=max_degree)
        if op == "cube":
            child_poly = self._extract_polynomial_in_base(node.children[0], base_node, consts, memo, max_degree=max_degree)
            if child_poly is None:
                return None
            return self._pow_poly_dict(child_poly, 3, max_degree=max_degree)
        if op == "^":
            left_poly = self._extract_polynomial_in_base(node.children[0], base_node, consts, memo, max_degree=max_degree)
            if left_poly is None:
                return None
            exp_val = self._eval_node(node.children[1], consts, memo)
            if not self._is_scalar_tensor(exp_val):
                return None
            exp_float = float(exp_val.item())
            exp_int = int(round(exp_float))
            if abs(exp_float - exp_int) > 1e-8:
                return None
            return self._pow_poly_dict(left_poly, exp_int, max_degree=max_degree)
        return None

    def _integrate_poly_dict(self, poly: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        return {deg + 1: coeff / float(deg + 1) for deg, coeff in poly.items()}

    def _eval_poly_dict(self, poly: Dict[int, torch.Tensor], base_val: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(base_val, dtype=torch.float64, device=self.ctx.device)
        for deg, coeff in poly.items():
            if deg == 0:
                out = out + coeff
            elif deg == 1:
                out = out + coeff * base_val
            else:
                out = out + coeff * torch.pow(base_val, deg)
        return out

    def _try_conservative_product_projection(self, node: _Node, consts: torch.Tensor, strong_memo: Dict) -> Optional[torch.Tensor]:
        """
        鎷︽埅褰㈠ f(q) * D(ax, q) 鐨勪繚瀹堝舰寮忎箻绉紝閫氳繃瑙ｆ瀽绉垎鏋勯€犻€氶噺 F(q)锛?        鐒跺悗鏃犲鏁版姇褰卞埌寮卞舰寮忥細 project_weak(F(q), {ax: 1})銆?        瀹屽叏閬垮厤楂橀鍣０琚己褰㈠紡瀵兼暟鍜岄潪绾挎€ч」鐩镐箻鏀惧ぇ銆?        """
        if node.op != "*" or len(node.children) != 2:
            return None

        for deriv_idx in (0, 1):
            deriv_node = node.children[deriv_idx]
            multiplier_node = node.children[1 - deriv_idx]

            if deriv_node.op != "D":
                continue

            base_node, axes = self._get_deriv_chain(deriv_node)
            if len(axes) != 1:
                continue

            axis = axes[0]
            if axis == "t" or axis not in self.ctx.coords:
                continue

            # IBP rewrite is valid for derivative-free state expressions.
            if not self._is_derivative_free_state_expr(base_node):
                continue

            poly = self._extract_polynomial_in_base(multiplier_node, base_node, consts, strong_memo)
            if poly is None:
                continue

            primitive = self._integrate_poly_dict(poly)
            base_val = self._eval_node(base_node, consts, strong_memo)
            flux_val = self._eval_poly_dict(primitive, base_val)

            # 銆愪慨澶嶇偣 2銆戯細灏嗙┖闂村鏁板畬鍏ㄨ浆绉诲埌娴嬭瘯鍑芥暟涓婏紝瀹岀編娑堥櫎寮哄舰寮忓鏁颁骇鐢熺殑鍚夊竷鏂櫔澹般€?            # 渚嬪瀵逛簬 u * u_x锛屾湰璐ㄤ笂鏄湪璁＄畻绉垎 -int( 0.5 * u^2 * phi_x )
            return self.ctx.project_weak(flux_val, {axis: 1})

        return None

    def evaluate_weak(self, token_seq: List[int], constants: Optional[np.ndarray] = None) -> torch.Tensor:
        if not is_valid_sequence(token_seq):
            return self.ctx.project_weak(self._nan.clone(), {})
        consts = self._as_tensor_constant(constants, token_seq)
        try:
            root = _parse_cached_tree(list(map(int, token_seq)))
        except Exception:
            return self.ctx.project_weak(self._nan.clone(), {})

        try:
            res = self._eval_target_node(root, consts, {}, {})
            return torch.clamp(res, -1e10, 1e10)
        except Exception:
            return self.ctx.project_weak(self._nan.clone(), {})

    def _eval_target_node(self, node: _Node, consts: torch.Tensor, strong_memo: Dict, target_memo: Dict) -> torch.Tensor:
        cache_key = f"{self.ctx.cache_id}_target_{node.key}"

        # Per-evaluation memo is safe for constant-bearing nodes.
        cached_local = target_memo.get(cache_key)
        if cached_local is not None:
            return cached_local

        if not node.has_const and cache_key in _WEAK_TARGET_CACHE:
            _EVAL_CACHE_STATS["hits"] += 1
            _EVAL_CACHE_STATS["saved_evals"] += 1
            return _WEAK_TARGET_CACHE[cache_key]

        op = node.op
        val = None

        if op == "+":
            val = self._eval_target_node(node.children[0], consts, strong_memo, target_memo) + self._eval_target_node(node.children[1], consts, strong_memo, target_memo)
        elif op == "neg":
            val = -self._eval_target_node(node.children[0], consts, strong_memo, target_memo)
        elif op == "*":
            conservative_val = self._try_conservative_product_projection(node, consts, strong_memo)
            if conservative_val is not None:
                val = conservative_val
            else:
                left_strong = self._eval_node(node.children[0], consts, strong_memo)
                right_strong = self._eval_node(node.children[1], consts, strong_memo)
                if self._is_scalar_tensor(left_strong):
                    val = left_strong * self._eval_target_node(node.children[1], consts, strong_memo, target_memo)
                elif self._is_scalar_tensor(right_strong):
                    val = self._eval_target_node(node.children[0], consts, strong_memo, target_memo) * right_strong
                else:
                    val = self.ctx.project_weak(left_strong * right_strong, {})
        elif op == "D":
            base_node, axes = self._get_deriv_chain(node)

            # --- 馃専 鎸佷箙鍖栫紦瀛橈細鍘熺敓鍦烘眰瀵?(璺‥poch澶嶇敤) ---
            is_native_field = base_node.op in self.ctx.fields
            ctx_cache_key = None
            if is_native_field and not node.has_const:
                axes_str = "".join(sorted(axes))
                ctx_cache_key = f"weak_D_{base_node.op}_{axes_str}"

                if not hasattr(self.ctx, "_cache"):
                    self.ctx._cache = {}
                cached_val = self.ctx._cache.get(ctx_cache_key, None)

                if cached_val is not None:
                    val = cached_val

            if val is None:
                derivative_orders: Dict[str, int] = {}
                for ax in axes:
                    derivative_orders[ax] = derivative_orders.get(ax, 0) + 1
                base_val = self._eval_node(base_node, consts, strong_memo)

                is_clean_periodic = all(self.ctx.periodic_axes.get(ax, False) for ax in axes if ax != "t")
                has_time_deriv = "t" in axes

                if is_clean_periodic and not has_time_deriv:
                    strong_val = base_val
                    for ax, order in derivative_orders.items():
                        idx = self.ctx.axes_order.index(ax)
                        L = self.ctx.domain_lengths.get(ax, 1.0)
                        strong_val = robust_strong_derivative(strong_val, idx, L, order, is_periodic=True)
                    val = self.ctx.project_weak(strong_val, {})
                else:
                    val = self.ctx.project_weak(base_val, derivative_orders)

                if ctx_cache_key is not None:
                    self.ctx._cache[ctx_cache_key] = val

        elif op == "lap":
            child_node = node.children[0] if node.children else None
            child_field = child_node.op if child_node else "u"

            # --- 馃専 鎸佷箙鍖栫紦瀛橈細Laplacian 瀹忕畻瀛?---
            ctx_cache_key = f"weak_lap_{child_field}"
            if not hasattr(self.ctx, "_cache"):
                self.ctx._cache = {}
            cached_val = self.ctx._cache.get(ctx_cache_key, None)
            if cached_val is not None:
                val = cached_val
            else:
                base_val = self.ctx.get(child_field)
                if base_val is None:
                    val = self._nan.clone()
                else:
                    is_clean_periodic = all(self.ctx.periodic_axes.get(ax, False) for ax in [a for a in self.ctx.axes_order if a != "t"])
                    acc = None
                    for ax in [a for a in self.ctx.axes_order if a != "t"]:
                        if is_clean_periodic:
                            idx = self.ctx.axes_order.index(ax)
                            L = self.ctx.domain_lengths.get(ax, 1.0)
                            term_strong = robust_strong_derivative(base_val, idx, L, 2, True)
                            term = self.ctx.project_weak(term_strong, {})
                        else:
                            term = self.ctx.project_weak(base_val, {ax: 2})
                        acc = term if acc is None else acc + term
                    val = acc if acc is not None else self._nan.clone()
                    self.ctx._cache[ctx_cache_key] = val

        elif op == "adv":
            child_node = node.children[0] if node.children else None
            if child_node is None or child_node.op not in self.ctx.fields:
                val = self._nan.clone()
            else:
                child_field = child_node.op

                # --- persistent cache for the physical advection macro ---
                ctx_cache_key = f"weak_adv_{child_field}"
                if not hasattr(self.ctx, "_cache"):
                    self.ctx._cache = {}
                cached_val = self.ctx._cache.get(ctx_cache_key, None)
                if cached_val is not None:
                    val = cached_val
                else:
                    child_val = self.ctx.get(child_field)
                    vel_map = {"x": "u", "y": "v", "z": "w"}
                    fallback = self.ctx.fields[0] if getattr(self.ctx, "fields", None) else "u"

                    acc = None
                    for ax in [a for a in self.ctx.axes_order if a != "t"]:
                        vel = vel_map.get(ax, fallback)
                        if vel not in self.ctx.fields:
                            vel = fallback
                        vel_val = self.ctx.get(vel)
                        if vel_val is None or child_val is None:
                            continue

                        if vel == child_field:
                            # Conservative self-advection path:
                            #     q * q_axis = 0.5 * D_axis(q^2)
                            # The derivative is carried by the weak-form test
                            # function, which is much less sensitive to noisy
                            # Burgers-like shocks than FFT(q_axis) * q.
                            term = self.ctx.project_weak(0.5 * vel_val * vel_val, {ax: 1})
                        else:
                            # Cross-advection, e.g. v*u_y or u*v_x.  Use the
                            # high-accuracy 5th/Scheme-D periodic spectral path
                            # whenever the axis is periodic; fall back to local
                            # finite differences only for genuinely non-periodic
                            # axes.  This restores clean 2D Burgers coefficient
                            # accuracy without changing the conservative
                            # self-advection weak rewrite above.
                            idx = self.ctx.axes_order.index(ax)
                            L = self.ctx.domain_lengths.get(ax, 1.0)
                            if bool(self.ctx.periodic_axes.get(ax, False)):
                                strong_dx = robust_strong_derivative(child_val, idx, L, order=1, is_periodic=True)
                            else:
                                n_points = int(child_val.shape[idx])
                                spacing = float(L) / float(max(1, n_points))
                                strong_dx = torch.gradient(child_val, spacing=(spacing,), dim=idx)[0]
                            term = self.ctx.project_weak(vel_val * strong_dx, {})

                        acc = term if acc is None else acc + term
                    val = acc if acc is not None else self._nan.clone()
                    self.ctx._cache[ctx_cache_key] = val
        else:
            strong_val = self._eval_node(node, consts, strong_memo)
            if isinstance(strong_val, torch.Tensor):
                if strong_val.ndim == 0:
                    strong_val = torch.full_like(self.ctx.get(self.ctx.fields[0]), strong_val.item())
                elif strong_val.shape != self.ctx.get(self.ctx.fields[0]).shape:
                    strong_val = torch.broadcast_to(strong_val, self.ctx.get(self.ctx.fields[0]).shape)
            val = self.ctx.project_weak(strong_val, {})

        # 褰撳墠 evaluation 鍐呯殑鎵€鏈夊瓙鏍戦兘缂撳瓨锛涜法 evaluation 鐨勫叏灞€缂撳瓨鍙繚瀛樻棤甯告暟瀛愭爲銆?        target_memo[cache_key] = val
        if not node.has_const:
            _WEAK_TARGET_CACHE[cache_key] = val
        return val

    def compile_fast_evaluator(self, token_seq: List[int]) -> callable:
        try: root_node = _Compiler(list(map(int, token_seq))).parse()
        except Exception: return lambda consts: self._nan.clone()

        def build_expr(n: _Node) -> str:
            if n is None: return "self._nan"
            op = n.op
            if op in self.ctx.fields or op in self.ctx.coords: return f"self.ctx.get('{op}')"
            elif op == "const": return f"consts[{n.const_index}]"
            elif op == "D":
                child_expr = build_expr(n.children[0])
                idx = self.ctx.axes_order.index(n.axis)
                L = self.ctx.domain_lengths.get(n.axis, 1.0)
                is_per = bool(self.ctx.periodic_axes.get(n.axis, False))
                return f"robust_strong_derivative({child_expr}, {idx}, {L}, 1, {is_per})"
            elif op == "neg": return f"(-{build_expr(n.children[0])})"
            elif op == "sq": return f"({build_expr(n.children[0])} * {build_expr(n.children[0])})"
            elif op == "cube": return f"({build_expr(n.children[0])} * {build_expr(n.children[0])} * {build_expr(n.children[0])})"
            elif op == "+": return f"({build_expr(n.children[0])} + {build_expr(n.children[1])})"
            elif op == "*": return f"({build_expr(n.children[0])} * {build_expr(n.children[1])})"
            elif op == "/": return f"({build_expr(n.children[0])} / (torch.abs({build_expr(n.children[1])}) + 1e-6))"
            elif op == "^": return f"torch.pow(torch.abs({build_expr(n.children[0])}) + 1e-8, torch.clamp({build_expr(n.children[1])}, -4.0, 4.0))"
            elif op == "sin": return f"torch.sin({build_expr(n.children[0])})"
            elif op == "cos": return f"torch.cos({build_expr(n.children[0])})"
            elif op == "exp": return f"torch.exp(torch.clamp({build_expr(n.children[0])}, -20.0, 20.0))"
            elif op == "log": return f"torch.log(torch.abs({build_expr(n.children[0])}) + 1e-8)"
            elif op == "lap":
                child_node = n.children[0] if n.children else None
                if child_node is None or child_node.op not in self.ctx.fields:
                    return "self._nan"
                child_expr = build_expr(child_node)
                terms = []
                for ax in [a for a in self.ctx.axes_order if a != "t"]:
                    idx = self.ctx.axes_order.index(ax)
                    L = self.ctx.domain_lengths.get(ax, 1.0)
                    is_per = bool(self.ctx.periodic_axes.get(ax, False))
                    terms.append(f"robust_strong_derivative({child_expr}, {idx}, {L}, 2, {is_per})")
                return "(" + " + ".join(terms) + ")" if terms else "self._nan"
            elif op == "adv":
                child_node = n.children[0] if n.children else None
                if child_node is None or child_node.op not in self.ctx.fields:
                    return "self._nan"
                child_expr = build_expr(child_node)
                terms = []
                vel_map = {"x": "u", "y": "v", "z": "w"}
                fallback = self.ctx.fields[0] if getattr(self.ctx, "fields", None) else "u"
                for ax in [a for a in self.ctx.axes_order if a != "t"]:
                    vel = vel_map.get(ax, fallback)
                    if vel not in self.ctx.fields:
                        vel = fallback
                    idx = self.ctx.axes_order.index(ax)
                    L = self.ctx.domain_lengths.get(ax, 1.0)
                    is_per = bool(self.ctx.periodic_axes.get(ax, False))
                    terms.append(f"self.ctx.get('{vel}') * robust_strong_derivative({child_expr}, {idx}, {L}, 1, {is_per})")
                return "(" + " + ".join(terms) + ")" if terms else "self._nan"
            return "self._nan"

        expr_str = build_expr(root_node)
        func_code = f"def compiled_eval(self, consts):\n    return {expr_str}"
        local_vars = {}
        exec(func_code, {"torch": torch, "robust_strong_derivative": robust_strong_derivative}, local_vars)
        func = local_vars['compiled_eval']
        return lambda consts: func(self, consts)

    # =========================================================================
    # 銆愪慨澶嶉噸鐐?2銆? 褰诲簳绉婚櫎 np.asarray锛屽叏閲忚繑鍥?torch.Tensor
    # =========================================================================
    def _eval_node(self, node: _Node, consts: torch.Tensor, memo: Dict) -> torch.Tensor:
        context_aware_key = f"{self.ctx.cache_id}_{node.key}"

        # Local strong-form memo is safe for the current constants.
        cached_local = memo.get(node.key)
        if cached_local is not None:
            return cached_local

        if not node.has_const:
            _EVAL_CACHE_STATS["queries"] += 1
            if context_aware_key in _SUBTREE_CACHE:
                _EVAL_CACHE_STATS["hits"] += 1
                _EVAL_CACHE_STATS["saved_evals"] += 1
                memo[node.key] = _SUBTREE_CACHE[context_aware_key]
                return _SUBTREE_CACHE[context_aware_key]
            _EVAL_CACHE_STATS["misses"] += 1

        op = node.op
        val = None

        if op in ("u", "v", "w") or op in self.ctx.coords:
            val = self.ctx.get(op)
        elif op == "const":
            val = consts[node.const_index] if node.const_index < len(consts) else torch.tensor(1.0, dtype=torch.float64, device=self.ctx.device)
        elif op == "D":
            base_node, axes = self._get_deriv_chain(node)

            # --- 馃専 鎸佷箙鍖栫紦瀛橈細鎷︽埅鍘熺敓鍦哄己姹傚 ---
            is_base_field = base_node.op in self.ctx.fields and len(set(axes)) == 1 and len(axes) <= 3 and axes[0] != 't'
            ctx_cache_key = None
            if is_base_field:
                ax = axes[0]
                ctx_cache_key = f"{base_node.op}_{ax * len(axes)}"
                if not hasattr(self.ctx, "_cache"):
                    self.ctx._cache = {}
                cached_val = self.ctx._cache.get(ctx_cache_key, None)
                if cached_val is not None:
                    if not node.has_const: _SUBTREE_CACHE[context_aware_key] = cached_val
                    return cached_val

            child = self._eval_node(node.children[0], consts, memo)
            ax = node.axis
            if ax not in self.ctx.coords:
                val = self._nan.clone()
            else:
                idx = self.ctx.axes_order.index(ax)
                L = self.ctx.domain_lengths.get(ax, 1.0)
                is_periodic = bool(self.ctx.periodic_axes.get(ax, False))
                if ax != "t":
                    val = robust_strong_derivative(child, idx, L, 1, is_periodic)
                else:
                    val = self._nan.clone()

            if ctx_cache_key is not None:
                self.ctx._cache[ctx_cache_key] = val

        elif op == "lap":
            val = self._eval_lap_strong(node, consts, memo)
        elif op == "adv":
            child_node = node.children[0] if node.children else None
            if child_node is None or child_node.op not in self.ctx.fields:
                val = self._nan.clone()
                memo[node.key] = val
                return val
            child = self._eval_node(child_node, consts, memo)
            vel_map = {"x": "u", "y": "v", "z": "w"}
            fallback = self.ctx.fields[0] if getattr(self.ctx, "fields", None) else "u"
            acc = torch.zeros_like(child)
            for ax in [a for a in self.ctx.axes_order if a != "t"]:
                vel = vel_map.get(ax, fallback)
                if vel not in self.ctx.fields:
                    vel = fallback
                idx = self.ctx.axes_order.index(ax)
                L = self.ctx.domain_lengths.get(ax, 1.0)
                is_periodic = bool(self.ctx.periodic_axes.get(ax, False))
                acc = acc + self.ctx.get(vel) * robust_strong_derivative(child, idx, L, 1, is_periodic)
            val = acc
        elif op == "neg": val = -self._eval_node(node.children[0], consts, memo)
        elif op == "sq":
            child_val = self._eval_node(node.children[0], consts, memo)
            val = child_val * child_val
        elif op == "cube":
            child_val = self._eval_node(node.children[0], consts, memo)
            val = child_val * child_val * child_val
        elif op == "+": val = self._eval_node(node.children[0], consts, memo) + self._eval_node(node.children[1], consts, memo)
        elif op == "*": val = self._eval_node(node.children[0], consts, memo) * self._eval_node(node.children[1], consts, memo)
        elif op == "/": val = self._eval_node(node.children[0], consts, memo) / (torch.abs(self._eval_node(node.children[1], consts, memo)) + 1e-6)
        elif op == "^": val = torch.pow(torch.abs(self._eval_node(node.children[0], consts, memo)) + 1e-8, torch.clamp(self._eval_node(node.children[1], consts, memo), -4.0, 4.0))
        elif op == "sin": val = torch.sin(self._eval_node(node.children[0], consts, memo))
        elif op == "cos": val = torch.cos(self._eval_node(node.children[0], consts, memo))
        elif op == "exp": val = torch.exp(torch.clamp(self._eval_node(node.children[0], consts, memo), -20.0, 20.0))
        elif op == "log": val = torch.log(torch.abs(self._eval_node(node.children[0], consts, memo)) + 1e-8)
        else: val = self._nan.clone()

        memo[node.key] = val
        if not node.has_const:
            _SUBTREE_CACHE[context_aware_key] = val
        return val

def warmup_shared_subtrees(ctx, subtree_sequences, max_items: Optional[int] = None) -> int:
    """Preload constant-free shared subtrees into the global subtree cache.

    The previous implementation collected candidate subtrees but returned before
    evaluating them, so `Warm` stayed at zero.  This version evaluates only
    short, valid, constant-free subtrees.  It preserves numerical precision: the
    same WeakFormEvaluator code path is used, and only deterministic tensor
    results are cached.
    """
    if subtree_sequences is None:
        return 0

    ctxs = list(ctx) if isinstance(ctx, (list, tuple)) else [ctx]
    if not ctxs:
        return 0

    seqs: List[List[int]] = []
    seen: set[Tuple[int, ...]] = set()
    for seq in subtree_sequences:
        tup = tuple(int(t) for t in seq)
        if not tup or tup in seen:
            continue
        candidate = list(tup)
        if count_constants(candidate) > 0:
            continue
        if not is_valid_sequence(candidate):
            continue
        seen.add(tup)
        seqs.append(candidate)
        if max_items is not None and len(seqs) >= int(max_items):
            break

    if not seqs:
        return 0

    warmed = 0
    with torch.no_grad():
        for cctx in ctxs:
            ev = WeakFormEvaluator(cctx)
            for seq in seqs:
                try:
                    out = ev.evaluate(seq, [])
                    if torch.is_tensor(out) and bool(torch.all(torch.isfinite(out)).item()):
                        warmed += 1
                except Exception:
                    continue
    return int(warmed)

def _structural_risk_stats(token_seq) -> Dict[str, float]:
    """Generic structural diagnostics used by the MDL score.

    This is deliberately *not* a task-specific prior.  It suppresses broad
    weak-form overfitting patterns that are numerically cheap ways to reduce
    residuals but rarely represent stable discovered PDEs: products/powers of
    explicit derivatives and derivatives applied to composite expressions.

    Compact physics primitives (``lap(field)``, ``adv(field)``) are treated as
    controlled operators and are not counted as raw derivative products.  This
    keeps Burgers/KdV/Burgers2D capacity intact while making reaction-diffusion
    searches less vulnerable to terms such as ``u*u_y*v_x`` or ``D_x(u*v)``.
    """
    stats = {
        "derivative_product_count": 0.0,
        "derivative_power_count": 0.0,
        "explicit_high_order_deriv_count": 0.0,
        "raw_derivative_count": 0.0,
        "composite_derivative_count": 0.0,
        "nested_composite_derivative_count": 0.0,
        "zero_order_field_count": 0.0,
    }
    try:
        root = _Compiler(list(map(int, token_seq))).parse()
    except Exception:
        return stats

    native_fields = {"u", "v", "w"}
    derivative_safe_macros = {"lap", "adv"}

    def count_zero_order_fields(n, parent_op: str = "") -> None:
        op = str(getattr(n, "op", ""))
        children = list(getattr(n, "children", []) or [])
        if op in native_fields and parent_op not in {"D", "lap", "adv"}:
            stats["zero_order_field_count"] += 1.0
        for child in children:
            count_zero_order_fields(child, op)

    def walk(n):
        op = str(getattr(n, "op", ""))
        children = list(getattr(n, "children", []) or [])
        child_infos = [walk(c) for c in children]

        is_native_field = op in native_fields
        contains_raw_deriv = False
        deriv_order = 0
        contains_field = is_native_field or any(info[3] for info in child_infos)

        if op == "D":
            stats["raw_derivative_count"] += 1.0
            child = children[0] if children else None
            child_op = str(getattr(child, "op", "")) if child is not None else ""
            child_info = child_infos[0] if child_infos else (False, 0, False, False)
            child_contains_deriv, child_order, child_is_native, _child_contains_field = child_info

            contains_raw_deriv = True
            deriv_order = int(child_order) + 1

            # ``D(D(D(u)))`` is a high-order native derivative and should remain
            # available for KdV-like equations.  ``D(u*v)``, ``D(sq(u))``,
            # ``D(lap(u))`` etc. are composite derivatives and receive a
            # stronger generic risk penalty because they often expand into many
            # high-variance derivative products.
            direct_native_chain = bool(child_is_native or child_op == "D")
            if not direct_native_chain:
                stats["composite_derivative_count"] += 1.0
                if child_contains_deriv or child_op in derivative_safe_macros:
                    stats["nested_composite_derivative_count"] += 1.0

            if deriv_order > 3:
                stats["explicit_high_order_deriv_count"] += 1.0

        elif op in derivative_safe_macros:
            # These operators are grammar-constrained to native fields.  Treat
            # them as compact PDE primitives, not as raw derivative products.
            contains_raw_deriv = False
            deriv_order = 0

        else:
            contains_raw_deriv = any(info[0] for info in child_infos)
            deriv_order = max([info[1] for info in child_infos], default=0)

        if op == "*" and len(child_infos) >= 2:
            n_deriv_children = sum(1 for info in child_infos if info[0])
            if n_deriv_children >= 2:
                stats["derivative_product_count"] += 1.0
                # Heavier penalty if both sides already contain higher-order or
                # composite derivatives, e.g. v_x * v_xxy.
                if any(info[1] >= 2 for info in child_infos if info[0]):
                    stats["derivative_product_count"] += 0.5
        elif op in {"sq", "cube", "^"} and child_infos and child_infos[0][0]:
            stats["derivative_power_count"] += 1.0
            if child_infos[0][1] >= 2:
                stats["derivative_power_count"] += 0.5

        return contains_raw_deriv, deriv_order, is_native_field, contains_field

    count_zero_order_fields(root)
    walk(root)
    return stats


def _parse_int_list(value, default=(3, 5)) -> List[int]:
    """Parse a small list of odd kernel sizes from config values."""
    if value is None:
        raw = list(default)
    elif isinstance(value, str):
        raw = [x.strip() for x in value.replace(';', ',').split(',') if x.strip()]
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raw = [value]
    out: List[int] = []
    for x in raw:
        try:
            k = int(x)
        except Exception:
            continue
        if k < 3:
            continue
        if k % 2 == 0:
            k += 1
        if k not in out:
            out.append(k)
    return out or list(default)


def _parse_symbol_set(value) -> set:
    if value is None:
        return set()
    if isinstance(value, str):
        return {x.strip() for x in value.replace(";", ",").split(",") if x.strip()}
    if isinstance(value, (list, tuple, set)):
        return {str(x).strip() for x in value if str(x).strip()}
    return set()


def _passes_diffusion_affine_guard(token_seq) -> Tuple[bool, str]:
    try:
        root = _Compiler(list(map(int, token_seq))).parse()
    except Exception:
        return False, "diffusion_guard_parse_failed"

    native_fields = {"u", "v", "w"}

    def is_lap_field(n) -> bool:
        children = list(getattr(n, "children", []) or [])
        if str(getattr(n, "op", "")) != "lap" or len(children) != 1:
            return False
        return str(getattr(children[0], "op", "")) in native_fields

    def is_const(n) -> bool:
        return str(getattr(n, "op", "")) == "const"

    def is_atom(n) -> bool:
        op = str(getattr(n, "op", ""))
        children = list(getattr(n, "children", []) or [])
        if is_const(n) or is_lap_field(n):
            return True
        if op == "neg" and len(children) == 1:
            return is_atom(children[0])
        if op == "*" and len(children) == 2:
            left, right = children
            return (is_const(left) and is_lap_field(right)) or (is_const(right) and is_lap_field(left))
        return False

    def is_expr(n) -> bool:
        op = str(getattr(n, "op", ""))
        children = list(getattr(n, "children", []) or [])
        if is_atom(n):
            return True
        if op == "+" and len(children) == 2:
            return is_expr(children[0]) and is_expr(children[1])
        if op == "neg" and len(children) == 1:
            return is_expr(children[0])
        return False

    return (True, "ok") if is_expr(root) else (False, "diffusion_non_affine_rhs")


def _target_field_from_time_derivative(target_field: str) -> str:
    target = normalize_target_field_name(target_field)
    if target.startswith("d") and target.endswith("_t"):
        return target[1:-2]
    return "u"


def _strong_time_lhs(ctx: DataContext, target_field: str) -> Optional[torch.Tensor]:
    field = _target_field_from_time_derivative(target_field)
    base = ctx.get(field)
    if base is None or "t" not in getattr(ctx, "axes_order", []):
        return None
    axis = int(ctx.axes_order.index("t"))
    step = float(ctx.steps.get("t", 1.0))
    return torch.gradient(base, spacing=(step,), dim=axis)[0]


def _apply_trim_mask(ctx: DataContext, lhs: torch.Tensor, rhs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    slices = []
    for ax in getattr(ctx, "axes_order", []):
        n = int(lhs.shape[len(slices)])
        trim = int(getattr(ctx, "trim_map", {}).get(ax, 0))
        if trim <= 0:
            slices.append(slice(None))
        elif 2 * trim >= n:
            slices.append(slice(None))
        else:
            slices.append(slice(trim, n - trim))
    if slices:
        key = tuple(slices)
        lhs = lhs[key]
        rhs = rhs[key]
    return lhs, rhs


def evaluate_scoring_tensors(
    ctx: DataContext,
    token_seq,
    constants=None,
    *,
    target_field: str = "du_t",
    scoring_form: Optional[str] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return aligned LHS/RHS tensors for the configured PDE score."""
    form = str(scoring_form or getattr(ctx, "scoring_form", "weak") or "weak").strip().lower()
    ev = WeakFormEvaluator(ctx)
    if form == "strong":
        lhs = _strong_time_lhs(ctx, target_field)
        rhs = ev.evaluate(token_seq, constants)
        if lhs is None or rhs is None:
            return None, None
        lhs, rhs = _apply_trim_mask(ctx, lhs, rhs)
        return lhs, rhs

    lhs = ctx.weak_lhs_cache.get(target_field)
    rhs = ev.evaluate_weak(token_seq, constants)
    return lhs, rhs


def _separable_roll_smooth(x: torch.Tensor, kernel_size: int, dims: Optional[List[int]] = None) -> torch.Tensor:
    """Cheap same-shape separable moving average used for multi-window scoring.

    This does not create new candidates and does not change the weak evaluator.
    It only asks whether the already-computed weak residual remains consistent
    after being viewed at coarser weak-feature scales.  Rolling is used for a
    same-shape, allocation-light implementation; for non-periodic temporal axes
    this is intentionally a weak diagnostic signal, not a derivative estimate.
    """
    if not torch.is_tensor(x) or x.ndim == 0:
        return x
    k = int(max(1, kernel_size))
    if k <= 1:
        return x
    half = k // 2
    out = x
    if dims is None:
        dims = list(range(int(x.ndim)))
    for dim in dims:
        dim = int(dim)
        if dim < 0 or dim >= out.ndim or int(out.shape[dim]) < k:
            continue
        acc = torch.zeros_like(out)
        for shift in range(-half, half + 1):
            acc = acc + torch.roll(out, shifts=int(shift), dims=dim)
        out = acc / float(k)
    return out


def _spectral_residual_stats(residual: torch.Tensor, ctx, denom: float, kwargs: Dict[str, Any]) -> Dict[str, float]:
    """Frequency-band diagnostic for weak residuals.

    Missing diffusion terms often leave a spatial high-frequency residual that
    a reaction-only RHS can hide in aggregate MSE.  This score is purely an
    evaluator/reward-shaping diagnostic: it does not inject terms, does not use
    time integration, and does not perform sparse regression.
    """
    out = {
        "spectral_high_fraction": 0.0,
        "spectral_mid_fraction": 0.0,
        "spectral_high_mse": 0.0,
        "spectral_mid_mse": 0.0,
    }
    if not bool(kwargs.get("spectral_residual_enable", False)):
        return out
    if not torch.is_tensor(residual) or residual.ndim < 2:
        return out
    try:
        axes_order = list(getattr(ctx, "axes_order", []))
        spatial_axes = [ax for ax in axes_order if ax != "t"]
        dims = [axes_order.index(ax) for ax in spatial_axes if ax in axes_order]
        dims = [d for d in dims if 0 <= int(d) < residual.ndim and int(residual.shape[int(d)]) >= 4]
        if not dims:
            return out
        # Diagnostic only: float32 is sufficient and much faster than float64.
        # Optionally downsample spatial axes before FFT to keep this out of the
        # candidate-evaluation hot path.
        r = residual.detach().to(dtype=torch.float32)
        subsample = int(max(1, kwargs.get("spectral_residual_subsample", 1)))
        if subsample > 1:
            slicer = [slice(None)] * r.ndim
            for dim in dims:
                if int(r.shape[int(dim)]) >= 2 * subsample:
                    slicer[int(dim)] = slice(None, None, subsample)
            r = r[tuple(slicer)]
            dims = [d for d in dims if 0 <= int(d) < r.ndim and int(r.shape[int(d)]) >= 4]
            if not dims:
                return out
        # Remove the residual mean so the zero-frequency bin does not dominate
        # band ratios for equations with source terms.
        r = r - torch.mean(r)
        power = torch.abs(torch.fft.fftn(r, dim=tuple(dims))) ** 2
        total = torch.sum(power) + 1e-20
        radius_sq = torch.zeros_like(power, dtype=torch.float32)
        for dim in dims:
            n = int(power.shape[int(dim)])
            freq = torch.fft.fftfreq(n, device=power.device, dtype=torch.float32).abs()
            maxf = torch.max(freq)
            if float(maxf.detach().cpu()) <= 0.0:
                continue
            freq = freq / (maxf + 1e-30)
            shape = [1] * power.ndim
            shape[int(dim)] = n
            radius_sq = radius_sq + freq.view(*shape) ** 2
        radius = torch.sqrt(radius_sq / max(1, len(dims)))
        high_th = float(kwargs.get("spectral_high_threshold", 0.45))
        mid_th = float(kwargs.get("spectral_mid_threshold", 0.20))
        high_mask = radius >= high_th
        mid_mask = (radius >= mid_th) & (radius < high_th)
        high_frac = torch.sum(power[high_mask]) / total if bool(torch.any(high_mask).item()) else torch.tensor(0.0, dtype=torch.float32, device=power.device)
        mid_frac = torch.sum(power[mid_mask]) / total if bool(torch.any(mid_mask).item()) else torch.tensor(0.0, dtype=torch.float32, device=power.device)
        mse_scaled = float((residual.detach().pow(2).mean() / (float(denom) + 1e-12)).item())
        out["spectral_high_fraction"] = float(high_frac.detach().cpu().item())
        out["spectral_mid_fraction"] = float(mid_frac.detach().cpu().item())
        out["spectral_high_mse"] = float(mse_scaled * out["spectral_high_fraction"])
        out["spectral_mid_mse"] = float(mse_scaled * out["spectral_mid_fraction"])
    except Exception:
        return out
    return out


def _multi_window_residual_stats(residual: torch.Tensor, ctx, denom: float, kwargs: Dict[str, Any]) -> Dict[str, float]:
    """Lightweight multi-window weak residual consistency score.

    Instead of rebuilding several DataContext objects with different weak
    windows, we view the current weak residual at several coarser local scales.
    True PDE structures should keep residuals low and comparatively stable under
    this scale change; reaction-only surrogates for reaction-diffusion equations
    tend to leave scale-sensitive residual energy.
    """
    out = {
        "multi_window_mean_mse": 0.0,
        "multi_window_logvar": 0.0,
        "multi_window_highfreq_proxy": 0.0,
    }
    if not bool(kwargs.get("multi_window_enable", False)):
        return out
    if not torch.is_tensor(residual) or residual.ndim == 0:
        return out
    try:
        kernels = _parse_int_list(kwargs.get("multi_window_kernels", (3, 5)), default=(3, 5))
        # Diagnostic only: use float32 and optionally downsample spatial axes.
        r = residual.detach().to(dtype=torch.float32)
        axes_order = list(getattr(ctx, "axes_order", []))
        if axes_order and len(axes_order) == r.ndim:
            if bool(kwargs.get("multi_window_spatial_only", True)):
                dims = [i for i, ax in enumerate(axes_order) if ax != "t" and int(r.shape[i]) >= 3]
            else:
                dims = [i for i, _ax in enumerate(axes_order) if int(r.shape[i]) >= 3]
        else:
            dims = list(range(r.ndim))
        subsample = int(max(1, kwargs.get("multi_window_subsample", 1)))
        if subsample > 1 and dims:
            slicer = [slice(None)] * r.ndim
            for dim in dims:
                if int(r.shape[int(dim)]) >= 2 * subsample:
                    slicer[int(dim)] = slice(None, None, subsample)
            r = r[tuple(slicer)]
            dims = [d for d in dims if 0 <= int(d) < r.ndim and int(r.shape[int(d)]) >= 3]
        base_mse = float((r.pow(2).mean() / (float(denom) + 1e-12)).item())
        vals = [base_mse]
        smooth_vals = []
        for k in kernels:
            sm = _separable_roll_smooth(r, int(k), dims=dims)
            val = float((sm.pow(2).mean() / (float(denom) + 1e-12)).item())
            if math.isfinite(val):
                vals.append(val)
                smooth_vals.append(val)
        if len(vals) <= 1:
            return out
        arr = np.asarray(vals, dtype=np.float64)
        logs = np.log10(np.maximum(arr, 1e-15))
        out["multi_window_mean_mse"] = float(np.mean(arr[1:])) if len(arr) > 1 else 0.0
        out["multi_window_logvar"] = float(np.var(logs)) if len(logs) > 1 else 0.0
        # Residual energy removed by coarse local averaging is a cheap proxy for
        # unresolved high-frequency error.  Missing diffusion terms typically
        # increase this quantity.
        if smooth_vals:
            out["multi_window_highfreq_proxy"] = float(max(0.0, base_mse - min(smooth_vals)))
    except Exception:
        return out
    return out

def compose_fitness(base_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified self-contained MDL fitness (Log-Additive BIC version).
    Accuracy-first scoring allows machine-precision weak residuals while still
    applying structural and MDL-style penalties.
    """
    out = dict(base_stats)
    nmse = out.get("residual_mse", 1e10)

    if nmse >= 1e9 or math.isnan(nmse) or math.isinf(nmse):
        out["fitness"] = 1e10
        out["pred_valid"] = False
        return out

    comp = out.get("complexity", 1.0)
    n_eff = out.get("n_eff", 50.0)
    vocab_size = out.get("vocab_size", 20)
    n_const = out.get("num_constants", 0)

    # =========================================================================
    # 1. 閲婃斁鏈哄櫒鏋侀檺 (Machine Precision Release)
    # 褰诲簳搴熼櫎 1e-4 鐨勭‖鎬у簳绾匡紝鍏佽娣卞叆鍒?1e-15 鐨勫弻绮惧害鐗╃悊鏋侀檺銆?    # =========================================================================
    eff_nmse = max(nmse, 1e-15)

    # =========================================================================
    # 2. 杞崲鍒?Log10 瀵规暟灏哄害 (娑堥櫎鎮礀鏁堝簲)
    # 骞崇Щ +15.0 淇濊瘉 fitness 濮嬬粓涓烘鏁?(鑼冨洿 0.0 ~ 15.0+)锛屽畬缇庡吋瀹逛笅娓?REINFORCE銆?    # =========================================================================
    log_base_score = math.log10(eff_nmse) + 15.0

    # =========================================================================
    # 3. 涓ユ牸 MDL 瀵规暟鍔犳硶鎯╃綒 (Log-Additive Penalty)
    # =========================================================================
    struct_bits = comp * math.log(max(vocab_size, 2))
    param_bits = n_const * 0.5 * math.log(max(n_eff, 2.0))

    # In accuracy-first mode the MDL weight is configurable.  Reaction systems
    # with weak small-coefficient terms often need a low base MDL weight, while
    # the structural-risk penalty below still suppresses derivative overfit.
    mdl_w = float(out.get("mdl_penalty_weight", 0.25))
    penalty = mdl_w * (struct_bits + param_bits) / max(n_eff, 1.0)

    structural_penalty = 0.0
    # Configurable generic risk weights.  Defaults are stronger than the first
    # FHN-general version because logs showed derivative products/composite
    # derivatives outranking simple reaction-diffusion structures.
    w_deriv_prod = float(out.get("derivative_product_penalty_weight", 1.20))
    w_deriv_pow = float(out.get("derivative_power_penalty_weight", 1.10))
    w_high_deriv = float(out.get("explicit_high_order_deriv_penalty_weight", 1.60))
    w_comp_deriv = float(out.get("composite_derivative_penalty_weight", 1.50))
    w_nested_comp = float(out.get("nested_composite_derivative_penalty_weight", 2.00))
    w_zero_order = float(out.get("zero_order_field_penalty_weight", 0.0))
    structural_penalty += w_deriv_prod * float(out.get("derivative_product_count", 0.0))
    structural_penalty += w_deriv_pow * float(out.get("derivative_power_count", 0.0))
    structural_penalty += w_high_deriv * float(out.get("explicit_high_order_deriv_count", 0.0))
    structural_penalty += w_comp_deriv * float(out.get("composite_derivative_count", 0.0))
    structural_penalty += w_nested_comp * float(out.get("nested_composite_derivative_count", 0.0))
    structural_penalty += w_zero_order * float(out.get("zero_order_field_count", 0.0))
    out["structural_risk_penalty"] = float(structural_penalty)

    # -------------------------------------------------------------------------
    # Multi-scale / spectral weak residual diagnostics.
    # These are generic evaluator terms: no atoms are injected, no sparse
    # regression is used, and time integration is not involved.  They simply reshape the
    # reward landscape so reaction-only surrogates are less likely to hide
    # missing spatial operators such as diffusion.
    # -------------------------------------------------------------------------
    spectral_penalty = 0.0
    spectral_penalty += float(out.get("spectral_residual_weight", 0.0)) * float(out.get("spectral_high_mse", 0.0))
    spectral_penalty += float(out.get("spectral_mid_residual_weight", 0.0)) * float(out.get("spectral_mid_mse", 0.0))
    spectral_penalty += float(out.get("spectral_high_fraction_weight", 0.0)) * float(out.get("spectral_high_fraction", 0.0))
    out["spectral_residual_penalty"] = float(spectral_penalty)

    multi_window_penalty = 0.0
    multi_window_penalty += float(out.get("multi_window_residual_weight", 0.0)) * float(out.get("multi_window_mean_mse", 0.0))
    multi_window_penalty += float(out.get("multi_window_variance_weight", 0.0)) * float(out.get("multi_window_logvar", 0.0))
    multi_window_penalty += float(out.get("multi_window_highfreq_weight", 0.0)) * float(out.get("multi_window_highfreq_proxy", 0.0))
    out["multi_window_penalty"] = float(multi_window_penalty)

    fitness = log_base_score + penalty + structural_penalty + spectral_penalty + multi_window_penalty

    # Contrastive reward score: used only by the decoder REINFORCE update when
    # enabled in wised_framework.  It can be stronger than the final selection
    # fitness without changing the evaluator's best-equation choice.
    contrastive_extra = 0.0
    contrastive_extra += float(out.get("contrastive_reward_spectral_weight", 0.0)) * (
        float(out.get("spectral_high_mse", 0.0)) + 0.25 * float(out.get("spectral_high_fraction", 0.0))
    )
    contrastive_extra += float(out.get("contrastive_reward_multi_window_weight", 0.0)) * (
        float(out.get("multi_window_logvar", 0.0)) + float(out.get("multi_window_highfreq_proxy", 0.0))
    )
    # Stored after hard constraints below.

    # =========================================================================
    # 4. 鐗╃悊鍏堥獙纭害鏉?    # =========================================================================
    max_deriv = out.get("max_deriv_depth", 0)
    if max_deriv > 3:
        fitness += 5.0 * (max_deriv - 2)

    if out.get("diverge_penalty", False):
        fitness += 10.0

    max_zero_order = out.get("max_zero_order_field_terms", None)
    if max_zero_order is not None:
        try:
            max_zero_order_val = float(max_zero_order)
            if float(out.get("zero_order_field_count", 0.0)) > max_zero_order_val:
                fitness += 1.0e6
        except Exception:
            pass

    out["fitness"] = min(max(0.0, fitness), 1e10)
    out["contrastive_reward_score"] = min(max(0.0, fitness + contrastive_extra), 1e10)
    return out

def compute_fitness(token_seq, ctx, constants=None, *args, **kwargs):
    from models.weak_form_evaluator import WeakFormEvaluator
    from models.symbol_vocabulary import IDX2SYM, COMPLEXITY_MAP, SYM2IDX
    # 娉ㄦ剰锛氱‘淇濊繖閲岃皟鐢ㄤ簡浣犵殑 normalize_target_field_name
    target_field = kwargs.get("target_field", "du_t")

    ev = WeakFormEvaluator(ctx)
    safe_constants = constants if constants is not None else []
    if str(kwargs.get("operator_mode", "") or "").strip().lower() in {"diffusion", "diffusion_only", "parabolic_diffusion"}:
        ok, reason = _passes_diffusion_affine_guard(token_seq)
        if not ok:
            return {
                "fitness": 1e10,
                "residual_mse": 1e10,
                "pred_valid": False,
                "struct_guard_reject": True,
                "struct_guard_reason": reason,
            }
    lhs_score, rhs_score = evaluate_scoring_tensors(
        ctx,
        token_seq,
        safe_constants,
        target_field=target_field,
        scoring_form=kwargs.get("scoring_form", getattr(ctx, "scoring_form", "weak")),
    )

    if lhs_score is None or rhs_score is None or torch.any(~torch.isfinite(rhs_score)):
        return {"fitness": 1e10, "pred_valid": False}

    if lhs_score is not None:
        residual = lhs_score - rhs_score
        mse_raw = float(residual.pow(2).mean().item())
        denom = float(torch.var(lhs_score).item()) if getattr(ctx, "normalize_mse", False) else 1.0

        # --- 鑾峰彇鎴戜滑鍦?DataContext 涓璁＄畻濂界殑鐗╃悊鑷敱搴?---
        # Fallback to a conservative physical effective sample size if unavailable.
        n_eff = getattr(ctx, "n_eff_physical", 50.0)
    else:
        return {"fitness": 1e10, "pred_valid": False}

    mse_scaled = mse_raw / (denom + 1e-12)
    syms = [IDX2SYM.get(int(t), "") for t in token_seq]
    forbidden_symbols = _parse_symbol_set(kwargs.get("forbidden_rhs_symbols", None))
    if forbidden_symbols and any(sym in forbidden_symbols for sym in syms):
        return {
            "fitness": 1e10,
            "residual_mse": 1e10,
            "pred_valid": False,
            "struct_guard_reject": True,
            "struct_guard_reason": "forbidden_rhs_symbol",
        }

    # 1. Symbolic complexity.
    comp = 0.0
    for s in syms:
        if s not in ["<START>", "<END>", "<PAD>", "<UNK>"]:
            comp += float(COMPLEXITY_MAP.get(s, 1.0))

    # 2. Consecutive derivative depth.
    deriv_streak = 0
    max_deriv_depth = 0
    for sym in syms:
        if sym in ['D', 'dx', 'dxx', 'dxxx', 'dt', 'dxt', 'dy', 'dyy']:
            deriv_streak += 1
            max_deriv_depth = max(max_deriv_depth, deriv_streak)
        elif sym in ['x', 'y', 'z', 't']:
            pass
        else:
            deriv_streak = 0

    # 3. Generic structural risk diagnostics.
    structural_stats = _structural_risk_stats(token_seq)

    # 3b. Multi-scale and spectral residual diagnostics.
    spectral_stats = _spectral_residual_stats(residual, ctx, denom, kwargs)
    multi_window_stats = _multi_window_residual_stats(residual, ctx, denom, kwargs)

    # 4. 鎷︽埅鍙戞暎鐨勫ぇ甯告暟
    diverge_penalty = False
    for c in safe_constants:
        try:
            val = float(c.item()) if hasattr(c, "item") else float(c)
        except Exception:
            val = 0.0
        if abs(val) > 100.0:
            diverge_penalty = True

    # --- 鏍稿績淇锛氳繕鍘熸鏋舵墍闇€鐨勫叏閮ㄩ敭鍊?---
    base = {
        "residual_mse": mse_scaled,
        "raw_mse": mse_raw,                 # 淇鏃ュ織 MSE=1e18
        "complexity": comp,
        "max_deriv_depth": max_deriv_depth,
        "num_constants": len(safe_constants),
        "diverge_penalty": diverge_penalty,
        "pred_valid": True,                 # 淇 Valid=0 宕╂簝
        "n_eff": n_eff,
        "vocab_size": len(SYM2IDX),
        "mdl_penalty_weight": float(kwargs.get("mdl_penalty_weight", 0.25)),
        "derivative_product_penalty_weight": float(kwargs.get("derivative_product_penalty_weight", 1.20)),
        "derivative_power_penalty_weight": float(kwargs.get("derivative_power_penalty_weight", 1.10)),
        "explicit_high_order_deriv_penalty_weight": float(kwargs.get("explicit_high_order_deriv_penalty_weight", 1.60)),
        "composite_derivative_penalty_weight": float(kwargs.get("composite_derivative_penalty_weight", 1.50)),
        "nested_composite_derivative_penalty_weight": float(kwargs.get("nested_composite_derivative_penalty_weight", 2.00)),
        "zero_order_field_penalty_weight": float(kwargs.get("zero_order_field_penalty_weight", 0.0)),
        "max_zero_order_field_terms": kwargs.get("max_zero_order_field_terms", None),
        "scoring_form": str(kwargs.get("scoring_form", getattr(ctx, "scoring_form", "weak"))),
        "spectral_residual_enable": bool(kwargs.get("spectral_residual_enable", False)),
        "spectral_residual_weight": float(kwargs.get("spectral_residual_weight", 0.0)),
        "spectral_mid_residual_weight": float(kwargs.get("spectral_mid_residual_weight", 0.0)),
        "spectral_high_fraction_weight": float(kwargs.get("spectral_high_fraction_weight", 0.0)),
        "spectral_high_threshold": float(kwargs.get("spectral_high_threshold", 0.45)),
        "spectral_mid_threshold": float(kwargs.get("spectral_mid_threshold", 0.20)),
        "spectral_residual_subsample": int(max(1, kwargs.get("spectral_residual_subsample", 1))),
        "multi_window_enable": bool(kwargs.get("multi_window_enable", False)),
        "multi_window_residual_weight": float(kwargs.get("multi_window_residual_weight", 0.0)),
        "multi_window_variance_weight": float(kwargs.get("multi_window_variance_weight", 0.0)),
        "multi_window_highfreq_weight": float(kwargs.get("multi_window_highfreq_weight", 0.0)),
        "multi_window_subsample": int(max(1, kwargs.get("multi_window_subsample", 1))),
        "multi_window_spatial_only": bool(kwargs.get("multi_window_spatial_only", True)),
        "contrastive_reward_spectral_weight": float(kwargs.get("contrastive_reward_spectral_weight", 0.0)),
        "contrastive_reward_multi_window_weight": float(kwargs.get("contrastive_reward_multi_window_weight", 0.0)),
        **structural_stats,
        **spectral_stats,
        **multi_window_stats,
    }

    return compose_fitness(base)


def get_valid_terminal_idxs(ctx: DataContext) -> List[int]:
    vocab = get_default_vocab()
    terminals: List[int] = []
    for tok, idx in SYM2IDX.items():
        if vocab.arity.get(tok, 0) != 0: continue
        if tok in {vocab.cfg.start_token, vocab.cfg.end_token, vocab.cfg.pad_token}: continue
        if getattr(vocab.cfg, "include_unk", False) and tok == vocab.cfg.unk_token: continue
        if tok in ("t", "x", "y", "z"):
            if not bool(getattr(ctx, "allow_coordinate_terminals", False)):
                continue
            if tok == "t":
                continue
            if tok not in getattr(ctx, "coords", {}):
                continue
        if tok == "v" and not getattr(ctx, "has_v", False): continue
        terminals.append(int(idx))
    return terminals

def get_valid_nonterminal_idxs_for_rhs(ctx: Optional[DataContext] = None) -> List[int]:
    vocab = get_default_vocab()
    nonterminals: List[int] = []
    for tok, idx in SYM2IDX.items():
        if vocab.arity.get(tok, 0) <= 0: continue
        if tok in {vocab.cfg.start_token, vocab.cfg.end_token, vocab.cfg.pad_token}: continue
        if getattr(vocab.cfg, "include_unk", False) and tok == vocab.cfg.unk_token: continue
        nonterminals.append(int(idx))
    return nonterminals

def get_valid_axis_idxs_for_D(ctx: DataContext, *, forbid_t_on_rhs: bool = True) -> List[int]:
    axes: List[int] = []
    for axis in ("t", "x", "y", "z"):
        if axis not in getattr(ctx, "coords", {}): continue
        if forbid_t_on_rhs and axis == "t": continue
        if axis in SYM2IDX: axes.append(int(SYM2IDX[axis]))
    return axes
