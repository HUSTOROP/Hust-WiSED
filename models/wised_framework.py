from __future__ import annotations
import os, math
from itertools import product
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from models.multiresolution_encoder_1d import MultiResolutionEncoder1D
from models.multiresolution_encoder_2d import MultiResolutionEncoder2D
from models.latent_encoder import PhysicsLatentEncoder
from models.symbolic_decoder import SymbolicDecoder, refresh_index_groups

from models.symbol_vocabulary import (
    IDX2SYM,
    SYM2IDX,
    count_constants,
    init_vocab_from_context,
    is_valid_sequence,
    sequence_to_str
)

from models.weak_form_evaluator import (
    DataContext,
    clear_evaluator_caches,
    get_evaluator_cache_stats,
    get_valid_terminal_idxs,
    warmup_shared_subtrees,
)

from models.symbolic_search import (
    EvolutionaryPopulation,
    clear_optimizer_caches,
    get_optimizer_cache_stats,
    optimize_constants,
    refresh_operator_groups,
)

from utils.structure_cache import get_structure_key_manager
from utils.equation_formatter import compile_equation
from models.operator_policy import build_operator_policy, OperatorPolicy
from utils.reference_probes import run_reference_probe



@dataclass
class SampleRecord:
    sample_idx: int
    seq: List[int]
    key: str
    latent_index: int
    valid: bool = False
    fitness: float = 1e18
    mse: float = float("nan")
    # Used only for REINFORCE when contrastive_reward_enable=True.
    # Final equation selection still uses evaluator fitness.
    reward_score: float = 1e18


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _open_text_for_write(path: str):
    """Open a text file for writing after creating its parent directory."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    try:
        return open(path, "w", encoding="utf-8")
    except FileNotFoundError:
        # Be defensive against concurrent cleanup between makedirs and open.
        os.makedirs(parent, exist_ok=True)
        return open(path, "w", encoding="utf-8")


def _safe_item(x: Any, default: float = 0.0) -> float:
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            return float(default)
    try:
        return float(x)
    except Exception:
        return float(default)

_STRUCTURE_PRIOR_DEFAULTS: Dict[str, Dict[str, float]] = {
    "pde_simple": {
        "derivative_product_penalty_weight": 1.20,
        "derivative_power_penalty_weight": 1.10,
        "explicit_high_order_deriv_penalty_weight": 1.60,
        "composite_derivative_penalty_weight": 1.50,
        "nested_composite_derivative_penalty_weight": 2.00,
    },
    "robust_pde": {
        "derivative_product_penalty_weight": 1.60,
        "derivative_power_penalty_weight": 1.35,
        "explicit_high_order_deriv_penalty_weight": 2.00,
        "composite_derivative_penalty_weight": 2.20,
        "nested_composite_derivative_penalty_weight": 2.80,
    },
}

_ROBUST_STRUCTURE_EVAL_DEFAULTS: Dict[str, Any] = {
    "spectral_residual_enable": True,
    "spectral_residual_weight": 1.20,
    "spectral_mid_residual_weight": 0.25,
    "spectral_high_fraction_weight": 0.08,
    "multi_window_enable": True,
    "multi_window_kernels": (3,),
    "multi_window_residual_weight": 0.20,
    "multi_window_variance_weight": 0.25,
    "multi_window_highfreq_weight": 0.70,
    "generic_affine_bootstrap_frac": 0.20,
    "nonlinear_lbfgs_refine_elite": 6,
}

@dataclass
class TrainerConfig:
    """Core discovery configuration.

    The trainer no longer exposes annealing schedules or many RL fallback knobs.
    Keep only the stable backbone needed by the discovery loop:
    neural symbol proposal, evolutionary structure search, weak residual
    evaluation, constant optimization, archive rerank and final equation
    selection.  Unknown legacy keys are ignored by ``from_dict`` so old scripts
    still run without becoming part of the public strategy surface.
    """

    # Fixed proposal-training backbone
    lr: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 40
    # Stop the outer training loop once best_fitness has not improved
    # for this many consecutive epochs. Final rerank and post-training
    # summaries still run after the break.
    early_stop_enabled: bool = True
    early_stop_patience: int = 8
    gamma_min: float = 0.45
    gamma_max: float = 0.45
    temperature: float = 1.0
    beta_kl: float = 0.05
    alpha_reinforce: float = 0.5
    alpha_sparsity: float = 1.0e-3
    grad_clip_norm: float = 1.0
    logprob_min: float = -50.0

    # Numeric residual / constant optimization
    normalize_mse: bool = True
    const_scaled_threshold: float = 1e-4
    small_coeff_prune_tol: float = 1e-4
    const_physical_prune_tol: Optional[float] = None
    const_l2: float = 5e-3
    const_ridge: float = 1e-8
    mdl_penalty_weight: float = 0.16
    enable_nonlinear_lbfgs: bool = True
    nonlinear_lbfgs_refine_elite: int = 8

    # Symbol generation and structural search budget
    n_eq_samples: int = 256
    pop_size: int = 120
    seed: int = 42
    evo_interval: int = 4
    n_offspring: int = 36
    init_random_trials: int = 512
    init_max_ops: int = 7
    init_p_const: float = 0.10
    full_refine_topk: int = 16
    full_refine_topk_max: int = 96
    population_reuse_k: int = 8
    search_subsample: Optional[Dict[str, int]] = None
    search_context_budget: Optional[int] = None
    refine_context_budget: Optional[int] = None

    # Evolutionary grammar moves: fixed structural family, not task schedules
    evo_tournament_k: int = 2
    evo_random_parent_prob: float = 0.25
    evo_neighborhood_trials: int = 4
    evo_neighborhood_keep: int = 1
    evo_term_crossover_prob: float = 0.20
    evo_drop_term_prob: float = 0.06
    evo_replace_term_prob: float = 0.10
    evo_residual_guided_prob: float = 0.12
    evo_residual_guided_basis_trials: int = 96
    evo_residual_guided_topk: int = 16
    evo_random_immigrant_frac: float = 0.10
    evo_primitive_completion_frac: float = 0.15
    evo_random_tree_immigrant_frac: float = 0.25
    evo_big_mutation_patience: int = 4
    evo_restart_patience: int = 8
    evo_max_restart_prob: float = 0.25
    evo_max_big_mutation_prob: float = 0.35
    evo_max_add_depth: int = 6
    evo_max_mut_depth: int = 12
    evo_structure_max_points: int = 256
    evo_refine_topk: int = 8
    evo_coarse_batch_size: int = 64
    evo_enable_prune_refit: bool = False

    # Generic structural family guard/evaluation weights
    struct_guard_enable: bool = True
    struct_guard_max_tokens: int = 96
    struct_guard_max_polynomial_degree: int = 3
    struct_guard_reject_power_operator: bool = True
    struct_guard_reject_derivative_powers: bool = True
    struct_guard_reject_multi_derivative_products: bool = True
    derivative_product_penalty_weight: float = 1.20
    derivative_power_penalty_weight: float = 1.10
    explicit_high_order_deriv_penalty_weight: float = 1.60
    composite_derivative_penalty_weight: float = 1.50
    nested_composite_derivative_penalty_weight: float = 2.00
    zero_order_field_penalty_weight: float = 0.0
    max_zero_order_field_terms: Optional[float] = None
    forbidden_rhs_symbols: Optional[Any] = None
    generic_affine_bootstrap_frac: float = 0.0
    reaction_diffusion_refine_bonus: bool = True
    # Compact semantic controls. Detailed weights are internal constants below,
    # not task-level hyperparameters. Legacy detailed fields remain accepted.
    robust_structure_eval: bool = False
    structure_prior: str = "pde_simple"
    budget_profile: str = "standard"
    operator_mode: str = "pde"
    scoring_form: str = "weak"

    # Coarse/refine evaluation pipeline and memory controls
    auto_task_strategy: bool = True
    dynamic_full_refine_topk: bool = False
    stagnation_expand_topk: bool = False
    stagnation_min_improve: float = 1e-4
    coarse_chunk_size: int = 128
    refine_chunk_size: int = 32
    subtree_warmup: bool = True
    subtree_warmup_max: int = 48
    subtree_warmup_min_support: int = 3
    subtree_warmup_max_tokens: int = 10
    subtree_warmup_cuda_2d_cap: int = 8
    dr_mask_enable: bool = True
    dr_mask_keep_per_signature: int = 2
    dr_mask_stage: str = "refine"
    clear_evaluator_cache_between_stages: bool = True
    release_eval_cache_before_backward: bool = True
    polish_during_train: bool = True
    reoptimize_pruned_best: bool = False

    # Final equation evaluation and selection
    coverage_archive_enable: bool = True
    coverage_population_keep_per_signature: int = 2
    coverage_archive_keep_per_signature: int = 4
    coverage_archive_max_size: int = 4096
    final_archive_rerank_enable: bool = True
    final_archive_rerank_topk: int = 256
    final_archive_rerank_per_signature: int = 3
    final_archive_rerank_use_diagnostics: bool = False
    final_rerank_context_budget: int = 0
    final_rerank_pre_simplify: bool = True
    final_rerank_prune_refit: bool = True
    final_rerank_prune_tol: float = 1e-4
    posthoc_pareto_screening: bool = True
    final_one_se_selection: bool = True
    final_one_se_rel_tol: float = 0.10
    final_one_se_abs_tol: float = 1e-10
    final_refit_accept_mode: str = "mdl"
    final_refit_score_tol: float = 1e-10
    final_refit_mdl_weight: float = 0.25
    final_refit_min_relative_improvement: float = 0.0

    # Diagnostics remain off by default; noisy profiles may enable them as part
    # of structure-family evaluation, not training schedules.
    spectral_residual_enable: bool = False
    spectral_residual_weight: float = 0.0
    spectral_mid_residual_weight: float = 0.0
    spectral_high_fraction_weight: float = 0.0
    spectral_high_threshold: float = 0.45
    spectral_mid_threshold: float = 0.20
    multi_window_enable: bool = False
    multi_window_kernels: Any = None
    multi_window_residual_weight: float = 0.0
    multi_window_variance_weight: float = 0.0
    multi_window_highfreq_weight: float = 0.0
    e2e_diagnostics_refine_topk: int = 24
    spectral_residual_subsample: int = 1
    multi_window_subsample: int = 1
    multi_window_spatial_only: bool = True
    contrastive_reward_enable: bool = False
    contrastive_reward_spectral_weight: float = 0.0
    contrastive_reward_multi_window_weight: float = 0.0
    training_residual_diagnostics: bool = False

    warn_on_error: bool = False
    console_log_every: int = 5
    show_candidate_progress: bool = False
    compact_epoch_log: bool = True
    seed_candidate_templates: Optional[List[str]] = None
    aggregate_mode: str = "mean"
    reference_probe_enable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TrainerConfig":
        cfg = TrainerConfig()
        for key, value in (d or {}).items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg


class WiSEDFramework(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        d_h: int = 128,
        d_z: int = 64,
        n_scales: int = 3,
        d_gru: int = 256,
        max_eq_len: int = 20,
        n_gru_layers: int = 2,
        enable_normalizer: bool = True,
        spatial_ndim: int = 1,
        spatial_padding_mode: str = "circular",
        op_ratio: float = 0.7,
        max_consts: int = 10,
        gate_logit_scale: float = 0.10,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.d_h = int(d_h)
        self.d_z = int(d_z)
        self.n_scales = int(n_scales)
        self.d_gru = int(d_gru)
        self.max_eq_len = int(max_eq_len)
        self.n_gru_layers = int(n_gru_layers)
        self.spatial_ndim = int(spatial_ndim)
        self.spatial_padding_mode = str(spatial_padding_mode)
        self.op_ratio = float(op_ratio)
        self.max_consts = int(max(1, max_consts))
        self.gate_logit_scale = float(gate_logit_scale)
        self.allowed_axes: List[str] = ["x"]
        if self.spatial_ndim == 2:
            self.mr_ste = MultiResolutionEncoder2D(
                in_channels=self.in_channels,
                d_h=self.d_h,
                n_scales=self.n_scales,
                spatial_padding_mode=self.spatial_padding_mode,
            )
        else:
            self.mr_ste = MultiResolutionEncoder1D(
                in_channels=self.in_channels,
                d_h=self.d_h,
                n_scales=self.n_scales,
                spatial_padding_mode=self.spatial_padding_mode,
            )
        self.ple = PhysicsLatentEncoder(d_h=self.d_h, d_z=self.d_z)
        self.gsd = SymbolicDecoder(
            d_z=self.d_z,
            d_hidden=self.d_gru,
            max_len=self.max_eq_len,
            n_gru_layers=self.n_gru_layers,
            allowed_axes=self.allowed_axes,
        )
        self.key_manager = None
        self.normalizer = None
        self.enable_normalizer = bool(enable_normalizer)

    def rebuild_decoder_from_current_vocab(self) -> None:
        refresh_index_groups()
        device = next(self.parameters()).device
        self.gsd = SymbolicDecoder(
            d_z=self.d_z,
            d_hidden=self.d_gru,
            max_len=self.max_eq_len,
            n_gru_layers=self.n_gru_layers,
            allowed_axes=self.allowed_axes,
        ).to(device)

    def ensure_normalizer(self) -> None:
        if not self.enable_normalizer:
            self.key_manager = None
            self.normalizer = None
            return
        try:
            self.key_manager = get_structure_key_manager()
            self.normalizer = self.key_manager
        except Exception:
            self.key_manager = None
            self.normalizer = None

    def encode(self, x: torch.Tensor):
        h = self.mr_ste(x)
        z, mu, logvar = self.ple(h)
        return z, mu, logvar, h

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return self.ple.kl_loss(mu, logvar)

    def struct_sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        return self.ple.struct_sparsity_loss(z)

    def total_parameter_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

class WiSEDTrainer:
    def __init__(
        self,
        model: WiSEDFramework,
        ctx: DataContext,
        config: Dict[str, Any],
        device: str = "cpu",
        log_dir: str = "logs",
        logger: Any = None,
        enable_population_init: bool = True,
    ):
        self.model = model.to(device)
        self.ctx = ctx
        self._ctx0 = ctx[0] if isinstance(ctx, (list, tuple)) and len(ctx) > 0 else ctx
        self.device = device
        self.log_dir = log_dir
        self.logger = logger
        self._reference_probe_done = False
        _ensure_dir(log_dir)
        self._raw_cfg = dict(config or {})
        self.cfg_obj = TrainerConfig.from_dict(config)
        self._apply_dimension_adaptive_strategy()
        self._apply_compact_semantic_defaults()
        self.cfg = self.cfg_obj.to_dict()
        init_vocab_from_context(self._ctx0)
        axes = list(getattr(self._ctx0, "coords", {}).keys()) if hasattr(self._ctx0, "coords") else ["t", "x"]
        spatial_axes = [a for a in axes if a != "t"] or ["x"]

        spatial_ndim = int(max(1, len(spatial_axes)))
        self.spatial_ndim = spatial_ndim

        self.model.allowed_axes = spatial_axes
        self.model.rebuild_decoder_from_current_vocab()
        self.model.ensure_normalizer()

        self.operator_policy: OperatorPolicy = build_operator_policy(
            epoch=0,
            spatial_ndim=self.spatial_ndim,
            mode=getattr(self.cfg_obj, "operator_mode", "pde"),
        )

        refresh_operator_groups(policy=self.operator_policy)


        self.model.gsd.set_allowed_axes(spatial_axes)
        self.model.gsd.set_spatial_ndim(self.spatial_ndim)
        self.model.gsd.set_allowed_fields(list(getattr(self._ctx0, "fields", ["u"])))
        self.model.gsd.set_operator_policy(self.operator_policy)

        try:
            target_fields_default = [f"d{f}_t" for f in getattr(self._ctx0, "fields", ["u"])]
            # 杩欓噷鍙槸涓轰簡鏄庣‘褰撳墠浠诲姟娲昏穬 target 鐨勯泦鍚堬紱embedding 鏈韩浠嶆敮鎸?du_t/dv_t/dw_t
            self.target_fields_default = target_fields_default
        except Exception:
            self.target_fields_default = ["du_t"]

        nm = bool(getattr(self.cfg_obj, "normalize_mse", True))
        if isinstance(self.ctx, (list, tuple)):
            for c in self.ctx:
                c.normalize_mse = nm
        else:
            self.ctx.normalize_mse = nm

        self.search_stride_map = self._sanitize_search_stride_map(
            self._ctx0,
            getattr(self.cfg_obj, "search_subsample", None),
        )

        self.ctx_search = self._maybe_build_search_ctx(self.ctx, self.search_stride_map)

        self.valid_terminals = [int(i) for i in get_valid_terminal_idxs(self._ctx0)]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.cfg_obj.lr),
            weight_decay=float(self.cfg_obj.weight_decay),
        )
        # Core backbone mode: no learning-rate scheduler/annealing.  The
        # symbolic generator is trained with a fixed optimizer while discovery
        # quality is controlled by candidate evaluation and structure search.
        self.scheduler = None

        self.populations: Dict[str, EvolutionaryPopulation] = {}
        self.population = EvolutionaryPopulation(
            pop_size=int(self.cfg_obj.pop_size),
            rng_seed=int(self.cfg_obj.seed),
        )
        self.enable_population_init = bool(enable_population_init)
        self.history: Dict[str, List[Any]] = {
            "epoch": [],
            "kl_loss": [],
            "reinforce_loss": [],
            "struct_loss": [],
            "total_loss": [],
            "best_fitness": [],
            "best_mse": [],
            "best_equation": [],
            "gamma": [],
            "lr": [],
            "n_valid_eqs": [],
            "diversity_score": [],
            "population_size": [],
            "n_sampled": [],
            "n_seq_valid": [],
            "batch_min_fitness": [],
            "batch_mean_fitness": [],
            "batch_std_fitness": [],
            "batch_min_mse": [],
            "batch_mean_mse": [],
            "batch_std_mse": [],
            "subtree_cache_queries": [],
            "subtree_cache_hits": [],
            "subtree_cache_misses": [],
            "subtree_cache_hit_rate": [],
            "subtree_cache_saved_evals": [],
            "subtree_cache_queries_epoch": [],
            "subtree_cache_hits_epoch": [],
            "subtree_cache_misses_epoch": [],
            "subtree_cache_hit_rate_epoch": [],
            "subtree_cache_saved_evals_epoch": [],
            "template_cache_queries": [],
            "template_cache_hits": [],
            "template_cache_misses": [],
            "template_cache_hit_rate": [],
            "template_cache_saved_evals": [],
            "template_cache_queries_epoch": [],
            "template_cache_hits_epoch": [],
            "template_cache_misses_epoch": [],
            "template_cache_hit_rate_epoch": [],
            "template_cache_saved_evals_epoch": [],
            "shared_subtrees_warmed": [],
            "dr_mask_pruned": [],
            "coarse_candidates": [],
            "refine_candidates": [],
            "stagnated_epochs": [],
            "epoch_time_sec": [],
        }
        self.best_equations: Dict[str, Dict[str, Any]] = {}
        self.best_equation: Dict[str, Any] = self._empty_best_equation()
        self.active_target_field: Optional[str] = None
        clear_evaluator_caches(reset_stats=True)
        clear_optimizer_caches(reset_stats=True)
        self._last_stage_stats: Dict[str, Any] = {}
        self._sync_logger_schema()

        self._last_best_fitness = 1e18
        self._stagnated_epochs = 0
        self.early_stopped = False
        self.stopped_epoch: Optional[int] = None

    # --------------------------------------------------------
    # internal helpers
    # --------------------------------------------------------
    def _warn(self, message: str) -> None:
        if not bool(getattr(self.cfg_obj, "warn_on_error", False)):
            return
        if self.logger is not None and hasattr(self.logger, "warning"):
            try:
                self.logger.warning(message)
                return
            except Exception:
                pass
        print(f"[WARN] {message}")

    def _release_eval_cache_before_backward(self) -> None:
        """Release equation-evaluation CUDA caches before autograd backward.

        Candidate scoring stores tensors in evaluator/subtree caches.  The
        REINFORCE loss only needs scalar fitness values and decoder log-probs,
        so those cached tensors are safe to drop before backpropagation.
        """
        if not bool(getattr(self.cfg_obj, "release_eval_cache_before_backward", True)):
            return
        try:
            clear_evaluator_caches(reset_stats=False)
        except TypeError:
            try:
                clear_evaluator_caches()
            except Exception:
                pass
        except Exception as exc:
            self._warn(f"failed to release evaluator cache before backward: {exc}")
        try:
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _set_cfg_default_if_missing(self, key: str, value: Any) -> None:
        if key not in self._raw_cfg:
            setattr(self.cfg_obj, key, value)

    def _apply_dimension_adaptive_strategy(self) -> None:
        if not bool(getattr(self.cfg_obj, "auto_task_strategy", True)):
            return
        axes_order = list(getattr(self._ctx0, "axes_order", []))
        spatial_axes = [ax for ax in axes_order if ax != "t"]
        spatial_ndim = int(max(1, len(spatial_axes)))
        if spatial_ndim <= 1:
            self._set_cfg_default_if_missing("subtree_warmup", False)
            self._set_cfg_default_if_missing("dr_mask_enable", False)
            self._set_cfg_default_if_missing("clear_evaluator_cache_between_stages", False)
            self._set_cfg_default_if_missing("init_random_trials", 128)
            self._set_cfg_default_if_missing("full_refine_topk", 24)
            self._set_cfg_default_if_missing("coarse_chunk_size", 256)
            self._set_cfg_default_if_missing("refine_chunk_size", 64)
            self._set_cfg_default_if_missing("population_reuse_k", 4)
        else:
            self._set_cfg_default_if_missing("subtree_warmup", True)
            self._set_cfg_default_if_missing("dr_mask_enable", True)
            # 2D candidate evaluation stores large CUDA tensors.  Keep cache use
            # within an epoch, but clear it around backward/stage boundaries unless
            # the user explicitly overrides this.
            self._set_cfg_default_if_missing("clear_evaluator_cache_between_stages", True)
            self._set_cfg_default_if_missing("release_eval_cache_before_backward", True)
            self._set_cfg_default_if_missing("subtree_warmup_cuda_2d_cap", 8)
            self._set_cfg_default_if_missing("init_random_trials", 768)
            self._set_cfg_default_if_missing("full_refine_topk", 12)
            self._set_cfg_default_if_missing("coarse_chunk_size", 96)
            self._set_cfg_default_if_missing("refine_chunk_size", 24)


    def _apply_compact_semantic_defaults(self) -> None:
        """Materialize compact semantic profiles into stable internal defaults.

        Public configs now expose high-level semantic controls such as
        ``structure_prior``, ``robust_structure_eval`` and ``budget_profile``.
        This method keeps those controls compact by applying the old detailed
        knobs internally only when the user did not explicitly provide them.
        """
        # Validate/normalize structure prior name without exposing individual
        # derivative-penalty weights as task hyperparameters.
        prior_name = str(getattr(self.cfg_obj, "structure_prior", "pde_simple") or "pde_simple")
        if prior_name not in _STRUCTURE_PRIOR_DEFAULTS:
            prior_name = "pde_simple"
        setattr(self.cfg_obj, "structure_prior", prior_name)

        # Compact robust evaluation switch.  The actual spectral/multi-window
        # weights stay in _ROBUST_STRUCTURE_EVAL_DEFAULTS and are used later by
        # _optimizer_common_kwargs().
        robust = bool(getattr(self.cfg_obj, "robust_structure_eval", False))
        setattr(self.cfg_obj, "robust_structure_eval", robust)
        if robust:
            self._set_cfg_default_if_missing("nonlinear_lbfgs_refine_elite", int(_ROBUST_STRUCTURE_EVAL_DEFAULTS.get("nonlinear_lbfgs_refine_elite", 6)))

        # Budget profiles are semantic aliases for a few stable engineering
        # choices. They are not discovery hyperparameters and remain fixed here.
        budget = str(getattr(self.cfg_obj, "budget_profile", "standard") or "standard").lower()
        setattr(self.cfg_obj, "budget_profile", budget)
        if budget == "small":
            self._set_cfg_default_if_missing("dynamic_full_refine_topk", False)
            self._set_cfg_default_if_missing("subtree_warmup", False)
            self._set_cfg_default_if_missing("dr_mask_stage", "pre_coarse")
            self._set_cfg_default_if_missing("dr_mask_keep_per_signature", 1)
            self._set_cfg_default_if_missing("polish_during_train", False)
            self._set_cfg_default_if_missing("mdl_penalty_weight", 0.0)
            self._set_cfg_default_if_missing("stagnation_expand_topk", False)
        else:
            # Keep the simplified backbone stable by default: no dynamic training
            # schedule/top-k expansion unless a legacy config explicitly asks.
            self._set_cfg_default_if_missing("dynamic_full_refine_topk", False)
            self._set_cfg_default_if_missing("stagnation_expand_topk", False)

    def _empty_best_equation(self) -> Dict[str, Any]:
        return {
            "seq": [],
            "consts": np.array([]),
            "fitness": 1e18,
            "mse": 1e18,
            "str": "",
            "readable": "",
            "complexity": 0.0,
        }

    def _get_population(self, target_field: str) -> EvolutionaryPopulation:
        tf = str(target_field)
        if tf not in self.populations:
            tf_seed = int(self.cfg_obj.seed) + 1009 * len(self.populations)
            self.populations[tf] = EvolutionaryPopulation(
                pop_size=int(self.cfg_obj.pop_size),
                rng_seed=tf_seed,
            )
            try:
                self.populations[tf].population_keep_per_signature = int(max(1, getattr(self.cfg_obj, "coverage_population_keep_per_signature", 2)))
                self.populations[tf].archive_keep_per_signature = int(max(1, getattr(self.cfg_obj, "coverage_archive_keep_per_signature", 4)))
                self.populations[tf].archive_max_size = int(max(128, getattr(self.cfg_obj, "coverage_archive_max_size", 4096)))
            except Exception:
                pass
        return self.populations[tf]

    def _activate_target(self, target_field: str) -> None:
        tf = str(target_field)
        self.active_target_field = tf
        self.population = self._get_population(tf)
        if tf not in self.best_equations:
            self.best_equations[tf] = self._empty_best_equation()
        self.best_equation = self.best_equations[tf]

    def _joint_best_summary(self, target_fields: List[str]) -> Tuple[float, float, str]:
        valid = []
        readable_parts = []
        for tf in target_fields:
            eq = self.best_equations.get(tf, self._empty_best_equation())
            fit = float(eq.get("fitness", 1e18))
            mse = float(eq.get("mse", 1e18))
            if np.isfinite(fit) and fit < 1e18:
                valid.append(eq)
                tag = self._tag_from_target_field(tf)
                readable_parts.append(f"{tag}:[{eq.get('readable', '')}]")
        if not valid:
            return 1e18, 1e18, ""
        mean_fit = float(np.mean([float(eq.get("fitness", 1e18)) for eq in valid]))
        mean_mse = float(np.mean([float(eq.get("mse", 1e18)) for eq in valid]))
        readable = " || ".join(readable_parts)
        return mean_fit, mean_mse, readable

    def _population_stats_for_targets(self, target_fields: List[str]) -> Dict[str, float]:
        pops = [self._get_population(tf) for tf in target_fields]
        if not pops:
            return {"diversity_score": 0.0, "pop_size": 0.0}
        divs = [float(p.get_stats().get("diversity_score", 0.0)) for p in pops]
        sizes = [float(p.get_stats().get("pop_size", 0.0)) for p in pops]
        return {
            "diversity_score": float(np.mean(divs)) if divs else 0.0,
            "pop_size": float(np.mean(sizes)) if sizes else 0.0,
        }

    def _log_progress(self, message: str) -> None:
        if self.logger is not None and hasattr(self.logger, "info"):
            try:
                self.logger.info(message)
                return
            except Exception:
                pass
        print(message)

    def _sync_logger_schema(self) -> None:
        if self.logger is None or not hasattr(self.logger, "_csv_fields"):
            return
        extra_fields = [
            "batch_min_fitness",
            "batch_mean_fitness",
            "batch_std_fitness",
            "batch_min_mse",
            "batch_mean_mse",
            "batch_std_mse",
            "shared_subtrees_warmed",
            "dr_mask_pruned",
            "coarse_candidates",
            "refine_candidates",
            "effective_full_refine_topk",
            "population_size",
            "stagnated_epochs",
        ]
        try:
            current = list(getattr(self.logger, "_csv_fields", []))
            changed = False
            for name in extra_fields:
                if name not in current:
                    current.append(name)
                    changed = True
            if changed:
                self.logger._csv_fields = current
        except Exception:
            pass

    def _update_stagnation_state(self, current_best_fitness: float) -> int:
        """Update structure-search stagnation counter from best fitness.

        The counter increments only when the best fitness fails to improve by a
        relative/absolute tolerance.  Evolution uses this value in the next
        offspring-generation call to widen exploration.
        """
        try:
            cur = float(current_best_fitness)
        except Exception:
            cur = 1e18

        if not np.isfinite(cur) or cur >= 1e18:
            self._stagnated_epochs = int(getattr(self, "_stagnated_epochs", 0)) + 1
            return int(self._stagnated_epochs)

        prev = float(getattr(self, "_last_best_fitness", 1e18))
        tol = float(max(0.0, getattr(self.cfg_obj, "stagnation_min_improve", 1e-4)))

        if not np.isfinite(prev) or prev >= 1e18:
            self._last_best_fitness = cur
            self._stagnated_epochs = 0
            return 0

        required_delta = max(abs(prev) * tol, tol)
        if (prev - cur) > required_delta:
            self._last_best_fitness = cur
            self._stagnated_epochs = 0
        else:
            # Keep the best reference monotone even if tiny numerical noise gives
            # a slightly smaller value that is below the improvement threshold.
            if cur < prev:
                self._last_best_fitness = cur
            self._stagnated_epochs = int(getattr(self, "_stagnated_epochs", 0)) + 1

        return int(self._stagnated_epochs)

    def _effective_refine_topk(self) -> int:
        """Return full-refine top-k enlarged when structure search stagnates.

        The counter is updated after each epoch, so this method uses the
        stagnation state accumulated up to the previous epoch.  It does not
        change the candidate grammar; it only lets more coarse-ranked
        candidates survive into expensive full refinement when the search has
        stopped making meaningful progress.
        """
        base = int(max(1, getattr(self.cfg_obj, "full_refine_topk", 32)))
        if (not bool(getattr(self.cfg_obj, "dynamic_full_refine_topk", True))) or (not bool(getattr(self.cfg_obj, "stagnation_expand_topk", True))):
            return base

        stagnated = int(max(0, getattr(self, "_stagnated_epochs", 0)))
        patience = int(max(1, getattr(self.cfg_obj, "full_refine_topk_stagnation_patience", 5)))
        hard_patience = int(max(patience, getattr(self.cfg_obj, "full_refine_topk_hard_patience", 10)))
        growth = float(max(1.0, getattr(self.cfg_obj, "full_refine_topk_growth", 2.0)))
        hard_growth = float(max(growth, getattr(self.cfg_obj, "full_refine_topk_hard_growth", 4.0)))
        max_topk = int(max(base, getattr(self.cfg_obj, "full_refine_topk_max", 96)))

        if stagnated >= hard_patience:
            return int(min(max_topk, math.ceil(base * hard_growth)))
        if stagnated >= patience:
            return int(min(max_topk, math.ceil(base * growth)))
        return base

    def _optimizer_common_kwargs(self) -> Dict[str, Any]:
        """Common scoring knobs forwarded to constant optimization.

        Public configs only choose semantic modes such as ``structure_prior`` and
        ``robust_structure_eval``.  The numerical weights below are stable
        internal defaults that preserve discovery ability without becoming
        task-level hyperparameters.
        """
        prior_name = str(getattr(self.cfg_obj, "structure_prior", "pde_simple") or "pde_simple")
        prior = dict(_STRUCTURE_PRIOR_DEFAULTS.get(prior_name, _STRUCTURE_PRIOR_DEFAULTS["pde_simple"]))

        # Backward compatibility: explicit legacy fields still win when old
        # scripts provide them, but new profiles do not emit these keys.
        for key in list(prior.keys()):
            prior[key] = float(getattr(self.cfg_obj, key, prior[key]))

        robust = bool(getattr(self.cfg_obj, "robust_structure_eval", False))
        diagnostics = dict(_ROBUST_STRUCTURE_EVAL_DEFAULTS) if robust else {
            "spectral_residual_enable": False,
            "spectral_residual_weight": 0.0,
            "spectral_mid_residual_weight": 0.0,
            "spectral_high_fraction_weight": 0.0,
            "multi_window_enable": False,
            "multi_window_kernels": None,
            "multi_window_residual_weight": 0.0,
            "multi_window_variance_weight": 0.0,
            "multi_window_highfreq_weight": 0.0,
            "generic_affine_bootstrap_frac": 0.0,
            "nonlinear_lbfgs_refine_elite": int(getattr(self.cfg_obj, "nonlinear_lbfgs_refine_elite", 8)),
        }

        # Backward compatibility for legacy detailed diagnostic keys.
        for key in (
            "spectral_residual_enable", "spectral_residual_weight",
            "spectral_mid_residual_weight", "spectral_high_fraction_weight",
            "multi_window_enable", "multi_window_kernels",
            "multi_window_residual_weight", "multi_window_variance_weight",
            "multi_window_highfreq_weight", "generic_affine_bootstrap_frac",
            "zero_order_field_penalty_weight", "max_zero_order_field_terms",
            "forbidden_rhs_symbols",
            "nonlinear_lbfgs_refine_elite",
        ):
            if hasattr(self.cfg_obj, key):
                val = getattr(self.cfg_obj, key)
                # Do not let dataclass defaults accidentally overwrite the
                # robust semantic mode unless an old config explicitly sets the
                # detailed key before from_dict. New compact configs won't.
                if key not in diagnostics or not robust:
                    diagnostics[key] = val

        forbidden_rhs_symbols = diagnostics.get("forbidden_rhs_symbols", None)
        if forbidden_rhs_symbols is None and str(getattr(self.cfg_obj, "operator_mode", "pde")) in {
            "diffusion",
            "diffusion_only",
            "parabolic_diffusion",
        }:
            forbidden_rhs_symbols = ("D", "adv", "sq", "cube", "/", "^", "sin", "cos", "exp", "log")

        return {
            **prior,
            "spectral_residual_enable": bool(diagnostics.get("spectral_residual_enable", False)),
            "spectral_residual_weight": float(diagnostics.get("spectral_residual_weight", 0.0)),
            "spectral_mid_residual_weight": float(diagnostics.get("spectral_mid_residual_weight", 0.0)),
            "spectral_high_fraction_weight": float(diagnostics.get("spectral_high_fraction_weight", 0.0)),
            "spectral_high_threshold": float(getattr(self.cfg_obj, "spectral_high_threshold", 0.45)),
            "spectral_mid_threshold": float(getattr(self.cfg_obj, "spectral_mid_threshold", 0.20)),
            "spectral_residual_subsample": int(max(1, getattr(self.cfg_obj, "spectral_residual_subsample", 1))),
            "multi_window_enable": bool(diagnostics.get("multi_window_enable", False)),
            "multi_window_kernels": diagnostics.get("multi_window_kernels", None) or (3, 5),
            "multi_window_residual_weight": float(diagnostics.get("multi_window_residual_weight", 0.0)),
            "multi_window_variance_weight": float(diagnostics.get("multi_window_variance_weight", 0.0)),
            "multi_window_highfreq_weight": float(diagnostics.get("multi_window_highfreq_weight", 0.0)),
            "multi_window_subsample": int(max(1, getattr(self.cfg_obj, "multi_window_subsample", 1))),
            "multi_window_spatial_only": bool(getattr(self.cfg_obj, "multi_window_spatial_only", True)),
            "contrastive_reward_spectral_weight": float(getattr(self.cfg_obj, "contrastive_reward_spectral_weight", 0.0)),
            "contrastive_reward_multi_window_weight": float(getattr(self.cfg_obj, "contrastive_reward_multi_window_weight", 0.0)),
            "struct_guard_enable": bool(getattr(self.cfg_obj, "struct_guard_enable", True)),
            "struct_guard_max_tokens": int(getattr(self.cfg_obj, "struct_guard_max_tokens", 96)),
            "struct_guard_max_polynomial_degree": int(getattr(self.cfg_obj, "struct_guard_max_polynomial_degree", 3)),
            "struct_guard_reject_power_operator": bool(getattr(self.cfg_obj, "struct_guard_reject_power_operator", True)),
            "struct_guard_reject_derivative_powers": bool(getattr(self.cfg_obj, "struct_guard_reject_derivative_powers", True)),
            "struct_guard_reject_multi_derivative_products": bool(getattr(self.cfg_obj, "struct_guard_reject_multi_derivative_products", True)),
            "zero_order_field_penalty_weight": float(diagnostics.get("zero_order_field_penalty_weight", 0.0)),
            "max_zero_order_field_terms": diagnostics.get("max_zero_order_field_terms", None),
            "forbidden_rhs_symbols": forbidden_rhs_symbols,
            "operator_mode": str(getattr(self.cfg_obj, "operator_mode", "pde")),
            "scoring_form": str(getattr(self.cfg_obj, "scoring_form", "weak")),
            "const_prune_tol": float(
                getattr(self.cfg_obj, "small_coeff_prune_tol", 1e-4)
                if getattr(self.cfg_obj, "const_physical_prune_tol", None) is None
                else getattr(self.cfg_obj, "const_physical_prune_tol")
            ),
        }

    @staticmethod
    def _with_residual_diagnostics_enabled(opt_kwargs: Dict[str, Any], enabled: bool) -> Dict[str, Any]:
        """Return scoring kwargs with expensive spectral/multi-window diagnostics toggled.

        Structural-risk penalties remain active in both modes.  Only the costly
        residual diagnostics and contrastive extras are disabled for coarse or
        non-elite candidates.  This preserves the end-to-end objective for the
        elite refinement path without making every candidate run FFT/smoothing.
        """
        out = dict(opt_kwargs)
        if not bool(enabled):
            out["spectral_residual_enable"] = False
            out["spectral_residual_weight"] = 0.0
            out["spectral_mid_residual_weight"] = 0.0
            out["spectral_high_fraction_weight"] = 0.0
            out["multi_window_enable"] = False
            out["multi_window_residual_weight"] = 0.0
            out["multi_window_variance_weight"] = 0.0
            out["multi_window_highfreq_weight"] = 0.0
            out["contrastive_reward_spectral_weight"] = 0.0
            out["contrastive_reward_multi_window_weight"] = 0.0
        return out

    def _emit_epoch_summary(self, epoch: int, n_epochs: int, metrics: Dict[str, Any], targets: List[str], joint_mode: bool) -> None:
        mode = "J" if joint_mode else "S"      # J = joint and S = single
        eq_preview = str(metrics.get("best_equation", "") or "").replace("\n", " ")
        line = (
            f"[Epoch {epoch:>3d}/{n_epochs:>3d}][{mode}] "
            f"KL={float(metrics.get('kl_loss', 0.0)):.4f} | "
            f"RF={float(metrics.get('reinforce_loss', 0.0)):.4f} | "
            f"Loss={float(metrics.get('total_loss', 0.0)):.4f} | "
            f"MSE={float(metrics.get('best_mse', 1e18)):.3e} | "
            f"Fit={float(metrics.get('best_fitness', 1e18)):.3e} | "
            f"Valid={int(metrics.get('n_valid_eqs', 0))} | "
            f"Subtree={float(metrics.get('subtree_cache_hit_rate_epoch', 0.0)):.1%}"
            f"({int(metrics.get('subtree_cache_hits_epoch', 0))}/{int(metrics.get('subtree_cache_queries_epoch', 0))}) | "
            f"Template={float(metrics.get('template_cache_hit_rate_epoch', 0.0)):.1%}"
            f"({int(metrics.get('template_cache_hits_epoch', 0))}/{int(metrics.get('template_cache_queries_epoch', 0))}) | "
            f"Warm={int(metrics.get('shared_subtrees_warmed', 0))} | "
            f"DRmask={int(metrics.get('dr_mask_pruned', 0))} | "
            f"Coarse/Refine={int(metrics.get('coarse_candidates', 0))}/{int(metrics.get('refine_candidates', 0))} | "
            f"TopK={int(metrics.get('effective_full_refine_topk', 0))} | "
            f"Stag={int(metrics.get('stagnated_epochs', 0))} | "
            f"t={float(metrics.get('epoch_time_sec', 0.0)):.1f}s | "
            f"Eq:[{eq_preview}]"
        )
        self._log_progress(line)

    def _pad_records_for_logprob(self, records: List[SampleRecord], device: torch.device):
        seqs = [list(map(int, r.seq)) for r in records]
        return self.model.gsd.pad_token_sequences(seqs, device=device)

    def _limit_ctx_budget(self, ctx, budget: Optional[int]):
        if not isinstance(ctx, (list, tuple)):
            return ctx
        if budget is None:
            return list(ctx)
        budget = int(max(1, budget))
        if len(ctx) <= budget:
            return list(ctx)
        return list(ctx[:budget])

    def _maybe_build_search_ctx(self, ctx, stride_map):
        if not stride_map:
            return ctx
        try:
            if isinstance(ctx, (list, tuple)):
                return [c.make_subsampled(stride_map) for c in ctx]
            return ctx.make_subsampled(stride_map)
        except Exception as exc:
            self._warn(f"failed to build search ctx: {exc}")
            return ctx

    def _sanitize_search_stride_map(self, ctx0: DataContext, stride_map: Optional[Dict[str, int]]) -> Optional[Dict[str, int]]:
        stride_map = {str(k): max(1, int(v)) for k, v in dict(stride_map or {}).items()}
        if not stride_map:
            return None
        axes_order = list(getattr(ctx0, "axes_order", []))
        spatial_axes = [ax for ax in axes_order if ax != "t"]
        # 2D/3D锛氱姝㈢┖闂寸矖閲囨牱锛屽彧淇濈暀鏃堕棿杞?subsample
        if len(spatial_axes) >= 2:
            sanitized: Dict[str, int] = {}
            if "t" in stride_map and "t" in axes_order and stride_map["t"] > 1:
                sanitized["t"] = stride_map["t"]
            return sanitized or None
        # 1D锛氬厑璁稿師鏉ョ殑琛屼负
        return {
            ax: stride
            for ax, stride in stride_map.items()
            if ax in axes_order and stride > 1
        }

    def _compute_gamma(self, epoch: int, n_epochs: int) -> float:
        """Fixed teacher-forcing / grammar-mixing strength.

        Schedules were removed from the public and internal training path.
        Keeping a fixed value makes the trainer easier to reason about while
        leaving equation discovery to the evaluator, optimizer and structural
        search backbone.
        """
        gmin = float(getattr(self.cfg_obj, "gamma_min", 0.45))
        gmax = float(getattr(self.cfg_obj, "gamma_max", gmin))
        return float(0.5 * (gmin + gmax))

    def _effective_temperature(self, epoch: int, n_epochs: int) -> float:
        return float(getattr(self.cfg_obj, "temperature", 1.0))

    def _canonical_seq(self, seq: List[int]) -> List[int]:
        """Return token-level canonical sequence used for real training state.

        This is stronger than readable-string polishing: equivalent expressions
        such as `neg(adv(v))`, `adv(neg(v))`, and
        `u*D(x,neg(v)) + v*D(y,neg(v))` are normalized to the same tokens before
        candidate de-duplication, population insertion, and best-equation storage.
        """
        try:
            km = get_structure_key_manager()
            out = list(map(int, km.normalizer.normalize(list(map(int, seq)))))
            return out if is_valid_sequence(out) else list(map(int, seq))
        except Exception:
            return list(map(int, seq))

    def _canonical_key(self, seq: List[int]) -> str:
        try:
            return get_structure_key_manager().expr_key(seq).key
        except Exception:
            return ",".join(str(int(t)) for t in seq)

    def _prefix_expr_spans(self, seq: List[int]) -> List[Tuple[int, int]]:
        """Return prefix spans [start, end) for all expression subtrees.

        This lightweight parser avoids relying on optional StructureKeyManager
        helpers.  It understands the project grammar, including D(axis, expr)
        where the axis token is not itself an expression subtree.
        """
        tokens = list(map(int, seq))
        spans: List[Tuple[int, int]] = []

        def arity(sym: str) -> int:
            if sym in {"+", "*", "/", "^"}:
                return 2
            if sym in {"neg", "sin", "cos", "exp", "log", "lap", "adv", "sq", "cube", "dx", "dxx", "dxxx", "dt", "dxt"}:
                return 1
            if sym == "D":
                return 2  # axis slot + expression slot
            return 0

        def parse_at(pos: int) -> int:
            if pos >= len(tokens):
                return pos
            start = pos
            sym = IDX2SYM.get(int(tokens[pos]), "")
            pos += 1
            if sym == "D":
                # Skip the typed axis slot, then parse the expression child.
                if pos < len(tokens):
                    pos += 1
                pos = parse_at(pos)
            else:
                for _ in range(arity(sym)):
                    pos = parse_at(pos)
            if pos > start:
                spans.append((start, min(pos, len(tokens))))
            return pos

        try:
            parse_at(0)
        except Exception:
            return [(0, len(tokens))] if tokens else []
        return spans

    def _select_shared_subtrees(self, seqs: List[List[int]]) -> List[List[int]]:
        """Select frequent short constant-free subtrees for cache warmup."""
        if not bool(getattr(self.cfg_obj, "subtree_warmup", True)) or not seqs:
            return []
        min_support = int(max(2, getattr(self.cfg_obj, "subtree_warmup_min_support", 3)))
        max_tokens = int(max(1, getattr(self.cfg_obj, "subtree_warmup_max_tokens", 10)))
        max_items = int(max(0, getattr(self.cfg_obj, "subtree_warmup_max", 48)))
        if int(getattr(self, "spatial_ndim", 1)) >= 2 and str(self.device).startswith("cuda"):
            max_items = min(max_items, int(max(0, getattr(self.cfg_obj, "subtree_warmup_cuda_2d_cap", 8))))

        counts: Dict[Tuple[int, ...], int] = {}
        for seq in seqs:
            seq_i = list(map(int, seq))
            for a, b in self._prefix_expr_spans(seq_i):
                sub = tuple(seq_i[a:b])
                if len(sub) < 2 or len(sub) > max_tokens:
                    continue
                sub_list = list(sub)
                if count_constants(sub_list) > 0:
                    continue
                if not is_valid_sequence(sub_list):
                    continue
                counts[sub] = counts.get(sub, 0) + 1

        rows = [(sub, cnt) for sub, cnt in counts.items() if cnt >= min_support]
        # Warm the most reusable subtrees first; shorter tensors are usually
        # cheaper and more likely to appear in larger expressions.
        rows.sort(key=lambda item: (-item[1], len(item[0]), item[0]))
        if max_items > 0:
            rows = rows[:max_items]
        return [list(sub) for sub, _ in rows]

    def _warmup_shared_subtrees(self, seqs: List[List[int]], ctx) -> int:
        selected = self._select_shared_subtrees(seqs)
        if not selected:
            return 0

        # First try the evaluator-level warmup if it is implemented in the
        # imported module.  Some older versions return None/0 for non-empty
        # input, so fall back to a local exact evaluator warmup below.
        try:
            warmed = warmup_shared_subtrees(ctx, selected, max_items=len(selected))
            warmed_i = int(warmed or 0)
            if warmed_i > 0:
                return warmed_i
        except Exception as exc:
            self._warn(f"shared-subtree module warmup failed: {exc}")

        # Local fallback: evaluate each selected constant-free subtree once.
        # WeakFormEvaluator._eval_node is responsible for writing the actual
        # subtree cache entries; this wrapper only counts successful warmups.
        warmed = 0
        try:
            from models.weak_form_evaluator import WeakFormEvaluator
            ctxs = list(ctx) if isinstance(ctx, (list, tuple)) else [ctx]
            with torch.no_grad():
                for cctx in ctxs:
                    ev = WeakFormEvaluator(cctx)
                    for seq in selected:
                        try:
                            try:
                                out = ev.evaluate(seq, [])
                            except TypeError:
                                out = ev.evaluate(seq)
                            if torch.is_tensor(out):
                                ok = bool(torch.isfinite(out).all().detach().cpu().item())
                            else:
                                ok = bool(np.all(np.isfinite(np.asarray(out))))
                            if ok:
                                warmed += 1
                        except Exception:
                            continue
        except Exception as exc:
            self._warn(f"shared-subtree local warmup failed: {exc}")
            return 0
        return int(warmed)

    def _dr_signature(self, seq: List[int]) -> str:
        """Conservative duplicate-reduction signature.

        It is intentionally finer than a simple operator bag so that physically
        distinct structures are unlikely to be merged.  The default use is at the
        refine stage after all candidates have already received coarse scores.
        """
        syms = [IDX2SYM.get(int(t), "") for t in seq]
        fields = "".join(sorted({s for s in syms if s in {"u", "v", "w"}}))
        axes = "".join(sorted({s for s in syms if s in {"x", "y", "z"}}))
        counts = {
            "L": len(seq) // 3,
            "C": syms.count("const"),
            "+": syms.count("+"),
            "*": syms.count("*"),
            "D": syms.count("D"),
            "lap": syms.count("lap"),
            "adv": syms.count("adv"),
            "poly": sum(1 for s in syms if s in {"sq", "cube", "^"}),
            "tr": sum(1 for s in syms if s in {"sin", "cos", "exp", "log"}),
        }
        return "|".join([f"{k}{v}" for k, v in counts.items()] + [f"F{fields}", f"A{axes}"])

    def _apply_dr_mask_to_unique(self, unique: Dict[str, List[int]]) -> Tuple[Dict[str, List[int]], int]:
        """Optional pre-coarse DRmask.  Disabled by default for precision."""
        if not bool(getattr(self.cfg_obj, "dr_mask_enable", True)):
            return unique, 0
        keep_per = int(max(1, getattr(self.cfg_obj, "dr_mask_keep_per_signature", 2)))
        buckets: Dict[str, List[Tuple[str, List[int]]]] = {}
        for key, seq in unique.items():
            buckets.setdefault(self._dr_signature(seq), []).append((key, seq))
        kept: Dict[str, List[int]] = {}
        for rows in buckets.values():
            rows.sort(key=lambda kv: (len(kv[1]), kv[0]))
            for key, seq in rows[:keep_per]:
                kept[key] = seq
        return kept, int(max(0, len(unique) - len(kept)))

    def _apply_dr_mask_to_search_rows(
        self,
        rows: List[Tuple[str, List[int], np.ndarray, Dict[str, Any]]],
    ) -> Tuple[List[Tuple[str, List[int], np.ndarray, Dict[str, Any]]], int]:
        """Post-coarse / pre-refine duplicate reduction.

        Input rows have already received coarse fitness scores, so this does not
        reduce coarse-stage recall.  It only prevents many near-duplicate rows
        from entering the expensive full-context refinement stage.
        """
        if not bool(getattr(self.cfg_obj, "dr_mask_enable", True)) or not rows:
            return list(rows), 0
        before = len(rows)
        kept = self._apply_dr_mask(rows)
        return kept, int(max(0, before - len(kept)))

    def _apply_dr_mask(self, ranked_candidates):
        """Refine-stage duplicate reduction for already scored candidates."""
        if not bool(getattr(self.cfg_obj, "dr_mask_enable", True)):
            return list(ranked_candidates)
        keep_per = int(max(1, getattr(self.cfg_obj, "dr_mask_keep_per_signature", 2)))
        seen: Dict[str, int] = {}
        kept = []
        # Sort by available fitness so each signature keeps its best coarse rows.
        def _fit(item):
            try:
                fit = item[3] if len(item) == 4 else item[4]
                return float(fit.get("fitness", 1e18)) if isinstance(fit, dict) else float(fit)
            except Exception:
                return 1e18
        for item in sorted(list(ranked_candidates), key=_fit):
            try:
                seq = item[1] if len(item) == 4 else item[2]
                sig = self._dr_signature(seq)
            except Exception:
                sig = str(item[0])
            n = int(seen.get(sig, 0))
            if n >= keep_per:
                continue
            seen[sig] = n + 1
            kept.append(item)
        return kept

    def _lhs_from_target_field(self, target_field: str) -> str:
        field = str(target_field)
        if field.startswith("d") and "_t" in field:
            var = field[1:].replace("_t", "")
        else:
            var = "u"
        return f"{var}_t"

    @staticmethod
    def _tag_from_target_field(target_field: str) -> str:
        field = str(target_field)
        if field.startswith("d") and "_t" in field:
            var = field[1:].replace("_t", "")
        else:
            var = "u"
        return var

    def _compile_candidate_equation(self, seq: List[int], consts: np.ndarray, target_field: str) -> str:
        lhs = self._lhs_from_target_field(target_field)
        try:
            return compile_equation(seq, consts, lhs=lhs)
        except Exception:
            return sequence_to_str(seq)

    def _update_best(
        self,
        seq: List[int],
        consts: np.ndarray,
        fit_dict: Dict[str, Any],
        target_field: str,
        *,
        force: bool = False,
        extra_fields: Optional[Dict[str, Any]] = None,
        log_update: bool = True,
    ) -> None:
        """Update the per-target best equation through a single validated path.

        Normal training and population sync use the default fitness-improving
        update rule. Post-hoc screeners may pass ``force=True`` when they
        intentionally choose a candidate by a secondary criterion such as
        post-selection checks. Even then, the sequence/constant pair is still
        canonicalized and validated here before touching best-equation state.
        """
        if not fit_dict.get("pred_valid", False):
            return
        tf = str(target_field)
        if tf not in self.best_equations:
            self.best_equations[tf] = self._empty_best_equation()

        # Store the canonical token sequence itself, not only a polished readable
        # string. This prevents equivalent sign/derivative/advection variants from
        # being treated as different best equations.
        seq_best = self._canonical_seq(list(map(int, seq)))
        consts_best = np.asarray(consts, dtype=np.float64).reshape(-1)
        fit_best = dict(fit_dict)

        if not self._validate_seq_const_pair(seq_best, consts_best):
            if self.logger is not None and hasattr(self.logger, "warning"):
                try:
                    self.logger.warning(
                        f"[BestUpdate] Skip invalid seq/const pair for {tf}: "
                        f"n_const(seq)={count_constants(seq_best)}, len(consts)={len(consts_best)}"
                    )
                except Exception:
                    pass
            return

        fitness = float(fit_best.get("fitness", 1e18))
        current_fitness = float(self.best_equations[tf].get("fitness", 1e18))
        if (not bool(force)) and fitness >= current_fitness:
            return
        try:
            lhs = self._lhs_from_target_field(tf)
            readable = compile_equation(seq_best, consts_best, lhs=lhs)
            if bool(getattr(self.cfg_obj, "polish_during_train", False)):
                from utils.equation_simplifier import polish_discovered_equation
                readable = polish_discovered_equation(readable, prune_tol=float(getattr(self.cfg_obj, "small_coeff_prune_tol", 1e-4)))
        except Exception:
            readable = ""

        updated = {
            "seq": seq_best,
            "consts": consts_best,
            "fitness": fitness,
            "mse": float(fit_best.get("residual_mse", fit_best.get("mse", 1e18))),
            "str": sequence_to_str(seq_best),
            "readable": readable,
            "complexity": float(fit_best.get("complexity", len(seq_best))),
        }
        if extra_fields:
            updated.update(dict(extra_fields))

        self.best_equations[tf] = updated
        if getattr(self, "active_target_field", tf) == tf:
            self.best_equation = updated

        if log_update and self.logger is not None and hasattr(self.logger, "log_equation"):
            try:
                self.logger.log_equation(
                    epoch=len(self.history.get("epoch", [])) + 1,
                    eq_str=updated.get("readable") or updated.get("str") or "",
                    consts=updated.get("consts", np.array([])),
                    mse=float(updated.get("mse", 1e18)),
                    fitness=float(updated.get("fitness", 1e18)),
                    tag=self._tag_from_target_field(tf),
                )
            except Exception:
                pass
    def _validate_seq_const_pair(self, seq: List[int], consts: np.ndarray) -> bool:
        try:
            seq_list = list(map(int, seq))
            const_arr = np.asarray(consts, dtype=np.float64).reshape(-1)
            if not is_valid_sequence(seq_list):
                return False
            if int(count_constants(seq_list)) != int(len(const_arr)):
                return False
            if const_arr.size > 0 and not np.all(np.isfinite(const_arr)):
                return False
            return True
        except Exception:
            return False

    def _sync_best_from_population(self, target_field: str) -> None:
        tf = str(target_field)
        pop = self._get_population(tf)
        if not pop.population:
            return
        try:
            cand = pop.get_best(1)[0]
        except Exception:
            return
        seq = list(map(int, cand.get("seq", [])))
        consts = np.asarray(cand.get("consts", np.array([])), dtype=np.float64)
        fit = {
            "pred_valid": True,
            "fitness": float(cand.get("fitness", 1e18)),
            "residual_mse": float(cand.get("mse", 1e18)),
            "complexity": float(cand.get("complexity", len(seq))),
        }
        self._update_best(seq, consts, fit, tf)

    def _evolve_population_once(self, target_field: str, epoch: int) -> None:
        tf = str(target_field)
        pop = self._get_population(tf)
        if not pop.population:
            self._sync_best_from_population(tf)
            return
        try:
            pop.evolve_with_diversity_control(
                self.ctx,
                target_field=tf,
                epoch=int(epoch),
                n_offspring=int(getattr(self.cfg_obj, "n_offspring", 36)),
                valid_terminals=self.valid_terminals,
                policy=self.operator_policy,
                spatial_ndim=self.spatial_ndim,
                tournament_k=int(getattr(self.cfg_obj, "evo_tournament_k", 2)),
                random_parent_prob=float(getattr(self.cfg_obj, "evo_random_parent_prob", 0.25)),
                stagnated=int(getattr(self, "_stagnated_epochs", 0)),
                neighborhood_trials=int(getattr(self.cfg_obj, "evo_neighborhood_trials", 4)),
                neighborhood_keep=int(getattr(self.cfg_obj, "evo_neighborhood_keep", 1)),
                big_mutation_patience=int(getattr(self.cfg_obj, "evo_big_mutation_patience", 4)),
                restart_patience=int(getattr(self.cfg_obj, "evo_restart_patience", 8)),
                max_restart_prob=float(getattr(self.cfg_obj, "evo_max_restart_prob", 0.25)),
                max_big_mutation_prob=float(getattr(self.cfg_obj, "evo_max_big_mutation_prob", 0.35)),
                max_add_depth=int(getattr(self.cfg_obj, "evo_max_add_depth", 6)),
                max_mut_depth=int(getattr(self.cfg_obj, "evo_max_mut_depth", 12)),
                term_crossover_prob=float(getattr(self.cfg_obj, "evo_term_crossover_prob", 0.20)),
                drop_term_prob=float(getattr(self.cfg_obj, "evo_drop_term_prob", 0.06)),
                replace_term_prob=float(getattr(self.cfg_obj, "evo_replace_term_prob", 0.10)),
                residual_guided_prob=float(getattr(self.cfg_obj, "evo_residual_guided_prob", 0.12)),
                residual_guided_basis_trials=int(getattr(self.cfg_obj, "evo_residual_guided_basis_trials", 96)),
                residual_guided_topk=int(getattr(self.cfg_obj, "evo_residual_guided_topk", 16)),
                random_immigrant_frac=float(getattr(self.cfg_obj, "evo_random_immigrant_frac", 0.10)),
                primitive_completion_frac=float(getattr(self.cfg_obj, "evo_primitive_completion_frac", 0.15)),
                random_tree_immigrant_frac=float(getattr(self.cfg_obj, "evo_random_tree_immigrant_frac", 0.25)),
                optimize_kwargs={
                    **self._with_residual_diagnostics_enabled(self._optimizer_common_kwargs(), False),
                    "enable_nonlinear_lbfgs": (
                        bool(getattr(self.cfg_obj, "enable_nonlinear_lbfgs", True))
                        and bool(getattr(self.cfg_obj, "nonlinear_lbfgs_evolution", False))
                    ),
                    "mdl_penalty_weight": float(getattr(self.cfg_obj, "mdl_penalty_weight", 0.25)),
                    "const_scaled_threshold": float(getattr(self.cfg_obj, "const_scaled_threshold", 1e-4)),
                    "const_l2": float(getattr(self.cfg_obj, "const_l2", 5e-3)),
                    "const_ridge": float(getattr(self.cfg_obj, "const_ridge", 1e-8)),
                    "const_context_budget": int(max(1, getattr(self.cfg_obj, "search_context_budget", 1) or 1)),
                },
            )
        except Exception as exc:
            self._warn(f"population evolve failed for {tf}: {exc}")
        self._sync_best_from_population(tf)

    def apply_pareto_screening(self, top_k: int = 20, target_field: str = "du_t") -> None:
        """Apply SRM-style post-training candidate selection."""
        tf = str(target_field)
        pop = self._get_population(tf)
        if not pop.population:
            return
        candidates = pop.get_best(top_k)
        if not candidates:
            return
        best_cand = candidates[0]
        baseline_mse = float(best_cand.get("mse", 1e18))
        baseline_comp = float(best_cand.get("complexity", 1.0))
        n_eff = float(best_cand.get("n_eff", 100.0))
        vocab_size = float(best_cand.get("vocab_size", 20.0))
        srm_penalty_slope = math.log(max(vocab_size, 2.0)) / math.sqrt(max(n_eff, 1.0))
        valid_candidates = [best_cand]
        for cand in candidates[1:]:
            c_mse = float(cand.get("mse", 1e18))
            c_comp = float(cand.get("complexity", 1.0))
            if c_comp < baseline_comp:
                delta_comp = baseline_comp - c_comp
                theoretical_ratio = 1.0 + delta_comp * srm_penalty_slope
                actual_ratio = min(theoretical_ratio, 1.15)
                if c_mse <= baseline_mse * actual_ratio:
                    valid_candidates.append(cand)
        final_choice = min(valid_candidates, key=lambda x: (float(x.get("complexity", 1e18)), float(x.get("mse", 1e18))))
        seq = list(map(int, final_choice.get("seq", [])))
        consts = np.asarray(final_choice.get("consts", np.array([])), dtype=np.float64)
        fit = {
            "pred_valid": True,
            "fitness": float(final_choice.get("fitness", 1e18)),
            "residual_mse": float(final_choice.get("mse", 1e18)),
            "complexity": float(final_choice.get("complexity", len(seq))),
        }
        self._update_best(seq, consts, fit, tf, force=True, extra_fields={}, log_update=False)

    def _result_payload_map(self, results: List[Tuple[str, List[int], np.ndarray, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        payload: Dict[str, Dict[str, Any]] = {}
        for key, seq, consts, fit in results:
            score = float(fit.get("fitness", 1e18))
            prev = payload.get(key)
            if prev is None or score < float(prev.get("fitness", 1e18)):
                payload[key] = {
                    "seq": list(map(int, seq)),
                    "consts": np.asarray(consts, dtype=np.float64),
                    "fitness": score,
                    "mse": float(fit.get("residual_mse", np.nan)),
                    "reward_score": float(fit.get("contrastive_reward_score", score)),
                    "spectral_high_mse": float(fit.get("spectral_high_mse", 0.0)),
                    "multi_window_logvar": float(fit.get("multi_window_logvar", 0.0)),
                }
        return payload

    def _attach_record_targets(self, records: List[SampleRecord], payload_map: Dict[str, Dict[str, Any]]) -> None:
        for rec in records:
            payload = payload_map.get(rec.key)
            if payload is None:
                continue
            rec.valid = True
            rec.fitness = float(payload.get("fitness", 1e18))
            rec.mse = float(payload.get("mse", np.nan))
            rec.reward_score = float(payload.get("reward_score", rec.fitness))

    # --------------------------------------------------------
    # REINFORCE
    # --------------------------------------------------------
    def _reinforce_loss(
        self,
        z_bank: torch.Tensor,
        records: List[SampleRecord],
        gamma: float,
        epoch: int,
        n_epochs: int,
        target_field: Optional[str] = None,
    ) -> torch.Tensor:
        """Minimal rank-based policy-gradient loss for symbol proposals.

        This is intentionally small: keep only the feedback needed for the
        neural decoder to prefer low-residual structures.  All complicated
        collapse handling, reward schedules and contrastive training knobs have
        been removed.  Final correctness still comes from weak-form evaluation,
        constant optimization, evolutionary structure search and archive rerank.
        """
        device = z_bank.device
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        valid_records = [r for r in records if r.valid and np.isfinite(r.fitness)]
        if len(valid_records) < 2:
            return zero

        best_by_key: Dict[str, SampleRecord] = {}
        for r in valid_records:
            prev = best_by_key.get(r.key)
            if prev is None or float(r.fitness) < float(prev.fitness):
                best_by_key[r.key] = r

        selected = sorted(best_by_key.values(), key=lambda r: float(r.fitness))
        if len(selected) < 2:
            selected = sorted(valid_records, key=lambda r: float(r.fitness))
        elite_k = int(max(2, min(12, len(selected))))
        selected = selected[:elite_k]
        if len(selected) < 2:
            return zero

        # Best candidate receives positive advantage, worst receives negative.
        advantages = np.linspace(1.0, -1.0, num=len(selected), dtype=np.float64)
        advantages = advantages - advantages.mean()
        std = float(np.std(advantages))
        if not np.isfinite(std) or std <= 0.0:
            return zero
        advantages = advantages / (std + 1e-8)

        idxs = torch.tensor([int(r.sample_idx) for r in selected], dtype=torch.long, device=device)
        z_sel = z_bank.index_select(0, idxs)
        seqs = [list(map(int, r.seq)) for r in selected]

        target_name = str(target_field or self.active_target_field or "du_t")
        target_ids = self.model.gsd.make_target_ids([target_name] * int(z_sel.shape[0]), device=device)
        log_probs = self.model.gsd.compute_sequence_logprob(
            z_sel,
            seqs,
            gamma=gamma,
            valid_terminals=self.valid_terminals,
            allowed_axes=self.model.allowed_axes,
            epoch=epoch,
            n_epochs=n_epochs,
            temperature=self._effective_temperature(epoch, n_epochs),
            target_ids=target_ids,
        )
        log_probs = torch.clamp(log_probs, min=float(getattr(self.cfg_obj, "logprob_min", -50.0)), max=0.0)
        adv_t = torch.as_tensor(advantages, dtype=log_probs.dtype, device=device)
        return -(adv_t * log_probs).mean()


    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------
    def _sample_mixed_sequences(
        self,
        z: torch.Tensor,
        gamma: float,
        temperature: float,
        epoch: int,
        n_epochs: int,
        target_field: Optional[str] = None,
        shared_z_bank: Optional[torch.Tensor] = None,
    ):
        K = int(self.cfg_obj.n_eq_samples)
        frac = float(getattr(self.cfg_obj, "latent_mixture_frac", 0.5))
        prior_frac = float(getattr(self.cfg_obj, "latent_prior_frac", 0.15))
        noise_std = float(getattr(self.cfg_obj, "latent_noise_std", 0.08))
        k_mean = int(round(K * max(0.0, 1.0 - frac - prior_frac)))
        k_mix = int(round(K * max(0.0, frac)))
        k_prior = max(0, K - k_mean - k_mix)
        z_mean = z.mean(dim=0, keepdim=True)
        banks: List[torch.Tensor] = []
        if k_mean > 0:
            banks.append(z_mean.expand(k_mean, -1))
        if k_mix > 0:
            if z.shape[0] > 1:
                idx = torch.randint(0, z.shape[0], (k_mix,), device=z.device)
                z_sel = z[idx]
            else:
                z_sel = z_mean.expand(k_mix, -1)
            noise = noise_std * torch.randn_like(z_sel)
            banks.append(z_sel + noise)
        if k_prior > 0:
            banks.append(torch.randn(k_prior, z.shape[-1], device=z.device))

        z_bank = shared_z_bank if shared_z_bank is not None else (torch.cat(banks, dim=0) if banks else z_mean.expand(K, -1))

        target_name = str(target_field or self.active_target_field or "du_t")
        target_ids = self.model.gsd.make_target_ids([target_name] * int(z_bank.shape[0]), device=z_bank.device)

        with torch.no_grad():
            seqs, lps = self.model.gsd.decode_sample(
                z_bank,
                gamma=gamma,
                temperature=temperature,
                valid_terminals=self.valid_terminals,
                allowed_axes=self.model.allowed_axes,
                epoch=epoch,
                n_epochs=n_epochs,
                target_ids=target_ids,
            )

        records: List[SampleRecord] = []

        for i, seq in enumerate(seqs):
            records.append(
                SampleRecord(
                    sample_idx=i,
                    seq=list(map(int, seq)),
                    key=self._canonical_key(seq) if is_valid_sequence(seq) else "",
                    latent_index=i,
                    valid=False,
                )
            )
        return z_bank, seqs, lps, records

    # --------------------------------------------------------
    # Candidate evaluation
    # --------------------------------------------------------
    def _collect_unique_candidates(self, sample_records: List[SampleRecord]) -> Dict[str, List[int]]:
        unique: Dict[str, List[int]] = {}
        for rec in sample_records:
            if not is_valid_sequence(rec.seq):
                continue
            canon_seq = self._canonical_seq(rec.seq)
            rec.seq = canon_seq
            rec.key = self._canonical_key(canon_seq)
            unique.setdefault(rec.key, canon_seq)

        reuse_k = int(getattr(self.cfg_obj, "population_reuse_k", 0))
        if reuse_k > 0 and self.population.population:
            for cand in self.population.get_best(reuse_k):
                seq = list(map(int, cand.get("seq", [])))
                if not is_valid_sequence(seq):
                    continue
                canon_seq = self._canonical_seq(seq)
                key = self._canonical_key(canon_seq)
                unique.setdefault(key, canon_seq)


        # Optional task-level seed templates.  This is disabled by default and is
        # only used by engineering-validation tasks where a very small real dataset
        # benefits from evaluating a few physically plausible basis combinations
        # (coefficients are still optimized from data).
        seed_templates = getattr(self.cfg_obj, "seed_candidate_templates", None)
        if seed_templates:
            for tpl in list(seed_templates):
                try:
                    if isinstance(tpl, str):
                        toks = [str(x).strip() for x in tpl.split() if str(x).strip()]
                    else:
                        toks = list(tpl)
                    seq = [int(SYM2IDX[t]) for t in toks if t in SYM2IDX]
                    if len(seq) != len(toks) or not is_valid_sequence(seq):
                        continue
                    canon_seq = self._canonical_seq(seq)
                    key = self._canonical_key(canon_seq)
                    unique.setdefault(key, canon_seq)
                except Exception:
                    continue

        return unique

    def _refine_selection_score(self, row: Tuple[str, List[int], np.ndarray, Dict[str, Any]]) -> float:
        """Generic coarse-to-refine priority score.

        The final equation is still selected by optimized fitness.  This score
        only decides which coarse candidates receive expensive full refinement.
        It gives a mild task-agnostic chance to additive affine structures while
        demoting explicit derivative products that often overfit weak residuals.
        """
        _, seq, _, fit = row
        base = float(fit.get("fitness", 1e18))
        if not bool(getattr(self.cfg_obj, "reaction_diffusion_refine_bonus", True)):
            return base
        syms = [IDX2SYM.get(int(t), "") for t in seq]
        bonus = 0.0
        if "+" in syms:
            bonus += 0.035
        if syms.count("const") >= 2:
            bonus += 0.035
        if any(s in syms for s in ("lap", "adv")):
            bonus += 0.025
        if any(s in syms for s in ("sq", "cube")):
            bonus += 0.020
        if len({s for s in syms if s in {"u", "v", "w"}}) >= 2:
            bonus += 0.020

        # Penalize raw-D-heavy multiplicative expressions only at selection time;
        # KdV-like explicit derivatives remain possible because elite rows are
        # always kept and the bonus is small.
        raw_d = syms.count("D")
        if raw_d and "*" in syms:
            bonus -= 0.040 * min(raw_d, 6)
        return float(base - bonus)

    def _candidate_bucket(self, seq: List[int]) -> str:
        """Generic structural bucket used for coarse-to-fine candidate diversity.

        This is intentionally not a PDE-template prior.  It does not promote
        Burgers-, diffusion-, advection-, or reaction-specific forms.  It only
        groups candidates by broad operator families so the full-refine stage is
        not monopolized by many near-duplicate low-coarse-loss expressions.
        """
        syms = [IDX2SYM.get(int(t), "") for t in seq]
        families: List[str] = []

        if any(s in syms for s in ("D", "lap", "adv")):
            families.append("derivative")
        if any(s in syms for s in ("sq", "cube", "^")):
            families.append("polynomial")
        if any(s in syms for s in ("sin", "cos", "exp", "log")):
            families.append("transcendental")
        if "/" in syms:
            families.append("rational")
        if "*" in syms:
            families.append("multiplicative")
        if "+" in syms:
            families.append("additive")

        return "+".join(families) if families else "terminal"

    def _select_refine_rows_by_structure(
        self,
        search_rows: List[Tuple[str, List[int], np.ndarray, Dict[str, Any]]],
        refine_topk: int,
        target_field: str = "du_t",
    ) -> List[Tuple[str, List[int], np.ndarray, Dict[str, Any]]]:
        if not search_rows:
            return []

        rows = sorted(search_rows, key=self._refine_selection_score)
        refine_topk = int(max(1, refine_topk))
        selected: List[Tuple[str, List[int], np.ndarray, Dict[str, Any]]] = []
        seen = set()

        # Preserve global elites before diversity bucketing.
        n_elite = max(1, refine_topk // 3)
        for row in rows[:n_elite]:
            key = row[0]
            if key not in seen:
                selected.append(row)
                seen.add(key)

        buckets: Dict[str, List[Tuple[str, List[int], np.ndarray, Dict[str, Any]]]] = {}
        for row in rows:
            bucket = self._candidate_bucket(row[1])
            buckets.setdefault(bucket, []).append(row)

        # Generic round-robin diversity: sort buckets by their best coarse
        # fitness, then take one candidate from each bucket per pass.  No
        # physics-specific bucket is privileged.
        bucket_order = sorted(
            buckets.keys(),
            key=lambda b: self._refine_selection_score(buckets[b][0]),
        )

        made_progress = True
        depth = 0
        while made_progress and len(selected) < refine_topk:
            made_progress = False
            for bucket in bucket_order:
                rows_b = buckets.get(bucket, [])
                if depth >= len(rows_b):
                    continue
                row = rows_b[depth]
                key = row[0]
                if key in seen:
                    continue
                selected.append(row)
                seen.add(key)
                made_progress = True
                if len(selected) >= refine_topk:
                    return selected
            depth += 1

        # Fill remaining slots by fitness order.
        for row in rows:
            key = row[0]
            if key in seen:
                continue
            selected.append(row)
            seen.add(key)
            if len(selected) >= refine_topk:
                break

        return selected

    def _evaluate_candidates(self, unique: Dict[str, List[int]], target_field: str):
        search_rows: List[Tuple[str, List[int], np.ndarray, Dict[str, Any]]] = []
        key_to_fit: Dict[str, Tuple[float, float]] = {}

        # Shared-subtree warmup: exact cache preload, no candidate pruning.
        n_warmed = 0
        if bool(self.cfg.get("subtree_warmup", True)):
            n_warmed = self._warmup_shared_subtrees(list(unique.values()), self.ctx_search)

        # DRmask can be used in two modes:
        #   refine     : default; coarse-scores every candidate, then reduces full-refine redundancy.
        #   pre_coarse : speed-first; prunes before optimize_constants.
        #   both       : applies both.
        n_pruned = 0
        dr_stage = str(getattr(self.cfg_obj, "dr_mask_stage", "refine")).lower()
        if bool(self.cfg.get("dr_mask_enable", True)) and dr_stage in {"pre", "pre_coarse", "both"}:
            unique, n_pre = self._apply_dr_mask_to_unique(unique)
            n_pruned += int(n_pre)

        n_unique_after_mask = len(unique)

        opt_common = self._optimizer_common_kwargs()
        # Expensive spectral/multi-window diagnostics are deliberately NOT run
        # during coarse scoring.  They are reserved for a small elite subset in
        # full refinement; otherwise every candidate pays for FFT/smoothing.
        opt_common_coarse = self._with_residual_diagnostics_enabled(opt_common, False)

        # ---------------- coarse search ----------------
        for key, seq in unique.items():
            try:
                consts_search, fit_search = optimize_constants(
                    seq,
                    self.ctx_search,
                    target_field=target_field,
                    n_init=2,
                    const_scaled_threshold=float(getattr(self.cfg_obj, "const_scaled_threshold", 1e-4)),
                    const_l2=float(getattr(self.cfg_obj, "const_l2", 5e-3)),
                    const_ridge=float(getattr(self.cfg_obj, "const_ridge", 1e-8)),
                    mdl_penalty_weight=float(getattr(self.cfg_obj, "mdl_penalty_weight", 0.25)),
                    enable_nonlinear_lbfgs=(
                        bool(getattr(self.cfg_obj, "enable_nonlinear_lbfgs", True))
                        and bool(getattr(self.cfg_obj, "nonlinear_lbfgs_coarse", False))
                    ),
                    const_context_budget=int(max(1, getattr(self.cfg_obj, "search_context_budget", 1) or 1)),
                    **opt_common_coarse,
                )
            except Exception as exc:
                self._warn(f"search optimize failed: {exc}")
                continue

            if not fit_search.get("pred_valid", False):
                continue

            search_rows.append((key, seq, consts_search, fit_search))
            key_to_fit[key] = (
                float(fit_search.get("fitness", 1e18)),
                float(fit_search.get("residual_mse", np.nan)),
            )

        # ---------------- post-coarse DRmask ----------------
        # Default precision-preserving mode: every candidate is coarse-scored,
        # then only redundant rows are removed before expensive full refinement.
        if (
            bool(getattr(self.cfg_obj, "dr_mask_enable", True))
            and search_rows
            and dr_stage in {"refine", "post", "post_coarse", "both"}
        ):
            try:
                search_rows, n_post = self._apply_dr_mask_to_search_rows(search_rows)
                n_pruned += int(n_post)
            except Exception as exc:
                self._warn(f"DRmask failed: {exc}")

        # ---------------- coarse-to-fine refinement ----------------
        use_same_ctx = self.ctx_search is self.ctx
        refine_topk = max(1, int(self._effective_refine_topk()))

        if use_same_ctx:
            refine_rows = self._select_refine_rows_by_structure(
                search_rows, min(refine_topk, len(search_rows)), target_field=target_field
            )
        else:
            refine_rows = self._select_refine_rows_by_structure(
                search_rows, min(refine_topk, len(search_rows)), target_field=target_field
            )

        full_results: List[Tuple[str, List[int], np.ndarray, Dict[str, Any]]] = []
        nonlinear_elite = int(max(0, getattr(self.cfg_obj, "nonlinear_lbfgs_refine_elite", 8)))
        diag_elite = int(max(0, getattr(self.cfg_obj, "e2e_diagnostics_refine_topk", 24))) if bool(getattr(self.cfg_obj, "training_residual_diagnostics", False)) else 0
        for refine_rank, (key, seq, consts_search, fit_search) in enumerate(refine_rows):
            try:
                opt_common_refine = self._with_residual_diagnostics_enabled(opt_common, int(refine_rank) < diag_elite)
                consts_full, fit_full = optimize_constants(
                    seq,
                    self.ctx,
                    target_field=target_field,
                    n_init=1,
                    init_consts=consts_search,
                    const_scaled_threshold=float(getattr(self.cfg_obj, "const_scaled_threshold", 1e-4)),
                    const_l2=float(getattr(self.cfg_obj, "const_l2", 5e-3)),
                    const_ridge=float(getattr(self.cfg_obj, "const_ridge", 1e-8)),
                    mdl_penalty_weight=float(getattr(self.cfg_obj, "mdl_penalty_weight", 0.25)),
                    enable_nonlinear_lbfgs=(
                        bool(getattr(self.cfg_obj, "enable_nonlinear_lbfgs", True))
                        and (int(refine_rank) < nonlinear_elite)
                    ),
                    const_context_budget=int(max(1, getattr(self.cfg_obj, "refine_context_budget", getattr(self.cfg_obj, "search_context_budget", 1)) or 1)),
                    **opt_common_refine,
                )
            except Exception as exc:
                self._warn(f"full optimize failed: {exc}")
                continue

            if not fit_full.get("pred_valid", False):
                continue

            full_results.append((key, seq, consts_full, fit_full))
            key_to_fit[key] = (
                float(fit_full.get("fitness", 1e18)),
                float(fit_full.get("residual_mse", np.nan)),
            )

            self._update_best(seq, consts_full, fit_full, target_field)
            try:
                self.population.add_evaluated_candidate(seq, consts_full, fit_full)
            except Exception as exc:
                self._warn(f"population insert failed: {exc}")

        self._last_stage_stats = {
            "shared_subtrees_warmed": int(n_warmed),
            "dr_mask_pruned": int(n_pruned),
            "coarse_candidates": int(n_unique_after_mask),
            "refine_candidates": int(len(refine_rows)),
            "effective_full_refine_topk": int(refine_topk),
        }
        return key_to_fit, search_rows, full_results

    # --------------------------------------------------------
    # One epoch
    # --------------------------------------------------------
    def train_epoch(self, data_tensor: torch.Tensor, epoch: int, n_epochs: int, target_field: str = "du_t") -> Dict[str, Any]:
        self._activate_target(target_field)
        self.model.train()

        self.operator_policy = build_operator_policy(
            epoch=epoch,
            spatial_ndim=getattr(self, "spatial_ndim", 1),
            mode=getattr(self.cfg_obj, "operator_mode", "pde"),
        )
        refresh_operator_groups(policy=self.operator_policy)

        try:
            self.model.gsd.set_operator_policy(self.operator_policy)
        except Exception:
            pass

        beta = float(self.cfg_obj.beta_kl)
        a_rf = float(self.cfg_obj.alpha_reinforce)
        a_sp = float(self.cfg_obj.alpha_sparsity)
        temperature = self._effective_temperature(epoch, n_epochs)
        gamma = self._compute_gamma(epoch, n_epochs)

        x = data_tensor.to(self.device)
        z, mu, logvar, _ = self.model.encode(x)
        kl = self.model.kl_loss(mu, logvar)
        struct_loss = self.model.struct_sparsity_loss(z)

        with torch.no_grad():
            z_bank, seqs, _, sample_records = self._sample_mixed_sequences(
                z, gamma, temperature, epoch, n_epochs, target_field=target_field
            )

        eval_before = get_evaluator_cache_stats(reset=False)
        opt_before = get_optimizer_cache_stats(reset=False)

        n_sampled = len(seqs)
        n_seq_valid = sum(1 for s in seqs if is_valid_sequence(s))
        unique = self._collect_unique_candidates(sample_records)
        key_to_fit, search_rows, full_results = self._evaluate_candidates(unique, target_field)

        fitnesses: List[float] = []
        mses: List[float] = []
        for _, _, _, fit in full_results if full_results else []:
            fitnesses.append(float(fit.get("fitness", 1e18)))
            mses.append(float(fit.get("residual_mse", np.nan)))
        if not full_results:
            for _, _, _, fit in search_rows:
                fitnesses.append(float(fit.get("fitness", 1e18)))
                mses.append(float(fit.get("residual_mse", np.nan)))

        for rec in sample_records:
            if rec.key in key_to_fit:
                rec.valid = True
                rec.fitness, rec.mse = key_to_fit[rec.key]

        n_valid_eqs = int(len(key_to_fit))
        f_arr = np.asarray(fitnesses, dtype=np.float64) if fitnesses else np.array([], dtype=np.float64)
        m_arr = np.asarray(mses, dtype=np.float64) if mses else np.array([], dtype=np.float64)
        batch_min_fit = float(np.min(f_arr)) if f_arr.size else float("nan")
        batch_mean_fit = float(np.mean(f_arr)) if f_arr.size else float("nan")
        batch_std_fit = float(np.std(f_arr)) if f_arr.size else float("nan")
        batch_min_mse = float(np.nanmin(m_arr)) if m_arr.size else float("nan")
        batch_mean_mse = float(np.nanmean(m_arr)) if m_arr.size else float("nan")
        batch_std_mse = float(np.nanstd(m_arr)) if m_arr.size else float("nan")

        rf_loss = self._reinforce_loss(
            z_bank, sample_records, gamma, epoch=epoch, n_epochs=n_epochs, target_field=target_field
        )

        total_loss = beta * kl + a_rf * rf_loss + a_sp * struct_loss

        # Drop evaluator CUDA caches before neural backward; candidate fitness
        # values have already been copied into Python/NumPy scalars.
        self._release_eval_cache_before_backward()

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.cfg_obj.grad_clip_norm))
        self.optimizer.step()
        if epoch % int(self.cfg_obj.evo_interval) == 0:
            try:
                if self.population.population:
                    self.population.evolve_with_diversity_control(
                        self.ctx,
                        target_field=target_field,
                        epoch=epoch,
                        n_offspring=int(self.cfg_obj.n_offspring),
                        valid_terminals=self.valid_terminals,
                        policy=self.operator_policy,
                        spatial_ndim=self.spatial_ndim,
                        tournament_k=int(getattr(self.cfg_obj, "evo_tournament_k", 2)),
                        random_parent_prob=float(getattr(self.cfg_obj, "evo_random_parent_prob", 0.25)),
                        stagnated=int(getattr(self, "_stagnated_epochs", 0)),
                        neighborhood_trials=int(getattr(self.cfg_obj, "evo_neighborhood_trials", 4)),
                        neighborhood_keep=int(getattr(self.cfg_obj, "evo_neighborhood_keep", 1)),
                        big_mutation_patience=int(getattr(self.cfg_obj, "evo_big_mutation_patience", 4)),
                        restart_patience=int(getattr(self.cfg_obj, "evo_restart_patience", 8)),
                        max_restart_prob=float(getattr(self.cfg_obj, "evo_max_restart_prob", 0.25)),
                        max_big_mutation_prob=float(getattr(self.cfg_obj, "evo_max_big_mutation_prob", 0.35)),
                        max_add_depth=int(getattr(self.cfg_obj, "evo_max_add_depth", 6)),
                        max_mut_depth=int(getattr(self.cfg_obj, "evo_max_mut_depth", 12)),
                        term_crossover_prob=float(getattr(self.cfg_obj, "evo_term_crossover_prob", 0.20)),
                        drop_term_prob=float(getattr(self.cfg_obj, "evo_drop_term_prob", 0.06)),
                        replace_term_prob=float(getattr(self.cfg_obj, "evo_replace_term_prob", 0.10)),
                        residual_guided_prob=float(getattr(self.cfg_obj, "evo_residual_guided_prob", 0.12)),
                        residual_guided_basis_trials=int(getattr(self.cfg_obj, "evo_residual_guided_basis_trials", 96)),
                        residual_guided_topk=int(getattr(self.cfg_obj, "evo_residual_guided_topk", 16)),
                        random_immigrant_frac=float(getattr(self.cfg_obj, "evo_random_immigrant_frac", 0.10)),
                        primitive_completion_frac=float(getattr(self.cfg_obj, "evo_primitive_completion_frac", 0.15)),
                        random_tree_immigrant_frac=float(getattr(self.cfg_obj, "evo_random_tree_immigrant_frac", 0.25)),
                        optimize_kwargs={
                            **self._with_residual_diagnostics_enabled(self._optimizer_common_kwargs(), False),
                            "enable_nonlinear_lbfgs": (
                                bool(getattr(self.cfg_obj, "enable_nonlinear_lbfgs", True))
                                and bool(getattr(self.cfg_obj, "nonlinear_lbfgs_evolution", False))
                            ),
                            "mdl_penalty_weight": float(getattr(self.cfg_obj, "mdl_penalty_weight", 0.25)),
                            "const_scaled_threshold": float(getattr(self.cfg_obj, "const_scaled_threshold", 1e-4)),
                            "const_l2": float(getattr(self.cfg_obj, "const_l2", 5e-3)),
                            "const_ridge": float(getattr(self.cfg_obj, "const_ridge", 1e-8)),
                            "const_context_budget": int(max(1, getattr(self.cfg_obj, "search_context_budget", 1) or 1)),
                        },
                    )
            except Exception as exc:
                self._warn(f"population evolve failed: {exc}")

        diversity_score = 0.0
        try:
            diversity_score = float(self.population.get_stats().get("diversity_score", 0.0))
        except Exception:
            pass

        eval_after = get_evaluator_cache_stats(reset=False)
        opt_after = get_optimizer_cache_stats(reset=False)

        if bool(getattr(self.cfg_obj, "clear_evaluator_cache_between_stages", True)):
            clear_evaluator_caches()

        # Keep the active target's public best-equation pointer in sync with
        # the per-target population best.  The actual state mutation goes through
        # _update_best(), so single-target and joint-target training share the
        # same validation, canonicalization and logging semantics.
        self._sync_best_from_population(target_field)

        subtree_queries_epoch = int(eval_after.get("queries", 0) - eval_before.get("queries", 0))
        subtree_hits_epoch = int(eval_after.get("hits", 0) - eval_before.get("hits", 0))
        subtree_misses_epoch = int(eval_after.get("misses", 0) - eval_before.get("misses", 0))
        subtree_saved_epoch = int(eval_after.get("saved_evals", 0) - eval_before.get("saved_evals", 0))
        subtree_hit_rate_epoch = float(subtree_hits_epoch / subtree_queries_epoch) if subtree_queries_epoch > 0 else 0.0
        template_queries_epoch = int(opt_after.get("queries", 0) - opt_before.get("queries", 0))
        template_hits_epoch = int(opt_after.get("hits", 0) - opt_before.get("hits", 0))
        template_misses_epoch = int(opt_after.get("misses", 0) - opt_before.get("misses", 0))
        template_saved_epoch = int(opt_after.get("saved_evals", 0) - opt_before.get("saved_evals", 0))
        template_hit_rate_epoch = float(template_hits_epoch / template_queries_epoch) if template_queries_epoch > 0 else 0.0

        best_eq_str = self.best_equation.get("readable") or self.best_equation.get("str") or ""
        stagnated_epochs = self._update_stagnation_state(float(self.best_equation.get("fitness", 1e18)))
        return {
            "kl_loss": _safe_item(kl),
            "reinforce_loss": _safe_item(rf_loss),
            "struct_loss": _safe_item(struct_loss),
            "total_loss": _safe_item(total_loss),
            "best_fitness": float(self.best_equation.get("fitness", 1e18)),
            "best_mse": float(self.best_equation.get("mse", 1e18)),
            "best_equation": best_eq_str,
            "gamma": float(gamma),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "n_valid_eqs": int(n_valid_eqs),
            "n_sampled_eqs": int(n_sampled),
            "n_sampled": int(n_sampled),
            "n_seq_valid": int(n_seq_valid),
            "batch_min_fitness": batch_min_fit,
            "batch_mean_fitness": batch_mean_fit,
            "batch_std_fitness": batch_std_fit,
            "batch_min_mse": batch_min_mse,
            "batch_mean_mse": batch_mean_mse,
            "batch_std_mse": batch_std_mse,
            "diversity_score": diversity_score,
            "population_size": float(self.population.get_stats().get("pop_size", 0.0)) if hasattr(self.population, "get_stats") else 0.0,
            "subtree_cache_queries": int(eval_after.get("queries", 0)),
            "subtree_cache_hits": int(eval_after.get("hits", 0)),
            "subtree_cache_misses": int(eval_after.get("misses", 0)),
            "subtree_cache_hit_rate": float(eval_after.get("hit_rate", 0.0)),
            "subtree_cache_saved_evals": int(eval_after.get("saved_evals", 0)),
            "subtree_cache_queries_epoch": subtree_queries_epoch,
            "subtree_cache_hits_epoch": subtree_hits_epoch,
            "subtree_cache_misses_epoch": subtree_misses_epoch,
            "subtree_cache_hit_rate_epoch": subtree_hit_rate_epoch,
            "subtree_cache_saved_evals_epoch": subtree_saved_epoch,
            "template_cache_queries": int(opt_after.get("queries", 0)),
            "template_cache_hits": int(opt_after.get("hits", 0)),
            "template_cache_misses": int(opt_after.get("misses", 0)),
            "template_cache_hit_rate": float(opt_after.get("hit_rate", 0.0)),
            "template_cache_saved_evals": int(opt_after.get("saved_evals", 0)),
            "template_cache_queries_epoch": template_queries_epoch,
            "template_cache_hits_epoch": template_hits_epoch,
            "template_cache_misses_epoch": template_misses_epoch,
            "template_cache_hit_rate_epoch": template_hit_rate_epoch,
            "template_cache_saved_evals_epoch": template_saved_epoch,
            "shared_subtrees_warmed": int(self._last_stage_stats.get("shared_subtrees_warmed", 0)),
            "dr_mask_pruned": int(self._last_stage_stats.get("dr_mask_pruned", 0)),
            "coarse_candidates": int(self._last_stage_stats.get("coarse_candidates", 0)),
            "refine_candidates": int(self._last_stage_stats.get("refine_candidates", 0)),
            "effective_full_refine_topk": int(self._last_stage_stats.get("effective_full_refine_topk", self._effective_refine_topk())),
            "stagnated_epochs": int(stagnated_epochs),
        }

    def train_epoch_joint(self, data_tensor: torch.Tensor, epoch: int, n_epochs: int, target_fields: List[str]) -> Dict[str, Any]:
        self.model.train()

        self.operator_policy = build_operator_policy(
            epoch=epoch,
            spatial_ndim=getattr(self, "spatial_ndim", 1),
            mode=getattr(self.cfg_obj, "operator_mode", "pde"),
        )
        refresh_operator_groups(policy=self.operator_policy)
        try:
            self.model.gsd.set_operator_policy(self.operator_policy)
        except Exception:
            pass

        beta = float(self.cfg_obj.beta_kl)
        a_rf = float(self.cfg_obj.alpha_reinforce)
        a_sp = float(self.cfg_obj.alpha_sparsity)
        temperature = self._effective_temperature(epoch, n_epochs)
        gamma = self._compute_gamma(epoch, n_epochs)

        x = data_tensor.to(self.device)
        z, mu, logvar, _ = self.model.encode(x)
        kl = self.model.kl_loss(mu, logvar)
        struct_loss = self.model.struct_sparsity_loss(z)

        eval_before = get_evaluator_cache_stats(reset=False)
        opt_before = get_optimizer_cache_stats(reset=False)

        per_target_rf_losses: List[torch.Tensor] = []
        all_fitnesses: List[float] = []
        all_mses: List[float] = []
        valid_eq_counts: List[int] = []
        n_sampled = 0
        n_seq_valid = 0
        total_coarse = 0
        total_refine = 0
        total_warmed = 0
        total_pruned = 0
        effective_refine_topks: List[int] = []

        for tf in target_fields:
            self._activate_target(tf)
            with torch.no_grad():
                z_bank, seqs_tf, _, target_records = self._sample_mixed_sequences(
                    z,
                    gamma,
                    temperature,
                    epoch,
                    n_epochs,
                    target_field=tf,
                )

            n_sampled += len(seqs_tf)
            n_seq_valid += sum(1 for s in seqs_tf if is_valid_sequence(s))

            unique = self._collect_unique_candidates(target_records)
            key_to_fit, search_rows, full_results = self._evaluate_candidates(unique, target_field=tf)

            stage_stats = dict(getattr(self, "_last_stage_stats", {}) or {})
            total_coarse += int(stage_stats.get("coarse_candidates", 0))
            total_refine += int(stage_stats.get("refine_candidates", 0))
            total_warmed += int(stage_stats.get("shared_subtrees_warmed", 0))
            total_pruned += int(stage_stats.get("dr_mask_pruned", 0))
            effective_refine_topks.append(int(stage_stats.get("effective_full_refine_topk", self._effective_refine_topk())))

            results_to_iter = full_results if full_results else search_rows
            for _, _, _, fit in results_to_iter:
                all_fitnesses.append(float(fit.get("fitness", 1e18)))
                all_mses.append(float(fit.get("residual_mse", np.nan)))

            for rec in target_records:
                if rec.key in key_to_fit:
                    rec.valid = True
                    rec.fitness, rec.mse = key_to_fit[rec.key]

            valid_eq_counts.append(len(key_to_fit))
            per_target_rf_losses.append(
                self._reinforce_loss(
                    z_bank,
                    target_records,
                    gamma,
                    epoch=epoch,
                    n_epochs=n_epochs,
                    target_field=tf,
                )
            )

        rf_loss = torch.stack(per_target_rf_losses).mean() if per_target_rf_losses else torch.tensor(0.0, device=self.device)
        total_loss = beta * kl + a_rf * rf_loss + a_sp * struct_loss

        # Drop evaluator CUDA caches before neural backward; candidate fitness
        # values have already been copied into Python/NumPy scalars.
        self._release_eval_cache_before_backward()

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg_obj.grad_clip_norm))
        self.optimizer.step()
        evo_interval = int(max(1, getattr(self.cfg_obj, "evo_interval", 4)))
        if epoch % evo_interval == 0:
            for tf in target_fields:
                self._evolve_population_once(tf, epoch)

        eval_after = get_evaluator_cache_stats(reset=False)
        opt_after = get_optimizer_cache_stats(reset=False)
        if bool(getattr(self.cfg_obj, "clear_evaluator_cache_between_stages", True)):
            clear_evaluator_caches(reset_stats=False)

        subtree_queries_epoch = int(eval_after.get("queries", 0)) - int(eval_before.get("queries", 0))
        subtree_hits_epoch = int(eval_after.get("hits", 0)) - int(eval_before.get("hits", 0))
        subtree_misses_epoch = int(eval_after.get("misses", 0)) - int(eval_before.get("misses", 0))
        subtree_saved_epoch = int(eval_after.get("saved_evals", 0)) - int(eval_before.get("saved_evals", 0))
        subtree_hit_rate_epoch = float(subtree_hits_epoch / max(1, subtree_queries_epoch))

        template_queries_epoch = int(opt_after.get("queries", 0)) - int(opt_before.get("queries", 0))
        template_hits_epoch = int(opt_after.get("hits", 0)) - int(opt_before.get("hits", 0))
        template_misses_epoch = int(opt_after.get("misses", 0)) - int(opt_before.get("misses", 0))
        template_saved_epoch = int(opt_after.get("saved_evals", 0)) - int(opt_before.get("saved_evals", 0))
        template_hit_rate_epoch = float(template_hits_epoch / max(1, template_queries_epoch))

        f_arr = np.asarray(all_fitnesses, dtype=np.float64) if all_fitnesses else np.array([], dtype=np.float64)
        m_arr = np.asarray(all_mses, dtype=np.float64) if all_mses else np.array([], dtype=np.float64)
        batch_min_fit = float(np.min(f_arr)) if f_arr.size else float("nan")
        batch_mean_fit = float(np.mean(f_arr)) if f_arr.size else float("nan")
        batch_std_fit = float(np.std(f_arr)) if f_arr.size else float("nan")
        batch_min_mse = float(np.nanmin(m_arr)) if m_arr.size else float("nan")
        batch_mean_mse = float(np.nanmean(m_arr)) if m_arr.size else float("nan")
        batch_std_mse = float(np.nanstd(m_arr)) if m_arr.size else float("nan")

        joint_best_fit, joint_best_mse, joint_best_readable = self._joint_best_summary(target_fields)
        pop_stats = self._population_stats_for_targets(target_fields)

        self._last_stage_stats = {
            "shared_subtrees_warmed": int(total_warmed),
            "dr_mask_pruned": int(total_pruned),
            "coarse_candidates": int(total_coarse),
            "refine_candidates": int(total_refine),
            "effective_full_refine_topk": int(max(effective_refine_topks) if effective_refine_topks else self._effective_refine_topk()),
        }
        stagnated_epochs = self._update_stagnation_state(float(joint_best_fit))

        return {
            "kl_loss": _safe_item(kl),
            "reinforce_loss": _safe_item(rf_loss),
            "struct_loss": _safe_item(struct_loss),
            "total_loss": _safe_item(total_loss),
            "best_fitness": float(joint_best_fit),
            "best_mse": float(joint_best_mse),
            "best_equation": joint_best_readable,
            "gamma": float(gamma),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "n_valid_eqs": int(np.mean(valid_eq_counts)) if valid_eq_counts else 0,
            "diversity_score": float(pop_stats.get("diversity_score", 0.0)),
            "population_size": float(pop_stats.get("pop_size", 0.0)),
            "n_sampled": int(n_sampled),
            "n_seq_valid": int(n_seq_valid),
            "batch_min_fitness": batch_min_fit,
            "batch_mean_fitness": batch_mean_fit,
            "batch_std_fitness": batch_std_fit,
            "batch_min_mse": batch_min_mse,
            "batch_mean_mse": batch_mean_mse,
            "batch_std_mse": batch_std_mse,
            "subtree_cache_queries": int(eval_after.get("queries", 0)),
            "subtree_cache_hits": int(eval_after.get("hits", 0)),
            "subtree_cache_misses": int(eval_after.get("misses", 0)),
            "subtree_cache_hit_rate": float(eval_after.get("hit_rate", 0.0)),
            "subtree_cache_saved_evals": int(eval_after.get("saved_evals", 0)),
            "subtree_cache_queries_epoch": subtree_queries_epoch,
            "subtree_cache_hits_epoch": subtree_hits_epoch,
            "subtree_cache_misses_epoch": subtree_misses_epoch,
            "subtree_cache_hit_rate_epoch": subtree_hit_rate_epoch,
            "subtree_cache_saved_evals_epoch": subtree_saved_epoch,
            "template_cache_queries": int(opt_after.get("queries", 0)),
            "template_cache_hits": int(opt_after.get("hits", 0)),
            "template_cache_misses": int(opt_after.get("misses", 0)),
            "template_cache_hit_rate": float(opt_after.get("hit_rate", 0.0)),
            "template_cache_saved_evals": int(opt_after.get("saved_evals", 0)),
            "template_cache_queries_epoch": template_queries_epoch,
            "template_cache_hits_epoch": template_hits_epoch,
            "template_cache_misses_epoch": template_misses_epoch,
            "template_cache_hit_rate_epoch": template_hit_rate_epoch,
            "template_cache_saved_evals_epoch": template_saved_epoch,
            "shared_subtrees_warmed": int(total_warmed),
            "dr_mask_pruned": int(total_pruned),
            "coarse_candidates": int(total_coarse),
            "refine_candidates": int(total_refine),
            "effective_full_refine_topk": int(max(effective_refine_topks) if effective_refine_topks else self._effective_refine_topk()),
            "stagnated_epochs": int(stagnated_epochs),
        }


    def _run_reference_probe_once(self) -> None:
        """Run the optional reference-equation probe once per trainer."""
        if not bool(getattr(self.cfg_obj, "reference_probe_enable", False)):
            return
        if self._reference_probe_done:
            return
        self._reference_probe_done = True

        equation_name = (
            self._raw_cfg.get("equation_name")
            or self._raw_cfg.get("task_name")
            or self._raw_cfg.get("equation_type")
            or getattr(self._ctx0, "equation_type", "")
        )
        if not equation_name:
            return

        try:
            run_reference_probe(self.ctx, equation_name, logger=self.logger)
        except Exception as exc:
            message = f"Reference probe skipped/failed for {equation_name}: {exc}"
            if self.logger is not None and hasattr(self.logger, "warning"):
                self.logger.warning(message)
            else:
                print(f"[WARN] {message}")

    def _final_rerank_context_budget(self) -> int:
        """Return context budget for final archive rerank; 0 means all contexts."""
        try:
            budget = int(getattr(self.cfg_obj, "final_rerank_context_budget", 0))
        except Exception:
            budget = 0
        if isinstance(self.ctx, (list, tuple)):
            if budget <= 0:
                return int(max(1, len(self.ctx)))
            return int(max(1, min(budget, len(self.ctx))))
        return 1

    def _final_simplify_with_consts(
        self,
        seq: List[int],
        consts: Optional[np.ndarray],
        *,
        prune_tol: Optional[float] = None,
    ) -> List[int]:
        """Constant-aware simplification used only by final rerank.

        The training population keeps raw end-to-end generated structures for
        exploration.  Before the final common rerank, however, we should compare
        candidates by their pruned/canonical structure rather than by hidden
        zero branches that the printer later suppresses.
        """
        try:
            seq0 = self._canonical_seq(list(map(int, seq)))
        except Exception:
            seq0 = list(map(int, seq))
        if not seq0 or not is_valid_sequence(seq0):
            return list(seq0)

        arr = None
        if consts is not None:
            try:
                arr = np.asarray(consts, dtype=np.float64).reshape(-1)
            except Exception:
                arr = None
        if arr is None or arr.size != count_constants(seq0):
            return list(seq0)

        try:
            km = get_structure_key_manager()
            tol = float(prune_tol if prune_tol is not None else getattr(self.cfg_obj, "final_rerank_prune_tol", getattr(self.cfg_obj, "small_coeff_prune_tol", 1e-4)))
            simp = km.normalizer.algebraic_simplify(seq0, consts=arr, prune_tol=tol)
            simp = self._canonical_seq(list(map(int, simp)))
            if simp and is_valid_sequence(simp):
                return list(simp)
        except Exception:
            pass
        return list(seq0)

    def apply_final_archive_rerank(self, target_field: str) -> None:
        """Final-only identifiability rerank over population + archive.

        This is deliberately post-training only.  It does not inject terms, does
        not run sparse regression, and does not alter training-time survival.
        It simply gives historically generated structures a common full-context
        diagnostic score before selecting the final equation.
        """
        if not bool(getattr(self.cfg_obj, "final_archive_rerank_enable", True)):
            return
        tf = str(target_field)
        pop = self._get_population(tf)
        rows: List[Dict[str, Any]] = []
        try:
            rows.extend([dict(r) for r in getattr(pop, "population", [])])
        except Exception:
            pass
        try:
            rows.extend(pop.get_archive_candidates(
                topk=int(max(1, getattr(self.cfg_obj, "final_archive_rerank_topk", 256))),
                per_signature=int(max(1, getattr(self.cfg_obj, "final_archive_rerank_per_signature", 3))),
            ))
        except Exception:
            pass

        # Deduplicate while preserving a diverse archive-first union.  Then cap
        # the total number of expensive final optimizations.
        seen = set()
        unique_rows: List[Dict[str, Any]] = []
        for row in rows:
            try:
                seq = list(map(int, row.get("seq", [])))
                key = pop._canonical_key(seq) if hasattr(pop, "_canonical_key") else "_".join(map(str, seq))
            except Exception:
                continue
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append(row)
        # Keep both good fitness and structural coverage: signature archive has
        # already diversified rows, so a stable sort by current fitness is enough
        # before the cap.
        unique_rows.sort(key=lambda r: float(r.get("fitness", 1e18)))
        max_rows = int(max(1, getattr(self.cfg_obj, "final_archive_rerank_topk", 256)))
        unique_rows = unique_rows[:max_rows]
        if not unique_rows:
            return

        opt_common = self._optimizer_common_kwargs()
        use_diag = bool(getattr(self.cfg_obj, "final_archive_rerank_use_diagnostics", False))
        opt_common = self._with_residual_diagnostics_enabled(opt_common, use_diag)
        ctx_budget = self._final_rerank_context_budget()
        nonlinear_elite = int(max(0, getattr(self.cfg_obj, "nonlinear_lbfgs_refine_elite", 8)))

        scored: List[Tuple[float, int, List[int], np.ndarray, Dict[str, Any]]] = []
        prune_tol = float(getattr(self.cfg_obj, "final_rerank_prune_tol", getattr(self.cfg_obj, "small_coeff_prune_tol", 1e-4)))
        pre_simplify = bool(getattr(self.cfg_obj, "final_rerank_pre_simplify", True))
        prune_refit = bool(getattr(self.cfg_obj, "final_rerank_prune_refit", True))

        for rank, row in enumerate(unique_rows):
            raw_seq = list(map(int, row.get("seq", [])))
            if not raw_seq or not is_valid_sequence(raw_seq):
                continue
            init_consts = np.asarray(row.get("consts", np.array([])), dtype=np.float64).reshape(-1)

            # Use the candidate's existing constants to canonicalize/prune before
            # the final common re-fit.  This avoids comparing a hidden redundant
            # raw tree against a short surrogate when their printed equations are
            # actually similar.
            seq = list(raw_seq)
            if pre_simplify and init_consts.size == count_constants(seq):
                seq = self._final_simplify_with_consts(seq, init_consts, prune_tol=prune_tol)
            if not seq or not is_valid_sequence(seq):
                seq = list(raw_seq)
            if init_consts.size != count_constants(seq):
                init_for_fit = None
            else:
                init_for_fit = init_consts

            try:
                consts, fit = optimize_constants(
                    seq,
                    self.ctx,
                    target_field=tf,
                    n_init=1,
                    init_consts=init_for_fit,
                    const_scaled_threshold=float(getattr(self.cfg_obj, "const_scaled_threshold", 1e-4)),
                    const_l2=float(getattr(self.cfg_obj, "const_l2", 5e-3)),
                    const_ridge=float(getattr(self.cfg_obj, "const_ridge", 1e-8)),
                    mdl_penalty_weight=float(getattr(self.cfg_obj, "mdl_penalty_weight", 0.25)),
                    enable_nonlinear_lbfgs=(
                        bool(getattr(self.cfg_obj, "enable_nonlinear_lbfgs", True))
                        and (int(rank) < nonlinear_elite)
                    ),
                    const_context_budget=int(max(1, ctx_budget)),
                    **opt_common,
                )
            except Exception as exc:
                self._warn(f"final archive rerank failed for {tf}: {exc}")
                continue

            if not isinstance(fit, dict) or not bool(fit.get("pred_valid", False)):
                continue

            final_seq = list(seq)
            final_consts = np.asarray(consts, dtype=np.float64).reshape(-1)
            final_fit = dict(fit)
            pruned_changed = False

            # One final constant-aware cleanup after full-context fitting.  Refit
            # only if the canonical structure actually changed.
            if prune_refit and final_consts.size == count_constants(final_seq):
                pruned_seq = self._final_simplify_with_consts(final_seq, final_consts, prune_tol=prune_tol)
                try:
                    key_before = pop._canonical_key(final_seq) if hasattr(pop, "_canonical_key") else "_".join(map(str, final_seq))
                    key_after = pop._canonical_key(pruned_seq) if hasattr(pop, "_canonical_key") else "_".join(map(str, pruned_seq))
                except Exception:
                    key_before = str(final_seq); key_after = str(pruned_seq)
                if pruned_seq and is_valid_sequence(pruned_seq) and key_after != key_before:
                    try:
                        p_consts, p_fit = optimize_constants(
                            pruned_seq,
                            self.ctx,
                            target_field=tf,
                            n_init=1,
                            init_consts=None,
                            const_scaled_threshold=float(getattr(self.cfg_obj, "const_scaled_threshold", 1e-4)),
                            const_l2=float(getattr(self.cfg_obj, "const_l2", 5e-3)),
                            const_ridge=float(getattr(self.cfg_obj, "const_ridge", 1e-8)),
                            mdl_penalty_weight=float(getattr(self.cfg_obj, "mdl_penalty_weight", 0.25)),
                            enable_nonlinear_lbfgs=(
                                bool(getattr(self.cfg_obj, "enable_nonlinear_lbfgs", True))
                                and (int(rank) < nonlinear_elite)
                            ),
                            const_context_budget=int(max(1, ctx_budget)),
                            **opt_common,
                        )
                        if isinstance(p_fit, dict) and bool(p_fit.get("pred_valid", False)):
                            final_seq = list(pruned_seq)
                            final_consts = np.asarray(p_consts, dtype=np.float64).reshape(-1)
                            final_fit = dict(p_fit)
                            pruned_changed = True
                    except Exception:
                        pass

            if not isinstance(final_fit, dict) or not bool(final_fit.get("pred_valid", False)):
                continue
            fval = float(final_fit.get("fitness", 1e18))
            if not np.isfinite(fval) or fval >= 1e18:
                continue
            final_fit["raw_complexity"] = float(row.get("complexity", len(raw_seq)))
            final_fit["pruned_complexity"] = float(final_fit.get("complexity", len(final_seq)))
            final_fit["final_pruned_changed"] = bool(pruned_changed or list(final_seq) != list(raw_seq))
            scored.append((fval, int(rank), final_seq, final_consts, final_fit))

        if not scored:
            return
        scored.sort(key=lambda x: (x[0], float(x[4].get("complexity", len(x[2]))), int(x[1])))

        # Final one-standard-error style selection: after all candidates have
        # been re-fitted on the same context budget, prefer the simplest
        # structure whose weak residual is statistically/operationally
        # indistinguishable from the best residual.  This is a final-only
        # identifiability rule; it does not change numerical evaluation or the
        # search grammar.
        selected = scored[0]
        one_se_used = False
        if bool(getattr(self.cfg_obj, "final_one_se_selection", True)):
            try:
                mses = [float(item[4].get("residual_mse", item[4].get("raw_mse", 1e18))) for item in scored]
                finite_mses = [m for m in mses if np.isfinite(m) and m < 1e18]
                if finite_mses:
                    best_mse = min(finite_mses)
                    rel_tol = float(getattr(self.cfg_obj, "final_one_se_rel_tol", 0.10))
                    abs_tol = float(getattr(self.cfg_obj, "final_one_se_abs_tol", 1e-10))
                    mse_tol = max(abs_tol, rel_tol * max(best_mse, abs_tol))
                    eligible = []
                    for item in scored:
                        fit = item[4]
                        mse = float(fit.get("residual_mse", fit.get("raw_mse", 1e18)))
                        if np.isfinite(mse) and mse <= best_mse + mse_tol:
                            eligible.append(item)
                    if eligible:
                        selected = min(
                            eligible,
                            key=lambda x: (
                                float(x[4].get("complexity", len(x[2]))),
                                int(count_constants(x[2])),
                                float(x[4].get("residual_mse", 1e18)),
                                float(x[0]),
                            ),
                        )
                        one_se_used = True
            except Exception:
                selected = scored[0]
                one_se_used = False

        best_f, _, best_seq, best_consts, best_fit = selected
        extra = {
            "final_archive_rerank": True,
            "final_archive_candidates": len(scored),
            "final_archive_use_diagnostics": bool(use_diag),
            "final_one_se_selection": bool(one_se_used),
        }
        self._update_best(best_seq, best_consts, best_fit, tf, force=True, extra_fields=extra, log_update=True)
        # Reinsert reranked elite into population so saved top_population reflects
        # the final common scoring where possible.
        try:
            for fval, _, seq, consts, fit in scored[: min(32, len(scored))]:
                pop.add_evaluated_candidate(seq, consts, fit)
        except Exception:
            pass
        if self.logger is not None and hasattr(self.logger, "info"):
            try:
                readable = self.best_equations.get(tf, {}).get("readable", "")
                self.logger.info(
                    f"[FinalArchiveRerank] target={tf} | candidates={len(scored)} | "
                    f"best_fitness={best_f:.6g} | best={readable}"
                )
            except Exception:
                pass

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    def train(
        self,
        data_tensor: torch.Tensor,
        n_epochs: Optional[int] = None,
        target_field: str = "du_t",
        target_fields: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        n_epochs = int(self.cfg_obj.n_epochs if n_epochs is None else n_epochs)
        targets = list(target_fields) if target_fields is not None else [target_field]
        joint_mode = len(targets) > 1
        if verbose:
            print("\n" + "=" * 78)
            print("  IMPROVED WiSED TRAINING")
            print(f"  Epochs: {n_epochs} | Targets: {targets}")
            print(f"  Device: {self.device}")
            print(f"  Model params: {self.model.total_parameter_count():,}")
            print(f"  Valid terminals: {[IDX2SYM[i] for i in self.valid_terminals]}")
            print(f"  StructureKeyManager: {'ON' if self.model.key_manager is not None else 'OFF'}")
            print(f"  Evaluator module: models.weak_form_evaluator")
            print(f"  Evolution module: models.symbolic_search")
            print("=" * 78 + "\n")

        self._run_reference_probe_once()

        if self.enable_population_init:
            try:
                init_ctx = self.ctx_search if self.ctx_search is not None else self.ctx
                init_ctx = self._limit_ctx_budget(
                    init_ctx,
                    getattr(self.cfg_obj, "search_context_budget", None),
                )
                for tf in targets:
                    self._activate_target(tf)
                    self.population.initialize_unbiased(
                        init_ctx,
                        target_field=tf,
                        n_epochs=n_epochs,
                        verbose=bool(verbose),
                        init_random_trials=int(getattr(self.cfg_obj, "init_random_trials", max(self.cfg_obj.pop_size * 16, 512))),
                        init_max_ops=int(getattr(self.cfg_obj, "init_max_ops", 7)),
                        init_p_const=float(getattr(self.cfg_obj, "init_p_const", 0.10)),
                        generic_affine_bootstrap_frac=float(getattr(self.cfg_obj, "generic_affine_bootstrap_frac", 0.0)),
                        policy=self.operator_policy,
                        spatial_ndim=self.spatial_ndim,
                        optimize_kwargs={
                            **self._with_residual_diagnostics_enabled(self._optimizer_common_kwargs(), False),
                            "enable_nonlinear_lbfgs": (
                                bool(getattr(self.cfg_obj, "enable_nonlinear_lbfgs", True))
                                and bool(getattr(self.cfg_obj, "nonlinear_lbfgs_coarse", False))
                            ),
                            "mdl_penalty_weight": float(getattr(self.cfg_obj, "mdl_penalty_weight", 0.25)),
                            "const_scaled_threshold": float(getattr(self.cfg_obj, "const_scaled_threshold", 1e-4)),
                            "const_l2": float(getattr(self.cfg_obj, "const_l2", 5e-3)),
                            "const_ridge": float(getattr(self.cfg_obj, "const_ridge", 1e-8)),
                            "const_context_budget": int(max(1, getattr(self.cfg_obj, "search_context_budget", 1) or 1)),
                        },
                    )
                    self._sync_best_from_population(tf)
            except Exception as exc:
                self._warn(f"population init failed: {exc}")
        print_every = max(1, n_epochs // 50)
        early_stop_enabled = bool(getattr(self.cfg_obj, "early_stop_enabled", True))
        early_stop_patience = int(max(0, getattr(self.cfg_obj, "early_stop_patience", 8)))
        stopped_epoch: Optional[int] = None
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            if joint_mode:
                metrics = self.train_epoch_joint(data_tensor, epoch, n_epochs, targets)
            else:
                metrics = self.train_epoch(data_tensor, epoch, n_epochs, targets[0])
            elapsed = time.time() - t0
            metrics["epoch_time_sec"] = elapsed
            for key in self.history.keys():
                if key in metrics:
                    self.history[key].append(metrics[key])
            self.history["epoch"].append(epoch)
            should_echo = bool(verbose and (epoch == 1 or epoch == n_epochs or epoch % print_every == 0))
            metrics["_should_print_to_console"] = should_echo
            if self.logger is not None and hasattr(self.logger, "log_epoch"):
                try:
                    self.logger.log_epoch(epoch, metrics)
                except Exception:
                    pass
            if should_echo:
                self._emit_epoch_summary(epoch, n_epochs, metrics, targets, joint_mode)

            if early_stop_enabled and early_stop_patience > 0:
                stagnated = int(metrics.get("stagnated_epochs", 0) or 0)
                if stagnated >= early_stop_patience:
                    stopped_epoch = int(epoch)
                    msg = (
                        f"EARLY STOPPING: best_fitness has not improved for {stagnated} consecutive epochs "
                        f"(patience={early_stop_patience}); stopped at epoch {epoch}/{n_epochs}. "
                        "Continuing with final rerank and post-training summaries."
                    )
                    self.early_stopped = True
                    self.stopped_epoch = stopped_epoch
                    if self.logger is not None and hasattr(self.logger, "info"):
                        try:
                            self.logger.info(msg)
                        except Exception:
                            print(msg)
                    else:
                        print(msg)
                    break
        # Final-only identifiability rerank.  Training has used base fitness and
        # coverage-preserving survival; this post-training pass uniformly scores
        # the union of current population and historical archive.
        for tf in targets:
            self._activate_target(tf)
            try:
                self.apply_final_archive_rerank(tf)
            except Exception as exc:
                self._warn(f"final archive rerank failed for {tf}: {exc}")

        if bool(getattr(self.cfg_obj, "posthoc_pareto_screening", True)):
            for tf in targets:
                self._activate_target(tf)
                self.apply_pareto_screening(top_k=20, target_field=tf)
        if joint_mode:
            return self.history, {tf: dict(self.best_equations.get(tf, self._empty_best_equation())) for tf in targets}
        self._activate_target(targets[0])
        return self.history, self.best_equation

    # --------------------------------------------------------
    # Saving
    # --------------------------------------------------------
    def save_results(
        self,
        result_dir: str,
        equation_name: str = "",
        target_field: Optional[str] = None,
    ):
        _ensure_dir(result_dir)

        if target_field is not None:
            self._activate_target(str(target_field))

        tf = str(self.active_target_field or target_field or "du_t")
        best = self.best_equations.get(tf, self.best_equation)
        pop = self._get_population(tf)

        hist_path = os.path.join(result_dir, f"{equation_name}_training_history.txt")
        _ensure_dir(os.path.dirname(hist_path) or result_dir)
        keys = [
            "epoch",
            "kl_loss",
            "reinforce_loss",
            "struct_loss",
            "total_loss",
            "best_fitness",
            "best_mse",
            "best_equation",
            "gamma",
            "lr",
            "n_valid_eqs",
            "diversity_score",
            "population_size",
            "n_seq_valid",
            "n_sampled",
            "batch_min_fitness",
            "batch_mean_fitness",
            "batch_std_fitness",
            "batch_min_mse",
            "batch_mean_mse",
            "batch_std_mse",
            "subtree_cache_queries",
            "subtree_cache_hits",
            "subtree_cache_misses",
            "subtree_cache_hit_rate",
            "subtree_cache_saved_evals",
            "subtree_cache_queries_epoch",
            "subtree_cache_hits_epoch",
            "subtree_cache_misses_epoch",
            "subtree_cache_hit_rate_epoch",
            "subtree_cache_saved_evals_epoch",
            "template_cache_queries",
            "template_cache_hits",
            "template_cache_misses",
            "template_cache_hit_rate",
            "template_cache_saved_evals",
            "template_cache_queries_epoch",
            "template_cache_hits_epoch",
            "template_cache_misses_epoch",
            "template_cache_hit_rate_epoch",
            "template_cache_saved_evals_epoch",
            "shared_subtrees_warmed",
            "dr_mask_pruned",
            "coarse_candidates",
            "refine_candidates",
            "effective_full_refine_topk",
            "stagnated_epochs",
            "epoch_time_sec",
        ]
        with _open_text_for_write(hist_path) as f:
            f.write(",".join(keys) + "\n")
            n = len(self.history.get("epoch", []))
            for i in range(n):
                row = [str(self.history.get(k, [""] * n)[i] if i < len(self.history.get(k, [])) else "") for k in keys]
                f.write(",".join(row) + "\n")
        eq_path = os.path.join(result_dir, f"{equation_name}_best_equation.txt")
        _ensure_dir(os.path.dirname(eq_path) or result_dir)

        with _open_text_for_write(eq_path) as f:
            f.write("Best Equation:\n")
            f.write(f"  Readable  : {best.get('readable', '')}\n")
            f.write(f"  Tokens    : {best.get('str', '')}\n")
            f.write(f"  Constants : {best.get('consts', np.array([]))}\n")
            f.write(f"  Fitness   : {float(best.get('fitness', 1e18)):.8e}\n")
            f.write(f"  MSE       : {float(best.get('mse', 1e18)):.8e}\n")
            f.write(f"  Complexity: {int(best.get('complexity', 0))}\n")
        pop_path = os.path.join(result_dir, f"{equation_name}_top_population.txt")
        _ensure_dir(os.path.dirname(pop_path) or result_dir)

        with _open_text_for_write(pop_path) as f:
            f.write("Rank | Fitness | MSE | Complexity | Constants | Equation\n")
            f.write("-" * 140 + "\n")
            try:
                from utils.equation_formatter import compile_equation
            except Exception:
                compile_equation = None  # type: ignore[assignment]
            topk = pop.get_best(10)
            if str(equation_name).endswith(("_u", "_v", "_w")):
                lhs = f"{str(equation_name).rsplit('_', 1)[-1]}_t"
            else:
                lhs = "u_t"
            for rank, ind in enumerate(topk, 1):
                seq = ind.get("seq", [])
                consts = ind.get("consts", np.array([]))
                readable = compile_equation(seq, consts, lhs=lhs) if compile_equation is not None else sequence_to_str(seq)

                # =====================================================
                # 馃専 娉ㄥ叆鎶涘厜鍣細鍦ㄥ啓鍏ユ帓琛屾鏃ュ織鍓嶏紝閫氳繃 SymPy 娓呮礂鎵€鏈夌瓑浠烽」
                # =====================================================
                try:
                    from utils.equation_simplifier import polish_discovered_equation
                    # 寮鸿娓呮礂 0.0 椤癸紝骞跺睍寮€宓屽瀵兼暟
                    readable = polish_discovered_equation(readable, prune_tol=float(getattr(self.cfg_obj, "small_coeff_prune_tol", 1e-4)))
                except Exception:
                    pass # 濡傛灉鎶涘厜澶辫触锛屼繚鐣欏師鏍风殑 readable

                # 浼樺寲甯告暟鐨勬樉绀烘牸寮忥紝闅愯棌 1e-18 绾у埆鐨?0.000 纰庣墖
                const_str = np.array2string(np.asarray(consts), precision=6, separator=' ', suppress_small=True, max_line_width=200)

                f.write(
                    f"{rank:4d} | {float(ind.get('fitness', 1e18)):.8e} | {float(ind.get('mse', 1e18)):.8e} | "
                    f"{int(ind.get('complexity', len(seq))):3d} | {const_str} | {readable}\n"
                )
        return hist_path, eq_path, pop_path

__all__ = [
    "WiSEDFramework",
    "WiSEDTrainer",
    "SampleRecord",
    "TrainerConfig",
]
