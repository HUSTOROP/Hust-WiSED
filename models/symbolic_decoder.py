import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict

from models.symbol_vocabulary import SYM2IDX, IDX2SYM
from models.operator_policy import OperatorPolicy, build_operator_policy

# -----------------------------
# Local arity + token groups
# -----------------------------
_SPECIALS = {"<START>", "<END>", "<PAD>", "<UNK>"}

# Physics-oriented constraint: exclude unsafe operators from the decoder's valid generation space.
_BINARY = {"+", "*", "/", "^"}  # ^ legacy-only; operator policy forbids it
_UNARY_SAFE = {"neg", "lap", "adv", "sq", "cube"}
_UNARY_TRANSC = {"sin", "cos", "exp", "log"}
_UNARY = _UNARY_SAFE | _UNARY_TRANSC

_GENERIC_DERIV = {"D"}
_LEGACY_DERIV_1 = {"dx", "dxx", "dxxx", "dt", "dxt"}

# RHS forbidden time-deriv:
# - legacy: dt, dxt
# - generic: D followed by axis 't' (enforced by decoding constraint)
_FORBIDDEN_RHS_OPS = {"dt", "dxt"}  # keep explicit forbid; generic handled by axis constraint

# axes tokens we recognize (if present in vocab)
_AXIS_TOKENS = ["t", "x", "y", "z"]


def sym_arity(sym: str) -> int:
    if sym in _BINARY:
        return 2
    if sym in _UNARY:
        return 1
    if sym in _GENERIC_DERIV:
        return 2  # axis + expr
    if sym in _LEGACY_DERIV_1:
        return 1
    return 0  # terminals, coords, const, etc.


def build_index_groups() -> Tuple[int, List[int], List[int], List[int], List[int]]:
    """
    Build:
      vocab_size
      terminal_idxs (arity 0, excluding specials)
      nonterminal_idxs (arity >0, excluding specials)
      rhs_nonterminal_idxs (exclude dt/dxt)
      axis_idxs (tokens t/x/y/z if present)
    """
    vocab_size = len(SYM2IDX)

    terminal_idxs = []
    nonterminal_idxs = []
    rhs_nonterminal_idxs = []
    axis_idxs = []

    for sym, idx in SYM2IDX.items():
        if sym in _SPECIALS:
            continue
        a = sym_arity(sym)
        if a == 0:
            terminal_idxs.append(idx)
        else:
            nonterminal_idxs.append(idx)
            if sym not in _FORBIDDEN_RHS_OPS:
                rhs_nonterminal_idxs.append(idx)

    for a in _AXIS_TOKENS:
        if a in SYM2IDX:
            axis_idxs.append(SYM2IDX[a])

    return vocab_size, terminal_idxs, nonterminal_idxs, rhs_nonterminal_idxs, axis_idxs



def refresh_index_groups() -> None:
    """Recompute cached index groups after vocab changes.

    The global SYM2IDX/IDX2SYM mappings can be updated at runtime (e.g., when switching from 1D to 2D
    or adding multiple fields u/v). This refresh keeps the decoder's module-level cached index groups
    consistent with the current vocab.
    """
    global _VOCAB_SIZE, _TERMINAL_IDXS, _NONTERMINAL_IDXS, _RHS_NONTERMINAL_IDXS, _AXIS_IDXS, _AXIS_IDXS_SET
    _VOCAB_SIZE, _TERMINAL_IDXS, _NONTERMINAL_IDXS, _RHS_NONTERMINAL_IDXS, _AXIS_IDXS = build_index_groups()
    _AXIS_IDXS_SET = set(_AXIS_IDXS)

refresh_index_groups()

_AXIS_IDXS_SET = set(_AXIS_IDXS)


def _is_axis_idx(idx: int) -> bool:
    return int(idx) in _AXIS_IDXS_SET


def _axis_is_time(idx: int) -> bool:
    return IDX2SYM.get(int(idx), "") == "t"


def _filter_terminals_remove_axis(valid_terminals: List[int]) -> List[int]:
    # Axis tokens are NOT variables; they should never be used as EXPR leaf.
    return [int(i) for i in valid_terminals if int(i) not in _AXIS_IDXS_SET]


# -----------------------------
# Liquid Gate
# -----------------------------
class LiquidGate(nn.Module):
    """
    Liquid sparse gate:
    gate(z) in [0,1]^{vocab}, used as a soft bias on logits.
    """
    def __init__(self, d_z: int, vocab_size: int):
        super().__init__()
        # Ensure cached vocab index groups are consistent with current SYM2IDX/IDX2SYM
        refresh_index_groups()
        self.gate_net = nn.Sequential(
            nn.Linear(d_z, d_z),
            nn.GELU(),
            nn.Linear(d_z, vocab_size),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.gate_net(z)


# -----------------------------
# Decoder
# -----------------------------
class SymbolicDecoder(nn.Module):
    """
    Autoregressive GRU decoder for equation token sequences.

    Key constraints:
      - Prefix syntactic validity via open_slots (stack slots)
      - RHS forbidden dt/dxt
      - After D, next token must be an axis
      - On RHS discovery, axis after D cannot be 't'

    Curriculum (optional):
      - Use epoch/n_epochs to compute progress in [0,1]
      - Early: discourage transcendental unary ops; encourage derivatives; mildly discourage ^
      - Late: relax biases to 0
    """
    def __init__(self, d_z: int = 64, d_hidden: int = 256,
                 max_len: int = 20, n_gru_layers: int = 2,
                 allowed_axes: Optional[List[str]] = None):
        super().__init__()
        self.d_z = d_z
        self.d_hidden = d_hidden
        self.max_len = max_len
        self.vocab_size = _VOCAB_SIZE
        self.n_gru_layers = n_gru_layers

        self.token_emb = nn.Embedding(self.vocab_size, d_z)
        self.z_to_h0 = nn.Sequential(
            nn.Linear(d_z, d_hidden * n_gru_layers),
            nn.Tanh()
        )
        self.gru = nn.GRU(
            input_size=d_z + d_z,
            hidden_size=d_hidden,
            num_layers=n_gru_layers,
            batch_first=True
        )
        # Shared decoder trunk; target-specific heads reduce cross-field token mixing.
        self.output_trunk = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
        )
        self.liquid_gate = LiquidGate(d_z, self.vocab_size)

        # Precompute operator index sets for curriculum bias
        self.idx_deriv = [SYM2IDX[s] for s in ["D", "dx", "dxx", "dxxx"] if s in SYM2IDX]
        self.idx_unary_transc = [SYM2IDX[s] for s in ["sin", "cos", "exp", "log"] if s in SYM2IDX]
        self.idx_unary_safe = [SYM2IDX[s] for s in ["neg", "lap", "adv", "sq", "cube"] if s in SYM2IDX]
        self.idx_spatial_macros = [SYM2IDX[s] for s in ["lap", "adv"] if s in SYM2IDX]
        self.spatial_ndim = max(1, len(allowed_axes or [])) if allowed_axes is not None else 1
        self.idx_pow = []  # generic ^ disabled; use sq/cube

        # Allowed spatial axes for the typed AXIS slot after D.
        # Example: 1D -> ['x'], 2D -> ['x','y'].
        # If None, defaults to all axis tokens present in vocab.
        self.allowed_fields: Optional[List[str]] = None
        self.allowed_axes: Optional[List[str]] = allowed_axes
        self.operator_policy: OperatorPolicy = build_operator_policy(
            epoch=0,
            spatial_ndim=max(1, len(allowed_axes or [])) if allowed_axes is not None else 1,
        )
        self.idx_div = [SYM2IDX["/"]] if "/" in SYM2IDX else []

        # target-conditioned decoding: du_t / dv_t / dw_t
        self.target_name_to_idx = {"du_t": 0, "dv_t": 1, "dw_t": 2}
        self.n_targets = 3
        self.target_emb = nn.Embedding(self.n_targets, self.d_z)
        self.target_proj = nn.Sequential(
            nn.Linear(self.d_z, self.d_z),
            nn.Tanh(),
        )
        # One token-logit head per target; GRU/latent conditioning remains shared.
        self.output_heads = nn.ModuleList([
            nn.Linear(self.d_hidden, self.vocab_size) for _ in range(self.n_targets)
        ])

    def set_allowed_fields(self, fields: Optional[List[str]]) -> None:
        self.allowed_fields = fields

    @staticmethod
    def _normalize_target_name(target_field: Optional[str]) -> str:
        tf = str(target_field or "du_t").strip()
        if tf in ("u", "v", "w"):
            return f"d{tf}_t"
        if tf in ("du_t", "dv_t", "dw_t"):
            return tf
        raise ValueError(
            f"Invalid target field '{tf}'. Only u/v/w or du_t/dv_t/dw_t are supported."
        )

    def make_target_ids(self, target_fields: List[str], device) -> torch.Tensor:
        ids = []
        for tf in target_fields:
            name = self._normalize_target_name(tf)
            ids.append(int(self.target_name_to_idx.get(name, 0)))
        return torch.as_tensor(ids, dtype=torch.long, device=device)

    def _sanitize_target_ids(self, batch_size: int, device, target_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if target_ids is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        target_ids = target_ids.to(device=device, dtype=torch.long).view(-1)
        if int(target_ids.numel()) == 1 and batch_size > 1:
            target_ids = target_ids.expand(batch_size)
        if int(target_ids.numel()) != batch_size:
            raise ValueError(f"target_ids length {int(target_ids.numel())} != batch_size {batch_size}")
        return target_ids

    def _condition_latent(self, z: torch.Tensor, target_ids: Optional[torch.Tensor]) -> torch.Tensor:
        target_ids = self._sanitize_target_ids(z.shape[0], z.device, target_ids)
        tgt = self.target_proj(self.target_emb(target_ids))
        return z + tgt

    def _project_logits(self, hidden: torch.Tensor, target_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """Shared decoder trunk with a separate output head for each target."""
        target_ids = self._sanitize_target_ids(hidden.shape[0], hidden.device, target_ids)
        trunk = self.output_trunk(hidden)
        logits_all = torch.stack([head(trunk) for head in self.output_heads], dim=1)
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        return logits_all[batch_idx, target_ids]

    def set_allowed_axes(self, axes: Optional[List[str]]) -> None:
        self.allowed_axes = axes

    def set_spatial_ndim(self, spatial_ndim: int) -> None:
        self.spatial_ndim = int(max(1, spatial_ndim))
        if self.spatial_ndim <= 1 and self.allowed_axes:
            self.allowed_axes = [a for a in self.allowed_axes if a == "x"] or ["x"]

    def set_operator_policy(self, policy: OperatorPolicy) -> None:
        self.operator_policy = policy
        self.set_spatial_ndim(int(max(1, policy.spatial_ndim)))

    def _allowed_nonterminal_idxs(self) -> List[int]:
        allowed = set(self.operator_policy.allowed_nonterminal_symbols)
        out: List[int] = []
        for idx in _RHS_NONTERMINAL_IDXS:
            sym = IDX2SYM.get(int(idx), "")
            if sym in allowed:
                out.append(int(idx))
        return out

    def _allowed_axis_idxs(self) -> List[int]:
        # Default: all axis tokens in vocab.
        if not self.allowed_axes:
            return [int(i) for i in _AXIS_IDXS]
        idxs: List[int] = []
        for a in self.allowed_axes:
            if a in SYM2IDX:
                idxs.append(int(SYM2IDX[a]))
        return idxs if idxs else [int(i) for i in _AXIS_IDXS]

    def _allowed_field_terminal_idxs(self) -> List[int]:
        fields = list(self.allowed_fields or [])
        if not fields:
            fields = [f for f in ("u", "v", "w") if f in SYM2IDX]
        return [int(SYM2IDX[f]) for f in fields if f in SYM2IDX]

    def _init_hidden(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        h0 = self.z_to_h0(z)
        return h0.reshape(B, self.n_gru_layers, self.d_hidden).permute(1, 0, 2).contiguous()

    def forward_step(
        self,
        token_idx: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
    ):
        emb = self.token_emb(token_idx)                  # [B, d_z]
        inp = torch.cat([emb, z], dim=-1).unsqueeze(1)   # [B, 1, 2*d_z]
        out, h_new = self.gru(inp, h)
        logits = self._project_logits(out.squeeze(1), target_ids)  # [B, vocab_size]
        return logits, h_new

    # --------------------------------------------------------
    # Curriculum bias (soft, not hard mask)
    # --------------------------------------------------------
    @staticmethod
    def _progress(epoch: Optional[int], n_epochs: Optional[int]) -> float:
        if epoch is None or n_epochs is None or n_epochs <= 1:
            return 1.0
        p = float(epoch) / float(max(n_epochs - 1, 1))
        return float(max(0.0, min(1.0, p)))

    def _curriculum_bias_vector(self, device, progress: float) -> torch.Tensor:
        bias = torch.zeros(self.vocab_size, device=device)
        for sym, val in dict(getattr(self.operator_policy, "logit_bias", {})).items():
            idx = SYM2IDX.get(sym)
            if idx is None:
                continue
            ii = int(idx)
            if 0 <= ii < self.vocab_size:
                bias[ii] = float(val)
        return bias

    # --------------------------------------------------------
    # Structural mask
    # --------------------------------------------------------
    def _make_struct_mask(
        self,
        open_slots: int,
        depth: int,
        max_depth: int,
        valid_terminals: List[int],
        device,
        pending_axis: bool,
        forbid_time_axis_after_D: bool = True,
        pending_adv_child: bool = False,
    ) -> torch.Tensor:
        """Binary mask: 1=allowed, 0=forbidden."""
        mask = torch.zeros(self.vocab_size, device=device)

        def _safe_mark(idx: int) -> None:
            ii = int(idx)
            if 0 <= ii < self.vocab_size:
                mask[ii] = 1.0

        if open_slots <= 0:
            return mask
        if pending_axis:
            for idx in self._allowed_axis_idxs():
                if forbid_time_axis_after_D and _axis_is_time(idx):
                    continue
                _safe_mark(idx)
            return mask
        if pending_adv_child:
            # adv is a physical template adv(field), not a generic unary expression.
            for idx in self._allowed_field_terminal_idxs():
                _safe_mark(idx)
            return mask
        if depth >= max_depth:
            for idx in valid_terminals:
                _safe_mark(idx)
            return mask
        # 1D PDEs still need physical macro templates:
        #   adv(u) -> u * u_x,  lap(u) -> u_xx.
        # Do not mask lap/adv just because spatial_ndim == 1. Semantic
        # restrictions such as adv(field) are enforced by pending_adv_child
        # and the global sequence validator.
        for idx in self._allowed_nonterminal_idxs():
            _safe_mark(idx)
        for idx in valid_terminals:
            _safe_mark(idx)
        return mask

    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------
    def decode_sample(
        self,
        z: torch.Tensor,
        gamma: float = 1.0,
        max_depth: int = 8,
        temperature: float = 1.0,
        valid_terminals: Optional[List[int]] = None,
        allowed_axes: Optional[List[str]] = None,
        # curriculum controls (optional)
        epoch: Optional[int] = None,
        n_epochs: Optional[int] = None,
        target_ids: Optional[torch.Tensor] = None,
    ):
        """
        Sample sequences from z under structural constraints + curriculum biases.
        """
        if valid_terminals is None:
            valid_terminals = _TERMINAL_IDXS
        valid_terminals = _filter_terminals_remove_axis(list(valid_terminals))

        if not valid_terminals:
            fallback_fields = [SYM2IDX[f] for f in ("u", "v", "w") if f in SYM2IDX]
            valid_terminals = fallback_fields or [SYM2IDX.get("const", 0)]

        # Update allowed axes from caller/context.
        if allowed_axes is None:
            allowed_axes = self.allowed_axes
        self.set_allowed_axes(allowed_axes)

        progress = self._progress(epoch, n_epochs)

        B = z.shape[0]
        device = z.device
        z_cond = self._condition_latent(z, target_ids)
        h = self._init_hidden(z_cond)
        gate = self.liquid_gate(z_cond) ** gamma

        # curriculum bias vector (shared across batch)
        cur_bias = self._curriculum_bias_vector(device, progress)

        start_tok = torch.full((B,), SYM2IDX.get("<START>", 0), dtype=torch.long, device=device)
        sequences = [[] for _ in range(B)]
        log_probs = [0.0 for _ in range(B)]

        open_slots = [1] * B
        depths = [0] * B
        done = [False] * B
        pending_axis = [False] * B  # True if last token was D and we need axis next
        pending_macro_child = [False] * B  # True if last token was adv and we need a field next

        current_tok = start_tok

        for step in range(self.max_len):
            if all(done):
                break

            logits, h = self.forward_step(current_tok, h, z_cond, target_ids=target_ids)
            next_tokens = []

            for b in range(B):
                if done[b] or open_slots[b] <= 0:
                    done[b] = True
                    next_tokens.append(SYM2IDX.get("<PAD>", 0))
                    continue

                struct_mask = self._make_struct_mask(
                    open_slots[b],
                    depths[b],
                    max_depth,
                    valid_terminals,
                    device,
                    pending_axis=pending_axis[b],
                    forbid_time_axis_after_D=True,   # RHS rule
                    pending_adv_child=pending_macro_child[b],
                )

                raw_logits = logits[b] / max(float(temperature), 1e-6)
                raw_logits = raw_logits + (1.0 - struct_mask) * (-1e9)

                # soft bias 1: liquid gate
                gate_bias = torch.log(gate[b] + 1e-10) * struct_mask
                raw_logits = raw_logits + 0.1 * gate_bias

                # soft bias 2: curriculum (masked)
                raw_logits = raw_logits + (cur_bias * struct_mask)

                probs = F.softmax(raw_logits, dim=-1)
                if torch.any(torch.isnan(probs)) or probs.sum() < 1e-10:
                    probs = struct_mask / (struct_mask.sum() + 1e-10)

                tok = torch.multinomial(probs, 1).item()
                sequences[b].append(tok)
                log_probs[b] += float(torch.log(probs[tok] + 1e-10).item())

                sym = IDX2SYM.get(tok, "")
                arity = sym_arity(sym)

                # Update pending_axis state
                if sym in _GENERIC_DERIV:
                    pending_axis[b] = True
                elif pending_axis[b]:
                    pending_axis[b] = False

                if sym in ["adv", "lap"]:
                    pending_macro_child[b] = True
                elif pending_macro_child[b]:
                    pending_macro_child[b] = False

                open_slots[b] = open_slots[b] - 1 + arity
                depths[b] = depths[b] + (1 if arity > 0 else -1)
                depths[b] = max(0, depths[b])

                if open_slots[b] <= 0:
                    done[b] = True

                next_tokens.append(tok)

            current_tok = torch.tensor(next_tokens, dtype=torch.long, device=device)

        return sequences, log_probs

    def decode_greedy(
        self,
        z: torch.Tensor,
        gamma: float = 1.0,
        max_depth: int = 8,
        valid_terminals: Optional[List[int]] = None,
        allowed_axes: Optional[List[str]] = None,
        # curriculum controls (optional)
        epoch: Optional[int] = None,
        n_epochs: Optional[int] = None,
        target_ids: Optional[torch.Tensor] = None,
    ):
        """
        Greedy decoding with same constraints + curriculum biases.
        """
        if valid_terminals is None:
            valid_terminals = _TERMINAL_IDXS
        valid_terminals = _filter_terminals_remove_axis(list(valid_terminals))
        if not valid_terminals:
            valid_terminals = [SYM2IDX.get("u", 0)]

        if allowed_axes is None:
            allowed_axes = self.allowed_axes
        self.set_allowed_axes(allowed_axes)

        progress = self._progress(epoch, n_epochs)

        B = z.shape[0]
        device = z.device
        z_cond = self._condition_latent(z, target_ids)
        h = self._init_hidden(z_cond)
        gate = self.liquid_gate(z_cond) ** gamma
        cur_bias = self._curriculum_bias_vector(device, progress)

        start_tok = torch.full((B,), SYM2IDX.get("<START>", 0), dtype=torch.long, device=device)
        sequences = [[] for _ in range(B)]
        open_slots = [1] * B
        depths = [0] * B
        done = [False] * B
        pending_axis = [False] * B
        pending_macro_child = [False] * B
        current_tok = start_tok

        for step in range(self.max_len):
            if all(done):
                break
            logits, h = self.forward_step(current_tok, h, z_cond, target_ids=target_ids)
            next_tokens = []

            for b in range(B):
                if done[b] or open_slots[b] <= 0:
                    done[b] = True
                    next_tokens.append(SYM2IDX.get("<PAD>", 0))
                    continue

                struct_mask = self._make_struct_mask(
                    open_slots[b],
                    depths[b],
                    max_depth,
                    valid_terminals,
                    device,
                    pending_axis=pending_axis[b],
                    forbid_time_axis_after_D=True,
                    pending_adv_child=pending_macro_child[b],
                )

                raw_logits = logits[b]
                raw_logits = raw_logits + (1.0 - struct_mask) * (-1e9)
                gate_bias = torch.log(gate[b] + 1e-10) * struct_mask
                raw_logits = raw_logits + 0.1 * gate_bias
                raw_logits = raw_logits + (cur_bias * struct_mask)

                tok = torch.argmax(raw_logits).item()
                sequences[b].append(tok)

                sym = IDX2SYM.get(tok, "")
                arity = sym_arity(sym)

                if sym in _GENERIC_DERIV:
                    pending_axis[b] = True
                elif pending_axis[b]:
                    pending_axis[b] = False

                if sym in ["adv", "lap"]:
                    pending_macro_child[b] = True
                elif pending_macro_child[b]:
                    pending_macro_child[b] = False

                open_slots[b] = open_slots[b] - 1 + arity
                depths[b] = max(0, depths[b] + (1 if arity > 0 else -1))
                if open_slots[b] <= 0:
                    done[b] = True
                next_tokens.append(tok)

            current_tok = torch.tensor(next_tokens, dtype=torch.long, device=device)

        return sequences

    # --------------------------------------------------------
    # Logprob (must match sampling constraints)
    # --------------------------------------------------------
    def compute_sequence_logprob(
        self,
        z: torch.Tensor,
        token_seqs: list,
        gamma: float = 1.0,
        valid_terminals: Optional[List[int]] = None,
        max_depth: int = 8,
        temperature: float = 1.0,
        allowed_axes: Optional[List[str]] = None,
        # curriculum controls (optional)
        epoch: Optional[int] = None,
        n_epochs: Optional[int] = None,
        target_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute constrained log-probabilities of given sequences under current model.
        IMPORTANT: Applies the same structural constraints and curriculum biases as decode_sample,
        otherwise REINFORCE becomes inconsistent.
        """
        if valid_terminals is None:
            valid_terminals = _TERMINAL_IDXS
        valid_terminals = _filter_terminals_remove_axis(list(valid_terminals))
        if not valid_terminals:
            valid_terminals = [SYM2IDX.get("u", 0)]

        if allowed_axes is None:
            allowed_axes = self.allowed_axes
        self.set_allowed_axes(allowed_axes)

        progress = self._progress(epoch, n_epochs)

        B = z.shape[0]
        device = z.device
        z_cond = self._condition_latent(z, target_ids)
        h = self._init_hidden(z_cond)
        gate = self.liquid_gate(z_cond) ** gamma
        cur_bias = self._curriculum_bias_vector(device, progress)

        max_seq_len = max((len(s) for s in token_seqs), default=1)
        log_probs = torch.zeros(B, device=device)

        current_tok = torch.full((B,), SYM2IDX.get("<START>", 0), dtype=torch.long, device=device)

        open_slots = [1] * B
        depths = [0] * B
        pending_axis = [False] * B
        pending_macro_child = [False] * B

        for step in range(max_seq_len):
            logits, h = self.forward_step(current_tok, h, z_cond, target_ids=target_ids)
            next_tokens = []

            for b in range(B):
                if step >= len(token_seqs[b]):
                    next_tokens.append(SYM2IDX.get("<PAD>", 0))
                    continue

                target_tok = int(token_seqs[b][step])

                struct_mask = self._make_struct_mask(
                    open_slots[b],
                    depths[b],
                    max_depth,
                    valid_terminals,
                    device,
                    pending_axis=pending_axis[b],
                    forbid_time_axis_after_D=True,
                    pending_adv_child=pending_macro_child[b],
                )

                raw_logits = logits[b] / max(float(temperature), 1e-6)
                raw_logits = raw_logits + (1.0 - struct_mask) * (-1e9)
                gate_bias = torch.log(gate[b] + 1e-10) * struct_mask
                raw_logits = raw_logits + 0.1 * gate_bias
                raw_logits = raw_logits + (cur_bias * struct_mask)

                probs = F.softmax(raw_logits, dim=-1)
                log_probs[b] = log_probs[b] + torch.log(probs[target_tok] + 1e-10)

                sym = IDX2SYM.get(target_tok, "")
                arity = sym_arity(sym)

                if sym in _GENERIC_DERIV:
                    pending_axis[b] = True
                elif pending_axis[b]:
                    pending_axis[b] = False

                if sym in ["adv", "lap"]:
                    pending_macro_child[b] = True
                elif pending_macro_child[b]:
                    pending_macro_child[b] = False

                open_slots[b] = open_slots[b] - 1 + arity
                depths[b] = depths[b] + (1 if arity > 0 else -1)
                depths[b] = max(0, depths[b])

                next_tokens.append(target_tok)

            current_tok = torch.tensor(next_tokens, dtype=torch.long, device=device)

        return log_probs
