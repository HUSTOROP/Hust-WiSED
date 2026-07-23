from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

COMPLEXITY_MAP = {
    "const": 1.0, "u": 1.0, "v": 1.0, "w": 1.0,
    "+": 1.0, "neg": 1.0,
    "*": 2.0, "D": 2.0,
    # Macro operators are physically meaningful numerical shortcuts.
    # Pricing them close to their minimal expanded forms keeps adv/lap searchable
    # without injecting a task-specific Burgers template.
    "lap": 2.0, "adv": 3.0,
    "/": 4.0, "^": 5.0,
    "sq": 2.0, "cube": 3.0,
    "sin": 10.0, "cos": 10.0, "exp": 10.0, "log": 10.0
}


class TokenType(Enum):
    FIELD = "field"
    AXIS = "axis"
    DERIV = "deriv"
    SCALAR = "scalar"
    UNARY = "unary"
    BINARY = "binary"


@dataclass
class VocabConfig:
    fields: Tuple[str, ...] = ("u",)
    axes: Tuple[str, ...] = ("t", "x")
    include_unk: bool = False
    forbid_time_derivative_in_rhs: bool = True
    allow_coordinate_terminals: bool = False
    forbid_time_coordinate_in_rhs: bool = True

    start_token: str = "<START>"
    end_token: str = "<END>"
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"


class SymbolVocab:
    """Typed-slot prefix grammar.

    EXPR terminals:
        - fields: u, v, ...
        - const
        - optional spatial coordinates: x, y, z when enabled by task config

    AXIS terminals:
        - t, x, y, z
        - only valid immediately after D in the AXIS slot

    Operators:
        - derivative: D axis expr
        - unary: sin cos exp log neg lap adv
        - binary: + * / ^
    """

    def __init__(self, cfg: Optional[VocabConfig] = None):
        self.cfg = cfg or VocabConfig()

        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self.token2type: Dict[str, TokenType] = {}
        self.arity = {
            "const": 0, "u": 0, "v": 0, "w": 0,
            "+": 2, "*": 2, "/": 2, "^": 2,
            "neg": 1, "sq": 1, "cube": 1, "sin": 1, "cos": 1, "exp": 1, "log": 1, "lap": 1, "adv": 1,
            "D": 2
        }

        self._build_vocab()

    def _add(self, token: str, ttype: Optional[TokenType], arity: int) -> None:
        if token in self.token2idx:
            return
        idx = len(self.token2idx)
        self.token2idx[token] = idx
        self.idx2token[idx] = token
        if ttype is not None:
            self.token2type[token] = ttype
        self.arity[token] = int(arity)

    def _build_vocab(self) -> None:
        self._add(self.cfg.start_token, None, 0)
        self._add(self.cfg.end_token, None, 0)
        self._add(self.cfg.pad_token, None, 0)
        if self.cfg.include_unk:
            self._add(self.cfg.unk_token, None, 0)

        for field in self.cfg.fields:
            self._add(field, TokenType.FIELD, 0)
        for axis in self.cfg.axes:
            self._add(axis, TokenType.AXIS, 0)

        self._add("const", TokenType.SCALAR, 0)
        self._add("D", TokenType.DERIV, 2)

        for unary in ("sin", "cos", "exp", "log", "neg", "sq", "cube", "lap", "adv"):
            self._add(unary, TokenType.UNARY, 1)
        for binary in ("+", "*", "/", "^"):
            self._add(binary, TokenType.BINARY, 2)

        self.vocab_size = len(self.token2idx)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token2idx[t] for t in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx2token[int(i)] for i in indices]

    def is_valid_prefix_expression(self, tokens: List[str]) -> Tuple[bool, Optional[str]]:
        """Validate typed-slot prefix syntax."""
        cleaned: List[str] = []
        for tok in tokens:
            if tok == self.cfg.end_token:
                break
            if tok in (self.cfg.start_token, self.cfg.pad_token):
                continue
            cleaned.append(tok)

        if not cleaned:
            return False, "EMPTY"

        EXPR = "EXPR"
        AXIS = "AXIS"
        slots: List[str] = [EXPR]

        for tok in cleaned:
            if not slots:
                return False, "SLOT_UNDERFLOW"

            expected = slots.pop()
            if tok not in self.arity:
                return False, f"UNKNOWN_TOKEN:{tok}"

            ttype = self.token2type.get(tok)
            if expected == AXIS:
                if ttype != TokenType.AXIS:
                    return False, f"EXPECTED_AXIS_GOT:{tok}"
                continue

            if ttype == TokenType.AXIS and not self.cfg.allow_coordinate_terminals:
                return False, f"AXIS_USED_AS_EXPR:{tok}"

            if tok == "D":
                slots.append(EXPR)
                slots.append(AXIS)
                continue

            ar = int(self.arity[tok])
            for _ in range(ar):
                slots.append(EXPR)

        if slots:
            return False, f"SLOT_MISMATCH:need={len(slots)}"
        return True, None

    def is_valid_rhs(self, tokens: List[str]) -> Tuple[bool, Optional[str]]:
        """Validate RHS-only semantic constraints.

        In addition to the typed prefix grammar, this enforces two physics-oriented
        constraints used by the evaluator:
        1. no time derivatives on the RHS;
        2. lap is a diffusion template and may only appear as lap(field), where
           field is one of u/v/w present in the current vocabulary;
        3. adv is an advection template and may only appear as adv(field), where
           field is one of u/v/w present in the current vocabulary. This keeps
           adv semantically equivalent to u路鈭噓 / u路鈭噕 / u路鈭噖 templates instead
           of opening a generic v路鈭噏 operator over arbitrary expressions.
        """
        cleaned: List[str] = []
        for tok in tokens:
            if tok == self.cfg.end_token:
                break
            if tok in (self.cfg.start_token, self.cfg.pad_token):
                continue
            cleaned.append(tok)

        if self.cfg.forbid_time_derivative_in_rhs:
            for i, tok in enumerate(cleaned):
                if tok == "D" and i + 1 < len(cleaned) and cleaned[i + 1] == "t":
                    return False, "TIME_DERIVATIVE_FORBIDDEN_IN_RHS"
        if self.cfg.forbid_time_coordinate_in_rhs and self.cfg.allow_coordinate_terminals:
            for i, tok in enumerate(cleaned):
                if tok == "t" and not (i > 0 and cleaned[i - 1] == "D"):
                    return False, "TIME_COORDINATE_FORBIDDEN_IN_RHS"

        field_tokens = {f for f in self.cfg.fields if f in {"u", "v", "w"}}

        def parse_expr(pos: int) -> Tuple[int, bool, Optional[str]]:
            if pos >= len(cleaned):
                return pos, False, "UNEXPECTED_END"
            tok = cleaned[pos]
            if tok not in self.arity:
                return pos + 1, False, f"UNKNOWN_TOKEN:{tok}"

            if tok == "D":
                if pos + 1 >= len(cleaned):
                    return pos + 1, False, "DERIVATIVE_MISSING_AXIS"
                axis = cleaned[pos + 1]
                if self.token2type.get(axis) != TokenType.AXIS:
                    return pos + 2, False, f"EXPECTED_AXIS_GOT:{axis}"
                return parse_expr(pos + 2)

            if tok in {"lap", "adv"}:
                child_pos = pos + 1
                op_name = str(tok).upper()
                if child_pos >= len(cleaned):
                    return child_pos, False, f"{op_name}_MISSING_CHILD"
                child = cleaned[child_pos]
                if child not in field_tokens:
                    return child_pos + 1, False, f"{op_name}_ONLY_SUPPORTS_NATIVE_FIELD"
                return child_pos + 1, True, None

            ar = int(self.arity.get(tok, 0))
            curr = pos + 1
            for _ in range(ar):
                curr, ok, err = parse_expr(curr)
                if not ok:
                    return curr, ok, err
            return curr, True, None

        if cleaned:
            end_pos, ok, err = parse_expr(0)
            if not ok:
                return False, err
            if end_pos != len(cleaned):
                return False, "EXTRA_TOKENS_AFTER_RHS"

        return True, None

    def get_complexity_score(self, tokens: List[str]) -> float:
        """缁熶竴浣跨敤 COMPLEXITY_MAP 璁＄畻澶嶆潅搴︺€?
        杩欓噷涓嶅啀缁存姢绗簩濂楃嫭绔嬫潈閲嶏紝閬垮厤璁粌闃舵锛坋quation_evaluator.py锛?        涓庣粺璁?灞曠ず闃舵锛坢etrics.py锛夊嚭鐜板鏉傚害鍙ｅ緞涓嶄竴鑷寸殑闂銆?        """
        score = 0.0
        for tok in tokens:
            if tok in (self.cfg.start_token, self.cfg.end_token, self.cfg.pad_token):
                continue
            if self.cfg.include_unk and tok == self.cfg.unk_token:
                continue
            score += float(COMPLEXITY_MAP.get(tok, 1.0))
        return float(score)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"SymbolVocab(size={self.vocab_size}, fields={self.cfg.fields}, axes={self.cfg.axes})"


_DEFAULT_VOCAB = SymbolVocab()
SYM2IDX: Dict[str, int] = dict(_DEFAULT_VOCAB.token2idx)
IDX2SYM: Dict[int, str] = dict(_DEFAULT_VOCAB.idx2token)


def _refresh_vocab_dependent_modules() -> None:
    """Keep all runtime vocab-dependent caches in sync.

    The project mutates the global vocabulary in-place when switching tasks
    (e.g. 1D -> 2D, or single-field -> coupled system). Any module that cached
    vocab-derived ids must be refreshed here.

    Refresh decoder/search/evaluator caches so the vocabulary can be swapped
    without leaving stale runtime ids.
    """
    for module_name in ("models.symbolic_decoder",):
        try:
            module = __import__(module_name, fromlist=["refresh_index_groups"])
            getattr(module, "refresh_index_groups")()
        except Exception:
            pass

    try:
        from models.symbolic_search import sync_operator_groups_with_vocab, clear_optimizer_caches
        sync_operator_groups_with_vocab()
        clear_optimizer_caches(reset_stats=True)
    except Exception:
        pass

    try:
        from models.weak_form_evaluator import clear_evaluator_caches
        clear_evaluator_caches(reset_stats=True)
    except Exception:
        pass

    try:
        from utils.structure_cache import reset_structure_key_manager
        reset_structure_key_manager()
    except Exception:
        pass


def init_vocab_from_context(ctx: Any) -> SymbolVocab:
    """Build a task-specific vocabulary from a DataContext.

    The global SYM2IDX / IDX2SYM dictionaries are updated *in place* so all
    modules that imported them remain synchronized.
    """
    raw_fields = list(getattr(ctx, "fields", ["u"]))
    fields = tuple(f for f in raw_fields if f in ("u", "v", "w"))
    if not fields:
        fields = ("u",)

    raw_axes = list(
        getattr(
            ctx,
            "axes_order",
            list(getattr(ctx, "coords", {}).keys()) or ["t", "x"],
        )
    )
    axes = tuple(ax for ax in raw_axes if ax in ("t", "x", "y", "z"))
    if not axes:
        axes = ("t", "x")
    elif "t" in axes:
        axes = ("t",) + tuple(ax for ax in axes if ax != "t")

    allow_coordinate_terminals = bool(getattr(ctx, "allow_coordinate_terminals", False))
    vocab = SymbolVocab(VocabConfig(
        fields=fields,
        axes=axes,
        allow_coordinate_terminals=allow_coordinate_terminals,
    ))

    global _DEFAULT_VOCAB
    _DEFAULT_VOCAB = vocab
    SYM2IDX.clear()
    SYM2IDX.update(vocab.token2idx)
    IDX2SYM.clear()
    IDX2SYM.update(vocab.idx2token)

    _refresh_vocab_dependent_modules()
    return vocab

def sequence_to_str(seq: List[int]) -> str:
    return " ".join(IDX2SYM.get(int(i), "?") for i in seq)


def count_constants(seq: List[int]) -> int:
    const_idx = SYM2IDX.get("const")
    return sum(1 for i in seq if int(i) == const_idx)


def symbolic_hash(seq: List[int]) -> str:
    return hashlib.sha1(",".join(str(int(i)) for i in seq).encode("utf-8")).hexdigest()


def is_valid_sequence(seq: List[int]) -> bool:
    tokens: List[str] = []
    for idx in seq:
        sym = IDX2SYM.get(int(idx))
        if sym is None:
            return False
        tokens.append(sym)
    ok_prefix, _ = _DEFAULT_VOCAB.is_valid_prefix_expression(tokens)
    if not ok_prefix:
        return False
    ok_rhs, _ = _DEFAULT_VOCAB.is_valid_rhs(tokens)
    return ok_rhs


def get_default_vocab() -> SymbolVocab:
    return _DEFAULT_VOCAB
