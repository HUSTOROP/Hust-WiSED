from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

from models.symbol_vocabulary import IDX2SYM, SYM2IDX

_SIMPLE_VARS = frozenset({"u", "v", "w", "x", "y", "z", "t"})
_FIELD_VARS = frozenset({"u", "v", "w"})
_UNARY_FUNCS = frozenset({"sin", "cos", "exp", "log"})
_BINARY_PRECEDENCE = {"+": 10, "*": 20, "/": 20, "^": 30}
_DERIV_PRECEDENCE = 90
_ATOM_PRECEDENCE = 100

@dataclass(frozen=True)
class ExprText:
    text: str
    precedence: int = _ATOM_PRECEDENCE
    deriv_base: Optional[str] = None
    deriv_axes: str = ""

    def with_axes(self, axes: str) -> "ExprText":
        axes = str(axes or "")
        if not axes:
            return self
        if self.deriv_base in _FIELD_VARS:
            all_axes = f"{self.deriv_axes}{axes}"
            return ExprText(
                text=f"{self.deriv_base}_{{{all_axes}}}",
                precedence=_DERIV_PRECEDENCE,
                deriv_base=self.deriv_base,
                deriv_axes=all_axes,
            )
        inner = self.text if self.precedence >= _DERIV_PRECEDENCE else f"({self.text})"
        return ExprText(text=f"{inner}_{{{axes}}}", precedence=_DERIV_PRECEDENCE)

def _format_const(value: float) -> str:
    v = float(value)
    av = abs(v)
    if av >= 0.1 and av < 1000.0 and abs(v - round(v)) < 1e-8:
        return str(int(round(v)))
    if av >= 1e-3 and av < 1000.0:
        return f"{v:.4g}"
    return f"{v:.4e}"

def _wrap(expr: ExprText, min_precedence: int) -> str:
    return expr.text if expr.precedence >= min_precedence else f"({expr.text})"

def _active_spatial_axes() -> List[str]:
    axes = [ax for ax in ("x", "y", "z") if ax in SYM2IDX]
    return axes or ["x"]

def _available_velocity_fields() -> Sequence[str]:
    fields = [f for f in ("u", "v", "w") if f in SYM2IDX]
    return fields or ["u"]

class PrefixPrinter:
    """Fast prefix-to-string compiler with derivative-aware formatting."""

    __slots__ = ("tokens", "consts", "pos", "const_pos", "axes", "velocity_fields")

    def __init__(self, tokens: Iterable[int], consts: Optional[np.ndarray] = None):
        self.tokens = [int(t) for t in tokens]
        self.consts = np.asarray(consts if consts is not None else [], dtype=np.float64).reshape(-1)
        self.pos = 0
        self.const_pos = 0
        self.axes = _active_spatial_axes()
        self.velocity_fields = list(_available_velocity_fields())

    def _next_const(self) -> str:
        if self.const_pos < int(self.consts.size):
            out = _format_const(float(self.consts[self.const_pos]))
            self.const_pos += 1
            return out
        return "C"

    def _parse_derivative(self) -> ExprText:
        ax_tok = self.tokens[self.pos] if self.pos < len(self.tokens) else -1
        if self.pos < len(self.tokens):
            self.pos += 1
        axis = IDX2SYM.get(int(ax_tok), "?")
        child = self.parse()
        if axis in {"x", "y", "z", "t"}:
            return child.with_axes(axis)
        return ExprText(text=f"D_?({_wrap(child, 0)})", precedence=_DERIV_PRECEDENCE)

    def _parse_laplacian(self) -> ExprText:
        child = self.parse()
        terms = [child.with_axes(ax + ax).text for ax in self.axes]
        if len(terms) == 1:
            return ExprText(text=terms[0], precedence=_BINARY_PRECEDENCE["+"])
        return ExprText(text=" + ".join(terms), precedence=_BINARY_PRECEDENCE["+"])

    def _parse_advection(self) -> ExprText:
        child = self.parse()
        vel_map = {"x": "u", "y": "v", "z": "w"}
        fallback = self.velocity_fields[0]
        terms: List[str] = []
        for axis in self.axes:
            vel = vel_map.get(axis, fallback)
            if vel not in self.velocity_fields:
                vel = fallback
            terms.append(f"{vel}*{child.with_axes(axis).text}")
        if len(terms) == 1:
            return ExprText(text=terms[0], precedence=_BINARY_PRECEDENCE["*"])
        return ExprText(text=" + ".join(terms), precedence=_BINARY_PRECEDENCE["+"])

    def _parse_neg(self) -> ExprText:
        child = self.parse()
        if child.text.startswith("-"):
            return ExprText(text=child.text[1:], precedence=child.precedence, deriv_base=child.deriv_base, deriv_axes=child.deriv_axes)
        return ExprText(text=f"-{_wrap(child, _BINARY_PRECEDENCE['^'])}", precedence=_BINARY_PRECEDENCE["^"])

    def _parse_binary(self, sym: str) -> ExprText:
        left = self.parse()
        right = self.parse()
        prec = _BINARY_PRECEDENCE[sym]
        if sym == "+":
            return ExprText(text=f"{_wrap(left, prec)} + {_wrap(right, prec)}", precedence=prec)
        if sym == "*":
            return ExprText(text=f"{_wrap(left, prec)}*{_wrap(right, prec)}", precedence=prec)
        if sym == "/":
            return ExprText(text=f"{_wrap(left, prec)} / {_wrap(right, prec + 1)}", precedence=prec)
        return ExprText(text=f"{_wrap(left, prec)} ^ {_wrap(right, prec)}", precedence=prec)

    def parse(self) -> ExprText:
        if self.pos >= len(self.tokens):
            return ExprText(text="")

        tok = self.tokens[self.pos]
        self.pos += 1
        sym = IDX2SYM.get(int(tok), f"UNK_{tok}")

        if sym in _SIMPLE_VARS:
            deriv_base = sym if sym in _FIELD_VARS else None
            return ExprText(text=sym, precedence=_ATOM_PRECEDENCE, deriv_base=deriv_base)

        if "_" in str(sym) and str(sym).split("_")[0] in _FIELD_VARS:
            base, axes = str(sym).split("_")
            return ExprText(
                text=f"{base}_{{{axes}}}",
                precedence=_ATOM_PRECEDENCE,
                deriv_base=base,
                deriv_axes=axes
            )

        if sym in {"const", "c"}:
            return ExprText(text=self._next_const())
        if sym in {"D"}:
            return self._parse_derivative()
        if sym == "lap":
            return self._parse_laplacian()
        if sym == "adv":
            return self._parse_advection()
        if sym == "neg":
            return self._parse_neg()
        if sym == "sq":
            child = self.parse()
            return ExprText(text=f"{_wrap(child, _BINARY_PRECEDENCE['^'])}^2", precedence=_BINARY_PRECEDENCE["^"])
        if sym == "cube":
            child = self.parse()
            return ExprText(text=f"{_wrap(child, _BINARY_PRECEDENCE['^'])}^3", precedence=_BINARY_PRECEDENCE["^"])
        if sym in _UNARY_FUNCS:
            child = self.parse()
            return ExprText(text=f"{sym}({_wrap(child, 0)})")
        if sym in _BINARY_PRECEDENCE:
            return self._parse_binary(sym)
        return ExprText(text=f"{sym}(...)")

def compile_equation(token_seq: List[int], consts=None, lhs: str = "u_t") -> str:
    if not token_seq:
        return f"{lhs} = ?"
    try:
        printer = PrefixPrinter(token_seq, consts)
        rhs = printer.parse().text.strip() or "?"
        return f"{lhs} = {rhs}"
    except Exception:
        parts = [IDX2SYM.get(int(t), "?") for t in token_seq]
        return f"{lhs} = [prefix: {' '.join(parts)}]"


__all__ = ["ExprText", "PrefixPrinter", "compile_equation"]

