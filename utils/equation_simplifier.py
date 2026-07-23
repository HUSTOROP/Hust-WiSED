from __future__ import annotations

import re
import sympy as sp
from typing import List, Optional, Tuple

_FIELD_RE = r"[uvw]"
_AXIS_RE = r"[xyz]"
_VALID_LHS_RE = re.compile(rf"^({_FIELD_RE})_t$")
_FIELD_DERIV_RE = re.compile(rf"\b({_FIELD_RE})_\{{(({_AXIS_RE})+)\}}")
_GENERIC_DERIV_MARK_RE = re.compile(rf"_\{{(({_AXIS_RE})+)\}}")
_DERIVATIVE_STR_RE = re.compile(
    rf"Derivative\(({_FIELD_RE})\(x, y, z\),\s*((?:\(({_AXIS_RE}),\s*\d+\)|{_AXIS_RE})(?:,\s*(?:\(({_AXIS_RE}),\s*\d+\)|{_AXIS_RE}))*)\)"
)
_WS_RE = re.compile(r"\s+")
_NUM_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?"
_NUMERIC_DERIV_RE = re.compile(rf"(?<![A-Za-z0-9_])(?:{_NUM_RE})_\{{(({_AXIS_RE})+)\}}")
_PAREN_NUMERIC_DERIV_RE = re.compile(rf"\(({_NUM_RE})\)_\{{(({_AXIS_RE})+)\}}")

_SP = None


def _get_sympy():
    global _SP
    if _SP is None:
        import sympy as sp

        _SP = sp
    return _SP

def _split_top_level_commas(text: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]

def _find_matching_left_paren(text: str, right_paren_idx: int) -> int:
    depth = 0
    for idx in range(int(right_paren_idx), -1, -1):
        ch = text[idx]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
            if depth == 0:
                return idx
    return -1

def _strip_outer_parens(text: str) -> str:
    s = str(text).strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        ok = True
        for idx, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    ok = False
                    break
            if depth == 0 and idx < len(s) - 1:
                ok = False
                break
        if not ok or depth != 0:
            break
        s = s[1:-1].strip()
    return s

def _balanced_delimiters(text: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: List[str] = []
    for ch in text:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack

def _sympy_function_pattern(field: str) -> re.Pattern[str]:
    return re.compile(rf"{field}\(x(?:,\s*y(?:,\s*z)?)?\)")

def _clean_rhs_text(rhs: str) -> str:
    rhs = str(rhs).strip()
    if "[prefix" in rhs:
        rhs = rhs.split("[prefix", 1)[0].strip()
    rhs = rhs.replace("−", "-")
    rhs = _strip_outer_parens(rhs)
    rhs = _WS_RE.sub(" ", rhs).strip()
    rhs = rhs.replace("+ -", "- ").replace("- -", "+ ")
    return rhs

def _pretty_rhs_to_sympy_text(rhs_str: str) -> str:
    parsed = str(rhs_str)
    for old, new in (("·", "*"), ("^", "**"), ("torch.abs", "abs"), ("−", "-")):
        parsed = parsed.replace(old, new)

    # Treat the derivative of a numeric constant as zero.
    parsed = _PAREN_NUMERIC_DERIV_RE.sub("0", parsed)
    parsed = _NUMERIC_DERIV_RE.sub("0", parsed)

    # Derivative of a field variable.
    parsed = _FIELD_DERIV_RE.sub(lambda m: f"D({m.group(1)}, '{m.group(2)}')", parsed)

    # Derivative of a general parenthesized expression.
    while True:
        match = _GENERIC_DERIV_MARK_RE.search(parsed)
        if match is None:
            break
        start = int(match.start())
        end = int(match.end())
        if start <= 0 or parsed[start - 1] != ")":
            break
        left = _find_matching_left_paren(parsed, start - 1)
        if left < 0:
            break
        axes = match.group(1)
        inner = parsed[left:start]
        parsed = parsed[:left] + f"D({inner}, '{axes}')" + parsed[end:]

    return parsed

def _eval_sympy_expr(parsed_expr: str):
    sp = _get_sympy()
    x, y, z = sp.symbols("x y z")
    func_map = {name: sp.Function(name)(x, y, z) for name in ("u", "v", "w")}

    def D_func(var, axes):
        axes_str = str(axes).replace("'", "").replace('"', "")
        out = var
        for ax in axes_str:
            if ax == "x":
                out = sp.Derivative(out, x)
            elif ax == "y":
                out = sp.Derivative(out, y)
            elif ax == "z":
                out = sp.Derivative(out, z)
            else:
                raise ValueError(f"Unsupported derivative axis: {ax}")
        return out

    env = {
        "D": D_func,
        "sin": sp.sin,
        "cos": sp.cos,
        "exp": sp.exp,
        "log": sp.log,
        "abs": sp.Abs,
        **func_map,
    }
    return eval(parsed_expr, {"__builtins__": {}}, env)

def _prune_small_numbers(expr, tol: float):
    sp = _get_sympy()
    tol = float(max(0.0, tol))
    if tol <= 0.0:
        return expr
    return expr.replace(
        lambda node: bool(getattr(node, "is_Float", False)) and abs(float(node)) < tol,
        lambda _node: sp.Float(0.0),
    )

def _derivative_repl(match: re.Match[str]) -> str:
    field = match.group(1)
    args = match.group(2)
    axes: List[str] = []
    for part in _split_top_level_commas(args):
        if re.fullmatch(rf"{_AXIS_RE}", part):
            axes.append(part)
            continue
        mm = re.fullmatch(rf"\(({_AXIS_RE}),\s*(\d+)\)", part)
        if mm:
            axes.extend([mm.group(1)] * int(mm.group(2)))
    return f"{field}_{{{''.join(axes)}}}" if axes else field

def _sympy_expr_to_pretty(expr) -> str:
    out = str(expr)
    out = _DERIVATIVE_STR_RE.sub(_derivative_repl, out)
    for field in ("u", "v", "w"):
        out = _sympy_function_pattern(field).sub(field, out)
    out = out.replace("**", "^")
    out = _WS_RE.sub(" ", out).strip()
    out = out.replace("+ -", "- ").replace("- -", "+ ")
    return _strip_outer_parens(out)

def _safe_polish_rhs(rhs: str, prune_tol: float) -> str:
    rhs = _clean_rhs_text(rhs)
    if (not rhs) or (not _balanced_delimiters(rhs)):
        return rhs

    parsed_expr = _pretty_rhs_to_sympy_text(rhs)
    expr = _eval_sympy_expr(parsed_expr)
    sp = _get_sympy()

    # Expand executable derivatives before simplification.
    try:
        expr = expr.doit()
    except Exception:
        pass

    expr = _prune_small_numbers(expr, prune_tol)

    try:
        n_ops = int(sp.count_ops(expr))
    except Exception:
        n_ops = 999999

    use_heavy = (len(rhs) <= 400 and n_ops <= 160)

    if use_heavy:
        try:
            expr = sp.expand(expr)
        except Exception:
            pass
        expr = _prune_small_numbers(expr, prune_tol)
        try:
            expr = sp.factor_terms(expr)
        except Exception:
            pass
        try:
            expr = sp.cancel(expr)
        except Exception:
            pass
        try:
            expr = sp.simplify(expr.evalf(6))
        except Exception:
            expr = expr.evalf(6)
    else:
        try:
            expr = sp.factor_terms(expr)
        except Exception:
            pass
        expr = _prune_small_numbers(expr.evalf(6), prune_tol)

    polished = _sympy_expr_to_pretty(expr)
    reparsed = _pretty_rhs_to_sympy_text(polished)
    _ = _eval_sympy_expr(reparsed)
    return polished or rhs

def polish_discovered_equation(eq_str: str, prune_tol: float = 5e-5) -> str:
        lhs = "u_t"
        rhs = str(eq_str).strip()
        if "=" in rhs:
            lhs_raw, rhs_raw = rhs.split("=", 1)
            lhs_raw = lhs_raw.strip()
            if _VALID_LHS_RE.fullmatch(lhs_raw):
                lhs = lhs_raw
            rhs = rhs_raw.strip()

        # =========================================================
        # Pre-clean equation strings before SymPy parsing.
        # =========================================================
        # 1. Replace every ^ with **.
        rhs = rhs.replace("^", "**")

        # 2. Merge consecutive derivative indices, for example (u^3)_{y}_{x} -> (u^3)_{yx}.
        import re
        while re.search(r"_\{([a-zA-Z]+)\}_\{([a-zA-Z]+)\}", rhs):
            rhs = re.sub(r"_\{([a-zA-Z]+)\}_\{([a-zA-Z]+)\}", r"_{\1\2}", rhs)
        # =========================================================

        try:
            expr = _eval_sympy_expr(_pretty_rhs_to_sympy_text(rhs))
            if expr is None:
                return eq_str

            # =========================================================
            # Evaluate symbolic derivatives and fold constants.
            # For example, Derivative(u**2, x) becomes 2*u*u_x.
            # Convert 0.000 * (...) directly to 0.
            # =========================================================
            expr = expr.doit()

            # Evaluate compound constant powers, for example 0.9985**3 -> 0.9955.
            expr = expr.evalf(6)

            n_ops = 999999
            try:
                n_ops = sp.count_ops(expr)
            except Exception:
                pass

            # Preserve the remaining workflow: expand, prune tiny terms, and combine like terms.
            use_heavy = (len(rhs) <= 400 and n_ops <= 160)
            if use_heavy:
                try:
                    expr = sp.expand(expr)
                except Exception:
                    pass
                # Remove exact zeros and tiny ordinary-least-squares artifacts.
                expr = _prune_small_numbers(expr, prune_tol)
                try:
                    expr = sp.factor_terms(expr)
                except Exception:
                    pass
                try:
                    expr = sp.cancel(expr)
                except Exception:
                    pass
            else:
                try:
                    expr = sp.factor_terms(expr)
                except Exception:
                    pass
                expr = _prune_small_numbers(expr, prune_tol)

            polished = _sympy_expr_to_pretty(expr)
            return f"{lhs} = {polished}"
        except Exception as e:
            return eq_str

__all__ = ["polish_discovered_equation"]
