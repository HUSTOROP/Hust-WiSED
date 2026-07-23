"""
Canonical equation normalizer for prefix PDE expressions.

This version is intentionally token-level first:
- Keeps the existing commutative normalization for + and *.
- Adds linear-operator / sign canonicalization:
    neg(neg(f))            -> f
    D(axis, neg(f))        -> neg(D(axis, f))
    lap(neg(field))        -> neg(lap(field))
- Adds advection-template recognition, but only for native fields:
    u * D(x, u) + v * D(y, u) [+ w * D(z, u)] -> adv(u)
  Therefore:
    u * D(x, neg(v)) + v * D(y, neg(v))       -> neg(adv(v))

  adv is intentionally a field-specific transport macro; adv(child) is valid only
  when child is one native field token u/v/w present in the active vocabulary.

The goal is to ensure equivalent forms share the same canonical key and the
same population/history entry, instead of only producing a prettier readable
string.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

_SPECIALS = {"<START>", "<END>", "<PAD>", "<UNK>"}
_COMMUTATIVE = {"+", "*"}
_BINARY = {"+", "*", "/", "^"}
_UNARY = {"sin", "cos", "exp", "log", "neg", "lap", "adv", "sq", "cube"}
_GENERIC_DERIV = {"D"}
_LEGACY_DERIV_1 = {"dx", "dxx", "dxxx", "dt", "dxt"}
_FIELDS = ("u", "v", "w")
_AXES = ("x", "y", "z")


def sym_arity(sym: str) -> int:
    if sym in _BINARY:
        return 2
    if sym in _UNARY:
        return 1
    if sym in _GENERIC_DERIV:
        return 2
    if sym in _LEGACY_DERIV_1:
        return 1
    return 0


@dataclass
class _Node:
    op: int
    children: List["_Node"] = field(default_factory=list)
    sort_key: Tuple[int, str, str] = (999, "", "")
    signature: str = ""


class EquationNormalizer:
    """Normalize prefix expression sequences into a canonical token form."""

    def __init__(self, sym2idx: Dict[str, int], idx2sym: Dict[int, str]):
        self.sym2idx = sym2idx
        self.idx2sym = idx2sym
        self.commutative_ops = {sym2idx[s] for s in _COMMUTATIVE if s in sym2idx}
        self.terminals = {
            idx for s, idx in sym2idx.items()
            if s not in _SPECIALS and sym_arity(s) == 0
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def normalize(self, seq: List[int]) -> List[int]:
        """Return a structurally canonical token sequence.

        This method is deliberately safe: if parsing fails or the transformed
        tokens are invalid, it returns the original sequence.
        """
        if not seq:
            return []
        try:
            tree, pos = self._build_tree(list(map(int, seq)), 0)
            if tree is None or pos != len(seq):
                return list(seq)
            tree = self._canonicalize_tree(tree)
            out = self._flatten_tree(tree)
            if self._is_valid_sequence(out):
                return out
            return list(seq)
        except Exception:
            return list(seq)

    def algebraic_simplify(
        self,
        seq: List[int],
        consts: Optional[np.ndarray] = None,
        prune_tol: float = 5e-5,
        const_mask: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Simplify and canonicalize an expression sequence with threshold pruning and optional constant masking."""
        # First canonicalize the structural sequence.
        canonical = self.normalize(list(seq))

        # Return the canonicalized sequence directly when no constants are present.
        if consts is None:
            return canonical

        # Accuracy-first speed mode: first apply only mathematically safe token-level
        # zero-branch pruning, then continue into the complete symbolic simplifier.
        # This preserves the original discovery capacity (e.g. adv/lap canonicalization
        # and nontrivial algebraic merging) while shrinking obvious zero branches.
        fast_pruned = self._fast_prune_by_constants(
            canonical,
            consts=consts,
            prune_tol=prune_tol,
            const_mask=const_mask,
        )
        if fast_pruned is not None:
            canonical = self.normalize(fast_pruned)

        try:
            import sympy as sp
        except ImportError:
            return canonical

        # 1. Build the AST.
        tree, pos = self._build_tree(list(canonical), 0)
        if tree is None or pos != len(canonical):
            return canonical

        # Define symbolic coordinates and fields.
        coords = [sp.Symbol(c) for c in ["t", "x", "y", "z"]]
        sp_funcs = {sym: sp.Function(sym)(*coords) for sym in _FIELDS if sym in self.sym2idx}

        # =========================================================
        # Register macro operators as SymPy placeholders to prevent unintended expansion.
        macro_funcs = {sym: sp.Function(sym) for sym in ["adv", "lap"] if sym in self.sym2idx}

        c_idx = 0

        def _to_sp(node):
            nonlocal c_idx
            if node is None:
                return None
            sym = self._sym(node)

            # Constant-node handling, including ablation masking.
            if sym in ("const", "c"):
                if consts is not None and c_idx < len(consts):
                    val = float(consts[c_idx])
                    current_idx = c_idx
                    c_idx += 1

                    if const_mask is not None:
                        if not const_mask[current_idx]:
                            return sp.Integer(0)

                    if abs(val) < float(prune_tol):
                        return sp.Integer(0)

                    return sp.Float(val)
                return sp.Float(random.uniform(1.1, 9.9))

            # Physical fields and coordinates.
            if sym in sp_funcs:
                return sp_funcs[sym]
            if sym in ["t", "x", "y", "z"]:
                return sp.Symbol(sym)

            # Operator handling.
            if sym in macro_funcs:
                if not node.get("children"): return None
                child = _to_sp(node["children"][0])
                return macro_funcs[sym](child) if child is not None else None

            if sym == "D":
                if len(node.get("children", [])) < 2:
                    return None
                ax_node = node["children"][0]
                ax = self._sym(ax_node)
                child = _to_sp(node["children"][1])
                if child is None or ax not in ["t", "x", "y", "z"]:
                    return None
                return sp.Derivative(child, sp.Symbol(ax))

            if sym == "+":
                l, r = _to_sp(node["children"][0]), _to_sp(node["children"][1])
                return (l + r) if l is not None and r is not None else None

            if sym == "*":
                l, r = _to_sp(node["children"][0]), _to_sp(node["children"][1])
                return (l * r) if l is not None and r is not None else None

            if sym == "neg":
                c = _to_sp(node["children"][0])
                return -c if c is not None else None

            if sym == "/":
                l, r = _to_sp(node["children"][0]), _to_sp(node["children"][1])
                if l is None or r is None or r == 0:
                    return None
                return l / r

            if sym == "sq":
                c = _to_sp(node["children"][0])
                return c**2 if c is not None else None

            if sym == "cube":
                c = _to_sp(node["children"][0])
                return c**3 if c is not None else None

            if sym == "^":
                l, r = _to_sp(node["children"][0]), _to_sp(node["children"][1])
                return l**r if l is not None and r is not None else None

            if sym in {"sin", "cos", "exp", "log"}:
                c = _to_sp(node["children"][0]) if node.get("children") else None
                if c is None:
                    return None
                return getattr(sp, sym)(c)

            return None

        # Convert the AST to a SymPy expression.
        sp_expr = _to_sp(tree)
        if sp_expr is None:
            return canonical

        try:
            # Expand derivatives, simplify the expression, and fold constants.
            sp_expr = sp.simplify(sp_expr.doit())
        except Exception:
            return canonical

        # 3. Convert the SymPy expression back to a token sequence.
        def _sp_to_seq(expr) -> Optional[List[int]]:
            if isinstance(expr, sp.Derivative):
                seq_child = _sp_to_seq(expr.args[0])
                if seq_child is None: return None
                for var in reversed(expr.variables):
                    ax_name = var.name
                    if ax_name not in self.sym2idx or "D" not in self.sym2idx:
                        return None
                    seq_child = [self.sym2idx["D"], self.sym2idx[ax_name]] + seq_child
                return seq_child

            if isinstance(expr, sp.Add):
                args = list(expr.args)
                if not args: return None
                out = _sp_to_seq(args[0])
                if out is None: return None
                for arg in args[1:]:
                    rhs = _sp_to_seq(arg)
                    if rhs is None or "+" not in self.sym2idx: return None
                    out = [self.sym2idx["+"]] + out + rhs
                return out

            if isinstance(expr, sp.Mul):
                args = list(expr.args)
                sign = 1
                if args and args[0] == -1:
                    sign = -1
                    args = args[1:]
                if not args:
                    return [self.sym2idx["const"]] if "const" in self.sym2idx else None
                out = _sp_to_seq(args[0])
                if out is None: return None
                for arg in args[1:]:
                    rhs = _sp_to_seq(arg)
                    if rhs is None or "*" not in self.sym2idx: return None
                    out = [self.sym2idx["*"]] + out + rhs
                if sign < 0 and "neg" in self.sym2idx:
                    out = [self.sym2idx["neg"]] + out
                return out

            if isinstance(expr, sp.Pow):
                base = _sp_to_seq(expr.base)
                exp = expr.exp
                if base is None: return None
                if isinstance(exp, sp.Integer) and int(exp) == 2 and "sq" in self.sym2idx:
                    return [self.sym2idx["sq"]] + base
                if isinstance(exp, sp.Integer) and int(exp) == 3 and "cube" in self.sym2idx:
                    return [self.sym2idx["cube"]] + base
                if isinstance(exp, sp.Integer) and int(exp) > 1 and "*" in self.sym2idx:
                    out = base
                    for _ in range(int(exp) - 1):
                        out = [self.sym2idx["*"]] + out + base
                    return out
                return None

            if isinstance(expr, sp.Function):
                name = str(expr.func)
                if name in ["adv", "lap"] and name in self.sym2idx:
                    if len(expr.args) == 1:
                        child_seq = _sp_to_seq(expr.args[0])
                        if child_seq is not None:
                            return [self.sym2idx[name]] + child_seq

                if name in {"sin", "cos", "exp", "log"} and name in self.sym2idx and len(expr.args) == 1:
                    child_seq = _sp_to_seq(expr.args[0])
                    if child_seq is not None:
                        return [self.sym2idx[name]] + child_seq
                if name in _FIELDS and name in self.sym2idx:
                    return [self.sym2idx[name]]
                return None

            if isinstance(expr, sp.Symbol):
                return [self.sym2idx[expr.name]] if expr.name in self.sym2idx else None

            if isinstance(expr, (sp.Number, sp.Float, sp.Integer)) or getattr(expr, "is_Number", False):
                if float(expr) == 0:
                    return None
                if float(expr) < 0 and "neg" in self.sym2idx and "const" in self.sym2idx:
                    return [self.sym2idx["neg"], self.sym2idx["const"]]
                return [self.sym2idx["const"]] if "const" in self.sym2idx else None

            return None

        new_seq = _sp_to_seq(sp_expr)

        if new_seq is None:
            return [self.sym2idx["const"]] if "const" in self.sym2idx else canonical

        final_seq = self.normalize(new_seq)
        return final_seq if self._is_valid_sequence(final_seq) else canonical


    def _fast_prune_by_constants(
        self,
        seq: List[int],
        consts: Optional[np.ndarray] = None,
        prune_tol: float = 5e-5,
        const_mask: Optional[np.ndarray] = None,
    ) -> Optional[List[int]]:
        """Token-level pruning for zero/small coefficient branches.

        This avoids importing SymPy in the hot evolutionary loop.  It is
        deliberately conservative: it only uses arithmetic identities that are
        safe for prefix PDE expressions, mainly 0 + a -> a and 0 * a -> 0.
        Nonzero constants are kept as `const` placeholders so coefficients are
        re-estimated by the normal optimizer after pruning.
        """
        if consts is None:
            return None
        try:
            arr = np.asarray(consts, dtype=np.float64).reshape(-1)
        except Exception:
            return None
        mask_arr = None
        if const_mask is not None:
            try:
                mask_arr = np.asarray(const_mask, dtype=bool).reshape(-1)
            except Exception:
                mask_arr = None

        try:
            tree, pos = self._build_tree(list(map(int, seq)), 0)
            if tree is None or pos != len(seq):
                return None
        except Exception:
            return None

        const_counter = 0
        changed = False

        def is_zero_const(idx: int) -> bool:
            if mask_arr is not None and idx < len(mask_arr) and not bool(mask_arr[idx]):
                return True
            if idx < len(arr):
                return abs(float(arr[idx])) < float(prune_tol)
            return False

        def prune(node):
            nonlocal const_counter, changed
            if node is None:
                return None
            sym = self._sym(node)

            if sym in ("const", "c"):
                idx = const_counter
                const_counter += 1
                if is_zero_const(idx):
                    changed = True
                    return None
                return {"op": int(node["op"]), "children": []}

            children = list(node.get("children", []))
            if not children:
                return {"op": int(node["op"]), "children": []}

            if sym == "D" and len(children) >= 2:
                ax_node = children[0]
                child = prune(children[1])
                if child is None:
                    changed = True
                    return None
                return {"op": int(node["op"]), "children": [ax_node, child]}

            pruned_children = [prune(c) for c in children]

            if sym == "+" and len(pruned_children) == 2:
                left, right = pruned_children
                if left is None and right is None:
                    changed = True
                    return None
                if left is None:
                    changed = True
                    return right
                if right is None:
                    changed = True
                    return left
                return {"op": int(node["op"]), "children": [left, right]}

            if sym == "*" and len(pruned_children) == 2:
                if pruned_children[0] is None or pruned_children[1] is None:
                    changed = True
                    return None
                return {"op": int(node["op"]), "children": pruned_children}

            if sym == "neg" and len(pruned_children) == 1:
                if pruned_children[0] is None:
                    changed = True
                    return None
                return {"op": int(node["op"]), "children": pruned_children}

            # Accuracy-first conservative zero propagation.  Only return a zero
            # branch for identities that are certainly zero.  Do NOT prune cos(0),
            # exp(0), log(0), denominator-zero, or exponent-zero cases here.
            if any(c is None for c in pruned_children):
                if sym in {"lap", "adv", "sq", "cube", "sin"} and len(pruned_children) == 1:
                    changed = True
                    return None
                if sym == "/" and len(pruned_children) == 2:
                    # 0 / f -> 0 is safe when denominator survived; f / 0 is not.
                    if pruned_children[0] is None and pruned_children[1] is not None:
                        changed = True
                        return None
                    return {"op": int(node["op"]), "children": children}
                if sym == "^":
                    return {"op": int(node["op"]), "children": children}
                if sym in {"cos", "exp", "log"}:
                    return {"op": int(node["op"]), "children": children}
                return {"op": int(node["op"]), "children": children}

            return {"op": int(node["op"]), "children": pruned_children}

        pruned = prune(tree)
        if not changed:
            return None

        if pruned is None:
            if "const" in self.sym2idx:
                out = [self.sym2idx["const"]]
            else:
                out = list(seq)
        else:
            out = self._flatten_tree(pruned)

        out = self.normalize(out)
        return out if self._is_valid_sequence(out) else list(seq)

    def _structural_simplify(self, tree: dict) -> dict:
        """Recursively apply constant folding to an AST."""
        if not tree or "children" not in tree or not tree["children"]:
            return tree

        # Simplify child nodes depth first.
        simplified_children = [self._structural_simplify(c) for c in tree["children"]]
        tree["children"] = simplified_children

        op_sym = self.idx2sym.get(tree["op"], "")

        # Rule A: shield against pathological operator nesting.
        if op_sym in {"sq", "cube", "exp", "log", "sin", "cos"}:
            child_sym = self.idx2sym.get(simplified_children[0]["op"], "")
            if child_sym in {"sq", "cube", "exp", "log", "sin", "cos", "neg"}:
                raise ValueError("Pathological operator nesting detected")

        # Rule B: absorb constants through unary operators.
        if op_sym in _UNARY and len(simplified_children) == 1:
            if self.idx2sym.get(simplified_children[0]["op"], "") in {"const", "c"}:
                return {"op": self.sym2idx["const"], "children": []}

        # Rule C: flatten associative operators and collapse constants.
        if op_sym in {"+", "*"}:
            operands = self._flatten_associative(tree, op_sym)

            # =========================================================
            # Rule C.1: conservative physical pruning.
            if op_sym == "+":
                # Addition: remove terms marked as zero.
                operands = [n for n in operands if not n.get("is_zero", False)]
                # Return a zero-marked constant node when every term is removed.
                if not operands:
                    return {"op": self.sym2idx["const"], "children": [], "is_zero": True}

            if op_sym == "*":
                # Multiplication: any zero factor makes the full product zero.
                if any(n.get("is_zero", False) for n in operands):
                    return {"op": self.sym2idx["const"], "children": [], "is_zero": True}
            # =========================================================

            const_nodes = [n for n in operands if self.idx2sym.get(n["op"], "") in {"const", "c"}]
            non_const_nodes = [n for n in operands if self.idx2sym.get(n["op"], "") not in {"const", "c"}]

            if len(const_nodes) > 1:
                const_nodes = [const_nodes[0]]

            if not non_const_nodes:
                return {"op": self.sym2idx["const"], "children": []}

            new_operands = const_nodes + non_const_nodes
            return self._build_binary_tree(new_operands, tree["op"])

        # Rule D: absorb constants through binary operators.
        if op_sym in {"/", "^"} and len(simplified_children) == 2:
            left_sym = self.idx2sym.get(simplified_children[0]["op"], "")
            right_sym = self.idx2sym.get(simplified_children[1]["op"], "")
            if left_sym in {"const", "c"} and right_sym in {"const", "c"}:
                return {"op": self.sym2idx["const"], "children": []}

        return tree

    def _flatten_associative(self, tree: dict, target_op_sym: str) -> List[dict]:
        """Flatten nested + or * operations into a one-dimensional operand list."""
        op_sym = self.idx2sym.get(tree["op"], "")
        if op_sym != target_op_sym:
            return [tree]

        operands = []
        for child in tree.get("children", []):
            operands.extend(self._flatten_associative(child, target_op_sym))
        return operands

    def _build_binary_tree(self, nodes: List[dict], op_idx: int) -> dict:
        """Fold an operand list into a binary tree that follows the parser grammar."""
        if len(nodes) == 1:
            return nodes[0]
        return {
            "op": op_idx,
            "children": [nodes[0], self._build_binary_tree(nodes[1:], op_idx)]
        }

    def _tree_to_seq(self, tree: dict) -> List[int]:
        """Convert an AST back into a prefix-token sequence."""
        if not tree:
            return []
        seq = [tree["op"]]
        for child in tree.get("children", []):
            seq.extend(self._tree_to_seq(child))
        return seq



    # ------------------------------------------------------------------
    # Tree construction / flattening
    # ------------------------------------------------------------------
    def _build_tree(self, tokens: List[int], pos: int) -> Tuple[Optional[Dict[str, Any]], int]:
        if pos >= len(tokens):
            return None, pos
        tok = int(tokens[pos])
        pos += 1
        sym = self.idx2sym.get(tok, str(tok))

        if sym in _SPECIALS:
            return {"op": tok, "children": []}, pos

        if sym == "D":
            if pos >= len(tokens):
                return None, pos
            ax_tok = int(tokens[pos])
            pos += 1
            expr_tree, pos = self._build_tree(tokens, pos)
            if expr_tree is None:
                return None, pos
            return {"op": tok, "children": [{"op": ax_tok, "children": []}, expr_tree]}, pos

        children = []
        for _ in range(sym_arity(sym)):
            child, pos = self._build_tree(tokens, pos)
            if child is None:
                return None, pos
            children.append(child)
        return {"op": tok, "children": children}, pos

    def _flatten_tree(self, tree: Dict[str, Any]) -> List[int]:
        if not tree:
            return []
        out = [int(tree["op"])]
        for child in tree.get("children", []):
            out.extend(self._flatten_tree(child))
        return out

    # ------------------------------------------------------------------
    # Canonicalization
    # ------------------------------------------------------------------
    def _canonicalize_tree(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        prev = None
        curr = tree
        # Iterate to a fixed point; local rewrites can expose new patterns.
        for _ in range(12):
            before = self._tree_to_string(curr)
            curr = self._canonicalize_once(curr)
            after = self._tree_to_string(curr)
            if after == before or after == prev:
                break
            prev = before
        return curr

    def _canonicalize_once(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        if not tree:
            return tree
        tree = {"op": int(tree["op"]), "children": [self._canonicalize_once(c) for c in tree.get("children", [])]}

        sym = self._sym(tree)

        # neg(neg(x)) -> x
        if sym == "neg" and len(tree["children"]) == 1:
            child = tree["children"][0]
            if self._sym(child) == "neg" and child.get("children"):
                return self._canonicalize_once(child["children"][0])

        # D(axis, neg(x)) -> neg(D(axis, x))
        if sym == "D" and len(tree["children"]) >= 2:
            ax, child = tree["children"][0], tree["children"][1]
            if self._sym(child) == "neg" and child.get("children") and "neg" in self.sym2idx:
                return self._canonicalize_once(self._make("neg", [self._make("D", [ax, child["children"][0]])]))

        # lap(neg(field)) -> neg(lap(field)).
        # Do not rewrite adv(neg(field)) here: adv is deliberately restricted
        # to adv(field) at the grammar/RHS-validator level. Product-form
        # advection with a negative field is still canonicalized by
        # _try_rewrite_advection_sum() into neg(adv(field)).
        if sym == "lap" and len(tree["children"]) == 1:
            child = tree["children"][0]
            if self._sym(child) == "neg" and child.get("children") and "neg" in self.sym2idx:
                inner = child["children"][0]
                return self._canonicalize_once(self._make("neg", [self._make(sym, [inner])]))


        # Hoist signs out of products so u*neg(Dx(v)) and neg(u*Dx(v))
        # share the same canonical representation.
        if sym == "*":
            sign, factors = self._flatten_mul_signed(tree)
            factors = [self._canonicalize_once(f) for f in factors]

            # ========================================================
            # Detect and merge one-dimensional advection factors in products.
            # This recognizes -6 * u * u_x as -6 * adv(u) without splitting its physical factor.
            active_fields = [f for f in ("u", "v", "w") if f in self.sym2idx]
            active_axes = [a for a in ("x", "y", "z") if a in self.sym2idx]

            # Restrict multiplicative advection merging to one-dimensional single-field systems.
            if len(active_fields) == 1 and len(active_axes) == 1:
                f_name = active_fields[0]
                a_name = active_axes[0]

                while True:
                    # 1. Find an independent field node, for example u.
                    idx_field = -1
                    for i, f in enumerate(factors):
                        if self._sym(f) == f_name and not f.get("children"):
                            idx_field = i
                            break

                    # 2. Find its matching derivative node, for example D_x(u).
                    idx_deriv = -1
                    if idx_field != -1:
                        for i, f in enumerate(factors):
                            if i == idx_field: continue
                            if self._sym(f) == "D" and len(f.get("children", [])) == 2:
                                c_ax, c_ex = f["children"]
                                if self._sym(c_ax) == a_name and self._sym(c_ex) == f_name and not c_ex.get("children"):
                                    idx_deriv = i
                                    break

                    # 3. Extract and merge the matching field-gradient pair as adv.
                    if idx_field != -1 and idx_deriv != -1:
                        f_f = factors.pop(max(idx_field, idx_deriv))
                        f_d = factors.pop(min(idx_field, idx_deriv))
                        adv_node = self._make("adv", [f_f])
                        factors.append(adv_node)
                    else:
                        break  # No additional advection factor can be merged.

            factors.sort(key=lambda c: self._get_subtree_priority(c))
            core = self._make_balanced("*", factors)

            if sign < 0 and "neg" in self.sym2idx:
                return self._canonicalize_once(self._make("neg", [core]))
            return core

        # Flatten and sort additions; if every term is negative, factor the sign.
        # This makes neg(a)+neg(b) canonical as neg(a+b).
        if sym == "+":
            terms = [self._canonicalize_once(t) for t in self._flatten_op(tree, "+")]
            signed_terms = [self._strip_full_neg(t) for t in terms]
            if signed_terms and all(sgn < 0 for sgn, _ in signed_terms) and "neg" in self.sym2idx:
                cores = [core for _, core in signed_terms]
                cores.sort(key=lambda c: self._get_subtree_priority(c))
                return self._canonicalize_once(self._make("neg", [self._make_balanced("+", cores)]))
            terms.sort(key=lambda c: self._get_subtree_priority(c))
            tree = self._make_balanced("+", terms)

        # Recognize u*D_x(f)+v*D_y(f)[+w*D_z(f)] as adv(f),
        # where f is an active native field token only.
        adv_tree = self._try_rewrite_advection_sum(tree)
        if adv_tree is not None:
            return self._canonicalize_once(adv_tree)

        # Sort commutative children after rewrites.
        if int(tree["op"]) in self.commutative_ops:
            tree["children"].sort(key=lambda c: self._get_subtree_priority(c))

        return tree

    def _try_rewrite_advection_sum(self, tree: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self._sym(tree) != "+" or "adv" not in self.sym2idx:
            return None
        terms = self._flatten_op(tree, "+")
        if len(terms) < 2:
            return None

        used = set()
        q_tree = None
        required = []
        for field, axis in zip(_FIELDS, _AXES):
            if field in self.sym2idx and axis in self.sym2idx:
                required.append((field, axis))
        if len(required) < 2:
            return None

        matches = {}
        signs = {}
        for idx, term in enumerate(terms):
            parsed = self._parse_velocity_gradient_product(term)
            if parsed is None:
                continue
            sign, field, axis, child = parsed
            if (field, axis) not in required:
                continue
            if q_tree is None:
                q_tree = child
            elif not self._same_tree(q_tree, child):
                continue
            matches[(field, axis)] = idx
            signs[(field, axis)] = int(sign)

        # Require all available spatial components up to current vocabulary.
        if q_tree is None or any(pair not in matches for pair in required):
            return None

        q_sign, q_core = self._strip_full_neg(q_tree)
        q_name = self._sym(q_core)
        if q_name not in _FIELDS or q_name not in self.sym2idx or q_core.get("children"):
            return None

        sign_values = {int(signs.get(pair, 1)) * int(q_sign) for pair in required}
        if len(sign_values) != 1:
            return None

        used = set(matches.values())
        adv = self._make("adv", [q_core])
        if next(iter(sign_values)) < 0 and "neg" in self.sym2idx:
            adv = self._make("neg", [adv])
        remaining = [terms[i] for i in range(len(terms)) if i not in used]
        out = adv
        for term in remaining:
            out = self._make("+", [out, term])
        return out

    def _try_rewrite_advection_product(self, tree: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Canonicalize the 1D advection macro: u * D_x(q) -> adv(q)."""
        parsed = self._parse_velocity_gradient_product(tree)
        if parsed is None:
            return None
        sign, field, axis, child = parsed

        # Active fields and axes in the current vocabulary.
        active_fields = [f for f in _FIELDS if f in self.sym2idx]
        active_axes = [a for a in _AXES if a in self.sym2idx]
        required = list(zip(active_fields, active_axes))

        # Enable single-term advection merging only for one-dimensional systems.
        if len(required) == 1 and (field, axis) == required[0]:
            q_sign, q_core = self._strip_full_neg(child)
            q_name = self._sym(q_core)
            if q_name in active_fields and not q_core.get("children"):
                adv = self._make("adv", [q_core])
                if sign * q_sign < 0 and "neg" in self.sym2idx:
                    adv = self._make("neg", [adv])
                return adv
        return None

    def _parse_velocity_gradient_product(self, term: Dict[str, Any]) -> Optional[Tuple[int, str, str, Dict[str, Any]]]:
        sign, core = self._strip_full_neg(term)
        if self._sym(core) != "*":
            return None
        factors = self._flatten_op(core, "*")
        if len(factors) != 2:
            return None
        a, b = factors
        parsed = self._field_times_derivative(a, b)
        if parsed is None:
            parsed = self._field_times_derivative(b, a)
        if parsed is None:
            return None
        p_sign, field, axis, child = parsed
        return int(sign * p_sign), field, axis, child

    def _field_times_derivative(self, field_node: Dict[str, Any], deriv_node: Dict[str, Any]) -> Optional[Tuple[int, str, str, Dict[str, Any]]]:
        field_sign, field_core = self._strip_full_neg(field_node)
        deriv_sign, deriv_core = self._strip_full_neg(deriv_node)
        field = self._sym(field_core)
        if field not in _FIELDS or field not in self.sym2idx or field_core.get("children"):
            return None
        if self._sym(deriv_core) != "D" or len(deriv_core.get("children", [])) < 2:
            return None
        axis_node = deriv_core["children"][0]
        axis = self._sym(axis_node)
        if axis not in _AXES or axis not in self.sym2idx or axis_node.get("children"):
            return None
        child = deriv_core["children"][1]
        return int(field_sign * deriv_sign), field, axis, child

    def _strip_one_neg(self, tree: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        if self._sym(tree) == "neg" and tree.get("children"):
            return -1, tree["children"][0]
        return 1, tree

    def _strip_full_neg(self, tree: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        sign = 1
        curr = tree
        while self._sym(curr) == "neg" and curr.get("children"):
            sign *= -1
            curr = curr["children"][0]
        return sign, curr

    def _flatten_mul_signed(self, tree: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]]]:
        sign = 1
        factors: List[Dict[str, Any]] = []
        for factor in self._flatten_op(tree, "*"):
            sgn, core = self._strip_full_neg(factor)
            sign *= int(sgn)
            factors.append(core)
        return sign, factors or [tree]

    def _make_balanced(self, sym: str, children: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not children:
            if "const" in self.sym2idx:
                return {"op": int(self.sym2idx["const"]), "children": []}
            return {"op": int(self.sym2idx[sym]), "children": []}
        if len(children) == 1:
            return children[0]
        out = self._make(sym, [children[0], children[1]])
        for child in children[2:]:
            out = self._make(sym, [out, child])
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make(self, sym: str, children: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"op": int(self.sym2idx[sym]), "children": children}

    def _sym(self, tree: Dict[str, Any]) -> str:
        return self.idx2sym.get(int(tree["op"]), str(int(tree["op"])))

    def _flatten_op(self, tree: Dict[str, Any], op_sym: str) -> List[Dict[str, Any]]:
        if self._sym(tree) != op_sym:
            return [tree]
        out = []
        for child in tree.get("children", []):
            out.extend(self._flatten_op(child, op_sym))
        return out

    def _same_tree(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        return self._tree_to_string(a) == self._tree_to_string(b)

    def _is_valid_sequence(self, seq: List[int]) -> bool:
        try:
            from models.symbol_vocabulary import is_valid_sequence
            return bool(is_valid_sequence(seq))
        except Exception:
            return True

    def _get_subtree_priority(self, subtree: Dict[str, Any]) -> Tuple[int, str, str]:
        if subtree is None:
            return (999, "", "")
        op = int(subtree["op"])
        sym = self.idx2sym.get(op, str(op))
        if op in self.terminals or sym_arity(sym) == 0:
            pri = 0
        elif sym in _GENERIC_DERIV or sym in _LEGACY_DERIV_1:
            pri = 1
        elif sym in _UNARY:
            pri = 2
        elif sym in _BINARY:
            pri = 3
        else:
            pri = 4
        return (pri, sym, self._tree_to_string(subtree))

    def _tree_to_string(self, tree) -> str:
        if tree is None:
            return ""
        sym = self._sym(tree)
        if not tree.get("children"):
            return "C" if sym in ("const", "c") else sym
        return f"({sym} {' '.join(self._tree_to_string(c) for c in tree['children'])})"


def deduplicate_population(population: List[Dict], normalizer: EquationNormalizer) -> List[Dict]:
    """Deduplicate by canonical form; keep the best candidate for each form."""
    best_by_form: Dict[str, Dict] = {}
    for ind in population:
        seq = list(map(int, ind.get("seq", [])))
        try:
            norm = normalizer.normalize(seq)
            form = "_".join(str(int(x)) for x in norm)
        except Exception:
            norm = seq
            form = "_".join(str(int(x)) for x in seq)
        ind = dict(ind)
        ind["seq"] = norm
        prev = best_by_form.get(form)
        if prev is None or float(ind.get("fitness", 1e18)) < float(prev.get("fitness", 1e18)):
            best_by_form[form] = ind
    return list(best_by_form.values())

