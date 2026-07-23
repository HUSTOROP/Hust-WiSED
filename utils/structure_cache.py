from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

from models.symbol_vocabulary import SYM2IDX, IDX2SYM, symbolic_hash
from utils.equation_canonicalizer import EquationNormalizer


@dataclass(frozen=True)
class StructureKey:
    kind: str
    canonical_tokens: Tuple[int, ...]
    canonical_form: str
    hash_id: str
    key: str


class StructureKeyManager:
    """Unified global structural ID system.

    Three levels are exposed:
      - expr_key: full expression canonical key
      - subtree_key: canonical key for a subtree token slice
      - template_key: canonical key for structure-level constant fitting caches

    All three use normalized prefix tokens plus a Godel hash so every module can share
    the same structural identity language.
    """

    def __init__(self, sym2idx: Dict[str, int], idx2sym: Dict[int, str]):
        self.sym2idx = sym2idx
        self.idx2sym = idx2sym
        self.normalizer = EquationNormalizer(sym2idx, idx2sym)

    @lru_cache(maxsize=200000)
    def _normalize_cached(self, seq_tuple: Tuple[int, ...]) -> Tuple[int, ...]:
        if not seq_tuple:
            return tuple()
        try:
            return tuple(int(t) for t in self.normalizer.normalize(list(seq_tuple)))
        except Exception:
            return tuple(int(t) for t in seq_tuple)

    def normalize(self, seq: Iterable[int]) -> Tuple[int, ...]:
        return self._normalize_cached(tuple(int(t) for t in seq))

    def canonical_form(self, seq: Iterable[int]) -> str:
        norm = self.normalize(seq)
        return "_".join(str(int(t)) for t in norm)

    def _make(self, kind: str, seq: Iterable[int]) -> StructureKey:
        norm = self.normalize(seq)
        canon = "_".join(str(int(t)) for t in norm)
        hash_id = symbolic_hash(list(norm))
        return StructureKey(kind=kind, canonical_tokens=norm, canonical_form=canon, hash_id=hash_id, key=f"{kind}|{hash_id}")

    def expr_key(self, seq: Iterable[int]) -> StructureKey:
        return self._make("expr", seq)

    def subtree_key(self, seq: Iterable[int]) -> StructureKey:
        return self._make("subtree", seq)

    def template_key(self, seq: Iterable[int]) -> StructureKey:
        # Constants are already represented structurally as `const` tokens. Keep a
        # dedicated key type so optimizer caches stay semantically separate.
        return self._make("template", seq)


_KEY_MANAGER: StructureKeyManager | None = None


def get_structure_key_manager() -> StructureKeyManager:
    global _KEY_MANAGER
    if _KEY_MANAGER is None:
        _KEY_MANAGER = StructureKeyManager(SYM2IDX, IDX2SYM)
    return _KEY_MANAGER

def reset_structure_key_manager() -> None:
    global _KEY_MANAGER
    _KEY_MANAGER = None

