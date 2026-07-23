from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from models.symbol_vocabulary import SYM2IDX


@dataclass
class OperatorPolicy:
    epoch: int
    spatial_ndim: int
    binary_symbols: List[str]
    unary_symbols: List[str]
    deriv_symbols: List[str]
    hard_forbid_symbols: List[str]
    logit_bias: Dict[str, float]

    @property
    def allowed_nonterminal_symbols(self) -> List[str]:
        out: List[str] = []
        for s in self.binary_symbols + self.unary_symbols + self.deriv_symbols:
            if s not in out:
                out.append(s)
        return out


def build_operator_policy(epoch: int = 0, spatial_ndim: int = 1, mode: str = "pde") -> OperatorPolicy:
    epoch = int(max(0, epoch))
    spatial_ndim = int(max(1, spatial_ndim))
    mode = str(mode or "pde").strip().lower()

    binary = ["+", "*"]
    # Fixed integer powers: replace generic ^ with sq/cube unary templates.
    #
    # adv/lap are operators, not answer templates. They are available in 1D/2D/3D:
    #   adv(q) is constrained elsewhere to adv(field), e.g. u*u_x in 1D
    #   lap(q) is constrained elsewhere to lap(field), e.g. u_xx in 1D
    #
    # Keeping them dimension-agnostic avoids the old failure mode where Burgers-1D
    # could not sample adv(u) even though that operator is semantically valid.
    unary = ["neg", "sq", "cube", "lap", "adv"]
    deriv = ["D"]
    hard_forbid: List[str] = []
    logit_bias: Dict[str, float] = {}

    if mode in {"ode", "no_spatial", "temporal_ode"}:
        unary = ["neg", "sq", "cube"]
        deriv = []
        hard_forbid.extend(["D", "lap", "adv", "^", "/", "sin", "cos", "exp", "log"])
        for s in hard_forbid:
            logit_bias[s] = -12.0
        return OperatorPolicy(
            epoch=epoch,
            spatial_ndim=spatial_ndim,
            binary_symbols=binary,
            unary_symbols=unary,
            deriv_symbols=deriv,
            hard_forbid_symbols=hard_forbid,
            logit_bias=logit_bias,
        )

    if mode in {"diffusion", "diffusion_only", "parabolic_diffusion"}:
        unary = ["neg", "lap"]
        deriv = []
        hard_forbid.extend(["D", "adv", "sq", "cube", "^", "/", "sin", "cos", "exp", "log"])
        for s in hard_forbid:
            logit_bias[s] = -12.0
        logit_bias["lap"] = +2.5
        return OperatorPolicy(
            epoch=epoch,
            spatial_ndim=spatial_ndim,
            binary_symbols=binary,
            unary_symbols=unary,
            deriv_symbols=deriv,
            hard_forbid_symbols=hard_forbid,
            logit_bias=logit_bias,
        )

    # Division is only enabled in later epochs.
    if epoch >= 30:
        binary.append("/")
    else:
        hard_forbid.append("/")
        logit_bias["/"] = -8.0

    # Generic ^ is permanently forbidden; sq/cube express integer powers safely.
    hard_forbid.append("^")
    logit_bias["^"] = -12.0

    # Transcendental functions keep the original late-stage curriculum.
    if epoch >= 30:
        unary.extend(["sin", "cos", "exp", "log"])
    else:
        hard_forbid.extend(["sin", "cos", "exp", "log"])
        for s in ("sin", "cos", "exp", "log"):
            logit_bias[s] = -8.0

    # Early curriculum remains generic: encourage the derivative operator D, but
    # do not inject or prefer a particular PDE template such as adv+lap.
    # Macro operators stay available with neutral logits.
    if epoch < 20:
        logit_bias["D"] = +2.0
    elif epoch < 35:
        logit_bias["D"] = +1.0

    return OperatorPolicy(
        epoch=epoch,
        spatial_ndim=spatial_ndim,
        binary_symbols=binary,
        unary_symbols=unary,
        deriv_symbols=deriv,
        hard_forbid_symbols=hard_forbid,
        logit_bias=logit_bias,
    )


def symbols_to_indices(symbols: List[str]) -> List[int]:
    return [int(SYM2IDX[s]) for s in symbols if s in SYM2IDX]
