from __future__ import annotations

from typing import Any, Callable, Dict

from .paths import ensure_project_root_on_path

ensure_project_root_on_path()

from data.burgers_1d import burgers_equation
from data.burgers_2d import burgers2d_equation
from data.fhn_2d import fitzhugh_nagumo2d_equation
from data.kdv_1d import kdv_equation

EquationGenerator = Callable[..., Any]


def _optional_generators() -> Dict[str, EquationGenerator]:
    """Return only the two engineering cases released with this archive."""
    from engineering_validation.sib_diffusion_raw_field import SIB_DIFFUSION_RAW_FIELD_REGISTRY
    from engineering_validation.soc_eis_impedance_manifold import SOC_EIS_IMPEDANCE_MANIFOLD_REGISTRY

    generators: Dict[str, EquationGenerator] = {}
    generators.update(SIB_DIFFUSION_RAW_FIELD_REGISTRY)
    generators.update(SOC_EIS_IMPEDANCE_MANIFOLD_REGISTRY)
    return generators


def get_equation_registry() -> Dict[str, EquationGenerator]:
    """Return all dataset generators available in the local checkout."""
    registry: Dict[str, EquationGenerator] = {
        "burgers_1d": burgers_equation,
        "kdv_1d": kdv_equation,
        "burgers_2d": burgers2d_equation,
        "fhn_2d": fitzhugh_nagumo2d_equation,
    }
    registry.update(_optional_generators())
    return registry


def get_equation_generator(task_name: str) -> EquationGenerator:
    """Resolve a task name to its dataset generator."""
    registry = get_equation_registry()
    try:
        return registry[str(task_name)]
    except KeyError as exc:
        raise ValueError(
            f"Task {task_name!r} is not available. Choose from {sorted(registry)}."
        ) from exc
