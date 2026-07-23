"""Public package for the weak-form symbolic search experiment code.

Import heavy experiment helpers from their concrete modules, for example:

    from src.wised.experiment import run_discovery_experiment
"""

__all__ = ["get_equation_registry", "run_discovery_experiment", "seed_everything"]


def __getattr__(name: str):
    if name == "run_discovery_experiment":
        from .experiment import run_discovery_experiment

        return run_discovery_experiment
    if name == "get_equation_registry":
        from .registries import get_equation_registry

        return get_equation_registry
    if name == "seed_everything":
        from .reproducibility import seed_everything

        return seed_everything
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

