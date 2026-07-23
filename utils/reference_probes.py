from __future__ import annotations

"""Optional reference-equation probes for numerical consistency checks.

These probes are deliberately outside the discovery loop.  They do not seed,
rank, inject, or mutate candidate equations; they only provide a hook for
developers who want to test the weak-form evaluator against a known equation in
separate diagnostics.
"""

from typing import Any


def run_reference_probe(
    ctx: Any,
    equation_name: Any,
    *,
    strict: bool = False,
    logger: Any = None,
) -> bool:
    """Placeholder hook for optional reference-equation diagnostics.

    Public reproducibility runs keep this disabled so paper experiments do not
    depend on task-specific ground-truth structures at training time.
    """
    _ = (ctx, strict)
    message = (
        f"Reference probe is disabled for {equation_name}; WiSED discovery "
        "uses only data-driven weak-form residuals."
    )
    if logger is not None and hasattr(logger, "info"):
        logger.info(message)
    return False


def reference_probe_test(ctx: Any, equation_name: Any = "burgers_1d", logger: Any = None) -> bool:
    """Backward-compatible diagnostic entry point."""
    return run_reference_probe(ctx, equation_name, logger=logger)


__all__ = ["run_reference_probe", "reference_probe_test"]

