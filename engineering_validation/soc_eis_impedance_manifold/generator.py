from __future__ import annotations

from typing import Any, Dict

from .dataset import make_soc_eis_impedance_manifold_dataset


def soc_eis_impedance_manifold_dataset(
    *,
    save_dir: str = "data/dataset",
    noise_level: float = 0.0,
    batch_size: int = 1,
    seed: int = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Generate the raw EIS manifold field dataset for WiSED discovery.

    Contract:
      t -> discharge progress q = 1 - SoC, coordinate only
      x -> log10(frequency_Hz), coordinate only
      u -> standardized Re(Z)
      v -> standardized -Im(Z)

    No scalar SoC feature, equivalent-circuit parameters, candidate term table,
    or precomputed derivative columns are exported.  WiSED estimates the target
    derivatives internally from the field and searches open-form RHS equations.
    """
    _ = batch_size
    return make_soc_eis_impedance_manifold_dataset(
        save_dir=save_dir,
        noise_level=noise_level,
        seed=seed,
        **kwargs,
    )


SOC_EIS_IMPEDANCE_MANIFOLD_REGISTRY = {
    "soc_eis_impedance_manifold": soc_eis_impedance_manifold_dataset,
}
