from __future__ import annotations

"""Discover an autonomous impedance-manifold law and evaluate it on B11."""

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engineering_validation.soc_eis_impedance_manifold.config import (  # noqa: E402
    DEFAULT_LOG_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_DATA_DIR,
)
from engineering_validation.soc_eis_impedance_manifold.wised_interface import (  # noqa: E402
    build_wised_overrides,
    discover_with_wised,
)

TEST_BATTERY = "B11"
VALIDATION_BATTERY = "B10"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the retained SOC-EIS impedance-manifold experiment (B11 test battery)."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory containing the public EIS files.")
    parser.add_argument("--device", default="cuda", help="WiSED device, e.g. cuda or cpu.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = build_wised_overrides(
        data_dir=args.data_dir,
        test_battery=TEST_BATTERY,
        val_battery=VALIDATION_BATTERY,
        frequency_band="full",
        n_logfreq_grid=48,
        freq_preprocess_mode="rc",
        n_rc_elements=12,
        soc_smooth_window=5,
        soc_smooth_polyorder=2,
        exclude_soc100=True,
        force_regenerate=True,
        n_train_samples=24,
        weak_window_t=5,
        weak_window_x=9,
        mask_trim_t=1,
        mask_trim_x=2,
        max_eq_len=34,
        n_epochs=30,
        n_eq_samples=72,
        pop_size=56,
        n_offspring=10,
        init_random_trials=180,
        full_refine_topk=10,
        search_context_budget=4,
    )
    discover_with_wised(
        seed=args.seed,
        device=args.device,
        results_root=Path(args.output_dir),
        logs_root=Path(args.log_dir),
        overrides=overrides,
    )
    print(f"SOC-EIS B11 run completed. Outputs: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
