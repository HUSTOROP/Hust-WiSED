from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the companion WiSED release data.")
    parser.add_argument("--dataset-root", required=True, help="Path to the cloned Hust-WiSED-Dataset repository.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    sib_source = dataset_root / "sib_diffusion" / "sib_diffusion_raw_field_noise0.03.npz"
    soc_source = dataset_root / "soc_eis"
    sib_target = ROOT / "data" / "dataset" / sib_source.name
    soc_target = ROOT / "engineering_validation" / "soc_eis_impedance_manifold" / "data"

    if not sib_source.is_file():
        raise FileNotFoundError(f"Missing SIB release field: {sib_source}")
    if not soc_source.is_dir():
        raise FileNotFoundError(f"Missing SOC-EIS release directory: {soc_source}")

    sib_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sib_source, sib_target)
    for battery_dir in sorted(path for path in soc_source.iterdir() if path.is_dir()):
        shutil.copytree(battery_dir, soc_target / battery_dir.name, dirs_exist_ok=True)

    print(f"Installed SIB field: {sib_target}")
    print(f"Installed SOC-EIS batteries: {soc_target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

