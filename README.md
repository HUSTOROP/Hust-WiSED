# WiSED

WiSED is a research implementation for discovering compact, open-form partial
differential equations (PDEs) from noisy spatiotemporal fields. The release
contains the framework used for the manuscript, four controlled PDE benchmarks,
and two engineering validations: spherical-particle ion diffusion and coupled
SOC-EIS impedance-manifold dynamics.

## Repository contents

- `data/`: benchmark-field generators. Large release data are installed from
  the companion dataset repository.
- `engineering_validation/`: runnable SIB and SOC-EIS experiments.
- `models/`, `src/`, `utils/`: WiSED implementation.
- `scripts/install_release_data.py`: installs benchmark and engineering data
  from the companion dataset repository.
- `REPRODUCIBILITY.md`: retained experimental settings and output locations.

## Installation

```bash
git clone https://github.com/HUSTOROP/Hust-WiSED.git
git clone https://github.com/HUSTOROP/Hust-WiSED-Dataset.git
cd Hust-WiSED
python -m venv .venv
```

Activate the virtual environment, install the pinned dependencies, and install
the companion data:

```bash
python -m pip install -r requirements.txt
cd ../Hust-WiSED-Dataset
git lfs install
git lfs pull
cd ../Hust-WiSED
python scripts/install_release_data.py --dataset-root ../Hust-WiSED-Dataset
```

On Windows PowerShell, activate the environment with
`.venv\\Scripts\\Activate.ps1`; on macOS or Linux, use `source .venv/bin/activate`.

## Reproducing the engineering examples

Run commands from the repository root. Use `cpu` when CUDA is unavailable.

```bash
# SIB: raw spherical-particle concentration field, 3% field noise
python -m engineering_validation.sib_diffusion_raw_field.run_sib_diffusion_discovery \
  --device cpu --seed 0 --noise-level 0.03 --use-cached-dataset

# SOC-EIS: B11 held-out battery experiment
python -m engineering_validation.soc_eis_impedance_manifold.run_soc_eis_b11_discovery \
  --device cpu --seed 0
```

## Data provenance

The SOC-EIS validation uses the public dataset of Mustafa *et al.* from Mendeley
Data (version 2, DOI: [10.17632/cb887gkmxw.2](https://doi.org/10.17632/cb887gkmxw.2)).
The data article is Mustafa *et al.*, *Data in Brief* **57**, 110947 (2024),
DOI: [10.1016/j.dib.2024.110947](https://doi.org/10.1016/j.dib.2024.110947).
The original authors and the CC BY 4.0 terms must be retained when reusing the
measurements. See the companion repository for the data README and license
boundaries.



