# Companion data layout

The engineering validations deliberately keep data separate from source code.
Clone or download
[`Hust-WiSED-Dataset`](https://github.com/HUSTOROP/Hust-WiSED-Dataset), then
run:

```bash
python scripts/install_release_data.py --dataset-root ../Hust-WiSED-Dataset
```

The installation script expects this minimal companion layout:

```text
Hust-WiSED-Dataset/
  sib_diffusion/sib_diffusion_raw_field_noise0.03.npz
  soc_eis/B01/ ... B11/
  figure_data/
```

The SIB `.npz` file is a generated simulation field used by the retained
noise-robustness experiment. The SOC-EIS directory contains the public
third-party measurements; cite their Mendeley Data record and data article as
specified in the dataset repository.

