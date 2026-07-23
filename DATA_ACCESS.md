# Companion data layout

Clone or download
[`Hust-WiSED-Dataset`](https://github.com/HUSTOROP/Hust-WiSED-Dataset), then
run:

```bash
python scripts/install_release_data.py --dataset-root ../Hust-WiSED-Dataset
```

The installation script expects this minimal companion layout:

```text
Hust-WiSED-Dataset/
  pde_benchmarks/burgers_1d_noise0.0.npz ... fhn_2d_noise0.1.npz
  sib_diffusion/sib_diffusion_raw_field_noise0.03.npz
  soc_eis/B01/ ... B11/
```

The 28 `pde_benchmarks/` archives contain the controlled numerical fields used
by the benchmark suite and are tracked with Git LFS. Run `git lfs pull` inside
the dataset repository before installing the data. The SIB `.npz` file is a
generated simulation field used by the retained noise-robustness experiment.
The SOC-EIS directory contains public third-party measurements; cite their
Mendeley Data record and data article as specified in the dataset repository.

