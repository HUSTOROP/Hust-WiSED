# Reproducibility guide

## Scope of this release

This repository contains the WiSED framework, retained benchmark generators and
the two engineering-validation entry points used in the manuscript. The associated numerical source data and
public raw SOC-EIS measurements are maintained in
[`Hust-WiSED-Dataset`](https://github.com/HUSTOROP/Hust-WiSED-Dataset).

## Data installation

After cloning both repositories, run:

```bash
python scripts/install_release_data.py --dataset-root ../Hust-WiSED-Dataset
```

The command installs the 28 controlled benchmark archives and the SIB cached
field in `data/dataset/`, and places the public SOC-EIS measurements in
`engineering_validation/soc_eis_impedance_manifold/data/`. It does not modify
any tracked source file.

The benchmark archives are stored with Git LFS in the dataset repository. Run
`git lfs pull` there before invoking the installer.

## Retained experiments

### SIB spherical diffusion

The SIB example receives the raw radial concentration field `u(tau, x)` with
the normalized time and radius coordinates. WiSED uses weak-form scoring to
discover a right-hand side for the radial diffusion equation. The retained
configuration uses one boundary-flux condition (`delta = 0.25`), seven initial
concentration profiles, `tau_max = 1.0`, a 3% additive field-noise level, and
seed 0. The numerical simulator clips the concentration to `[0, 1]` after each
time step.

### SOC-EIS impedance manifold

The SOC-EIS example represents the measured impedance as coupled fields over
discharge progress and log-frequency:

```text
q = 1 - SoC
x = log10(frequency / Hz)
u(q, x) = standardized Re(Z)
v(q, x) = standardized -Im(Z)
```

B01-B09 supply discovery surfaces, B10 is used for validation, and B11 is the
retained held-out test battery. WiSED estimates field derivatives internally
and searches coupled evolution equations for `u_q` and `v_q`.

## Computational environment

The dependency versions are pinned in `requirements.txt`. The experiments can
run on CPU; CUDA is optional and changes runtime but not the documented
experimental settings. Set the seed explicitly in every command when comparing
runs.

