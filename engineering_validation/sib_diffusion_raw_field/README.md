# SIB spherical-diffusion discovery

This engineering example asks WiSED to recover the radial diffusion law from a
raw concentration field, without the `u = xC` transformation or a manually
constructed candidate library:

```text
input:  u(tau, x), tau and x
target: u_tau = u_xx + 2 u_x / x
```

The retained setting uses 3% field noise, one boundary-flux condition
(`delta = 0.25`), seven initial concentration profiles and weak-form scoring.
The simulator applies `0 <= u <= 1` clipping after each numerical step.

Run from `code/`:

```powershell
python -m engineering_validation.sib_diffusion_raw_field.run_sib_diffusion_discovery `
  --device cuda --seed 0 --noise-level 0.03 --use-cached-dataset
```

Outputs are written to `code/outputs/sib_diffusion/noise_0.03__seed_0/`. The downstream analysis evaluates
the WiSED-selected equation by field reconstruction over the discovery window
and future rollout from the final observed field.
