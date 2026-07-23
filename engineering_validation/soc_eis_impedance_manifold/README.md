# SOC-EIS impedance-manifold discovery

This example uses the public LFP EIS dataset in `data/` to test whether WiSED
can discover a coupled, executable impedance evolution law. B01-B09 supply
the discovery surfaces, B10 is used for validation and B11 is the held-out
test battery.

```text
q = 1 - SoC
x = log10(frequency / Hz)
u(q, x) = standardized Re(Z)
v(q, x) = standardized -Im(Z)
```

WiSED receives the two impedance fields and their coordinates. It estimates
derivatives internally and searches the coupled open-form equations
`u_q = F(...)` and `v_q = G(...)`; SoC is an ordering coordinate, not a scalar
right-hand-side feature.

Run the retained B11 experiment from `code/`:

```powershell
python -m engineering_validation.soc_eis_impedance_manifold.run_soc_eis_b11_discovery `
  --device cpu --seed 0
```

New outputs are written to `code/outputs/soc_eis_b11/noise_0__seed_0/`. Dataset provenance,
license, citation and acknowledgment are provided in `data/README.md`.
