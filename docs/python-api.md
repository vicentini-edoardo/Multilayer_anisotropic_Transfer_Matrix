# Python API

```python
from multilayer_atm import (
    DopingSpec, LayerSpec, StackSpec,
    compute_rpp_map, compute_isofreq_map,
    material_catalog, material_notes, axes_for_material,
)
from multilayer_atm.solver import compute_mode_dispersion
from multilayer_atm.surface_wave import compute_surface_wave
```

Units: wavenumbers and in-plane wavevectors in **cm⁻¹**, thickness in **metres**, Euler angles in **degrees** (convert internally).

## Models (`multilayer_atm.models`)

Frozen dataclasses.

- `DopingSpec(enabled=False, wp_cm1=0.0, gp_cm1=0.0)`: Drude term.
- `LayerSpec(material, thickness_m, euler_deg=(0, 0, 0), doping=DopingSpec())`.
- `StackSpec.from_layers(layers)`: ordered superstrate → substrate. The first and last layers are forced to be semi-infinite (`enforce_boundary_layers`).

`material` is a name from `material_catalog()` or a custom material name passed through `custom_materials=`.

## Solvers (`multilayer_atm.solver`)

Common keyword arguments: `workers` (process count, `None` = automatic, `1` = serial), `fast` (vectorised solver), `custom_materials` (registry mapping), `progress` (callback).

### `compute_rpp_map(stack, w_min, w_max, nw, kx_min, kx_max, nk, ...)`

Returns `(w_cm1, kx_cm1, im_rpp)`; `im_rpp` has shape `(nw, nk)`.

### `compute_isofreq_map(stack, w0, kx_min, kx_max, nk, nphi, phi_min_deg=0, phi_max_deg=360, ..., return_complex=False)`

Returns `(phi_rad, kx_cm1, values)`; `values` has shape `(nphi, nk)`. φ is sampled with `endpoint=False`. `return_complex=True` returns complex `rpp` (needed for the dipole map).

### `compute_mode_dispersion(stack, w_min, w_max, nw, kx_min, kx_max, nk, ...)`

Returns `(w_values, kx_mode_cm1, residual, converged)`. `kx_mode_cm1` is complex: real part is the mode wavevector, imaginary part the modal loss. Drop frequencies where `converged` is false.

## Surface-wave map (`multilayer_atm.surface_wave`)

```python
phi, k, rpp = compute_isofreq_map(stack, 900, 0, 20000, 256, 128, return_complex=True)
x_um, ez, intensity = compute_surface_wave(phi, k, rpp, w0_cm1=900,
                                           source_height_nm=25, fft_size=1024)
```

Requires a full 360° sweep without a repeated endpoint and a lossless dielectric superstrate. See [theory](theory.md#vertical-dipole-surface-wave-map) for assumptions.

## Example: loop over a Drude-doped layer

```python
stack = StackSpec.from_layers([
    LayerSpec("vacuum", 0.0),
    LayerSpec("SiC3C", 100e-9, doping=DopingSpec(True, wp_cm1=200, gp_cm1=50)),
    LayerSpec("Si", 0.0),
])
w, kx, im_rpp = compute_rpp_map(stack, 700, 1100, 160, 0, 5000, 180, fast=True)
```

Authoritative signatures live in [`src/multilayer_atm/solver.py`](../src/multilayer_atm/solver.py).
