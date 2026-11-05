# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/); versions follow [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-05

First archived release.

### Solver
- Generalized 4 × 4 transfer-matrix formalism of Passler & Paarmann (2017) for layered anisotropic media, with Euler angles per layer and semi-infinite boundary layers.
- Two code paths: a per-point engine and a vectorised fast solver. Tests check them against each other and against an independent Berreman/Yeh reference built on a matrix exponential.
- Fast path: layer transfer matrices from `exp(−iαΔ)` without mode sorting, closed forms for isotropic and block-diagonal (β = 0) layers, and 4 × 2 block propagation. Results agree with the per-point engine to about 1e-12. Typical presets run 3–20× faster than a naive evaluation; tilted anisotropic layers about 2–3×.

### Calculations
- Dispersion map `Im(rpp)(ω, kx)`, with vectorised permittivity evaluation.
- Isofrequency map `Im(rpp)(φ, kx)`, with azimuth-symmetry reduction.
- Mode trace: complex poles of `rpp`, found with batched Newton steps over all frequencies at once and seeded from the `|rpp|` grid of the preceding dispersion map (5–13× faster after a map). Robust pole tracing with convergence flags.
- Vertical-dipole surface-wave map.
- Pole-based real-space polariton maps.

### Materials
- Built-in catalog (isotropic, uniaxial and biaxial) through the [Materials Library](https://github.com/vicentini-edoardo/Materials_Library), pinned by commit.
- Custom isotropic and diagonal-anisotropic materials (Lorentz oscillators or tabulated ε).
- Optional Drude doping per layer.

### App and package
- Streamlit app with layer-stack editor and preview, presets, a 2 × 2 plot grid with shared controls, and CSV/PNG export.
- Python API: `compute_rpp_map`, `compute_isofreq_map`, `StackSpec`, `LayerSpec`, `DopingSpec`, `material_catalog`, and `__version__`.
- Jobs shorter than process-pool start-up run serially.
- Double-click install and open launchers for macOS and Windows.

### Project
- Documentation (theory, user guide, Python API, materials, install guide, releasing) and a project website.
- Test suite with CI on Python 3.11 and 3.12.
- `CITATION.cff` and `.zenodo.json` for archiving on Zenodo.
