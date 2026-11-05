# User guide (Streamlit app)

Run locally with `python -m streamlit run app.py` or use the [hosted app](https://multilayeranisotropictransfermatrix-ev.streamlit.app).

## 1. Build the stack

- Load the example stack (vacuum / 100 nm SiC3C / Si) or edit the layer panel.
- Layers are ordered **superstrate → substrate**. The first and last entries are semi-infinite; their thickness is ignored. Entries between them are finite layers.
- For each layer choose a material, thickness and orientation (Euler angles α, β, γ in degrees, z–x–z convention, see [theory](theory.md#euler-angles)).
- Optionally enable a Drude term (plasma frequency ωp and damping γp, cm⁻¹) to model doping.
- Define a custom material if the catalog lacks one ([materials](materials.md)).

## 2. Choose a calculation

| Calculation | Output | Axes |
|---|---|---|
| Dispersion map | `Im(rpp)` | ω (cm⁻¹) × kx |
| Isofrequency map | `Im(rpp)` | φ (azimuth) × kx at fixed ω |
| Mode trace | complex kx of the `rpp` pole | overlay on the dispersion map |
| Vertical-dipole wave map | normalized \|Ez\|² | x, y (µm) |

The horizontal axis can be shown as kx in cm⁻¹ or normalized kx/k0. Grid presets (Coarse / Normal / Fine) set the number of ω, kx and φ samples; adjust ranges manually for narrow features.

## 3. Run and inspect

- **Fast vectorised solver** (default on) uses the batched path; turn it off for the per-point reference path. Results agree to round-off.
- Interactive plots (Plotly) support zoom and hover; colour limits are adjustable.

## 4. Vertical-dipole map

Run a full 360° isofrequency sweep (the app keeps the complex `rpp` samples automatically), then set source height, observation height and FFT size. Requires a lossless dielectric superstrate such as vacuum. Increase momentum/angular sampling and FFT size to test convergence; a finite band causes ringing and periodic-FFT artifacts.

## 5. Export

Download result arrays as CSV and figures as PNG from the export controls beneath each plot.
