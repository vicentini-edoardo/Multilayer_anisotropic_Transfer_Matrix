# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit application for computing optical properties of multilayer anisotropic stacks using the generalized 4×4 transfer-matrix formalism (Passler & Paarmann 2017). It computes Im(rpp) dispersion maps and isofrequency diagrams with an in-house solver and the shared `Materials_Library` catalog.

## Commands

**Install dependencies:**
```bash
python3 -m pip install -r requirements.txt
# or editable install
python3 -m pip install -e .
```

**Run the app:**
```bash
python3 -m streamlit run app.py
```

The root `app.py` adds `src/` to `sys.path`, so a full package install is not required after dependencies are in place.

**Run tests:** `python3 -m pytest -q`

## Architecture

Three-layer design:

### 1. Models (`src/multilayer_atm/models.py`)
Frozen dataclasses: `DopingSpec` → `LayerSpec` → `StackSpec`. These are the single source of truth for optical configurations. All computation and UI work flows through these structures.

### 2. Computation (`src/multilayer_atm/solver.py`, `engine.py`, `solver_fast.py`)
`solver.py` exposes three public entry points and orchestrates parallelism:
- `compute_rpp_map()` — dispersion map: Im(rpp) as f(ω, kx)
- `compute_isofreq_map()` — isofrequency diagram: Im(rpp) as f(φ, kx) at fixed ω
- `compute_mode_dispersion()` — mode trace: the complex in-plane wavevector kx of the
  optical mode (the **pole** of rpp, where the p-polarised dispersion determinant
  vanishes — the bright band of the Im(rpp) map), as f(ω). Per frequency it seeds from
  the |rpp| maximum on the real kx grid and refines `1/rpp = 0` into the complex plane
  via `solver_fast.find_rpp_pole` (damped Newton). The UI overlays Re(kx) on the
  dispersion map ("Mode trace (rpp pole)" toggle). `solver_fast.find_rpp_zero` (the
  reflection-zero finder) is retained for completeness but is not the mode condition.

Uses `ProcessPoolExecutor` (up to 4 workers) with serial fallback. Converts the
Passler z-x-z Euler angles to the (theta, phi, psi) ordering the engine expects
(`passler_euler_radians`, an angle relabelling only).

The numerical engine is **in-house**:
- `engine.py` — lightweight `Layer`/`System` data containers, the Euler rotation
  matrix and the rotated permittivity tensor, plus shared constants/thresholds.
- `solver_fast.py` — the 4×4 transfer-matrix algorithm (Passler & Paarmann 2017)
  in batched NumPy. `compute_row_batched()` evaluates a whole ω/φ row across the
  kx axis at once (one `np.linalg.eig` over all kx, batched 4×4 inverses/matmuls);
  `compute_row_pointwise()` evaluates each kx sample independently. Isotropic
  layers (detected via `Layer.is_isotropic`) skip the eigen-decomposition entirely
  and use the closed-form modes `q = ±√(εμ − ζ²)` — `eig` is ~half the batched
  runtime, so this saves ~25% per isotropic layer (air boundaries are ubiquitous).

`fast=True` (default in the GUI, "Fast vectorised solver" toggle) uses the batched
path; `fast=False` uses the per-point path. They agree to round-off, and the
batched path falls back to per-point for any row it can't handle (modes that don't
split 2 forward / 2 backward). The batched path is ~10-20× faster on dense grids.

### 3. UI (`src/multilayer_atm/ui/`)
Streamlit composition split across:
- `app.py` — top-level layout orchestration
- `layer_builder.py` — stack editing (add/remove/reorder layers, Euler angles, Drude terms)
- `material_builder.py` — built-in material selection + custom material dialogs
- `calculation_views.py` — computation controls, result rendering, PNG/CSV export
- `theme_layout.py` — responsive CSS, page config, dark/light theming

### Cross-cutting modules
- `materials.py` — adapter for the shared `Materials_Library` catalog
- `custom_materials.py` — JSON-serialized custom material registry (isotropic/anisotropic-diagonal tensors)
- `plotting.py` — matplotlib pseudo-3D stack preview + plotly interactive heatmaps/polar plots
- `presets.py` — grid resolution presets (Coarse/Normal/Fine) and example stacks

## Key Conventions

- **Euler angles**: z-x-z convention (Passler/paper convention); `solver.py` maps degrees to the engine's angle ordering in radians.
- **Units**: Wavenumber (cm⁻¹) for frequency, normalized momentum kx/k0 for in-plane wavevector.
- **Streamlit state**: UI state is managed via `st.session_state`; computation results and stack configurations are stored there.
- **Custom materials** are persisted as JSON via `custom_materials.py` and survive session reloads.
