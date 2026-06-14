# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit application for computing optical properties of multilayer anisotropic stacks using the generalized 4×4 transfer-matrix formalism (Passler & Paarmann 2017), with [pyGTM](https://github.com/nskrypnik/pyGTM) as the numerical backend. It computes Im(rpp) dispersion maps and isofrequency diagrams.

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

**No test suite exists** — this is a research/exploration tool without automated testing.

## Architecture

Three-layer design:

### 1. Models (`src/multilayer_atm/models.py`)
Frozen dataclasses: `DopingSpec` → `LayerSpec` → `StackSpec`. These are the single source of truth for optical configurations. All computation and UI work flows through these structures.

### 2. Computation (`src/multilayer_atm/solver.py`)
Two public entry points:
- `compute_rpp_map()` — dispersion map: Im(rpp) as f(ω, kx)
- `compute_isofreq_map()` — isofrequency diagram: Im(rpp) as f(φ, kx) at fixed ω

Uses `ProcessPoolExecutor` (up to 4 workers) with serial fallback. Converts between Passler z-x-z Euler conventions and pyGTM conventions internally.

Both entry points accept `fast=True` to use the vectorised solver in
`solver_fast.py`, which evaluates an entire frequency/angle row across the kx axis
in batched NumPy (one `np.linalg.eig` over all kx, batched 4×4 inverses/matmuls)
instead of the per-point Python loop. It is numerically identical to the per-point
path (matches to round-off) and ~10-20× faster on dense grids; rows it cannot
handle fall back to the exact per-point evaluation. The GUI exposes this as the
"Fast vectorised solver" toggle (default on).

Note: pyGTM's `Layer` carries history-dependent state (a never-reset
`_useBerreman` latch and a `gamma`/`Berreman` array alias). `_layer_ai` resets the
flag and hands each evaluation a fresh `gamma` array so results are independent of
grid iteration order and parallel chunking; the batched path reproduces the same
clean per-point result.

### 3. UI (`src/multilayer_atm/ui/`)
Streamlit composition split across:
- `app.py` — top-level layout orchestration
- `layer_builder.py` — stack editing (add/remove/reorder layers, Euler angles, Drude terms)
- `material_builder.py` — built-in material selection + custom material dialogs
- `calculation_views.py` — computation controls, result rendering, PNG/CSV export
- `theme_layout.py` — responsive CSS, page config, dark/light theming

### Cross-cutting modules
- `materials.py` — catalog of 50+ Passler built-in materials, wraps pyGTM permittivities
- `custom_materials.py` — JSON-serialized custom material registry (isotropic/anisotropic-diagonal tensors)
- `plotting.py` — matplotlib pseudo-3D stack preview + plotly interactive heatmaps/polar plots
- `presets.py` — grid resolution presets (Coarse/Normal/Fine) and example stacks

## Key Conventions

- **Euler angles**: z-x-z convention (Passler/paper convention); pyGTM uses a different convention — `solver.py` handles the conversion.
- **Units**: Wavenumber (cm⁻¹) for frequency, normalized momentum kx/k0 for in-plane wavevector.
- **Streamlit state**: UI state is managed via `st.session_state`; computation results and stack configurations are stored there.
- **Custom materials** are persisted as JSON via `custom_materials.py` and survive session reloads.
