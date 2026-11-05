# Development

```bash
python -m pip install -e .
python -m pip install pytest
python -m pytest -q
```

## Layout

```
app.py                      Streamlit entry point (adds src/ to sys.path)
src/multilayer_atm/
  models.py                 DopingSpec, LayerSpec, StackSpec (frozen dataclasses)
  solver.py                 public entry points + process-pool orchestration
  solver_fast.py            batched 4x4 transfer-matrix algorithm, pole finder
  engine.py                 Layer/System containers, Euler rotation, constants
  surface_wave.py           vertical-dipole field reconstruction
  materials.py              adapter to Materials_Library
  custom_materials.py       custom material registry and JSON IO
  plotting.py, presets.py
  ui/                       Streamlit layout, layer/material builders, results views
tests/                      pytest suite
```

Dependencies flow models → solver/engine → UI. Computation never imports Streamlit.

## Conventions

- Keep units: cm⁻¹ for frequency, metres for thickness, degrees at the API boundary.
- The batched and per-point solvers must agree to round-off; add a comparison test for any solver change.
- UI state lives in `st.session_state`.

## Contributing

See [`CONTRIBUTING.md`](../CONTRIBUTING.md).
