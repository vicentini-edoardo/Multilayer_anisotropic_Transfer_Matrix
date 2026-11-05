# Materials

## Built-in catalog

Built-in materials come from the [Materials Library](https://github.com/vicentini-edoardo/Materials_Library), pinned by commit in `pyproject.toml` and `requirements.txt`. `material_catalog()` lists the entries the solver supports (isotropic, uniaxial and biaxial; `pending` entries are excluded). `material_notes()` returns caveats for models that are not fully verified.

```python
from multilayer_atm import material_catalog, material_notes
print(material_catalog())
```

## Custom materials

Defined in the app (or in code) as isotropic or diagonal-anisotropic (`xx`, `yy`, `zz` axes). Each axis is either

- **Lorentz model:** `eps_inf` plus oscillators with `strength`, `w0_cm1`, `gamma_cm1`, or
- **Tabulated ε:** `frequency_cm1`, `eps_real`, `eps_imag`, linearly interpolated and held constant outside the table range.

### JSON interchange

Materials export/import as JSON with the top-level keys `name`, `tensor_mode` (`isotropic` or `anisotropic_diagonal`) and `axes` (mapping `xx`/`yy`/`zz` to an axis definition). Use the app's export button to obtain a template. Registries persist across sessions through `custom_materials.py`.

### Table format

Plain text, three columns, `#` comments allowed, whitespace or comma delimited:

```
# frequency_cm1  Re(eps)  Im(eps)
700   -3.1   0.8
710   -2.7   0.7
```

Frequencies must be strictly increasing, with at least two rows.
