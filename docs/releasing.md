# Releasing and archiving on Zenodo

Zenodo archives each GitHub *release* and mints a DOI. Metadata comes from [`.zenodo.json`](../.zenodo.json) (which takes precedence over [`CITATION.cff`](../CITATION.cff)); keep the two consistent.

## One-time setup

1. Log in to [zenodo.org](https://zenodo.org) with GitHub.
2. Open *Account → GitHub*, find `Multilayer_anisotropic_Transfer_Matrix` and switch it **on**.

## Each release

1. Bump `version` in `pyproject.toml` and `CITATION.cff`; set `date-released` in `CITATION.cff`.
2. Move the *Unreleased* entries in `CHANGELOG.md` under the new version and date.
3. `python -m pytest -q` passes; commit to `main`.
4. Tag and publish a GitHub release:
   ```bash
   git tag -a v0.1.0 -m "v0.1.0" && git push origin v0.1.0
   gh release create v0.1.0 --title "v0.1.0" --notes-file CHANGELOG.md
   ```
5. Zenodo builds the record within minutes. It creates a concept DOI (all versions) and a version DOI.
6. The concept DOI (all versions) is `10.5281/zenodo.23981616`; it is already in `README.md`, `docs/index.html`, `CITATION.cff` and `ZENODO_DOI_URL` in `src/multilayer_atm/ui/theme_layout.py`. Add the version DOI of each new release to `CHANGELOG.md` if wanted.

## Notes

- The release archive is the repository snapshot at the tag. The pinned Materials Library dependency is fetched from GitHub at install time and is not bundled; cite it separately if you rely on specific materials.
- Edit published Zenodo metadata on zenodo.org if needed; file contents are immutable per version.
