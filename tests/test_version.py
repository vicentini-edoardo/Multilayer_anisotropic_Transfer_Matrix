import tomllib
from pathlib import Path

import multilayer_atm


def test_version_matches_pyproject():
    root = Path(__file__).resolve().parents[1]
    declared = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    assert multilayer_atm.__version__ == declared
