#!/bin/bash
# Double-click launcher: first run installs everything (no Python/Git needed), then starts the app.
set -o pipefail
cd "$(dirname "$0")" || exit 1
UV="$PWD/.uv/uv"
if [ ! -x "$UV" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | UV_UNMANAGED_INSTALL="$PWD/.uv" sh || { echo "Download failed. Check your internet connection."; read -r -p "Press Enter to close"; exit 1; }
  [ -x "$UV" ] || { echo "Download failed: uv was not installed."; read -r -p "Press Enter to close"; exit 1; }
fi
MOVED=0
if [ -f .venv/atm-install-path ] && [ "$(cat .venv/atm-install-path)" != "$PWD" ]; then
  MOVED=1
fi
if [ "$MOVED" = 1 ] || ! .venv/bin/python -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14); import multilayer_atm' >/dev/null 2>&1 \
    || ! "$UV" pip check --python .venv/bin/python >/dev/null 2>&1; then
  echo "Installing or repairing the environment (a few minutes)..."
  if [ "$MOVED" = 1 ] || ! .venv/bin/python -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14)' >/dev/null 2>&1; then
    { "$UV" venv --clear --no-python-downloads --python ">=3.11,<3.14" .venv || "$UV" venv --clear --python 3.13 .venv; } || { echo "Python setup failed."; read -r -p "Press Enter to close"; exit 1; }
  fi
  "$UV" pip install --python .venv/bin/python --override overrides.txt -e . \
    || { echo "Install failed. Run this launcher again to retry."; read -r -p "Press Enter to close"; exit 1; }
fi
printf '%s\n' "$PWD" > .venv/atm-install-path || exit 1
exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
