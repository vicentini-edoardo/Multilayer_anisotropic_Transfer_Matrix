#!/bin/bash
# Start the installed app; installation and repairs belong to the Install launcher.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ] || { [ -f .venv/atm-install-path ] && [ "$(cat .venv/atm-install-path)" != "$PWD" ]; }; then
  echo "Run Install_Multilayer_ATM_macOS.command first to install or repair the environment."
  read -r -p "Press Enter to close"
  exit 1
fi
exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
