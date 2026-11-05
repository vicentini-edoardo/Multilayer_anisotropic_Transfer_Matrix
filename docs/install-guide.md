# Install and run Multilayer ATM locally

## Quick start (no Python or Git needed)

1. On GitHub click **Code → Download ZIP** and unzip it.
2. Double-click **Install_Multilayer_ATM_Windows.bat** (Windows) or **Install_Multilayer_ATM_macOS.command** (macOS).
   - macOS first time: the launcher may be blocked. Click **Done**, open **System Settings → Privacy & Security**, scroll to **Security**, click **Open Anyway** next to the launcher and confirm. On older macOS: right-click → Open.
   - Windows first time: if SmartScreen shows "Windows protected your PC", click **More info** → **Run anyway**.
3. The installer uses an installed Python 3.11–3.13 if found (otherwise downloads 3.13) and downloads the packages (a few minutes, internet needed). It then starts the app.
4. Your browser opens at <http://localhost:8501>. Close the terminal window to stop the app.

For later launches, double-click **Open_Multilayer_ATM_Windows.bat** or **Open_Multilayer_ATM_macOS.command**. Open only starts Streamlit and installs nothing.

The app binds to `127.0.0.1`, so it is reachable only from your computer.

## Moving, updating, repairing

- Keep the whole folder together. After moving it, run the Install launcher once to repair `.venv` (internet needed).
- To update, download the new ZIP, or `git pull`, then run the Install launcher again. Deleting `.venv` forces a clean reinstall.
- The Materials Library is pinned to a tested commit (`overrides.txt` for the launchers, `pyproject.toml` for pip). The launchers need no Git because `overrides.txt` points to the zip of that commit.

## Manual install

Requires Python 3.11 or newer and Git.

```bash
git clone https://github.com/vicentini-edoardo/Multilayer_anisotropic_Transfer_Matrix.git
cd Multilayer_anisotropic_Transfer_Matrix
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m streamlit run app.py
```

Check the install with `python -m pytest -q` (needs the test extras if declared in `pyproject.toml`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Download failed" | Check network/proxy access to astral.sh and github.com, then re-run Install. |
| "Install failed" | Re-run Install to retry. |
| Open says to run Install | `.venv` is missing or the folder moved. Run Install. |
| Port 8501 busy | Close the other Streamlit instance, or run `python -m streamlit run app.py --server.port 8502`. |
