@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto not_installed
if not exist ".venv\atm-install-path" goto ready
set /p ATM_INSTALL_PATH=<.venv\atm-install-path
if /i not "%ATM_INSTALL_PATH%"=="%CD%" goto not_installed
:ready
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
set "ATM_EXIT_CODE=%ERRORLEVEL%"
pause
exit /b %ATM_EXIT_CODE%
:not_installed
echo Run Install_Multilayer_ATM_Windows.bat first to install or repair the environment.
pause
exit /b 1
