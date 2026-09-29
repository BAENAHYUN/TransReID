@echo off
rem TransReID GUI launcher: runs search_gui.py with the project .venv. Arguments are passed through
rem (e.g. run_gui.bat --config pipeline_image.yaml). ASCII only on purpose (cmd.exe reads .bat in the OEM code page).
setlocal
set "ROOT=%~dp0"
if not exist "%ROOT%.venv\Scripts\python.exe" (
    echo [TransReID] .venv not found. Run first:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
    pause
    exit /b 1
)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%ROOT%"
"%ROOT%.venv\Scripts\python.exe" "%ROOT%search_gui.py" %*
if errorlevel 1 pause
