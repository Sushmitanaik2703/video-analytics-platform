@echo off
echo ========================================================
echo  Starting Smart CCTV Monitoring System (Live UI)
echo ========================================================
cd /d "%~dp0"
if exist ".venv-gpu\Scripts\python.exe" (
    ".venv-gpu\Scripts\python.exe" app.py
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    python app.py
)
pause
