@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Ambiente .venv nao encontrado.
  echo Execute instalar_app.bat primeiro.
  pause
  exit /b 1
)

call ".venv\Scripts\activate.bat"
python app.py
pause
