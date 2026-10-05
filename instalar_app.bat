@echo off
setlocal
cd /d "%~dp0"

echo Criando ambiente da interface/Kokoro...
py -3.11 -m venv .venv
call ".venv\Scripts\activate.bat"

python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

echo.
echo Instalacao da interface concluida.
echo Ainda e necessario instalar:
echo 1. eSpeak-NG no Windows
echo 2. FFmpeg
echo 3. SadTalker e seus checkpoints 512 em ambiente separado (motor Standard)
echo 4. Opcional: EchoMimic em ambiente separado (motor Experimental HD)
echo Com GPU NVIDIA: troque o PyTorch do SadTalker pela versao CUDA (README, secao GPU).
echo Veja o README.md.
echo.
pause
