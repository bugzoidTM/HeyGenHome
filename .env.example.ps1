# Exemplo (PowerShell). Rode antes de iniciar o app; só é preciso o que difere do padrão.

# Motor Standard (SadTalkerNext)
# $env:SADTALKER_DIR="C:\AI\SadTalker"
# $env:SADTALKER_PYTHON="C:\AI\SadTalker\venv\Scripts\python.exe"
# $env:SADTALKER_DEVICE="auto"     # auto | cpu | cuda | cuda:N
# $env:SADTALKER_FP16="auto"       # auto | on | off
# $env:SADTALKER_BATCH="0"         # 0 = automático pela VRAM

# Motor Experimental HD (EchoMimic)
# $env:ECHOMIMIC_DIR="C:\AI\EchoMimic"
# $env:ECHOMIMIC_PYTHON="C:\AI\EchoMimic\venv\Scripts\python.exe"
# $env:ECHOMIMIC_DEVICE="auto"     # auto usa GPU de 12 GB+
# $env:ECHOMIMIC_ACCELERATED="1"

# Geral
# $env:HEYGEN_MAX_SIDE="1920"     # 0 = nunca reduzir a foto
# $env:HEYGEN_CPU_THREADS="4"
# $env:HEYGEN_SEED="42"           # muda o movimento gerado para o mesmo texto
