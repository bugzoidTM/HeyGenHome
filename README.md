# HeyGen caseiro — CPU / Windows

MVP local para transformar:

**foto + texto -> voz em português brasileiro -> avatar falante -> MP4**

## Arquitetura

- Interface: Gradio
- TTS: Kokoro-82M
- Português brasileiro: `lang_code="p"`
- Vozes incluídas no app: `pf_dora`, `pm_alex`, `pm_santa`
- Animação: SadTalker
- Execução do avatar: CPU (`--cpu`)
- Áudio/vídeo: FFmpeg + SoundFile

O Kokoro e o SadTalker ficam separados porque o SadTalker usa dependências antigas.
Isso reduz conflitos de Python/PyTorch.

## 1. Pré-requisitos

Instale no Windows:

1. Git
2. Python 3.11 (interface/Kokoro)
3. Python 3.8 (SadTalker)
4. FFmpeg no PATH
5. eSpeak-NG no PATH

Teste:

```powershell
ffmpeg -version
espeak-ng --version
py -3.11 --version
py -3.8 --version
```

## 2. Instalar a interface e Kokoro

Dê dois cliques em:

`instalar_app.bat`

Ou manualmente:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 3. Instalar SadTalker separado

Exemplo de pasta:

`C:\AI\SadTalker`

Clone o repositório oficial:

```powershell
cd C:\AI
git clone https://github.com/OpenTalker/SadTalker.git
cd SadTalker
py -3.8 -m venv venv
.\venv\Scripts\Activate.ps1
```

A instalação do SadTalker pode exigir versões específicas das dependências.
Siga o README oficial do projeto para instalar requirements e baixar os checkpoints.

Confirme que este arquivo existe:

`C:\AI\SadTalker\inference.py`

E que os modelos/checkpoints foram colocados nas pastas solicitadas pelo projeto.

## 4. Configurar caminhos

O `app.py` usa por padrão:

- `C:\AI\SadTalker`
- `C:\AI\SadTalker\venv\Scripts\python.exe`

Se usar outro caminho, configure no PowerShell antes de iniciar:

```powershell
$env:SADTALKER_DIR="D:\IA\SadTalker"
$env:SADTALKER_PYTHON="D:\IA\SadTalker\venv\Scripts\python.exe"
```

## 5. Rodar

Dê dois cliques em:

`iniciar.bat`

Ou:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

A interface abre em:

`http://127.0.0.1:7860`

## 6. Primeiro teste

Use:
- foto frontal;
- rosto bem iluminado;
- fundo simples;
- texto curto (1 a 3 frases);
- modo "Apresentador (mais estável)".

Em CPU, a animação é o gargalo. Para vídeos longos, gere blocos curtos
e depois concatene com FFmpeg.

## 7. Próxima evolução recomendada

Depois do MVP funcional:

1. histórico de avatares;
2. geração em segmentos;
3. legendas automáticas;
4. fundo 9:16 para TikTok/Reels;
5. fila de geração;
6. presets de voz;
7. API REST;
8. n8n;
9. opção de usar GPU remota sem alterar a interface;
10. trocar apenas o motor de animação por um modelo melhor no futuro.

## Observação

Use imagens e vozes próprias ou com autorização. Se o sistema for oferecido a
terceiros, inclua consentimento, política de privacidade e mecanismos para evitar
uso enganoso da imagem de outras pessoas.
