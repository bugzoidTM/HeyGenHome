# HeyGen caseiro — Windows (CPU, com GPU NVIDIA opcional)

MVP local para transformar:

**foto + texto -> voz em português brasileiro -> avatar falante -> MP4**

## Arquitetura

```
                    HeyGenHome  (interface Gradio, Kokoro, exportação)
                        │
                 TalkingHeadEngine
                        │
          ┌─────────────┼──────────────┐
          ↓             ↓              ↓
     SadTalkerNext   EchoMimic       futuro
      "Standard"   "Experimental HD"
```

- Interface: Gradio
- TTS: Kokoro-82M, português brasileiro (`lang_code="p"`), vozes `pf_dora`, `pm_alex`, `pm_santa`
- Motor do avatar: escolhido na interface (**MOTOR DO AVATAR**), atrás de uma interface única
  (`heygenhome/engines`). Trocar ou adicionar motor não mexe em Kokoro, interface ou exportação.
- Execução do avatar: GPU NVIDIA automaticamente quando disponível (a CPU compõe e
  codifica o vídeo em paralelo); senão, só CPU
- Áudio/vídeo: FFmpeg + SoundFile

Cada motor roda no seu próprio ambiente Python (eles usam dependências antigas e
incompatíveis entre si). O HeyGenHome só chama o Python de cada um.

| Pasta/arquivo | Papel |
|---|---|
| `app.py` | Interface Gradio |
| `heygenhome/tts.py` | Voz (Kokoro) |
| `heygenhome/engines/engine.py` | `TalkingHeadEngine`: prepara a foto, chama o motor e valida o vídeo |
| `heygenhome/engines/sadtalker_next.py` | Motor **Standard** (SadTalkerNext) |
| `heygenhome/engines/sadtalker_runner.py` | O "fork melhorado": roda dentro do ambiente do SadTalker oficial |
| `heygenhome/engines/echomimic.py` | Motor **Experimental HD** (EchoMimic) |
| `heygenhome/engines/echomimic_runner.py` | Roda dentro do ambiente do EchoMimic (CPU/GPU) |
| `heygenhome/engines/gpu.py` | Detecta a GPU e mostra na interface se ela será usada |
| `heygenhome/media.py` | Preparo da foto, leitura do MP4 (ffprobe), colagem do rosto |
| `heygenhome/export.py` | Exportação Original / 9:16 e validação final |

## Motores

| | Standard — SadTalkerNext | Experimental HD — EchoMimic |
|---|---|---|
| Base | [SadTalker](https://github.com/OpenTalker/SadTalker) oficial | [EchoMimic v1](https://github.com/antgroup/echomimic) (difusão) |
| CPU (medido, 4 núcleos) | ~15 s por quadro a 512 (~6 min por segundo de fala); ~3,5 s a 256 | ~14 min por segundo de vídeo |
| GPU NVIDIA (estimativa, não medida) | ~1–3 min para 7,5 s de fala numa RTX 3060 | precisa de ~12 GB de VRAM |
| RAM (pico medido) | não medido | ~13 GB |
| Resolução do rosto | 512 (padrão) ou 256 | 512 |
| Foto inteira | sim (cola o rosto na foto original) | sim (o HeyGenHome cola o rosto 512x512 de volta na foto) |
| Movimento (Estável / Natural / Expressivo) | sim | vem do próprio modelo |
| Melhorar rosto (GFPGAN) | opcional (só no rosto, sem mudar a resolução) | não se aplica |

### SadTalkerNext: o "fork melhorado" do SadTalker

O motor Standard usa o repositório oficial <https://github.com/OpenTalker/SadTalker>
(modelos e código) sem modificá-lo: o `sadtalker_runner.py` roda com o Python do
SadTalker no lugar do `inference.py` e muda o que fazia o vídeo sair "parado" e lento:

- **Cabeça:** o `--still` do SadTalker congela toda a pose da cabeça. O runner usa o
  movimento previsto pelo áudio, suavizado e normalizado para uma amplitude-alvo
  (preset), aplicado direto aos ângulos do renderizador, com a translação congelada
  (o pescoço não "descola"). Começa e termina na pose da foto.
- **Lábios:** ganho relativo à expressão da própria foto (em silêncio o rosto é o
  da foto; ao falar, a boca abre mais).
- **Piscadas:** curva natural (fecha rápido, abre devagar, ~3 s entre piscadas),
  aplicada na direção exata que o modelo usa para piscar.
- **Colagem:** recorte quadrado próprio (corrige bugs do SadTalker com rostos grandes
  ou perto da borda), máscara elíptica suave em vez do `seamlessClone` retangular,
  correção de cor fixa e **uma única** codificação H.264 (o SadTalker fazia 4, com perdas).
- **Desempenho:** GPU automática (fp16 nas placas com tensor cores, lote ajustado à
  VRAM, volta para CPU se faltar memória), quadros em ordem saindo da GPU enquanto a
  CPU compõe e codifica, foto codificada uma vez só, lote 1 em CPU.
- **Compatibilidade:** funciona com o PyTorch 1.12 do README do SadTalker e com
  PyTorch 2.x (inclui os ajustes para Pillow 10, torchvision e numpy novos) e com
  pastas com acento no Windows (ex.: `C:\Users\João\...`).

Medido com a mesma foto (1434x1920) e a mesma fala de 7,5 s, antes (`--still`) e
depois (preset Natural), por marcos faciais: movimento horizontal da cabeça de 5,7 para
22,9 px, abertura média da boca de 10,3 para 20,1 px, e 2 piscadas onde antes não havia
nenhuma. A saída continua com a resolução exata da foto.

## Padrões da interface

- **Resolução do rosto: 512** (256 em "Mais ajustes", ~4x mais rápido em CPU).
- **Enquadramento: foto inteira.** O recorte do rosto só quando escolhido ("Só o rosto (recorte)").
- **Movimento: Natural** — cabeça se mexe de leve, lábios marcados e piscadas.
  *Estável* deixa só rosto e lábios (como o antigo `--still`); *Expressivo* mexe mais.
- **Resolução e proporção da foto preservadas**: a foto é girada pelo EXIF, convertida
  para PNG e, se tiver largura/altura ímpar, ganha 1 px de borda (H.264 exige pares).
  Nunca é cortada. Fotos com lado maior que 1920 px são reduzidas proporcionalmente
  para não estourar RAM/tempo em CPU (`HEYGEN_MAX_SIDE`, `0` desliga).
- **Formatos sem corte**:
  - *Original*: mesma resolução da foto (ou do recorte, no modo recorte);
  - *9:16 vertical*: 1080x1920 com o vídeo inteiro encaixado; a sobra é preenchida
    com fundo desfocado (padrão) ou preto.
- **Melhorar rosto (GFPGAN)**: desligado por padrão (em CPU deixa bem mais lento).
- **Player com `object-fit: contain`**: o vídeo e a prévia da foto aparecem inteiros.

## Validação do MP4

Depois de cada vídeo, o `ffprobe` confere:

1. **Saída do motor** — no modo foto inteira, a resolução/proporção deve ser a da foto.
2. **MP4 final** — resolução exata do formato escolhido, H.264 `yuv420p`, áudio AAC.
   Se não bater, a geração falha com a explicação em vez de entregar um vídeo errado.

O resultado aparece no **Relatório** da interface (inclui onde rodou — GPU ou CPU — e
os segundos por quadro) e em `outputs/<data>-<motor>-xxxx/relatorio.md`, junto com
`voz.wav`, `foto.png`, o MP4 final e o log do motor.

O MP4 final usa o áudio original do Kokoro (24 kHz), e não o áudio reamostrado
para 16 kHz pelo SadTalker.

## 1. Pré-requisitos

Instale no Windows:

1. Git
2. Python 3.11 (interface/Kokoro)
3. Python 3.8 (SadTalker) — ou 3.10 se tiver uma RTX 50xx — e, se for usar o
   Experimental HD, Python 3.10 (EchoMimic)
4. FFmpeg no PATH (inclui o `ffprobe`)
5. eSpeak-NG no PATH

Teste:

```powershell
ffmpeg -version
ffprobe -version
espeak-ng --version
py -3.11 --version
py -3.8 --version
py -3.10 --version   # só para o EchoMimic
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

## 3. Instalar o motor Standard (SadTalker)

Exemplo de pasta: `C:\AI\SadTalker`

```powershell
cd C:\AI
git clone https://github.com/OpenTalker/SadTalker.git
cd SadTalker
py -3.8 -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch==1.12.1+cpu torchvision==0.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
# a interface web do SadTalker (gradio) não é usada pelo HeyGenHome
(Get-Content requirements.txt) | Where-Object { $_ -notmatch '^gradio' } | Set-Content requirements-cpu.txt
pip install -r requirements-cpu.txt
```

Checkpoints (pasta `C:\AI\SadTalker\checkpoints`), de
<https://github.com/OpenTalker/SadTalker/releases/tag/v0.0.2-rc>:

- `SadTalker_V0.0.2_512.safetensors` (**obrigatório para o padrão 512**)
- `SadTalker_V0.0.2_256.safetensors` (só se for usar 256)
- `mapping_00109-model.pth.tar`

E os modelos de detecção de rosto em `C:\AI\SadTalker\gfpgan\weights` (veja
`scripts/download_models.sh` do SadTalker). O GFPGAN é baixado sozinho no primeiro uso
de "Melhorar rosto".

### GPU NVIDIA (recomendado: de ~50 min para poucos minutos)

O HeyGenHome usa a GPU sozinho quando o PyTorch **do ambiente do motor** tem CUDA.
Abaixo de **MOTOR DO AVATAR** a interface mostra se a GPU será usada ou por que não.
O `instalar_app.bat` e os comandos acima instalam a versão só-CPU; para usar a GPU,
troque o PyTorch do SadTalker (driver NVIDIA atualizado, Game Ready ou Studio):

GTX 10xx até RTX 40xx — no mesmo `venv` Python 3.8:

```powershell
cd C:\AI\SadTalker
.\venv\Scripts\python -m pip uninstall -y torch torchvision
.\venv\Scripts\python -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
.\venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

RTX 50xx — exige PyTorch 2.7+ (CUDA 12.8), que não existe para Python 3.8. Crie um
segundo ambiente do SadTalker, com Python 3.10, e aponte o HeyGenHome para ele:

```powershell
cd C:\AI\SadTalker
py -3.10 -m venv venv310
.\venv310\Scripts\python -m pip install "setuptools<81"   # o librosa do SadTalker usa pkg_resources
.\venv310\Scripts\python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
(Get-Content requirements.txt) | Where-Object { $_ -notmatch '^gradio' } | Set-Content requirements-cpu.txt
.\venv310\Scripts\python -m pip install -r requirements-cpu.txt
# grava a variável para o usuário (vale para o iniciar.bat); feche e abra o terminal depois
[Environment]::SetEnvironmentVariable("SADTALKER_PYTHON", "C:\AI\SadTalker\venv310\Scripts\python.exe", "User")
```

(O runner já traz os ajustes que o SadTalker precisa para PyTorch 2.x. Testado aqui com
Python 3.10 + PyTorch 2.7.1 em CPU; a variante CUDA 12.8 não pôde ser testada sem GPU.)

Dicas:
- **AMD/Intel:** não suportadas por este motor (o DirectML não roda as convoluções 3D
  do SadTalker); ele usa a CPU.
- **Pouca VRAM** (4–6 GB): funciona com lote 1; se faltar memória o runner diminui o
  lote, tenta fp16 e, por fim, continua em CPU sem perder o que já renderizou.
- No **Painel de Controle NVIDIA**, em "CUDA - Política de fallback da memória do
  sistema", escolha "Preferir sem fallback" para o `python.exe` do SadTalker: senão o
  driver pode usar a RAM do PC no lugar da VRAM e ficar muito lento sem avisar.
- Se a GPU der erro no meio, o vídeo é refeito automaticamente em CPU e o relatório avisa.

## 4. (Opcional) Instalar o motor Experimental HD (EchoMimic)

Exemplo de pasta: `C:\AI\EchoMimic`

```powershell
cd C:\AI
git clone https://github.com/antgroup/echomimic.git EchoMimic
cd EchoMimic
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
# sem gradio (não usado) e com versões que funcionam com o diffusers 0.24 do EchoMimic
(Get-Content requirements.txt) | Where-Object { $_ -notmatch '^(gradio|av==)' } | Set-Content requirements-cpu.txt
pip install -r requirements-cpu.txt "av<13" "numpy<2" "huggingface_hub==0.25.2" "transformers==4.46.3" "accelerate<1"
```

Por que os ajustes: o `diffusers==0.24.0` do EchoMimic não importa com o
`huggingface_hub` atual; o PyTorch 2.2.2 não funciona com NumPy 2; o
`av==11.0.0` pode não ter instalador pronto para o seu Python; e o `accelerate`
reduz o pico de memória ao carregar os modelos.

Pesos (~12 GB; o `git clone` do repositório inteiro tem 34 GB e inclui modelos que
o modo CPU não usa):

```powershell
huggingface-cli download BadToBest/EchoMimic `
  face_locator.pth denoising_unet_acc.pth motion_module_acc.pth reference_unet.pth `
  audio_processor/whisper_tiny.pt `
  sd-image-variations-diffusers/unet/config.json sd-image-variations-diffusers/unet/diffusion_pytorch_model.bin `
  sd-vae-ft-mse/config.json sd-vae-ft-mse/diffusion_pytorch_model.safetensors `
  --local-dir pretrained_weights
```

Com uma GPU NVIDIA de 12 GB ou mais, troque o PyTorch por
`torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121`
(`--extra-index-url https://download.pytorch.org/whl/cu121`): o EchoMimic passa a usar a
GPU sozinho, em fp16. Sem GPU também funciona, porque o HeyGenHome usa o próprio
`echomimic_runner.py` (o `infer_audio2vid.py` oficial força CUDA). Em CPU usa os pesos
acelerados (`*_acc.pth`, 6 passos) e precisa de ~13 GB de RAM (máquina com 16 GB) —
mesmo assim é lento.
Com `ECHOMIMIC_ACCELERATED=0` baixe também `denoising_unet.pth` e `motion_module.pth`.

## 5. Configurar caminhos

O `app.py` usa por padrão `C:\AI\SadTalker` e `C:\AI\EchoMimic`, com o Python em
`venv\Scripts\python.exe` dentro de cada pasta. Para mudar, configure no PowerShell
antes de iniciar (veja `.env.example.ps1`). Variáveis definidas com `$env:` valem só
naquela janela; para valerem também no `iniciar.bat`, grave-as para o usuário e abra um
novo terminal, por exemplo:
`[Environment]::SetEnvironmentVariable("SADTALKER_DIR", "D:\IA\SadTalker", "User")`.

| Variável | Padrão | Uso |
|---|---|---|
| `SADTALKER_DIR` | `C:\AI\SadTalker` | Pasta do SadTalker |
| `SADTALKER_PYTHON` | `<SADTALKER_DIR>\venv\Scripts\python.exe` | Python do SadTalker |
| `SADTALKER_DEVICE` | `auto` | `auto` (GPU se houver), `cpu`, `cuda` ou `cuda:N` |
| `SADTALKER_FP16` | `auto` | `auto` (só GPUs com tensor cores), `on`, `off` |
| `SADTALKER_BATCH` | `0` | Quadros por lote na GPU (`0` = pela VRAM livre) |
| `SADTALKER_POSE_STYLE` | `0` | Estilo de movimento da cabeça do SadTalker (0–45) |
| `HEYGEN_SEED` | `42` | Semente: mesmo texto e foto geram o mesmo movimento |
| `ECHOMIMIC_DIR` | `C:\AI\EchoMimic` | Pasta do EchoMimic |
| `ECHOMIMIC_PYTHON` | `<ECHOMIMIC_DIR>\venv\Scripts\python.exe` | Python do EchoMimic |
| `ECHOMIMIC_DEVICE` | `auto` | `auto` (GPU de 12 GB+), `cpu`, `cuda` ou `cuda:N` |
| `ECHOMIMIC_ACCELERATED` | `1` | `0` usa os pesos normais (30 passos) |
| `ECHOMIMIC_STEPS` | do modo | Força o número de passos |
| `ECHOMIMIC_FPS` / `ECHOMIMIC_SEED` | `24` / `420` | Quadros por segundo / semente |
| `HEYGEN_MAX_SIDE` | `1920` | Maior lado da foto antes de reduzir (`0` = nunca reduzir) |
| `HEYGEN_CPU_THREADS` | (automático) | Threads de CPU dos motores (em CPUs Intel híbridas, teste o nº de núcleos P) |

Ao iniciar, o console mostra se cada motor está pronto; a interface também mostra
o status abaixo de **MOTOR DO AVATAR**.

## 6. Rodar

Dê dois cliques em:

`iniciar.bat`

Ou:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

A interface abre em:

`http://127.0.0.1:7860`

## 7. Primeiro teste

Use:
- foto frontal;
- rosto bem iluminado;
- fundo simples;
- texto curto (1 a 3 frases);
- motor **Standard**, foto inteira, movimento Natural (os padrões).

Só com CPU, a animação é o gargalo (a barra de progresso mostra o quadro atual e o
tempo restante). Para vídeos longos sem GPU, use 256 em "Mais ajustes" ou gere
blocos curtos.

## 8. Adicionar um motor novo

1. Crie `heygenhome/engines/meu_motor.py` com uma subclasse de `TalkingHeadBackend`
   (`key`, `label`, `name`, `description`, `capabilities`, `status()` e `animate()`).
2. Inclua-a na lista de `create_engine()` em `heygenhome/engines/__init__.py`.

O `animate()` recebe a foto já preparada (PNG RGB, dimensões pares) e o WAV, e
devolve um MP4. Contrato: no modo foto inteira, o vídeo tem a proporção da foto.
O resto (voz, validação, formatos, interface) já funciona.

## 9. Testes

```powershell
pip install pytest
python -m pytest
```

Os testes usam um runner simulado do SadTalker e exigem FFmpeg no PATH.

## 10. Próxima evolução recomendada

1. histórico de avatares;
2. geração em segmentos;
3. legendas automáticas;
4. fila de geração;
5. presets de voz;
6. API REST;
7. n8n;
8. opção de usar GPU remota sem alterar a interface;
9. novos motores atrás do `TalkingHeadEngine`.

## Observação

Use imagens e vozes próprias ou com autorização. Se o sistema for oferecido a
terceiros, inclua consentimento, política de privacidade e mecanismos para evitar
uso enganoso da imagem de outras pessoas.
