"""SadTalkerNext: o SadTalker oficial com as melhorias do HeyGenHome.

Executado com o Python do SadTalker (SADTALKER_PYTHON) e com o diretório de
trabalho na raiz do repositório oficial (https://github.com/OpenTalker/SadTalker).
Reaproveita os modelos e o código ``src.*`` do SadTalker sem alterar o clone e
troca o ``inference.py`` por um pipeline próprio:

Movimento
- a cabeça se move de verdade: o movimento previsto pelo audio2pose é suavizado,
  limitado a uma amplitude-alvo (preset) e aplicado aos ângulos do renderizador,
  com a translação congelada para não "descolar" o pescoço;
- os lábios usam ganho relativo à expressão da própria foto (em silêncio o rosto
  é exatamente o da foto);
- piscadas naturais (perfil assimétrico, intervalos realistas), controladas na
  direção exata que o modelo usa para piscar.

Desempenho
- GPU NVIDIA usada automaticamente quando disponível (com teste real de
  conv3d + grid_sample), fp16 em placas com tensor cores, lote ajustado à VRAM e
  fallback (lote menor -> fp16 -> CPU) em falta de memória;
- a CPU trabalha em paralelo: colagem e codificação em outra thread;
- a foto é codificada uma única vez (o SadTalker refaz isso a cada quadro);
- quadros saem da GPU à medida que ficam prontos (memória constante).

Composição
- recorte quadrado próprio (sem os bugs de recorte do SadTalker para rostos
  grandes ou perto da borda), colagem com máscara elíptica suavizada e correção
  de cor fixa, e uma única codificação H.264 na resolução exata da foto.

Saídas: --output (MP4 só com vídeo) e --meta (JSON). Progresso no stdout em
linhas "HG_PROGRESS ...". Compatível com Python 3.8.
"""

import argparse
import importlib
import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time

# Códigos próprios, longe de 1-4 (no Windows, abort() do runtime C sai com 3).
EXIT_CONFIG = 12
EXIT_NO_FACE = 13
EXIT_DEVICE = 14
NO_FACE_MARKER = "HG_ERROR no_face"

# Dispositivo em uso agora (para o run() saber se um erro veio da GPU).
STATE = {"device": "cpu"}

FPS = 25
SAMPLES_PER_FRAME = 640  # 16 kHz / 25 fps, como no SadTalker
SEMANTIC_RADIUS = 13  # janela de 27 quadros do MappingNet

# Presets de movimento. Ângulos em graus, na ordem (pitch, yaw, roll).
#   pose_rms: amplitude-alvo (RMS) do movimento vindo do áudio; o audio2pose varia
#             muito entre execuções (1° a 7° medidos), então normalizamos para o alvo
#   pose_limit: limite suave (tanh)
#   idle_rms: micro-movimento de base, para a cabeça não congelar nas pausas
#   mouth: ganho dos lábios relativo à expressão da foto
#   blink: intensidade das piscadas
#   margin: margem do recorte em volta do rosto (cada lado, fração do rosto)
MOTION_PRESETS = {
    "estavel": {
        "pose_rms": (0.0, 0.0, 0.0), "pose_limit": (0.0, 0.0, 0.0), "idle_rms": (0.0, 0.0, 0.0),
        "mouth": 1.0, "blink": 1.0, "margin": 0.10,
    },
    "natural": {
        "pose_rms": (1.2, 1.8, 0.8), "pose_limit": (3.5, 5.0, 2.5), "idle_rms": (0.4, 0.6, 0.3),
        "mouth": 1.3, "blink": 1.15, "margin": 0.10,
    },
    "expressivo": {
        "pose_rms": (2.0, 3.0, 1.2), "pose_limit": (5.0, 8.0, 3.5), "idle_rms": (0.5, 0.8, 0.4),
        "mouth": 1.45, "blink": 1.25, "margin": 0.15,
    },
}

BLINK_PROFILE = (0.35, 0.8, 1.0, 1.0, 0.75, 0.45, 0.2)  # fecha rápido, abre devagar (~280 ms)
MAX_POSE_GAIN = 2.5  # até quanto o movimento do áudio pode ser ampliado para chegar ao alvo


# --------------------------------------------------------------------------
# Funções puras (só numpy): testáveis fora do ambiente do SadTalker.
# --------------------------------------------------------------------------

def gaussian_smooth(x, sigma):
    """Suaviza ao longo do eixo 0 (bordas repetidas)."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    if sigma <= 0 or len(x) < 2:
        return x.copy()
    radius = max(1, int(round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    pad = [(radius, radius)] + [(0, 0)] * (x.ndim - 1)
    xp = np.pad(x, pad, mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.tensordot(k, xp[i:i + 2 * radius + 1], axes=(0, 0))
    return out


def ease_envelope(n, ease_in=8, ease_out=10):
    """Envelope 0->1->0 (smoothstep): o primeiro e o último quadro ficam iguais à foto."""
    import numpy as np

    env = np.ones(n)
    for i in range(min(ease_in, n)):
        t = i / float(ease_in)
        env[i] = min(env[i], t * t * (3 - 2 * t))
    for i in range(min(ease_out, n)):
        t = i / float(ease_out)
        env[n - 1 - i] = min(env[n - 1 - i], t * t * (3 - 2 * t))
    return env


def idle_motion(n, rms, rng, sigma=12.0):
    """Ruído suave (graus) com RMS dado por eixo."""
    import numpy as np

    rms = np.asarray(rms, dtype=np.float64)
    if n < 2 or not rms.any():
        return np.zeros((n, len(rms)))
    noise = gaussian_smooth(rng.standard_normal((n, len(rms))), sigma)
    noise -= noise.mean(axis=0)
    std = noise.std(axis=0)
    std[std == 0] = 1.0
    return noise / std * rms


def head_motion(delta_deg, preset, rng):
    """Converte o movimento previsto pelo audio2pose no movimento final da cabeça.

    ``delta_deg``: (T, 3) diferença em graus para a pose da foto (pitch, yaw, roll).
    Suaviza, normaliza para a amplitude-alvo (ampliando no máximo MAX_POSE_GAIN
    vezes), soma o micro-movimento, limita (tanh) e aplica o envelope de entrada/saída.
    """
    import numpy as np

    d = np.asarray(delta_deg, dtype=np.float64)
    n = len(d)
    target = np.asarray(preset["pose_rms"], dtype=np.float64)
    limit = np.asarray(preset["pose_limit"], dtype=np.float64)
    if n == 0 or not (target.any() or np.any(preset["idle_rms"])):
        return np.zeros((n, 3))

    d = gaussian_smooth(d, 3.0)
    d -= d[:1]  # começa exatamente na pose da foto
    rms = np.sqrt((d ** 2).mean(axis=0))
    gain = np.where(rms > 1e-3, np.minimum(MAX_POSE_GAIN, target / np.maximum(rms, 1e-3)), 0.0)
    d = d * gain + idle_motion(n, preset["idle_rms"], rng)
    safe = np.where(limit > 0, limit, 1.0)
    d = np.where(limit > 0, safe * np.tanh(d / safe), 0.0)
    return d * ease_envelope(n)[:, None]


def blink_schedule(n, rng, fps=FPS, peak=1.0):
    """Curva de piscadas (0..~1.3) por quadro, com intervalos realistas (~3 s)."""
    import numpy as np

    r = np.zeros(n)
    profile = np.asarray(BLINK_PROFILE)
    t = rng.uniform(0.6, 1.6) * fps
    last_allowed = n - len(profile) - 8
    while t <= last_allowed:
        start = int(t)
        amp = peak * rng.uniform(0.95, 1.25)
        r[start:start + len(profile)] = np.maximum(r[start:start + len(profile)], profile * amp)
        if rng.random() < 0.1 and start + 7 + len(profile) <= last_allowed:  # piscada dupla
            s2 = start + 7
            r[s2:s2 + len(profile)] = np.maximum(r[s2:s2 + len(profile)], profile * amp * 0.9)
        t += float(np.clip(rng.gamma(4.0, 0.75), 1.4, 6.0)) * fps
    return r


def face_box(lm, margin):
    """Quadrado (x0, y0, lado) em volta do rosto, em pixels da foto.

    Usa a mesma geometria do recorte do SadTalker (quad estilo FFHQ a partir de
    olhos e boca), sem a redução de imagem nem as limitações de borda dele.
    """
    import numpy as np

    lm = np.asarray(lm, dtype=np.float64)
    eye_left = lm[36:42].mean(axis=0)
    eye_right = lm[42:48].mean(axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_avg = (lm[48] + lm[54]) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    c = eye_avg + eye_to_mouth * 0.1
    half = abs(x[0]) + abs(x[1])  # meia-largura da caixa alinhada do quad (rotacionado)
    side = int(round(2 * half * (1 + 2 * margin)))
    side += side % 2
    return int(round(c[0] - side / 2.0)), int(round(c[1] - side / 2.0)), side


def crop_padded(img, x0, y0, side):
    """Recorta o quadrado, repetindo a borda onde ele sai da imagem."""
    import cv2

    h, w = img.shape[:2]
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x0 + side - w), max(0, y0 + side - h)
    sub = img[max(0, y0):min(h, y0 + side), max(0, x0):min(w, x0 + side)]
    if pad_l or pad_t or pad_r or pad_b:
        sub = cv2.copyMakeBorder(sub, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
    return sub


def head_mask(side, ax=0.40, ay=0.46, dy=-0.03, feather=0.07, bottom_feather=0.14, inset=0.02):
    """Máscara alfa (side x side) da cabeça: elipse com borda suave, mais larga no pescoço."""
    import numpy as np

    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32) + 0.5
    cx, cy = side / 2.0, side * (0.5 + dy)
    r = np.sqrt(((xx - cx) / (ax * side)) ** 2 + ((yy - cy) / (ay * side)) ** 2)
    width = np.where(yy > cy, bottom_feather, feather) / min(ax, ay)
    t = np.clip((1.0 - r) / width, 0.0, 1.0)
    alpha = t * t * (3 - 2 * t)
    edge = np.minimum(np.minimum(xx, yy), np.minimum(side - xx, side - yy))
    alpha *= np.clip(edge / max(1.0, inset * side) - 1.0, 0.0, 1.0)
    return alpha.astype(np.float32)


def color_match(generated, original, alpha, valid):
    """Ganho/deslocamento por canal que aproxima a cor gerada da foto (anel da máscara)."""
    import numpy as np

    ring = (alpha > 0.15) & (alpha < 0.85) & valid
    if ring.sum() < 50:
        return np.ones(3, np.float32), np.zeros(3, np.float32)
    g = generated[ring].astype(np.float64)
    o = original[ring].astype(np.float64)
    gain = np.clip(o.std(axis=0) / np.maximum(g.std(axis=0), 1e-3), 0.85, 1.15)
    bias = o.mean(axis=0) - gain * g.mean(axis=0)
    return gain.astype(np.float32), bias.astype(np.float32)


def auto_batch(free_bytes, size, fp16):
    """Quantos quadros por lote cabem na VRAM livre (estimativa conservadora)."""
    per_frame = (2.0 if size >= 512 else 0.5) * (0.55 if fp16 else 1.0) * 2 ** 30
    reserve = 1.0 * 2 ** 30
    return int(max(1, min(4, (free_bytes - reserve) // per_frame)))


# --------------------------------------------------------------------------
# Ambiente
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SadTalkerNext para o HeyGenHome")
    p.add_argument("--image", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--size", type=int, default=512, choices=[256, 512])
    p.add_argument("--framing", default="full", choices=["full", "crop"])
    p.add_argument("--motion", default="natural", choices=sorted(MOTION_PRESETS))
    p.add_argument("--enhancer", action="store_true")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--fp16", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--batch", type=int, default=0, help="0 = automático")
    p.add_argument("--threads", type=int, default=0, help="0 = padrão do PyTorch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pose_style", type=int, default=0)
    p.add_argument("--ffmpeg", default="ffmpeg")
    return p.parse_args(argv)


def emit(tag, **fields):
    print(tag + " " + json.dumps(fields, ensure_ascii=False), flush=True)


def prefer_fast_scheduling():
    """No Windows 11, impede que o processo seja estrangulado (EcoQoS) com a janela minimizada."""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class PowerThrottlingState(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetProcessInformation.restype = wintypes.BOOL
        state = PowerThrottlingState(1, 0x1, 0)  # ProcessPowerThrottling: EXECUTION_SPEED desligado
        if not k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state)):
            print("Aviso: não foi possível desligar o EcoQoS (erro %d)." % ctypes.get_last_error(), flush=True)
    except Exception:  # Windows antigo sem a API
        pass


def install_shims():
    """Compatibilidade do SadTalker com bibliotecas mais novas e com o Windows, sem editar o clone."""
    import numpy as np
    from PIL import Image

    if not hasattr(Image, "ANTIALIAS"):  # Pillow >= 10 (usado em src/utils/croper.py)
        Image.ANTIALIAS = Image.LANCZOS
    for name, typ in (("float", float), ("int", int), ("bool", bool), ("complex", complex)):
        if name not in np.__dict__:  # numpy >= 1.24 removeu os aliases (usados em src/face3d)
            setattr(np, name, typ)
    if os.name == "nt":
        unicode_safe_cv2()
    try:
        importlib.import_module("torchvision.transforms.functional_tensor")
    except ImportError:  # torchvision >= 0.17 removeu; o basicsr (GFPGAN) ainda importa
        try:
            import types

            import torchvision.transforms.functional as F

            mod = types.ModuleType("torchvision.transforms.functional_tensor")
            mod.rgb_to_grayscale = F.rgb_to_grayscale
            sys.modules["torchvision.transforms.functional_tensor"] = mod
        except ImportError:
            pass


def unicode_safe_cv2():
    """cv2.imread/imwrite do Windows não abrem caminhos com acento (C:\\Users\\João\\...)."""
    import cv2
    import numpy as np

    if getattr(cv2.imread, "_hg_safe", False):
        return
    original_read, original_write = cv2.imread, cv2.imwrite

    def imread(path, flags=cv2.IMREAD_COLOR):
        try:
            data = np.fromfile(path, dtype=np.uint8)
        except (OSError, ValueError):
            return original_read(path, flags)
        return cv2.imdecode(data, flags) if data.size else None

    def imwrite(path, img, params=None):
        ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img, params or [])
        if not ok:
            return original_write(path, img, params or [])
        buf.tofile(path)
        return True

    imread._hg_safe = True
    cv2.imread, cv2.imwrite = imread, imwrite


OOM_MARKERS = ("out of memory", "cublas_status_alloc_failed", "unable to find a valid cudnn algorithm")


def is_oom(exc):
    oom_type = getattr(getattr(sys.modules.get("torch"), "cuda", None), "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in OOM_MARKERS)


def pick_device(torch, request):
    """Escolhe a GPU (a de mais VRAM, ou a pedida) e testa de verdade as operações do renderizador."""
    import torch.nn.functional as F

    info = {"requested": request}
    if request == "cpu":
        return torch.device("cpu"), info
    if torch.version.cuda is None:
        info["cpu_build"] = True  # PyTorch só CPU: a interface explica como instalar a versão CUDA
        return torch.device("cpu"), info
    if not torch.cuda.is_available():
        info["gpu_error"] = "CUDA indisponível (sem GPU NVIDIA ou driver antigo para o CUDA %s)" % torch.version.cuda
        return torch.device("cpu"), info

    if request in ("auto", "cuda"):
        # get_device_properties não cria contexto CUDA; mesma regra da interface (mais VRAM primeiro).
        count = torch.cuda.device_count()
        order = sorted(range(count), key=lambda i: -torch.cuda.get_device_properties(i).total_memory)
        candidates = [torch.device("cuda", i) for i in order]
    else:
        dev = torch.device(request)
        candidates = [torch.device("cuda", dev.index or 0)]

    errors = []
    for dev in candidates:
        try:
            torch.cuda.set_device(dev)  # o SadTalker cria tensores no dispositivo "atual"
            x = torch.randn(1, 4, 4, 16, 16, device=dev)
            w = torch.randn(4, 4, 3, 3, 3, device=dev)
            g = torch.rand(1, 4, 16, 16, 3, device=dev) * 2 - 1
            y = F.grid_sample(F.conv3d(x, w, padding=1), g, align_corners=False)
            torch.cuda.synchronize(dev)
            if not bool(torch.isfinite(y).all()):
                raise RuntimeError("resultado inválido no teste da GPU")
            props = torch.cuda.get_device_properties(dev)
            info.update(name=props.name, vram_gb=round(props.total_memory / 2 ** 30, 1),
                        capability="%d.%d" % (props.major, props.minor))
            return dev, info
        except Exception as exc:  # GPU sem suporte (ex.: RTX 50 com PyTorch antigo), driver etc.
            errors.append("%s: %s" % (dev, str(exc)[:200]))
    info["gpu_error"] = "; ".join(errors)[:300]
    print("GPU indisponível para o SadTalker (%s); usando CPU." % info["gpu_error"], flush=True)
    return torch.device("cpu"), info


def fp16_supported(info, mode):
    if mode == "off" or "name" not in info:
        return False
    if mode == "on":
        return True
    major = int(info["capability"].split(".")[0])
    name = info["name"].upper()
    if "GTX" in name or " MX" in name:  # sem tensor cores: fp16 gera quadros pretos/NaN
        return False
    return major >= 8 or (major == 7 and ("RTX" in name or "TITAN" in name or "TESLA" in name or "V100" in name))


# --------------------------------------------------------------------------
# Renderizador (cópia enxuta de OcclusionAwareSPADEGenerator.forward/make_animation)
# --------------------------------------------------------------------------

class Renderer:
    def __init__(self, torch, paths, device):
        import yaml
        from src.facerender.modules.generator import OcclusionAwareSPADEGenerator
        from src.facerender.modules.keypoint_detector import KPDetector
        from src.facerender.modules.mapping import MappingNet

        self.torch = torch
        with open(paths["facerender_yaml"]) as fh:
            cfg = yaml.safe_load(fh)["model_params"]
        common = cfg["common_params"]
        self.gen = OcclusionAwareSPADEGenerator(**cfg["generator_params"], **common)
        self.kpd = KPDetector(**cfg["kp_detector_params"], **common)
        self.mapping = MappingNet(**cfg["mapping_params"])

        if "checkpoint" in paths:
            import safetensors.torch

            ckpt = safetensors.torch.load_file(paths["checkpoint"])
            self.gen.load_state_dict(_strip(ckpt, "generator."))
            self.kpd.load_state_dict(_strip(ckpt, "kp_extractor."))
            del ckpt
        else:
            ckpt = torch.load(paths["free_view_checkpoint"], map_location="cpu")
            self.gen.load_state_dict(ckpt["generator"])
            self.kpd.load_state_dict(ckpt["kp_detector"])
            del ckpt
        # Os mapping_*.pth.tar foram salvos em cuda:0: map_location é obrigatório.
        self.mapping.load_state_dict(torch.load(paths["mappingnet_checkpoint"], map_location="cpu")["mapping"])
        for m in (self.gen, self.kpd, self.mapping):
            m.eval()
            m.requires_grad_(False)
        self.fp16 = False
        self.to(device)

    def to(self, device):
        self.device = device
        for m in (self.gen, self.kpd, self.mapping):
            m.to(device)
        if getattr(self, "src", None) is not None:
            self.prepare(self.src_cpu, self.src_win_cpu)

    def prepare(self, src, src_win):
        """Tudo que depende só da foto é calculado uma vez."""
        from src.facerender.modules.make_animation import headpose_pred_to_degree, keypoint_transformation

        self.src_cpu, self.src_win_cpu = src, src_win
        self.src = src.to(self.device)
        win = src_win.to(self.device)
        self.kp_can = self.kpd(self.src)["value"]
        he = self.mapping(win)
        self.t_src = he["t"].clone()
        self.kp_src = keypoint_transformation({"value": self.kp_can}, he)["value"]
        self.t_src[:, 0] = 0
        self.t_src[:, 2] = 0
        self.src_deg = [float(headpose_pred_to_degree(he[k])[0]) for k in ("pitch", "yaw", "roll")]
        g = self.gen
        f = g.first(self.src)
        for block in g.down_blocks:
            f = block(f)
        f = g.second(f)
        b, c, h, w = f.shape
        self.feature = g.resblocks_3d(f.view(b, g.reshape_channel, g.reshape_depth, h, w))

    def render(self, windows, angles):
        """windows: (n, C, 27); angles: (n, 3) graus absolutos. Devolve uint8 (n, S, S, 3) RGB."""
        import torch.nn.functional as F
        from src.facerender.modules.make_animation import keypoint_transformation

        torch = self.torch
        n = windows.shape[0]
        win = windows.to(self.device)
        ang = angles.to(self.device)
        he = self.mapping(win)
        he["pitch_in"], he["yaw_in"], he["roll_in"] = ang[:, 0], ang[:, 1], ang[:, 2]
        he["t"] = self.t_src.expand(n, -1).clone()
        kp_drv = keypoint_transformation({"value": self.kp_can.expand(n, -1, -1)}, he)
        kp_src = {"value": self.kp_src.expand(n, -1, -1).contiguous()}
        feat = self.feature.expand(n, -1, -1, -1, -1).contiguous()
        g = self.gen
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16 and self.device.type == "cuda"):
            dm = g.dense_motion_network(feature=feat, kp_driving=kp_drv, kp_source=kp_src)
            out = g.deform_input(feat, dm["deformation"])
            bs, c, d, h, w = out.shape
            out = g.fourth(g.third(out.view(bs, c * d, h, w)))
            occ = dm.get("occlusion_map")
            if occ is not None:
                if occ.shape[2:] != out.shape[2:]:
                    occ = F.interpolate(occ, size=out.shape[2:], mode="bilinear")
                out = out * occ
            pred = g.decoder(out).float()
        if not bool(torch.isfinite(pred).all()):
            raise FloatingPointError("quadro inválido (NaN)")
        return (pred.clamp(0, 1) * 255 + 0.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def _strip(state, prefix):
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


# --------------------------------------------------------------------------
# Composição e escrita (thread da CPU)
# --------------------------------------------------------------------------

class CompositeError(RuntimeError):
    """Falha ao compor/codificar o vídeo (ffmpeg). Não é erro de GPU."""


def _log_tail(path, lines=12):
    try:
        with open(path, "rb") as fh:
            data = fh.read()[-6000:].decode("utf-8", "replace")
    except OSError:
        return ""
    return "\n".join(data.strip().splitlines()[-lines:])


class Compositor(threading.Thread):
    """Recebe quadros renderizados em ordem, compõe na foto e envia ao ffmpeg."""

    def __init__(self, ffmpeg, out_path, log_path, photo, box, size, framing, enhancer):
        super().__init__(daemon=True)
        import numpy as np

        self.np = np
        self.queue = queue.Queue(maxsize=8)
        self.error = None
        self.frames = 0
        self.size = size
        self.enhancer = enhancer
        self.framing = framing
        self.log_path = log_path
        if framing == "full":
            self.photo = photo
            self.base = photo.copy()
            x0, y0, side = box
            h, w = photo.shape[:2]
            self.side = side
            self.dst = (max(0, x0), max(0, y0), min(w, x0 + side), min(h, y0 + side))
            self.src = (self.dst[0] - x0, self.dst[1] - y0, self.dst[2] - x0, self.dst[3] - y0)
            sx0, sy0, sx1, sy1 = self.src
            self.alpha = head_mask(side)[sy0:sy1, sx0:sx1, None]
            self.gain = None
            out_w, out_h = w, h
        else:
            out_w = out_h = size
        self.log = open(log_path, "ab")
        self.proc = subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (out_w, out_h), "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", out_path],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log,
        )

    def put(self, item):
        # Nunca bloqueia para sempre: se a thread morreu (ex.: ffmpeg caiu), o erro sobe.
        while True:
            if self.error is not None:
                raise self.error
            if not self.is_alive():
                raise RuntimeError("a composição do vídeo parou inesperadamente")
            try:
                self.queue.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    def compose(self, rgb):
        import cv2

        np = self.np
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        if self.enhancer is not None:
            bgr = self.enhancer(bgr)
        if self.framing != "full":
            return bgr
        side = self.side
        interp = cv2.INTER_CUBIC if side > self.size else cv2.INTER_AREA
        face = cv2.resize(bgr, (side, side), interpolation=interp)
        sx0, sy0, sx1, sy1 = self.src
        dx0, dy0, dx1, dy1 = self.dst
        face = face[sy0:sy1, sx0:sx1].astype(np.float32)
        orig = self.photo[dy0:dy1, dx0:dx1].astype(np.float32)
        if self.gain is None:  # correção de cor calculada uma vez (sem cintilação)
            valid = np.ones(face.shape[:2], bool)
            self.gain, self.bias = color_match(face, orig, self.alpha[:, :, 0], valid)
        face = face * self.gain + self.bias
        a = self.alpha
        self.base[dy0:dy1, dx0:dx1] = np.clip(a * face + (1 - a) * orig + 0.5, 0, 255).astype(np.uint8)
        return self.base

    def ffmpeg_failed(self):
        try:
            code = self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            code = "?"
        return CompositeError("ffmpeg falhou ao gravar o vídeo (código %s). Final do %s:\n%s"
                              % (code, self.log_path, _log_tail(self.log_path)))

    def run(self):
        try:
            while True:
                batch = self.queue.get()
                if batch is None:
                    break
                for rgb in batch:
                    frame = self.compose(rgb)
                    try:
                        self.proc.stdin.write(frame.tobytes())
                    except OSError:  # BrokenPipe (Linux) ou Errno 22 (Windows): o ffmpeg caiu
                        raise self.ffmpeg_failed()
                    self.frames += 1
        except BaseException as exc:  # repassado para a thread principal
            self.error = exc
            try:
                while self.queue.get_nowait() is not None:
                    pass
            except queue.Empty:
                pass

    def finish(self):
        if self.is_alive():
            try:
                self.put(None)
            except Exception:
                pass
        self.join()
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        code = self.proc.wait()
        self.log.close()
        if isinstance(self.error, CompositeError):
            raise self.error
        if code != 0:
            raise CompositeError("ffmpeg falhou ao gravar o vídeo (código %d). Final do %s:\n%s"
                                 % (code, self.log_path, _log_tail(self.log_path)))
        if self.error is not None:
            raise self.error


class Enhancer:
    """GFPGAN no recorte do rosto (upscale=1): mantém a resolução e custa bem menos."""

    URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"

    def __init__(self, torch, device):
        self.torch = torch
        self.device = device
        self.lock = threading.Lock()
        self.restorer = self._build(device)

    @classmethod
    def model_path(cls):
        for candidate in ("gfpgan/weights/GFPGANv1.4.pth", "checkpoints/GFPGANv1.4.pth"):
            if os.path.isfile(candidate):
                return candidate
        # Baixa para gfpgan/weights do SadTalker (onde o HeyGenHome confere), não para o site-packages.
        from basicsr.utils.download_util import load_file_from_url

        return load_file_from_url(cls.URL, model_dir=os.path.join(os.getcwd(), "gfpgan", "weights"))

    def _build(self, device):
        from gfpgan import GFPGANer

        torch = self.torch
        original_load = torch.load

        def load_on_cpu(*a, **kw):  # o GFPGANer chama torch.load sem map_location
            kw.setdefault("map_location", "cpu")
            return original_load(*a, **kw)

        torch.load = load_on_cpu
        try:
            return GFPGANer(model_path=self.model_path(), upscale=1, arch="clean", channel_multiplier=2,
                            bg_upsampler=None, device=device)
        finally:
            torch.load = original_load

    def to_cpu(self):
        with self.lock:
            if self.device.type != "cpu":
                self.device = self.torch.device("cpu")
                self.restorer = self._build(self.device)

    def _enhance(self, bgr):
        _, _, out = self.restorer.enhance(bgr, has_aligned=False, only_center_face=True,
                                          paste_back=True, weight=0.5)
        return out

    def __call__(self, bgr):
        with self.lock:
            try:
                out = self._enhance(bgr)
            except RuntimeError as exc:
                if self.device.type != "cuda" or not is_oom(exc):
                    raise
                print("GFPGAN sem VRAM; o GFPGAN continua em CPU.", flush=True)
                self.torch.cuda.empty_cache()
                self.device = self.torch.device("cpu")
                self.restorer = self._build(self.device)
                out = self._enhance(bgr)
        return bgr if out is None else out


class RenderPolicy(object):
    """O que fazer quando a renderização falha na GPU.

    Nunca entra em ciclo: cada falha reduz o lote, liga o fp16 (uma vez, se ainda
    permitido) ou desiste da GPU; NaN em fp16 proíbe o fp16 de vez.
    """

    def __init__(self, batch, fp16, fp16_allowed):
        self.batch, self.fp16, self.fp16_allowed = batch, fp16, fp16_allowed

    def on_oom(self):
        if self.batch > 1:
            self.batch = max(1, self.batch // 2)
            return "retry"
        if self.fp16_allowed and not self.fp16:
            self.fp16 = True
            return "retry"
        self.fp16 = self.fp16_allowed = False
        return "cpu"

    def on_nan(self):
        if self.fp16:
            self.fp16 = self.fp16_allowed = False
            return "retry"
        return "cpu"

    def on_error(self):
        if self.fp16:
            self.fp16 = self.fp16_allowed = False
            return "retry"
        return "raise"


def detect_landmarks(preprocess, rgb):
    """68 pontos do rosto na foto, como o get_landmark do SadTalker, mas com a caixa
    do detector limitada à imagem (rostos encostados na borda quebravam o original)."""
    import numpy as np
    import torch
    from facexlib.alignment import landmark_98_to_68

    predictor = preprocess.propress.predictor
    with torch.no_grad():
        dets = predictor.det_net.detect_faces(rgb, 0.97)
    if len(dets) == 0:
        return None
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in dets[0][:4]]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    lm = landmark_98_to_68(predictor.detector.get_landmarks(np.ascontiguousarray(rgb[y0:y1, x0:x1])))
    lm[:, 0] += x0
    lm[:, 1] += y0
    return lm


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def no_face():
    print(NO_FACE_MARKER, flush=True)
    print("ERRO: nenhum rosto detectado na foto. Use uma foto frontal e bem iluminada.", flush=True)
    return EXIT_NO_FACE


def main(argv=None):
    args = parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
    sys.path.insert(0, os.getcwd())
    prefer_fast_scheduling()

    import cv2
    import numpy as np
    import torch

    install_shims()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    import safetensors.torch
    from scipy.io import loadmat

    from src.generate_batch import get_data
    from src.test_audio2coeff import Audio2Coeff
    from src.utils.audio import load_wav
    from src.utils.init_path import init_path
    from src.utils.preprocess import CropAndExtract

    preset = MOTION_PRESETS[args.motion]
    t_start = time.time()
    device, dev_info = pick_device(torch, args.device)
    STATE["device"] = str(device)
    emit("HG_DEVICE", device=str(device), **dev_info)

    paths = init_path(args.checkpoint_dir, os.path.join(os.getcwd(), "src", "config"), args.size, False, "full")
    needed = [paths.get("checkpoint") or paths.get("free_view_checkpoint"), paths["mappingnet_checkpoint"]]
    missing = [p for p in needed if not p or not os.path.isfile(p)]
    if missing:
        print("ERRO: checkpoints do SadTalker não encontrados: " + ", ".join(map(str, missing)), flush=True)
        return EXIT_CONFIG

    photo = cv2.imread(args.image)
    if photo is None:
        print("ERRO: não foi possível ler a foto %s" % args.image, flush=True)
        return EXIT_CONFIG
    work = tempfile.mkdtemp(prefix="sadtalkernext-", dir=os.path.dirname(os.path.abspath(args.meta)))

    # O safetensors de ~700 MB seria lido 4 vezes; lê uma.
    original_load = safetensors.torch.load_file
    cache = {}

    def cached_load(path, *a, **kw):
        if path not in cache:
            cache[path] = original_load(path, *a, **kw)
        return cache[path]

    safetensors.torch.load_file = cached_load
    try:
        emit("HG_PROGRESS", stage="load", done=0, total=1)
        preprocess = CropAndExtract(paths, str(device))
        a2c = Audio2Coeff(paths, str(device))
        renderer = Renderer(torch, paths, device)
    finally:
        safetensors.torch.load_file = original_load
        cache.clear()

    # 1) Rosto: recorte quadrado próprio e coeficientes 3DMM pelo caminho oficial.
    emit("HG_PROGRESS", stage="face", done=0, total=1)
    lm = detect_landmarks(preprocess, cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))
    if lm is None:
        return no_face()
    box = face_box(lm, preset["margin"])
    crop = crop_padded(photo, *box)
    face = cv2.resize(crop, (args.size, args.size),
                      interpolation=cv2.INTER_AREA if box[2] > args.size else cv2.INTER_CUBIC)
    face_png = os.path.join(work, "rosto.png")
    if not cv2.imwrite(face_png, face):
        print("ERRO: não foi possível gravar %s" % face_png, flush=True)
        return EXIT_CONFIG
    # Marcos já conhecidos (convertidos para o recorte): o SadTalker não detecta de novo.
    lm_crop = (np.asarray(lm, np.float64) - [box[0], box[1]]) * (args.size / float(box[2]))
    np.savetxt(os.path.join(work, "rosto_landmarks.txt"), lm_crop.reshape(-1))
    coeff_path, src_png, _ = preprocess.generate(face_png, work, "resize", source_image_flag=True,
                                                 pic_size=args.size)
    if coeff_path is None:
        return no_face()
    S = loadmat(coeff_path)["coeff_3dmm"][0].astype(np.float64)  # (73,)
    del preprocess
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2) Áudio -> coeficientes, com ganhos e piscadas controlados.
    emit("HG_PROGRESS", stage="audio", done=0, total=1)
    import soundfile as sf

    wav = load_wav(args.audio, 16000)
    frames = max(2, int(math.ceil(len(wav) / float(SAMPLES_PER_FRAME))))
    wav = np.pad(wav, (0, frames * SAMPLES_PER_FRAME - len(wav)))
    speech_path = os.path.join(work, "fala16k.wav")
    silence_path = os.path.join(work, "silencio16k.wav")
    sf.write(speech_path, wav.astype(np.float32), 16000)
    sf.write(silence_path, np.zeros_like(wav, dtype=np.float32), 16000)

    batch = get_data(coeff_path, speech_path, str(device), None, use_blink=False)
    T = int(batch["num_frames"])
    exp_speech = a2c.audio2exp_model.test(batch)["exp_coeff_pred"][0].float().cpu().numpy()
    batch_sil = get_data(coeff_path, silence_path, str(device), None, use_blink=False)
    exp_sil = a2c.audio2exp_model.test(batch_sil)["exp_coeff_pred"][0].float().cpu().numpy().mean(axis=0)
    blink_dir = a2c.audio2exp_model.netG.mapping1.weight[:, -1].float().cpu().numpy()
    blinks = blink_schedule(T, rng, peak=preset["blink"])
    exp = S[:64] + preset["mouth"] * (exp_speech - exp_sil) + blinks[:, None] * blink_dir

    torch.manual_seed(args.seed)
    batch["class"] = torch.LongTensor([args.pose_style]).to(device)
    pose = a2c.audio2pose_model.test(batch)["pose_pred"][0].float().cpu().numpy()  # (T, 6) radianos
    motion = head_motion(np.degrees(pose[:, :3] - S[64:67]), preset, rng)
    del a2c, batch, batch_sil
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Coeficientes de expressão por quadro; a pose fica a da foto (o mapeamento
    # "still" do modo foto inteira) e o movimento da cabeça entra direto nos ângulos.
    coeffs = np.concatenate([exp, np.repeat(S[None, 64:73], T, axis=0)], axis=1).astype(np.float32)
    src_win = torch.from_numpy(np.repeat(S[None, :73, None], 2 * SEMANTIC_RADIUS + 1, axis=2).astype(np.float32))
    src_bgr = cv2.imread(src_png)
    if src_bgr is None:
        print("ERRO: não foi possível ler %s" % src_png, flush=True)
        return EXIT_CONFIG
    src = torch.from_numpy(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)).float().div(255).permute(2, 0, 1)[None]
    renderer.prepare(src, src_win)
    angles = (np.asarray(renderer.src_deg)[None, :] + motion).astype(np.float32)  # (T, 3) graus

    def windows(start, n):
        idx = np.clip(np.arange(start, start + n)[:, None] + np.arange(-SEMANTIC_RADIUS, SEMANTIC_RADIUS + 1),
                      0, T - 1)
        return torch.from_numpy(coeffs[idx].transpose(0, 2, 1).copy())

    def angles_at(start, n):
        return torch.from_numpy(angles[start:start + n])

    # 3) GFPGAN antes de medir a VRAM; depois fp16 (validado contra fp32) e o lote.
    enhancer = Enhancer(torch, device) if args.enhancer else None
    renderer.fp16 = False
    fp16_allowed = device.type == "cuda" and fp16_supported(dev_info, args.fp16)
    if fp16_allowed:
        try:
            ref = renderer.render(windows(0, 1), angles_at(0, 1)).astype(np.float64)
        except RuntimeError as exc:  # fp32 sem VRAM nem para 1 quadro: a política decide
            if not is_oom(exc):
                raise
            torch.cuda.empty_cache()
            ref = None
        if ref is not None:
            try:
                renderer.fp16 = True
                half = renderer.render(windows(0, 1), angles_at(0, 1)).astype(np.float64)
                mse = ((ref - half) ** 2).mean()
                psnr = 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)
                dev_info["fp16_psnr"] = round(psnr, 1)
                fp16_allowed = psnr >= 38.0
            except Exception as exc:  # fp32 funciona; só o fp16 não
                dev_info["fp16_error"] = str(exc)[:200]
                fp16_allowed = False
                torch.cuda.empty_cache()
            renderer.fp16 = fp16_allowed
    batch_size = args.batch or 1
    if device.type == "cuda" and not args.batch:
        batch_size = auto_batch(torch.cuda.mem_get_info(device)[0], args.size, renderer.fp16)
    policy = RenderPolicy(batch_size, renderer.fp16, fp16_allowed)
    emit("HG_DEVICE", device=str(device), fp16=policy.fp16, batch=policy.batch, **dev_info)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    compositor = Compositor(args.ffmpeg, args.output, os.path.join(out_dir, "ffmpeg.log"),
                            photo, box, args.size, args.framing, enhancer)
    compositor.start()

    # 4) Renderização em ordem; a CPU compõe e codifica em paralelo.
    t_render = time.time()
    t_base, done_base = t_render, 0  # velocidade medida desde a última mudança de dispositivo/lote
    done = 0
    fallback = None
    try:
        while done < T:
            n = min(policy.batch, T - done)
            renderer.fp16 = policy.fp16
            try:
                rgb = renderer.render(windows(done, n), angles_at(done, n))
            except (RuntimeError, FloatingPointError) as exc:
                if renderer.device.type != "cuda":
                    raise
                oom = is_oom(exc)
                if isinstance(exc, FloatingPointError):
                    action = policy.on_nan()
                elif oom:
                    torch.cuda.empty_cache()
                    action = policy.on_oom()
                else:
                    action = policy.on_error()
                if action == "raise":
                    raise
                if action == "cpu":
                    fallback = "VRAM insuficiente" if oom else "a GPU gerou quadros inválidos"
                    print("%s; continuando em CPU." % fallback.capitalize(), flush=True)
                    renderer.fp16 = False
                    policy.batch = 1  # em CPU lote maior só gasta memória
                    renderer.to(torch.device("cpu"))
                    torch.cuda.empty_cache()
                    STATE["device"] = "cpu"
                    if enhancer is not None:
                        enhancer.to_cpu()
                on_gpu = renderer.device.type == "cuda"
                emit("HG_DEVICE", device=str(renderer.device), name=dev_info.get("name") if on_gpu else "CPU",
                     fp16=policy.fp16 and on_gpu, batch=policy.batch, fallback=fallback)
                t_base, done_base = time.time(), done
                continue
            compositor.put(rgb)
            done += n
            rate = (time.time() - t_base) / max(1, done - done_base)
            emit("HG_PROGRESS", stage="render", done=done, total=T, sec_per_frame=round(rate, 3))
    except BaseException:
        try:
            compositor.finish()
        except CompositeError:
            raise  # o ffmpeg caiu: esta é a causa real
        except Exception:
            pass
        raise
    compositor.finish()
    render_seconds = time.time() - t_render

    on_gpu = renderer.device.type == "cuda"
    meta = {
        "frames": T,
        "fps": FPS,
        "size": args.size,
        "framing": args.framing,
        "motion": args.motion,
        "seed": args.seed,
        "box": list(box),
        "device": str(renderer.device),
        "device_name": dev_info.get("name", "CPU") if on_gpu else "CPU",
        "fp16": bool(renderer.fp16 and on_gpu),
        "batch": policy.batch,
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "render_seconds": round(render_seconds, 1),
        "sec_per_frame": round(render_seconds / max(T, 1), 3),
        "total_seconds": round(time.time() - t_start, 1),
        "head_motion_deg_rms": [round(float(v), 2) for v in np.sqrt((motion ** 2).mean(axis=0))],
        "blinks": int(((blinks[1:] > 0) & (blinks[:-1] == 0)).sum()),
    }
    for key in ("gpu_error", "cpu_build", "fp16_psnr", "fp16_error"):
        if key in dev_info:
            meta[key] = dev_info[key]
    if fallback:
        meta["fallback"] = fallback
    with open(args.meta, "w") as fh:
        json.dump(meta, fh)
    print("SadTalkerNext OK: %s" % args.output, flush=True)
    return 0


def run():
    try:
        return main()
    except CompositeError:
        raise  # ffmpeg: refazer em CPU não ajudaria
    except Exception as exc:
        # Qualquer falha enquanto a GPU estava em uso: o HeyGenHome refaz tudo em CPU.
        if STATE["device"].startswith("cuda"):
            import traceback

            traceback.print_exc()
            print("HG_DEVICE_FAILED " + json.dumps({"error": str(exc)[:500]}), flush=True)
            return EXIT_DEVICE
        raise


if __name__ == "__main__":
    sys.exit(run())
