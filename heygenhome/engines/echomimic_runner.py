"""Roda o EchoMimic (v1) para o HeyGenHome, em CPU ou GPU.

Este script é executado com o Python do EchoMimic (ECHOMIMIC_PYTHON) e com
o diretório de trabalho na raiz do repositório EchoMimic. Ele substitui o
infer_audio2vid.py oficial porque aquele:
- fixa "cuda" no VAE, no face locator, na pipeline e na máscara (não roda em CPU);
- carrega o face locator sem map_location (falha em máquina sem GPU);
- não deixa escolher o arquivo de saída nem informa onde o rosto foi recortado.

Saídas:
  --output  vídeo do rosto, quadrado (size x size), sem áudio;
  --meta    JSON com o retângulo do rosto na foto, para colar de volta.

Compatível com Python 3.8 (versão usada pelo EchoMimic).
"""

import argparse
import gc
import json
import os
import sys

# Códigos próprios, longe de 1-4 (no Windows, abort() do runtime C sai com 3).
EXIT_CONFIG = 12
EXIT_NO_FACE = 13
NO_FACE_MARKER = "HG_ERROR no_face"
# Placas de "12 GB" reportam ~11,99 GiB: a mesma tolerância é usada na interface (gpu.py).
VRAM_TOLERANCE_GB = 0.5

PRESETS = {
    # modo: (config, passos, cfg)
    "accelerated": ("./configs/prompts/animation_acc.yaml", 6, 1.0),
    "standard": ("./configs/prompts/animation.yaml", 30, 2.5),
}
WEIGHT_KEYS = (
    "pretrained_base_model_path",
    "pretrained_vae_path",
    "audio_model_path",
    "denoising_unet_path",
    "reference_unet_path",
    "face_locator_path",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="EchoMimic para o HeyGenHome (CPU/GPU).")
    p.add_argument("--image", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--accelerated", action="store_true", help="pesos *_acc: 6 passos em vez de 30")
    p.add_argument("--config", default=None)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--min_vram_gb", type=float, default=12.0, help="auto só usa GPU com essa VRAM")
    p.add_argument("--fp16", action="store_true", help="só tem efeito em GPU")
    p.add_argument("--steps", type=int, default=0, help="0 = padrão do modo")
    p.add_argument("--cfg", type=float, default=0.0, help="0 = padrão do modo")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--seed", type=int, default=420)
    p.add_argument("--frames", type=int, default=0, help="0 = duração do áudio")
    p.add_argument("--facemask_dilation_ratio", type=float, default=0.1)
    p.add_argument("--facecrop_dilation_ratio", type=float, default=0.5)
    p.add_argument("--context_frames", type=int, default=12)
    p.add_argument("--context_overlap", type=int, default=3)
    p.add_argument("--sample_rate", type=int, default=16000)
    return p.parse_args(argv)


def emit_progress(fraction, message):
    print("HG_PROGRESS " + json.dumps({"fraction": round(fraction, 4), "message": message},
                                      ensure_ascii=False), flush=True)


def unicode_safe_cv2():
    """cv2.imread do Windows não abre caminhos com acento (C:\\Users\\João\\...)."""
    import cv2
    import numpy as np

    if os.name != "nt" or getattr(cv2.imread, "_hg_safe", False):
        return
    original_read = cv2.imread

    def imread(path, flags=cv2.IMREAD_COLOR):
        try:
            data = np.fromfile(path, dtype=np.uint8)
        except (OSError, ValueError):
            return original_read(path, flags)
        return cv2.imdecode(data, flags) if data.size else None

    imread._hg_safe = True
    cv2.imread = imread


def pick_device(torch, request, min_vram_gb):
    """auto: a GPU NVIDIA de mais VRAM (com VRAM suficiente) que passe num teste real; senão CPU."""
    if request == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print("CUDA indisponível neste PyTorch (%s); usando CPU." % torch.__version__)
        return torch.device("cpu")
    if request in ("auto", "cuda"):
        order = sorted(range(torch.cuda.device_count()),
                       key=lambda i: -torch.cuda.get_device_properties(i).total_memory)
        candidates = [torch.device("cuda", i) for i in order]
    else:
        candidates = [torch.device("cuda", torch.device(request).index or 0)]
    for dev in candidates:
        props = torch.cuda.get_device_properties(dev)
        vram = props.total_memory / 2 ** 30
        if request == "auto" and vram + VRAM_TOLERANCE_GB < min_vram_gb:
            print("GPU %s tem %.1f GB (< %.0f GB recomendados); ignorada." % (props.name, vram, min_vram_gb))
            continue
        try:
            torch.cuda.set_device(dev)
            x = torch.randn(1, 4, 32, 32, device=dev)
            y = torch.nn.functional.conv2d(x, torch.randn(4, 4, 3, 3, device=dev), padding=1)
            torch.cuda.synchronize(dev)
            if not bool(torch.isfinite(y).all()):
                raise RuntimeError("resultado inválido no teste da GPU")
            print("GPU: %s (%.1f GB)" % (props.name, vram))
            return dev
        except Exception as exc:
            print("GPU %s indisponível (%s)." % (props.name, str(exc)[:300]))
    print("Nenhuma GPU utilizável; usando CPU.")
    return torch.device("cpu")


def load_state(torch, path):
    """Carrega um checkpoint mapeado do disco (mmap) para não duplicar ~3 GB na RAM."""
    try:
        return torch.load(path, map_location="cpu", mmap=True)
    except (TypeError, RuntimeError):  # PyTorch < 2.1 ou checkpoint em formato antigo
        return torch.load(path, map_location="cpu")


def load_into(torch, model, path, strict=True):
    state = load_state(torch, path)
    model.load_state_dict(state, strict=strict)
    del state
    gc.collect()


def select_face(det_bboxes, probs):
    """Maior rosto com confiança > 0.8 (mesma regra do EchoMimic oficial)."""
    if det_bboxes is None or probs is None:
        return None
    faces = [det_bboxes[i] for i in range(len(det_bboxes)) if probs[i] > 0.8]
    if not faces:
        return None
    return max(faces, key=lambda b: (b[3] - b[1]) * (b[2] - b[0]))


def main(argv=None):
    args = parse_args(argv)
    # Importa o "src" do EchoMimic (cwd), nunca os módulos vizinhos deste arquivo.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
    sys.path.insert(0, os.getcwd())

    import cv2
    import numpy as np
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler
    from facenet_pytorch import MTCNN
    from omegaconf import OmegaConf
    from PIL import Image

    from src.models import mutual_self_attention
    from src.models.face_locator import FaceLocator
    from src.models.unet_2d_condition import UNet2DConditionModel
    from src.models.unet_3d_echo import EchoUNet3DConditionModel
    from src.models.whisper.audio2feature import load_audio_model
    from src.utils.util import crop_and_pad, save_videos_grid

    unicode_safe_cv2()
    mode = "accelerated" if args.accelerated else "standard"
    if args.accelerated:
        from src.pipelines.pipeline_echo_mimic_acc import Audio2VideoPipeline
    else:
        from src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline
    default_config, default_steps, default_cfg = PRESETS[mode]
    steps = args.steps or default_steps
    cfg = args.cfg or default_cfg

    emit_progress(0.01, "Carregando o EchoMimic…")
    device = pick_device(torch, args.device, args.min_vram_gb)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    # fp16 em CPU não é suportado por várias operações do PyTorch.
    dtype = torch.float16 if (args.fp16 and device.type == "cuda") else torch.float32
    print("EchoMimic: modo={} device={} dtype={} passos={} cfg={} threads={}".format(
        mode, device, dtype, steps, cfg, torch.get_num_threads()))

    # O EchoMimic usa torch.device("cuda") como valor padrão neste método.
    hooks = mutual_self_attention.ReferenceAttentionControl.register_reference_hooks
    hooks.__defaults__ = tuple(
        device if isinstance(d, torch.device) else d for d in (hooks.__defaults__ or ())
    )

    config = OmegaConf.load(args.config or default_config)
    missing = [config[k] for k in WEIGHT_KEYS if not os.path.exists(config[k])]
    if missing:
        print("ERRO: pesos do EchoMimic não encontrados:\n  " + "\n  ".join(missing))
        print("Baixe com: git clone https://huggingface.co/BadToBest/EchoMimic pretrained_weights")
        return EXIT_CONFIG
    infer_config = OmegaConf.load(config.inference_config)

    # 1) Rosto: detecta antes de carregar os modelos (falha rápido se não houver rosto).
    image = cv2.imread(args.image)
    if image is None:
        print("ERRO: não foi possível ler a imagem {}".format(args.image))
        return EXIT_CONFIG
    h, w = image.shape[:2]
    detector = MTCNN(image_size=320, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7],
                     factor=0.709, post_process=True, device=device)
    det_bboxes, probs = detector.detect(np.ascontiguousarray(image[:, :, ::-1]))
    bbox = select_face(det_bboxes, probs)
    if bbox is None:
        print(NO_FACE_MARKER)
        print("ERRO: nenhum rosto detectado na foto. Use uma foto frontal, bem iluminada.")
        return EXIT_NO_FACE

    x0, y0, x1, y1 = [int(v) for v in np.round(bbox[:4])]
    bw, bh = x1 - x0, y1 - y0
    mask = np.zeros((h, w), dtype="uint8")
    mx, my = int(bw * args.facemask_dilation_ratio), int(bh * args.facemask_dilation_ratio)
    mask[max(0, y0 - my):y1 + my, max(0, x0 - mx):x1 + mx] = 255
    cx, cy = int(bw * args.facecrop_dilation_ratio), int(bh * args.facecrop_dilation_ratio)
    crop_rect = [max(0, x0 - cx), max(0, y0 - cy), min(x1 + cx, w), min(y1 + cy, h)]
    face_img, rect = crop_and_pad(image, crop_rect)
    face_mask, _ = crop_and_pad(mask, crop_rect)
    face_img = cv2.resize(face_img, (args.size, args.size))
    face_mask = cv2.resize(face_mask, (args.size, args.size))
    print("Rosto em {} (foto {}x{})".format(list(rect), w, h))

    # 2) Modelos, no device escolhido.
    vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path).to(device, dtype=dtype)
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path, subfolder="unet"
    ).to(dtype=dtype, device=device)
    load_into(torch, reference_unet, config.reference_unet_path)

    if os.path.exists(config.motion_module_path):
        denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(dtype=dtype, device=device)
    else:
        denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            "",
            subfolder="unet",
            unet_additional_kwargs={
                "use_motion_module": False,
                "unet_use_temporal_attention": False,
                "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim,
            },
        ).to(dtype=dtype, device=device)
    load_into(torch, denoising_unet, config.denoising_unet_path, strict=False)

    face_locator = FaceLocator(320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)).to(
        dtype=dtype, device=device
    )
    load_into(torch, face_locator, config.face_locator_path)
    audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)

    scheduler = DDIMScheduler(**OmegaConf.to_container(infer_config.noise_scheduler_kwargs))
    pipe = Audio2VideoPipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        face_locator=face_locator,
        scheduler=scheduler,
    ).to(device, dtype=dtype)

    # 3) Geração (progresso por passo de difusão).
    class _Progress(object):
        def __init__(self, total):
            self.total, self.n = max(1, total or 1), 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def update(self, k=1):
            self.n += k
            emit_progress(0.15 + 0.75 * self.n / self.total,
                          "Difusão: passo {}/{} em {}".format(self.n, self.total, device))

    pipe.progress_bar = lambda iterable=None, total=None: _Progress(total)
    emit_progress(0.15, "Gerando o rosto ({})…".format(device))
    face_mask_tensor = (
        torch.Tensor(face_mask).to(dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        / 255.0
    )
    video = pipe(
        Image.fromarray(face_img[:, :, [2, 1, 0]]),
        args.audio,
        face_mask_tensor,
        args.size,
        args.size,
        args.frames or 10 ** 6,  # a pipeline limita à duração do áudio
        steps,
        cfg,
        generator=torch.manual_seed(args.seed),
        audio_sample_rate=args.sample_rate,
        context_frames=args.context_frames,
        fps=args.fps,
        context_overlap=args.context_overlap,
    ).videos

    emit_progress(0.92, "Gravando o vídeo do rosto…")
    save_videos_grid(video, args.output, n_rows=1, fps=args.fps)
    with open(args.meta, "w") as fh:
        json.dump(
            {
                "crop_rect": [int(v) for v in rect],
                "image_size": [w, h],
                "size": args.size,
                "fps": args.fps,
                "frames": int(video.shape[2]),
                "mode": mode,
                "device": str(device),
                "steps": steps,
            },
            fh,
        )
    print("EchoMimic OK: {}".format(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
