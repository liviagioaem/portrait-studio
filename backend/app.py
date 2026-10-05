import io
import os
from threading import Lock, Thread
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageChops, ImageDraw, ImageFilter
from dotenv import load_dotenv


load_dotenv()


app = FastAPI(title="Portrait Studio FLUX API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_lock = Lock()
_pipeline = None
_inpaint_pipeline = None
_torch = None
_AutoPipelineForImage2Image = None
_AutoPipelineForInpainting = None
_img2img_pipeline_name = None
_inpaint_pipeline_name = None
_frontend_dir = Path(__file__).resolve().parent.parent
_frontend_index = _frontend_dir / "index.html"


def _ensure_ml_imports() -> None:
    global _torch, _AutoPipelineForImage2Image, _AutoPipelineForInpainting, _img2img_pipeline_name, _inpaint_pipeline_name
    if _torch is not None and _AutoPipelineForImage2Image is not None and _AutoPipelineForInpainting is not None:
        return

    import torch as _torch_mod

    try:
        from diffusers import AutoPipelineForInpainting as _inpaint_cls

        _inpaint_pipeline_name = "AutoPipelineForInpainting"
    except ImportError:
        # Fallback para versoes antigas do diffusers sem auto pipeline de inpainting.
        from diffusers import StableDiffusionInpaintPipeline as _inpaint_cls

        _inpaint_pipeline_name = "StableDiffusionInpaintPipeline"

    try:
        from diffusers import AutoPipelineForImage2Image as _img2img_cls

        _img2img_pipeline_name = "AutoPipelineForImage2Image"
    except ImportError:
        # Fallback para ambientes onde AutoPipelineForImage2Image nao esta disponivel.
        from diffusers import StableDiffusionImg2ImgPipeline as _img2img_cls

        _img2img_pipeline_name = "StableDiffusionImg2ImgPipeline"

    _torch = _torch_mod
    _AutoPipelineForImage2Image = _img2img_cls
    _AutoPipelineForInpainting = _inpaint_cls


def _get_pipeline():
    _ensure_ml_imports()

    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _model_lock:
        if _pipeline is not None:
            return _pipeline

        model_id = os.getenv("IMG2IMG_MODEL_ID", "runwayml/stable-diffusion-v1-5")
        token = os.getenv("HF_TOKEN") or None
        device = os.getenv("DEVICE", "cuda").lower()

        if device == "cuda" and not _torch.cuda.is_available():
            raise RuntimeError("CUDA nao disponivel. Configure DEVICE=cpu (lento) ou use GPU NVIDIA.")

        dtype = _torch.float16 if device == "cuda" else _torch.float32
        load_kwargs = {"torch_dtype": dtype, "token": token, "local_files_only": False}

        try:
            pipe = _AutoPipelineForImage2Image.from_pretrained(
                model_id,
                variant="fp16" if dtype == _torch.float16 else None,
                **load_kwargs,
            )
        except (OSError, ValueError, TypeError):
            # Alguns pipelines/versoes nao aceitam `variant` ou exigem assinatura diferente.
            pipe = _AutoPipelineForImage2Image.from_pretrained(model_id, **load_kwargs)

        if device == "cuda" and os.getenv("LOW_VRAM", "0") == "1":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

        _pipeline = pipe
        return _pipeline


def _get_inpaint_pipeline():
    _ensure_ml_imports()

    global _inpaint_pipeline
    if _inpaint_pipeline is not None:
        return _inpaint_pipeline

    with _model_lock:
        if _inpaint_pipeline is not None:
            return _inpaint_pipeline

        # SDXL inpainting (~7 GB VRAM) com AutoPipeline. Se o ambiente tiver
        # somente StableDiffusionInpaintPipeline, usamos modelo SD1.5 compativel.
        default_inpaint_model = (
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
            if _inpaint_pipeline_name == "AutoPipelineForInpainting"
            else "runwayml/stable-diffusion-inpainting"
        )
        model_id = os.getenv("INPAINT_MODEL_ID", default_inpaint_model)
        token = os.getenv("HF_TOKEN") or None
        device = os.getenv("DEVICE", "cuda").lower()

        if device == "cuda" and not _torch.cuda.is_available():
            raise RuntimeError("CUDA nao disponivel. Configure DEVICE=cpu (lento) ou use GPU NVIDIA.")

        dtype = _torch.float16 if device == "cuda" else _torch.float32
        load_kwargs = {"torch_dtype": dtype, "token": token, "local_files_only": False}

        try:
            pipe = _AutoPipelineForInpainting.from_pretrained(
                model_id, variant="fp16" if dtype == _torch.float16 else None, **load_kwargs
            )
        except (OSError, ValueError, TypeError):
            # Nem todo repositorio publica pesos na variante fp16.
            pipe = _AutoPipelineForInpainting.from_pretrained(model_id, **load_kwargs)

        if device == "cuda" and os.getenv("LOW_VRAM", "0") == "1":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

        _inpaint_pipeline = pipe
        return _inpaint_pipeline


def _load_image_from_upload(upload: UploadFile, *, mode: str, label: str) -> Image.Image:
    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"{label} precisa ser um arquivo de imagem valido.")

    try:
        data = upload.file.read()
        return Image.open(io.BytesIO(data)).convert(mode)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Nao foi possivel ler {label}: {exc}") from exc


def _opaque_bounds(alpha: Image.Image) -> tuple[int, int, int, int] | None:
    return alpha.getbbox()


def _touches_edges(bounds: tuple[int, int, int, int], size: tuple[int, int], margin: int) -> dict:
    left, top, right, bottom = bounds
    width, height = size
    return {
        "top": top <= margin,
        "left": left <= margin,
        "right": right >= width - margin,
        "bottom": bottom >= height - margin,
    }


def _auto_repair_mask(alpha: Image.Image, touches: dict) -> Image.Image:
    width, height = alpha.size
    edge_band = max(16, int(min(width, height) * 0.11))
    seam_blend = max(8, int(min(width, height) * 0.02))

    edge_mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(edge_mask)

    if touches["top"]:
        draw.rectangle((0, 0, width, edge_band + seam_blend), fill=255)
    if touches["left"]:
        draw.rectangle((0, 0, edge_band + seam_blend, height), fill=255)
    if touches["right"]:
        draw.rectangle((width - edge_band - seam_blend, 0, width, height), fill=255)
    if touches["bottom"]:
        draw.rectangle((0, height - edge_band - seam_blend, width, height), fill=255)

    # Restrict edits to the portrait neighborhood while still allowing slight extension.
    alpha_neighborhood = alpha.filter(ImageFilter.MaxFilter(size=31))
    mask = ImageChops.multiply(edge_mask, alpha_neighborhood)
    return mask.filter(ImageFilter.GaussianBlur(radius=max(3, seam_blend // 2)))


def _outpaint_canvas(source: Image.Image, pads: dict, overlap: int) -> tuple[Image.Image, Image.Image]:
    """Expande a tela nos lados pedidos e devolve (imagem inicial, mascara branca = area a gerar)."""
    top, left, right, bottom = pads["top"], pads["left"], pads["right"], pads["bottom"]

    # Estica as bordas e desfoca: da cores plausiveis de partida para o modelo.
    stretched = np.pad(np.asarray(source), ((top, bottom), (left, right), (0, 0)), mode="edge")
    canvas = Image.fromarray(stretched).filter(ImageFilter.GaussianBlur(radius=max(8, min(source.size) // 20)))
    canvas.paste(source, (left, top))

    # Um pouco de sobreposicao nos lados expandidos para a IA costurar a emenda.
    keep = (
        left + (overlap if left else 0),
        top + (overlap if top else 0),
        left + source.width - (overlap if right else 0),
        top + source.height - (overlap if bottom else 0),
    )
    mask = Image.new("L", canvas.size, 255)
    ImageDraw.Draw(mask).rectangle((keep[0], keep[1], keep[2] - 1, keep[3] - 1), fill=0)
    return canvas, mask


def _working_size(size: tuple[int, int], target: int) -> tuple[int, int]:
    width, height = size
    scale = target / max(width, height)
    return max(64, round(width * scale / 8) * 8), max(64, round(height * scale / 8) * 8)


def _fit_image_for_model(image: Image.Image, target: int) -> Image.Image:
    return image.resize(_working_size(image.size, target), Image.Resampling.LANCZOS)


def _is_cpu_mode() -> bool:
    return os.getenv("DEVICE", "cuda").lower() != "cuda"


def _tune_enhance_params(strength: float, guidance_scale: float, num_inference_steps: int) -> tuple[float, float, int]:
    # Em CPU do Render, forca parametros mais leves para reduzir 502 por timeout/OOM.
    if _is_cpu_mode() and os.getenv("CPU_SAFE_MODE", "1") == "1":
        cpu_max_steps = int(os.getenv("CPU_MAX_STEPS", "6"))
        cpu_max_guidance = float(os.getenv("CPU_MAX_GUIDANCE", "2.0"))
        strength = min(strength, 0.4)
        guidance_scale = min(guidance_scale, cpu_max_guidance)
        num_inference_steps = min(num_inference_steps, cpu_max_steps)
    return strength, guidance_scale, num_inference_steps


def _tune_inpaint_params(strength: float, guidance_scale: float, num_inference_steps: int) -> tuple[float, float, int]:
    # Inpainting costuma ser mais pesado; em CPU aplicamos teto ainda mais conservador.
    if _is_cpu_mode() and os.getenv("CPU_SAFE_MODE", "1") == "1":
        cpu_max_steps = int(os.getenv("CPU_MAX_INPAINT_STEPS", os.getenv("CPU_MAX_STEPS", "6")))
        cpu_max_guidance = float(os.getenv("CPU_MAX_INPAINT_GUIDANCE", "3.0"))
        strength = min(strength, 0.75)
        guidance_scale = min(guidance_scale, cpu_max_guidance)
        num_inference_steps = min(num_inference_steps, cpu_max_steps)
    return strength, guidance_scale, num_inference_steps


def _prepare_inpaint_mask(mask_image: Image.Image, target_size: tuple[int, int], blur_radius: float) -> Image.Image:
    mask = mask_image.convert("L")
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.Resampling.BILINEAR)

    threshold = int(os.getenv("INPAINT_MASK_THRESHOLD", "16"))
    mask = mask.point(lambda value: 255 if value >= threshold else 0)

    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return mask


def _composite_inpaint_result(
    original_rgb: Image.Image,
    generated_rgb: Image.Image,
    mask_image: Image.Image,
    blend_feather: float,
) -> Image.Image:
    blend_mask = mask_image
    if blend_feather > 0:
        blend_mask = blend_mask.filter(ImageFilter.GaussianBlur(radius=blend_feather))
    return Image.composite(generated_rgb.convert("RGB"), original_rgb.convert("RGB"), blend_mask)


def _warmup_models_background() -> None:
    # Em hospedagens com limite de boot (ex.: Render free/starter),
    # aquecer modelo no startup pode causar 502 intermitente.
    if os.getenv("WARMUP_ON_START", "0") != "1":
        return

    def _run() -> None:
        try:
            _get_pipeline()
        except Exception as exc:
            print(f"[warmup] img2img warmup failed: {exc}")

    Thread(target=_run, daemon=True).start()


@app.on_event("startup")
def on_startup() -> None:
    _warmup_models_background()


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/")
def root() -> dict:
    if _frontend_index.exists():
        return FileResponse(str(_frontend_index))

    return {
        "service": "portrait-studio-api",
        "ok": True,
        "health": "/health",
        "enhance": "/enhance",
        "inpaint": "/inpaint",
        "outpaint": "/outpaint",
    }


@app.get("/config.js")
def frontend_config() -> Response:
    config_path = _frontend_dir / "config.js"
    if config_path.exists():
        return FileResponse(str(config_path), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="config.js nao encontrado")


@app.post("/enhance")
async def enhance(
    file: UploadFile = File(...),
    prompt: str = Form(
        "professional studio headshot of the same person, preserve identity and facial geometry, "
        "natural skin texture with realistic pores, neutral white balance, soft even lighting, "
        "sharp eyes, clean edges, high realism"
    ),
    negative_prompt: str = Form(
        "cartoon, anime, painting, cgi, plastic skin, waxy skin, over-smoothing, lowres, blurry, "
        "noise, jpeg artifacts, deformed face, asymmetrical eyes, crossed eyes, extra eyes, extra limbs, "
        "duplicate person, text, watermark, logo"
    ),
    strength: float = Form(0.35),
    guidance_scale: float = Form(5.0),
    num_inference_steps: int = Form(10),
    seed: int = Form(-1),
) -> Response:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de imagem valido.")

    if strength < 0.05 or strength > 0.95:
        raise HTTPException(status_code=400, detail="strength deve estar entre 0.05 e 0.95.")

    if num_inference_steps < 1 or num_inference_steps > 80:
        raise HTTPException(status_code=400, detail="num_inference_steps deve estar entre 1 e 80.")

    try:
        data = await file.read()
        source = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Nao foi possivel ler a imagem: {exc}") from exc

    strength, guidance_scale, num_inference_steps = _tune_enhance_params(
        strength, guidance_scale, num_inference_steps
    )

    default_resolution = "256" if _is_cpu_mode() else "768"
    work_resolution = int(os.getenv("IMG2IMG_RESOLUTION", default_resolution))
    source_for_model = _fit_image_for_model(source, work_resolution)

    try:
        pipe = _get_pipeline()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha ao carregar modelo: {exc}") from exc

    generator = None
    if seed >= 0:
        _ensure_ml_imports()
        dev = "cuda" if os.getenv("DEVICE", "cuda").lower() == "cuda" else "cpu"
        generator = _torch.Generator(device=dev).manual_seed(seed)

    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source_for_model,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        output = result.images[0].resize(source.size, Image.Resampling.LANCZOS)
    except RuntimeError as exc:
        # Erro comum: OOM em GPU
        raise HTTPException(status_code=500, detail=f"Erro de inferencia: {exc}") from exc

    out_buf = io.BytesIO()
    output.save(out_buf, format="PNG")
    return Response(content=out_buf.getvalue(), media_type="image/png")


@app.post("/inpaint")
async def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(
        "repair only the masked region of the same person, preserve identity, expression, and pose, "
        "natural skin texture, realistic anatomy, seamless blend with surrounding pixels, "
        "matching lighting, color, and noise level"
    ),
    negative_prompt: str = Form(
        "change identity, different person, age change, gender change, unrealistic skin, plastic skin, "
        "deformed face, asymmetrical eyes, extra eyes, extra mouth, extra limbs, seam, halo, blur, "
        "text, watermark, logo"
    ),
    strength: float = Form(0.6),
    guidance_scale: float = Form(4.0),
    num_inference_steps: int = Form(10),
    seed: int = Form(-1),
    preserve_unmasked: bool = Form(True),
    mask_blur: float = Form(2.0),
    blend_feather: float = Form(1.25),
) -> Response:
    if strength < 0.05 or strength > 1.0:
        raise HTTPException(status_code=400, detail="strength deve estar entre 0.05 e 1.0.")

    if num_inference_steps < 1 or num_inference_steps > 100:
        raise HTTPException(status_code=400, detail="num_inference_steps deve estar entre 1 e 100.")

    if mask_blur < 0 or mask_blur > 24:
        raise HTTPException(status_code=400, detail="mask_blur deve estar entre 0 e 24.")

    if blend_feather < 0 or blend_feather > 24:
        raise HTTPException(status_code=400, detail="blend_feather deve estar entre 0 e 24.")

    source = _load_image_from_upload(image, mode="RGB", label="image")
    mask_image = _load_image_from_upload(mask, mode="L", label="mask")

    strength, guidance_scale, num_inference_steps = _tune_inpaint_params(
        strength, guidance_scale, num_inference_steps
    )

    mask_image = _prepare_inpaint_mask(mask_image, source.size, blur_radius=mask_blur)

    if mask_image.getbbox() is None:
        out_passthrough = io.BytesIO()
        source.save(out_passthrough, format="PNG")
        return Response(content=out_passthrough.getvalue(), media_type="image/png")

    default_resolution = "256" if _is_cpu_mode() else "512"
    inpaint_resolution = int(os.getenv("INPAINT_RESOLUTION", default_resolution))
    work_size = _working_size(source.size, inpaint_resolution)
    source_work = source.resize(work_size, Image.Resampling.LANCZOS)
    mask_work = mask_image.resize(work_size, Image.Resampling.BILINEAR)

    try:
        pipe = _get_inpaint_pipeline()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha ao carregar modelo de inpainting: {exc}") from exc

    generator = None
    if seed >= 0:
        _ensure_ml_imports()
        dev = "cuda" if os.getenv("DEVICE", "cuda").lower() == "cuda" else "cpu"
        generator = _torch.Generator(device=dev).manual_seed(seed)

    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source_work,
            mask_image=mask_work,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        generated = result.images[0].resize(source.size, Image.Resampling.LANCZOS)
        output = (
            _composite_inpaint_result(source, generated, mask_image, blend_feather)
            if preserve_unmasked
            else generated
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"Erro de inferencia no inpainting: {exc}") from exc

    out_buf = io.BytesIO()
    output.save(out_buf, format="PNG")
    return Response(content=out_buf.getvalue(), media_type="image/png")


@app.post("/auto-repair")
async def auto_repair(
    file: UploadFile = File(...),
    prompt: str = Form(
        "complete cropped portrait borders of the same person, restore missing hair, head contour, and shoulders, "
        "maintain identity and realistic anatomy, seamless continuation, natural texture and lighting"
    ),
    negative_prompt: str = Form(
        "different person, identity drift, artifacts, seam, halo, blur, deformed face, extra limbs, "
        "duplicate head, text, watermark, logo"
    ),
    strength: float = Form(0.82),
    guidance_scale: float = Form(7.0),
    num_inference_steps: int = Form(28),
    seed: int = Form(-1),
) -> Response:
    if strength < 0.05 or strength > 1.0:
        raise HTTPException(status_code=400, detail="strength deve estar entre 0.05 e 1.0.")

    if num_inference_steps < 1 or num_inference_steps > 100:
        raise HTTPException(status_code=400, detail="num_inference_steps deve estar entre 1 e 100.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de imagem valido.")

    try:
        source_data = await file.read()
        source_rgba = Image.open(io.BytesIO(source_data)).convert("RGBA")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Nao foi possivel ler a imagem: {exc}") from exc

    alpha = source_rgba.split()[3]
    bounds = _opaque_bounds(alpha)
    if not bounds:
        return Response(content=source_data, media_type="image/png")

    touches = _touches_edges(bounds, source_rgba.size, margin=max(2, int(min(source_rgba.size) * 0.015)))
    if not any(touches.values()):
        out_passthrough = io.BytesIO()
        source_rgba.save(out_passthrough, format="PNG")
        return Response(content=out_passthrough.getvalue(), media_type="image/png")

    mask_image = _auto_repair_mask(alpha, touches)
    if mask_image.getbbox() is None:
        out_passthrough = io.BytesIO()
        source_rgba.save(out_passthrough, format="PNG")
        return Response(content=out_passthrough.getvalue(), media_type="image/png")

    # Flatten over neutral background for stable inpainting, then recover transparency.
    base_rgb = Image.new("RGB", source_rgba.size, (235, 235, 235))
    base_rgb.paste(source_rgba.convert("RGB"), mask=alpha)

    strength, guidance_scale, num_inference_steps = _tune_inpaint_params(
        strength, guidance_scale, num_inference_steps
    )

    default_resolution = "256" if _is_cpu_mode() else "512"
    inpaint_resolution = int(os.getenv("INPAINT_RESOLUTION", default_resolution))
    work_size = _working_size(base_rgb.size, inpaint_resolution)
    base_work = base_rgb.resize(work_size, Image.Resampling.LANCZOS)
    mask_work = mask_image.resize(work_size, Image.Resampling.BILINEAR)

    try:
        pipe = _get_inpaint_pipeline()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha ao carregar modelo de inpainting: {exc}") from exc

    generator = None
    if seed >= 0:
        _ensure_ml_imports()
        dev = "cuda" if os.getenv("DEVICE", "cuda").lower() == "cuda" else "cpu"
        generator = _torch.Generator(device=dev).manual_seed(seed)

    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=base_work,
            mask_image=mask_work,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        repaired_rgb = result.images[0].convert("RGB")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"Erro de inferencia no auto-repair: {exc}") from exc

    if repaired_rgb.size != source_rgba.size:
        repaired_rgb = repaired_rgb.resize(source_rgba.size, Image.Resampling.LANCZOS)

    expanded_alpha = ImageChops.lighter(alpha, mask_image.point(lambda value: int(min(255, value * 1.25))))
    repaired_rgba = repaired_rgb.convert("RGBA")
    repaired_rgba.putalpha(expanded_alpha)

    out_buf = io.BytesIO()
    repaired_rgba.save(out_buf, format="PNG")
    return Response(content=out_buf.getvalue(), media_type="image/png")


@app.post("/outpaint")
async def outpaint(
    file: UploadFile = File(...),
    pad_top: int = Form(0),
    pad_left: int = Form(0),
    pad_right: int = Form(0),
    pad_bottom: int = Form(0),
    prompt: str = Form(
        "professional headshot photo of the same person, extend canvas naturally, complete missing head or hair "
        "only where needed, preserve facial identity and expression, natural shoulders and clothing, "
        "seamless continuation of background, matching perspective, color, and lighting, photorealistic"
    ),
    negative_prompt: str = Form(
        "different person, identity drift, cropped, frame, border, seam, halo, duplicate face, extra head, "
        "extra limbs, deformed anatomy, blurry, cartoon, text, watermark, logo"
    ),
    guidance_scale: float = Form(7.0),
    num_inference_steps: int = Form(30),
    seed: int = Form(-1),
) -> Response:
    """Completa partes cortadas (cabeca/ombros) expandindo a foto original nas bordas indicadas."""
    if num_inference_steps < 1 or num_inference_steps > 100:
        raise HTTPException(status_code=400, detail="num_inference_steps deve estar entre 1 e 100.")

    source = _load_image_from_upload(file, mode="RGB", label="file")
    pads = {"top": pad_top, "left": pad_left, "right": pad_right, "bottom": pad_bottom}
    max_pad = max(source.size)
    if any(value < 0 or value > max_pad for value in pads.values()):
        raise HTTPException(status_code=400, detail=f"Cada pad deve estar entre 0 e {max_pad} px.")

    if not any(pads.values()):
        out_passthrough = io.BytesIO()
        source.save(out_passthrough, format="PNG")
        return Response(content=out_passthrough.getvalue(), media_type="image/png")

    _, guidance_scale, num_inference_steps = _tune_inpaint_params(0.99, guidance_scale, num_inference_steps)

    overlap = max(8, int(min(source.size) * 0.02))
    canvas, mask = _outpaint_canvas(source, pads, overlap)
    default_resolution = "256" if _is_cpu_mode() else "1024"
    work_size = _working_size(canvas.size, int(os.getenv("INPAINT_RESOLUTION", default_resolution)))

    try:
        pipe = _get_inpaint_pipeline()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha ao carregar modelo de inpainting: {exc}") from exc

    generator = None
    if seed >= 0:
        _ensure_ml_imports()
        dev = "cuda" if os.getenv("DEVICE", "cuda").lower() == "cuda" else "cpu"
        generator = _torch.Generator(device=dev).manual_seed(seed)

    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=canvas.resize(work_size, Image.Resampling.LANCZOS),
            mask_image=mask.resize(work_size, Image.Resampling.BILINEAR),
            width=work_size[0],
            height=work_size[1],
            strength=0.99,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        generated = result.images[0].convert("RGB")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"Erro de inferencia no outpainting: {exc}") from exc

    # Mantem os pixels originais intactos; so a area nova (e a emenda) vem da IA.
    generated = generated.resize(canvas.size, Image.Resampling.LANCZOS)
    blend_mask = mask.filter(ImageFilter.GaussianBlur(radius=overlap / 2))
    final = Image.composite(generated, canvas, blend_mask)

    out_buf = io.BytesIO()
    final.save(out_buf, format="PNG")
    return Response(content=out_buf.getvalue(), media_type="image/png")
