import io
import os
from threading import Lock

import numpy as np
import torch
from diffusers import AutoPipelineForInpainting, FluxImg2ImgPipeline
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
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


def _get_pipeline() -> FluxImg2ImgPipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _model_lock:
        if _pipeline is not None:
            return _pipeline

        model_id = os.getenv("FLUX_MODEL_ID", "black-forest-labs/FLUX.2-klein-base-9b-fp8")
        token = os.getenv("HF_TOKEN") or None
        device = os.getenv("DEVICE", "cuda").lower()

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA nao disponivel. Configure DEVICE=cpu (lento) ou use GPU NVIDIA.")

        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        pipe = FluxImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=token,
            local_files_only=False,
        )

        if device == "cuda":
            pipe = pipe.to("cuda")
        else:
            pipe = pipe.to("cpu")

        _pipeline = pipe
        return _pipeline


def _get_inpaint_pipeline():
    global _inpaint_pipeline
    if _inpaint_pipeline is not None:
        return _inpaint_pipeline

    with _model_lock:
        if _inpaint_pipeline is not None:
            return _inpaint_pipeline

        # SDXL inpainting (~7 GB VRAM). Para GPUs menores use
        # INPAINT_MODEL_ID=stable-diffusion-v1-5/stable-diffusion-inpainting e INPAINT_RESOLUTION=512.
        model_id = os.getenv("INPAINT_MODEL_ID", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
        token = os.getenv("HF_TOKEN") or None
        device = os.getenv("DEVICE", "cuda").lower()

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA nao disponivel. Configure DEVICE=cpu (lento) ou use GPU NVIDIA.")

        dtype = torch.float16 if device == "cuda" else torch.float32
        load_kwargs = {"torch_dtype": dtype, "token": token, "local_files_only": False}

        try:
            pipe = AutoPipelineForInpainting.from_pretrained(
                model_id, variant="fp16" if dtype == torch.float16 else None, **load_kwargs
            )
        except (OSError, ValueError):
            # Nem todo repositorio publica pesos na variante fp16.
            pipe = AutoPipelineForInpainting.from_pretrained(model_id, **load_kwargs)

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


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/enhance")
async def enhance(
    file: UploadFile = File(...),
    prompt: str = Form("high quality portrait, realistic skin detail, balanced lighting, natural colors"),
    negative_prompt: str = Form("artifacts, blur, extra fingers, distorted face, oversharpen, watermark"),
    strength: float = Form(0.35),
    guidance_scale: float = Form(3.5),
    num_inference_steps: int = Form(30),
    seed: int = Form(-1),
) -> Response:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de imagem valido.")

    if strength < 0.05 or strength > 0.95:
        raise HTTPException(status_code=400, detail="strength deve estar entre 0.05 e 0.95.")

    if num_inference_steps < 10 or num_inference_steps > 80:
        raise HTTPException(status_code=400, detail="num_inference_steps deve estar entre 10 e 80.")

    try:
        data = await file.read()
        source = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Nao foi possivel ler a imagem: {exc}") from exc

    try:
        pipe = _get_pipeline()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha ao carregar modelo: {exc}") from exc

    generator = None
    if seed >= 0:
        dev = "cuda" if os.getenv("DEVICE", "cuda").lower() == "cuda" else "cpu"
        generator = torch.Generator(device=dev).manual_seed(seed)

    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        output = result.images[0]
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
    prompt: str = Form("clean natural portrait skin, realistic details, consistent lighting"),
    negative_prompt: str = Form("artifacts, blur, deformed face, text, watermark"),
    strength: float = Form(0.75),
    guidance_scale: float = Form(7.5),
    num_inference_steps: int = Form(30),
    seed: int = Form(-1),
) -> Response:
    if strength < 0.05 or strength > 1.0:
        raise HTTPException(status_code=400, detail="strength deve estar entre 0.05 e 1.0.")

    if num_inference_steps < 10 or num_inference_steps > 100:
        raise HTTPException(status_code=400, detail="num_inference_steps deve estar entre 10 e 100.")

    source = _load_image_from_upload(image, mode="RGB", label="image")
    mask_image = _load_image_from_upload(mask, mode="L", label="mask")

    if source.size != mask_image.size:
        mask_image = mask_image.resize(source.size)

    try:
        pipe = _get_inpaint_pipeline()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha ao carregar modelo de inpainting: {exc}") from exc

    generator = None
    if seed >= 0:
        dev = "cuda" if os.getenv("DEVICE", "cuda").lower() == "cuda" else "cpu"
        generator = torch.Generator(device=dev).manual_seed(seed)

    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source,
            mask_image=mask_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        output = result.images[0]
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"Erro de inferencia no inpainting: {exc}") from exc

    out_buf = io.BytesIO()
    output.save(out_buf, format="PNG")
    return Response(content=out_buf.getvalue(), media_type="image/png")


@app.post("/auto-repair")
async def auto_repair(
    file: UploadFile = File(...),
    prompt: str = Form("complete cropped portrait edges, restore missing head or shoulders, realistic anatomy"),
    negative_prompt: str = Form("artifacts, blur, extra limbs, deformed face, text, watermark"),
    strength: float = Form(0.82),
    guidance_scale: float = Form(7.0),
    num_inference_steps: int = Form(28),
    seed: int = Form(-1),
) -> Response:
    if strength < 0.05 or strength > 1.0:
        raise HTTPException(status_code=400, detail="strength deve estar entre 0.05 e 1.0.")

    if num_inference_steps < 10 or num_inference_steps > 100:
        raise HTTPException(status_code=400, detail="num_inference_steps deve estar entre 10 e 100.")

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

    try:
        pipe = _get_inpaint_pipeline()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha ao carregar modelo de inpainting: {exc}") from exc

    generator = None
    if seed >= 0:
        dev = "cuda" if os.getenv("DEVICE", "cuda").lower() == "cuda" else "cpu"
        generator = torch.Generator(device=dev).manual_seed(seed)

    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=base_rgb,
            mask_image=mask_image,
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
        "professional headshot photo of the same person, complete head and hair, natural shoulders and clothing, "
        "seamless continuation of the photo, consistent lighting and background, realistic"
    ),
    negative_prompt: str = Form(
        "cropped, frame, border, seam, text, watermark, extra head, extra limbs, deformed, blurry, cartoon"
    ),
    guidance_scale: float = Form(7.0),
    num_inference_steps: int = Form(30),
    seed: int = Form(-1),
) -> Response:
    """Completa partes cortadas (cabeca/ombros) expandindo a foto original nas bordas indicadas."""
    if num_inference_steps < 10 or num_inference_steps > 100:
        raise HTTPException(status_code=400, detail="num_inference_steps deve estar entre 10 e 100.")

    source = _load_image_from_upload(file, mode="RGB", label="file")
    pads = {"top": pad_top, "left": pad_left, "right": pad_right, "bottom": pad_bottom}
    max_pad = max(source.size)
    if any(value < 0 or value > max_pad for value in pads.values()):
        raise HTTPException(status_code=400, detail=f"Cada pad deve estar entre 0 e {max_pad} px.")

    if not any(pads.values()):
        out_passthrough = io.BytesIO()
        source.save(out_passthrough, format="PNG")
        return Response(content=out_passthrough.getvalue(), media_type="image/png")

    overlap = max(8, int(min(source.size) * 0.02))
    canvas, mask = _outpaint_canvas(source, pads, overlap)
    work_size = _working_size(canvas.size, int(os.getenv("INPAINT_RESOLUTION", "1024")))

    try:
        pipe = _get_inpaint_pipeline()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha ao carregar modelo de inpainting: {exc}") from exc

    generator = None
    if seed >= 0:
        dev = "cuda" if os.getenv("DEVICE", "cuda").lower() == "cuda" else "cpu"
        generator = torch.Generator(device=dev).manual_seed(seed)

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
