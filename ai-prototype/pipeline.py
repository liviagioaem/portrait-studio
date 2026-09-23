"""Shared inpainting logic used by the local API (server.py) and the CLI test (inpaint_test.py)."""

from functools import lru_cache

import numpy as np
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image, ImageFilter


DEFAULT_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"

# SD 1.5 was trained at 512px. Running far above that degrades quality and costs
# a lot of memory, so the diffusion step is capped and the result is scaled back.
MAX_DIFFUSION_SIDE = 768
MIN_DIFFUSION_SIDE = 512

DEFAULT_PROMPT = (
    "natural continuation of the person's hairstyle and portrait edges, "
    "realistic hair strands, matching original lighting and hair color, "
    "clean studio background, professional portrait"
)
DEFAULT_NEGATIVE_PROMPT = (
    "distorted face, deformed anatomy, duplicated hair, extra limbs, "
    "unnatural edges, seams, blurry details, artifacts, text, watermark"
)


@lru_cache(maxsize=2)
def get_pipeline(model_id: str = DEFAULT_MODEL):
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    pipe = AutoPipelineForInpainting.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda" if use_cuda else "cpu")
    pipe.set_progress_bar_config(disable=True)

    # Memory hints: allow larger canvases without OOM on modest GPUs.
    if use_cuda:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    return pipe


def device_of(pipe) -> str:
    return str(pipe.device)


def binarize_mask(mask: Image.Image) -> Image.Image:
    """White (>=50% luma) means inpaint, black means keep. Avoids weak/ambiguous regions."""
    luma = np.asarray(mask.convert("L"), dtype=np.uint8)
    return Image.fromarray(np.where(luma >= 128, 255, 0).astype(np.uint8), mode="L")


def _diffusion_size(width: int, height: int) -> tuple[int, int]:
    longest = max(width, height)
    scale = 1.0
    if longest > MAX_DIFFUSION_SIDE:
        scale = MAX_DIFFUSION_SIDE / longest
    elif longest < MIN_DIFFUSION_SIDE:
        scale = MIN_DIFFUSION_SIDE / longest

    # Latent space works in blocks of 8 pixels.
    target_w = max(8, int(round(width * scale / 8)) * 8)
    target_h = max(8, int(round(height * scale / 8)) * 8)
    return target_w, target_h


def _composite(base: Image.Image, filled: Image.Image, mask: Image.Image) -> Image.Image:
    """Keep the original pixels outside the mask, feathering only the seam.

    The VAE round-trip softens the whole frame, so pasting the untouched original
    back is what preserves facial detail; the feather hides the hard mask edge.
    """
    radius = max(1.5, min(base.size) / 220)
    feather = mask.filter(ImageFilter.GaussianBlur(radius=radius))

    alpha = np.asarray(feather, dtype=np.float32)[:, :, None] / 255.0
    base_arr = np.asarray(base, dtype=np.float32)
    filled_arr = np.asarray(filled, dtype=np.float32)
    blended = base_arr * (1.0 - alpha) + filled_arr * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def run_inpaint(
    base: Image.Image,
    mask: Image.Image,
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.0,
    seed: int | None = None,
    model_id: str = DEFAULT_MODEL,
) -> Image.Image:
    """Inpaint `mask` (white=fill) over `base`, returning an image at the original size."""
    base = base.convert("RGB")
    original_size = base.size
    mask = binarize_mask(mask.resize(original_size, Image.NEAREST))

    work_size = _diffusion_size(*original_size)
    work_base = base.resize(work_size, Image.LANCZOS)
    work_mask = mask.resize(work_size, Image.NEAREST)

    pipe = get_pipeline(model_id)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(int(seed))

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=work_base,
        mask_image=work_mask,
        width=work_size[0],
        height=work_size[1],
        num_inference_steps=max(8, min(60, int(num_inference_steps))),
        guidance_scale=max(1.0, min(15.0, float(guidance_scale))),
        generator=generator,
    ).images[0]

    filled = result.convert("RGB").resize(original_size, Image.LANCZOS)
    return _composite(base, filled, mask)
