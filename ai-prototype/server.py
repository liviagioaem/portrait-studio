from io import BytesIO

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image

from pipeline import (
    DEFAULT_MODEL,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    get_pipeline,
    run_inpaint,
)


app = FastAPI(title="Portrait Studio Inpainting API", version="0.2.0")

# The frontend runs from file:// (origin "null") or from a static host, so the
# browser needs an explicit CORS allowance to read the response.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "cuda": torch.cuda.is_available(), "model": DEFAULT_MODEL}


@app.post("/warmup")
def warmup():
    """Load the model ahead of the first real request so the UI does not time out."""
    pipe = get_pipeline()
    return {"ok": True, "device": str(pipe.device)}


@app.post("/inpaint")
async def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
    negative_prompt: str = Form(DEFAULT_NEGATIVE_PROMPT),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(7.0),
    seed: int = Form(-1),
):
    try:
        base = Image.open(BytesIO(await image.read())).convert("RGB")
        mask_image = Image.open(BytesIO(await mask.read()))

        out = run_inpaint(
            base,
            mask_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=None if seed is None or int(seed) < 0 else int(seed),
        )

        buffer = BytesIO()
        out.save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
