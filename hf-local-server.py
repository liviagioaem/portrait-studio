from io import BytesIO
from threading import Lock

import torch
from diffusers import AutoPipelineForImage2Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from starlette.responses import Response

app = FastAPI(title="HF Local Image Enhancer")

_PIPELINE = None
_PIPELINE_MODEL_ID = None
_PIPELINE_LOCK = Lock()


def _load_pipeline(model_id: str):
    global _PIPELINE, _PIPELINE_MODEL_ID

    with _PIPELINE_LOCK:
        if _PIPELINE is not None and _PIPELINE_MODEL_ID == model_id:
            return _PIPELINE

        has_cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if has_cuda else torch.float32

        pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
        pipe = pipe.to("cuda" if has_cuda else "cpu")

        _PIPELINE = pipe
        _PIPELINE_MODEL_ID = model_id
        return _PIPELINE


@app.get("/health")
def health():
    return {"ok": True, "loaded_model": _PIPELINE_MODEL_ID}


@app.post("/enhance")
async def enhance(
    image: UploadFile = File(...),
    model: str = Form("black-forest-labs/FLUX.2-klein-base-9b-fp8"),
    prompt: str = Form(""),
    strength: float = Form(0.8),
    guidance_scale: float = Form(3.5),
    num_inference_steps: int = Form(28),
):
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Campo 'image' deve ser uma imagem.")

    try:
        payload = await image.read()
        src = Image.open(BytesIO(payload)).convert("RGB")

        pipe = _load_pipeline(model)
        effective_prompt = prompt.strip() or "Enhance details while preserving composition."

        result = pipe(
            prompt=effective_prompt,
            image=src,
            strength=max(0.0, min(1.0, strength)),
            guidance_scale=max(0.0, guidance_scale),
            num_inference_steps=max(1, num_inference_steps),
        )

        out = result.images[0]
        out_buf = BytesIO()
        out.save(out_buf, format="PNG")
        return Response(content=out_buf.getvalue(), media_type="image/png")
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
