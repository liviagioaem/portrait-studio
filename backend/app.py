from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageChops, ImageDraw, ImageFilter

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


# ============================================================================
# Configuration
# ============================================================================

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("portrait-studio")


@dataclass(frozen=True)
class Settings:
    device: str
    low_vram: bool
    max_quality: bool
    cpu_safe_mode: bool
    max_upload_mb: int
    cors_origins: list[str]
    cors_allow_credentials: bool

    img2img_model_id: str
    inpaint_model_id: str
    img2img_resolution: int
    inpaint_resolution: int

    cpu_max_steps: int
    cpu_max_guidance: float
    cpu_max_inpaint_steps: int
    cpu_max_inpaint_guidance: float

    gfpgan_enabled: bool
    gfpgan_model_path: str
    gfpgan_upscale: int
    gfpgan_arch: str
    gfpgan_channel_multiplier: int

    face_protect: bool
    face_protect_strength: float
    face_identity_lock: bool
    face_identity_lock_strength: float
    face_lock_min_mask: int

    color_harmonize: bool
    color_harmonize_strength: float
    local_sharpen: bool
    opencv_detail: bool
    gfpgan_restore: bool

    warmup_on_start: bool
    auto_repair_inner_transparency: bool

    credential_output_size: int
    credential_eye_target_x: float
    credential_eye_target_y: float
    credential_top_head_margin: float
    credential_bottom_shoulder_margin: float
    credential_face_scale: float
    credential_min_head_margin: float
    credential_min_shoulder_visibility: float
    credential_max_crop_ratio: float
    credential_background_threshold: int
    credential_artifact_threshold: float
    credential_validation_tolerance: float
    credential_face_center_weight: float
    credential_max_side_from_face_ratio: float
    credential_min_side_from_face_ratio: float
    credential_clothing_enabled: bool
    credential_clothing_min_required: float
    credential_clothing_mask_padding: int
    credential_clothing_generation_strength: float
    credential_clothing_steps: int
    credential_clothing_guidance_scale: float
    credential_clothing_type: str
    credential_clothing_neck_protect_ratio: float
    credential_clothing_head_protect_ratio: float
    credential_clothing_max_mask_area_ratio: float
    credential_debug_mask: bool

    @classmethod
    def from_env(cls) -> "Settings":
        device = os.getenv("DEVICE", "cuda").strip().lower()
        if device not in {"cuda", "cpu"}:
            raise RuntimeError("DEVICE deve ser 'cuda' ou 'cpu'.")

        def boolean(name: str, default: bool = False) -> bool:
            value = os.getenv(name, "1" if default else "0")
            return value.strip().lower() in {"1", "true", "yes", "on"}

        def integer(name: str, default: int) -> int:
            return int(os.getenv(name, str(default)))

        def number(name: str, default: float) -> float:
            return float(os.getenv(name, str(default)))

        origins = os.getenv("CORS_ORIGINS", "*").strip()
        cors_origins = (
            ["*"]
            if origins == "*"
            else [item.strip() for item in origins.split(",") if item.strip()]
        )

        max_quality = boolean("MAX_QUALITY_MODE", True)

        return cls(
            device=device,
            low_vram=boolean("LOW_VRAM"),
            max_quality=max_quality,
            cpu_safe_mode=boolean("CPU_SAFE_MODE", True),
            max_upload_mb=integer("MAX_UPLOAD_MB", 15),
            cors_origins=cors_origins,
            cors_allow_credentials=boolean("CORS_ALLOW_CREDENTIALS"),
            img2img_model_id=os.getenv(
                "IMG2IMG_MODEL_ID",
                "runwayml/stable-diffusion-v1-5",
            ),
            inpaint_model_id=os.getenv("INPAINT_MODEL_ID", ""),
            img2img_resolution=integer(
                "IMG2IMG_RESOLUTION",
                256 if device == "cpu" else (1024 if max_quality else 768),
            ),
            inpaint_resolution=integer(
                "INPAINT_RESOLUTION",
                256 if device == "cpu" else (1024 if max_quality else 512),
            ),
            cpu_max_steps=integer("CPU_MAX_STEPS", 6),
            cpu_max_guidance=number("CPU_MAX_GUIDANCE", 2.0),
            cpu_max_inpaint_steps=integer(
                "CPU_MAX_INPAINT_STEPS",
                integer("CPU_MAX_STEPS", 6),
            ),
            cpu_max_inpaint_guidance=number("CPU_MAX_INPAINT_GUIDANCE", 3.0),
            gfpgan_enabled=boolean("GFPGAN_ENABLED", True),
            gfpgan_model_path=os.getenv(
                "GFPGAN_MODEL_PATH",
                "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            ),
            gfpgan_upscale=integer("GFPGAN_UPSCALE", 1),
            gfpgan_arch=os.getenv("GFPGAN_ARCH", "clean"),
            gfpgan_channel_multiplier=integer("GFPGAN_CHANNEL_MULTIPLIER", 2),
            face_protect=boolean("INPAINT_FACE_PROTECT", True),
            face_protect_strength=max(
                0.0, min(1.0, number("INPAINT_FACE_PROTECT_STRENGTH", 0.88))
            ),
            face_identity_lock=boolean("INPAINT_FACE_IDENTITY_LOCK", True),
            face_identity_lock_strength=max(
                0.0,
                min(
                    0.85,
                    number("INPAINT_FACE_IDENTITY_LOCK_STRENGTH", 0.46),
                ),
            ),
            face_lock_min_mask=integer("INPAINT_FACE_LOCK_MIN_MASK", 64),
            color_harmonize=boolean("INPAINT_COLOR_HARMONIZE", True),
            color_harmonize_strength=max(
                0.0,
                min(0.7, number("INPAINT_COLOR_HARMONIZE_STRENGTH", 0.22)),
            ),
            local_sharpen=boolean("INPAINT_LOCAL_SHARPEN", True),
            opencv_detail=boolean("INPAINT_OPENCV_DETAIL", True),
            gfpgan_restore=boolean("INPAINT_GFPGAN_RESTORE", True),
            warmup_on_start=boolean("WARMUP_ON_START"),
            auto_repair_inner_transparency=boolean(
                "AUTO_REPAIR_ALLOW_INNER_TRANSPARENCY",
                True,
            ),
            credential_output_size=integer("CREDENTIAL_OUTPUT_SIZE", 1024),
            credential_eye_target_x=max(0.2, min(0.8, number("CREDENTIAL_EYE_TARGET_X", 0.5))),
            credential_eye_target_y=max(0.2, min(0.55, number("CREDENTIAL_EYE_TARGET_Y", 0.40))),
            credential_top_head_margin=max(0.04, min(0.30, number("CREDENTIAL_TOP_HEAD_MARGIN", 0.12))),
            credential_bottom_shoulder_margin=max(0.05, min(0.35, number("CREDENTIAL_BOTTOM_SHOULDER_MARGIN", 0.16))),
            credential_face_scale=max(0.20, min(0.70, number("CREDENTIAL_FACE_SCALE", 0.36))),
            credential_min_head_margin=max(0.03, min(0.20, number("CREDENTIAL_MIN_HEAD_MARGIN", 0.07))),
            credential_min_shoulder_visibility=max(0.04, min(0.40, number("CREDENTIAL_MIN_SHOULDER_VISIBILITY", 0.11))),
            credential_max_crop_ratio=max(1.0, min(3.0, number("CREDENTIAL_MAX_CROP_RATIO", 1.75))),
            credential_background_threshold=max(0, min(255, integer("CREDENTIAL_BACKGROUND_THRESHOLD", 20))),
            credential_artifact_threshold=max(0.0, min(1.0, number("CREDENTIAL_ARTIFACT_THRESHOLD", 0.20))),
            credential_validation_tolerance=max(0.01, min(0.20, number("CREDENTIAL_VALIDATION_TOLERANCE", 0.06))),
            credential_face_center_weight=max(0.0, min(1.0, number("CREDENTIAL_FACE_CENTER_WEIGHT", 0.38))),
            credential_max_side_from_face_ratio=max(1.1, min(3.5, number("CREDENTIAL_MAX_SIDE_FROM_FACE_RATIO", 1.85))),
            credential_min_side_from_face_ratio=max(0.3, min(1.2, number("CREDENTIAL_MIN_SIDE_FROM_FACE_RATIO", 0.72))),
            credential_clothing_enabled=boolean("CREDENTIAL_CLOTHING_ENABLED", True),
            credential_clothing_min_required=max(0.01, min(0.95, number("CREDENTIAL_CLOTHING_MIN_REQUIRED", 0.28))),
            credential_clothing_mask_padding=max(0, min(64, integer("CREDENTIAL_CLOTHING_MASK_PADDING", 10))),
            credential_clothing_generation_strength=max(0.05, min(0.95, number("CREDENTIAL_CLOTHING_GENERATION_STRENGTH", 0.42))),
            credential_clothing_steps=max(8, min(120, integer("CREDENTIAL_CLOTHING_STEPS", 28))),
            credential_clothing_guidance_scale=max(0.0, min(20.0, number("CREDENTIAL_CLOTHING_GUIDANCE_SCALE", 3.2))),
            credential_clothing_type=os.getenv("CREDENTIAL_CLOTHING_TYPE", "auto").strip().lower(),
            credential_clothing_neck_protect_ratio=max(0.02, min(0.40, number("CREDENTIAL_CLOTHING_NECK_PROTECT_RATIO", 0.22))),
            credential_clothing_head_protect_ratio=max(0.15, min(1.0, number("CREDENTIAL_CLOTHING_HEAD_PROTECT_RATIO", 0.65))),
            credential_clothing_max_mask_area_ratio=max(0.01, min(0.50, number("CREDENTIAL_CLOTHING_MAX_MASK_AREA_RATIO", 0.16))),
            credential_debug_mask=boolean("CREDENTIAL_DEBUG_MASK"),
        )


SETTINGS = Settings.from_env()

if SETTINGS.credential_clothing_type not in {"auto", "male_formal", "female_formal", "disabled"}:
    raise RuntimeError(
        "CREDENTIAL_CLOTHING_TYPE deve ser auto, male_formal, female_formal ou disabled."
    )


# ============================================================================
# Prompts
# ============================================================================

DEFAULT_ENHANCE_PROMPT = (
    "ultra-clean professional studio headshot of the same person, "
    "preserve identity and facial geometry, accurate eye shape and gaze, "
    "natural lips and nose proportions, realistic skin pores and fine detail, "
    "balanced facial symmetry, neutral white balance, soft diffused studio "
    "lighting, subtle micro-contrast, sharp but natural focus, photorealistic, "
    "no overprocessing"
)

DEFAULT_ENHANCE_NEGATIVE = (
    "cartoon, anime, painting, cgi, plastic skin, waxy skin, oversharpen, "
    "over-smoothing, lowres, blurry, noise, jpeg artifacts, dark blotch, "
    "black smudge, muddy shadows, deformed face, asymmetrical eyes, crossed "
    "eyes, extra eyes, extra limbs, duplicate person, uncanny face, text, "
    "watermark, logo"
)

DEFAULT_INPAINT_PROMPT = (
    "reconstruct only the masked region as a premium standardized professional "
    "portrait, create missing parts with plausible anatomy and coherent "
    "proportions, preserve person identity when visible, rebuild coherent "
    "hairline, forehead, ears, jawline, neck and shoulders, centered headshot "
    "composition, if upper-body clothing is missing, generate formal business "
    "attire with a dark suit jacket and white shirt (optional subtle tie), "
    "natural skin microtexture, realistic pores, subtle skin variation, "
    "consistent color temperature, lighting and grain, seamless transition "
    "with surrounding pixels, photorealistic"
)

DEFAULT_INPAINT_NEGATIVE = (
    "painting, cartoon, anime, cgi, doll face, plastic skin, waxy skin, "
    "black smudge, dark blotch, muddy texture, blur, seam, halo, ghosting, "
    "double exposure, duplicated face, extra eyes, extra mouth, extra nose, "
    "extra limbs, deformed anatomy, distorted perspective, patchy skin, text, "
    "watermark, logo"
)

DEFAULT_REPAIR_PROMPT = (
    "complete cropped portrait borders by constructing missing head and upper "
    "body regions with photorealistic quality, restore natural hair volume, "
    "skull contour, ears, neck and shoulders with realistic anatomy and "
    "proportion, when clothing is missing in generated areas, create formal "
    "studio-portrait attire (dark suit jacket, white shirt, optional tie), "
    "preserve person identity when visible, maintain standardized studio "
    "portrait style, coherent lighting, consistent perspective and neutral "
    "background"
)

DEFAULT_REPAIR_NEGATIVE = (
    "cartoon, cgi, black patch, dark stain, seam, halo, blur, muddy shadows, "
    "identity drift, duplicated head, deformed face, extra limbs, stretched "
    "anatomy, text, watermark, logo"
)

DEFAULT_OUTPAINT_PROMPT = (
    "professional standardized headshot, extend canvas and construct missing "
    "portrait areas with high realism, complete top of head, hair, neck and "
    "shoulders when absent, preserve person identity when visible, if the "
    "chest/shoulder clothing region is missing, generate formal business "
    "clothing: dark suit jacket and white shirt with clean collar, optional "
    "subtle tie, natural proportions, centered passport-style framing, clean "
    "neutral studio background, consistent perspective, color and soft "
    "diffused lighting, seamless transitions, photorealistic"
)

DEFAULT_OUTPAINT_NEGATIVE = (
    "cartoon, cgi, black smudge, muddy texture, dark patch, cropped, border, "
    "seam, halo, duplicate face, extra head, extra limbs, deformed anatomy, "
    "blurry, warped shoulders, text, watermark, logo"
)

DEFAULT_CLOTHING_PROMPT_MALE = (
    "simple professional corporate portrait, formal dark neutral suit jacket, "
    "clean white dress shirt, realistic natural fabric, understated business attire, "
    "professional corporate headshot"
)

DEFAULT_CLOTHING_PROMPT_FEMALE = (
    "professional corporate portrait, elegant formal blouse, professional "
    "business attire, subtle neutral colors, clean tailored appearance, corporate portrait"
)

DEFAULT_CLOTHING_NEGATIVE = (
    "tie, scarf, neck accessory, colorful clothing, patterned clothing, bright blue clothing, "
    "bright green clothing, ornaments, jewelry, logo, text, watermark, fashion clothing, "
    "casual clothing, sportswear, fantasy clothing, deformed clothing, painted clothing, "
    "abstract clothing, fabric artifacts, face alteration, different person, face regeneration, "
    "hair alteration, hair artifacts, deformed neck, extra neck, extra head, extra face, skin artifacts, "
    "extra limbs, hands, arms, unnatural shoulders, deformed torso"
)


# ============================================================================
# Application
# ============================================================================

app = FastAPI(
    title="Portrait Studio FLUX API",
    version="1.0.0",
    description="API for standardized portrait enhancement, repair and outpainting.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS.cors_origins,
    allow_credentials=SETTINGS.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent
FRONTEND_INDEX = FRONTEND_DIR / "index.html"


# ============================================================================
# Runtime / model management
# ============================================================================

class ModelManager:
    """Lazy, thread-safe model lifecycle manager."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._torch: Any = None
        self._img2img_cls: Any = None
        self._inpaint_cls: Any = None
        self._img2img_pipeline: Any = None
        self._inpaint_pipeline: Any = None
        self._gfpgan: Any = None
        self.img2img_pipeline_name: str | None = None
        self.inpaint_pipeline_name: str | None = None

    def ensure_imports(self) -> None:
        if (
            self._torch is not None
            and self._img2img_cls is not None
            and self._inpaint_cls is not None
        ):
            return

        import torch

        try:
            from diffusers import AutoPipelineForInpainting

            self._inpaint_cls = AutoPipelineForInpainting
            self.inpaint_pipeline_name = "AutoPipelineForInpainting"
        except ImportError:
            from diffusers import StableDiffusionInpaintPipeline

            self._inpaint_cls = StableDiffusionInpaintPipeline
            self.inpaint_pipeline_name = "StableDiffusionInpaintPipeline"

        try:
            from diffusers import AutoPipelineForImage2Image

            self._img2img_cls = AutoPipelineForImage2Image
            self.img2img_pipeline_name = "AutoPipelineForImage2Image"
        except ImportError:
            from diffusers import StableDiffusionImg2ImgPipeline

            self._img2img_cls = StableDiffusionImg2ImgPipeline
            self.img2img_pipeline_name = "StableDiffusionImg2ImgPipeline"

        self._torch = torch

    def validate_device(self) -> None:
        self.ensure_imports()

        if SETTINGS.device == "cuda" and not self._torch.cuda.is_available():
            raise RuntimeError(
                "CUDA nao disponivel. Configure DEVICE=cpu ou use uma GPU NVIDIA."
            )

    def dtype(self) -> Any:
        self.ensure_imports()
        return (
            self._torch.float16
            if SETTINGS.device == "cuda"
            else self._torch.float32
        )

    def load_pipeline(self, cls: Any, model_id: str) -> Any:
        self.ensure_imports()

        kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype(),
            "local_files_only": False,
        }

        token = os.getenv("HF_TOKEN")
        if token:
            kwargs["token"] = token

        if self.dtype() == self._torch.float16:
            try:
                return cls.from_pretrained(
                    model_id,
                    variant="fp16",
                    **kwargs,
                )
            except (OSError, ValueError, TypeError):
                logger.warning(
                    "Modelo %s nao possui variante fp16 compatível; "
                    "tentando carregamento padrão.",
                    model_id,
                )

        return cls.from_pretrained(model_id, **kwargs)

    def prepare_pipeline(self, pipeline: Any) -> Any:
        if SETTINGS.device == "cuda" and SETTINGS.low_vram:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline = pipeline.to(SETTINGS.device)

        return pipeline

    def get_img2img(self) -> Any:
        self.ensure_imports()

        if self._img2img_pipeline is not None:
            return self._img2img_pipeline

        with self._lock:
            if self._img2img_pipeline is not None:
                return self._img2img_pipeline

            self.validate_device()
            pipeline = self.load_pipeline(
                self._img2img_cls,
                SETTINGS.img2img_model_id,
            )
            self._img2img_pipeline = self.prepare_pipeline(pipeline)

            logger.info(
                "img2img carregado: model=%s device=%s",
                SETTINGS.img2img_model_id,
                SETTINGS.device,
            )
            return self._img2img_pipeline

    def get_inpaint(self) -> Any:
        self.ensure_imports()

        if self._inpaint_pipeline is not None:
            return self._inpaint_pipeline

        with self._lock:
            if self._inpaint_pipeline is not None:
                return self._inpaint_pipeline

            self.validate_device()

            default_model = (
                "black-forest-labs/FLUX.1-Fill-dev"
                if self.inpaint_pipeline_name == "AutoPipelineForInpainting"
                else "runwayml/stable-diffusion-inpainting"
            )
            model_id = SETTINGS.inpaint_model_id or default_model

            pipeline = self.load_pipeline(self._inpaint_cls, model_id)
            self._inpaint_pipeline = self.prepare_pipeline(pipeline)

            logger.info(
                "inpaint carregado: model=%s device=%s",
                model_id,
                SETTINGS.device,
            )
            return self._inpaint_pipeline

    def get_gfpgan(self) -> Any | None:
        if not SETTINGS.gfpgan_enabled or cv2 is None:
            return None

        if self._gfpgan is not None:
            return self._gfpgan

        with self._lock:
            if self._gfpgan is not None:
                return self._gfpgan

            try:
                from gfpgan import GFPGANer  # type: ignore

                self._gfpgan = GFPGANer(
                    model_path=SETTINGS.gfpgan_model_path,
                    upscale=SETTINGS.gfpgan_upscale,
                    arch=SETTINGS.gfpgan_arch,
                    channel_multiplier=SETTINGS.gfpgan_channel_multiplier,
                    bg_upsampler=None,
                )
                logger.info("GFPGAN carregado.")
            except Exception:
                logger.exception("Falha ao carregar GFPGAN.")
                self._gfpgan = None

        return self._gfpgan

    def seed_generator(self, seed: int) -> Any | None:
        if seed < 0:
            return None

        self.ensure_imports()
        return self._torch.Generator(device=SETTINGS.device).manual_seed(seed)

    def status(self) -> dict[str, Any]:
        return {
            "device": SETTINGS.device,
            "img2img_loaded": self._img2img_pipeline is not None,
            "inpaint_loaded": self._inpaint_pipeline is not None,
            "gfpgan_loaded": self._gfpgan is not None,
            "opencv_available": cv2 is not None,
        }


MODELS = ModelManager()


# ============================================================================
# General utilities
# ============================================================================

def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "1" if default else "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def png_response(image: Image.Image) -> Response:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
    )


def working_size(
    size: tuple[int, int],
    target: int,
) -> tuple[int, int]:
    width, height = size
    if target <= 0:
        raise ValueError("target deve ser maior que zero.")

    scale = target / max(width, height)

    return (
        max(64, round(width * scale / 8) * 8),
        max(64, round(height * scale / 8) * 8),
    )


def fit_image_for_model(
    image: Image.Image,
    target: int,
) -> Image.Image:
    return image.resize(
        working_size(image.size, target),
        Image.Resampling.LANCZOS,
    )


def validate_steps(steps: int, maximum: int = 180) -> None:
    if steps < 1 or steps > maximum:
        raise HTTPException(
            status_code=400,
            detail=f"num_inference_steps deve estar entre 1 e {maximum}.",
        )


def validate_guidance(guidance: float) -> None:
    if guidance < 0 or guidance > 30:
        raise HTTPException(
            status_code=400,
            detail="guidance_scale deve estar entre 0 e 30.",
        )


def validate_strength(
    strength: float,
    maximum: float = 1.0,
) -> None:
    if strength < 0.05 or strength > maximum:
        raise HTTPException(
            status_code=400,
            detail=f"strength deve estar entre 0.05 e {maximum}.",
        )


# ============================================================================
# Upload / image validation
# ============================================================================

def read_upload_bytes(
    upload: UploadFile,
    label: str,
) -> bytes:
    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"{label} precisa ser um arquivo de imagem valido.",
        )

    max_bytes = SETTINGS.max_upload_mb * 1024 * 1024
    data = upload.file.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail=f"{label} esta vazio.",
        )

    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"{label} excede o limite de "
                f"{SETTINGS.max_upload_mb} MB."
            ),
        )

    return data


def load_image_from_bytes(
    data: bytes,
    mode: str,
    label: str,
) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()

        with Image.open(io.BytesIO(data)) as image:
            loaded = image.convert(mode)

        if loaded.width < 64 or loaded.height < 64:
            raise HTTPException(
                status_code=400,
                detail=f"{label} precisa ter pelo menos 64x64 pixels.",
            )

        return loaded
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Nao foi possivel ler {label}: {exc}",
        ) from exc


def load_upload(
    upload: UploadFile,
    *,
    mode: str,
    label: str,
) -> Image.Image:
    return load_image_from_bytes(
        read_upload_bytes(upload, label),
        mode,
        label,
    )


# ============================================================================
# Face detection
# ============================================================================

@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class EyePair:
    left: tuple[float, float]
    right: tuple[float, float]
    source: str


@dataclass(frozen=True)
class CredentialFramingResult:
    square: Image.Image
    round_image: Image.Image
    status: str
    reasons: list[str]
    metrics: dict[str, float]
    clothing_status: str
    clothing_generated: bool


def detect_primary_face(image: Image.Image) -> FaceBox | None:
    """Detecta o maior rosto frontal disponível.

    A implementação atual usa Haar Cascade para manter as dependências leves.
    O restante do sistema não depende diretamente do detector e pode receber
    MediaPipe/landmarks posteriormente.
    """
    if cv2 is None:
        return None

    try:
        source = np.asarray(image.convert("RGB"), dtype=np.uint8)
        gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)

        cascade_path = cv2.data.haarcascades + (
            "haarcascade_frontalface_default.xml"
        )
        cascade = cv2.CascadeClassifier(cascade_path)

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(
                max(32, source.shape[1] // 8),
                max(32, source.shape[0] // 8),
            ),
        )

        if len(faces) == 0:
            return None

        x, y, width, height = max(
            faces,
            key=lambda box: int(box[2] * box[3]),
        )

        return FaceBox(int(x), int(y), int(width), int(height))
    except Exception:
        logger.exception("Falha na deteccao facial.")
        return None


def face_ellipse_mask(
    size: tuple[int, int],
    face: FaceBox,
    *,
    width_ratio: float,
    height_ratio: float,
    center_y_ratio: float,
    feather_ratio: float,
) -> np.ndarray:
    if cv2 is None:
        width, height = size
        mask_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        center = (
            int(face.x + face.width * 0.5),
            int(face.y + face.height * center_y_ratio),
        )
        axes = (
            max(10, int(face.width * width_ratio)),
            max(10, int(face.height * height_ratio)),
        )
        draw.ellipse(
            (
                center[0] - axes[0],
                center[1] - axes[1],
                center[0] + axes[0],
                center[1] + axes[1],
            ),
            fill=255,
        )
        feather = max(3, int(min(face.width, face.height) * feather_ratio))
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather))
        return np.asarray(mask_img, dtype=np.float32) / 255.0

    width, height = size
    result = np.zeros((height, width), dtype=np.float32)

    center = (
        int(face.x + face.width * 0.5),
        int(face.y + face.height * center_y_ratio),
    )
    axes = (
        max(10, int(face.width * width_ratio)),
        max(10, int(face.height * height_ratio)),
    )

    cv2.ellipse(
        result,
        center,
        axes,
        0,
        0,
        360,
        1.0,
        thickness=-1,
    )

    feather = max(
        3,
        int(min(face.width, face.height) * feather_ratio),
    )
    kernel_size = feather * 2 + 1
    return cv2.GaussianBlur(
        result,
        (kernel_size, kernel_size),
        0,
    )


def detect_eyes(image: Image.Image, face: FaceBox | None) -> EyePair | None:
    """Try to detect eyes; fallback to face-proportional estimate if needed."""
    if cv2 is None:
        if face is None:
            return None
        return estimate_eyes_from_face(face)

    source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)

    if face is not None:
        x0 = max(0, face.x)
        y0 = max(0, face.y)
        x1 = min(source.shape[1], face.x + face.width)
        y1 = min(source.shape[0], face.y + int(face.height * 0.72))
        roi = gray[y0:y1, x0:x1]
    else:
        x0 = 0
        y0 = 0
        roi = gray

    if roi.size == 0:
        return estimate_eyes_from_face(face) if face is not None else None

    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        eyes = cascade.detectMultiScale(
            roi,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(12, 12),
        )
        if len(eyes) >= 2:
            boxes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:4]
            centers = [
                (float(x0 + ex + ew / 2.0), float(y0 + ey + eh / 2.0))
                for ex, ey, ew, eh in boxes
            ]
            # Pick the pair with best horizontal separation and reasonable vertical alignment.
            best_score = -1e9
            best_pair: tuple[tuple[float, float], tuple[float, float]] | None = None
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    c1 = centers[i]
                    c2 = centers[j]
                    dx = abs(c2[0] - c1[0])
                    dy = abs(c2[1] - c1[1])
                    score = dx - 1.8 * dy
                    if score > best_score:
                        best_score = score
                        best_pair = (c1, c2)

            if best_pair is not None:
                left, right = best_pair if best_pair[0][0] <= best_pair[1][0] else (best_pair[1], best_pair[0])
                return EyePair(left=left, right=right, source="detected")
    except Exception:
        logger.exception("Falha na deteccao de olhos.")

    return estimate_eyes_from_face(face) if face is not None else None


def estimate_eyes_from_face(face: FaceBox) -> EyePair:
    return EyePair(
        left=(face.x + face.width * 0.32, face.y + face.height * 0.42),
        right=(face.x + face.width * 0.68, face.y + face.height * 0.42),
        source="estimated",
    )


def estimate_person_bounds(image: Image.Image, face: FaceBox, alpha: Image.Image | None) -> tuple[float, float, float, float]:
    """Estimate visible person extent (head+shoulders) using alpha when available, otherwise face-guided expansion."""
    width, height = image.size

    if alpha is not None:
        bbox = alpha.getbbox()
        if bbox is not None:
            left, top, right, bottom = bbox
            return float(left), float(top), float(right), float(bottom)

    # Face-guided bounds tuned for credential framing.
    left = max(0.0, face.x - 0.55 * face.width)
    right = min(float(width), face.x + 1.55 * face.width)
    head_top = max(0.0, face.y - 0.52 * face.height)
    shoulder_bottom = min(float(height), face.y + 2.05 * face.height)
    return left, head_top, right, shoulder_bottom


def _composite_on_neutral_background(image: Image.Image, bg_color: tuple[int, int, int] = (238, 238, 238)) -> Image.Image:
    if image.mode == "RGBA":
        alpha = image.getchannel("A")
        rgb = Image.new("RGB", image.size, bg_color)
        rgb.paste(image.convert("RGB"), mask=alpha)
        return rgb
    return image.convert("RGB")


def _apply_round_mask(square_rgb: Image.Image, background: tuple[int, int, int] = (238, 238, 238)) -> Image.Image:
    output = Image.new("RGB", square_rgb.size, background)
    mask = Image.new("L", square_rgb.size, 0)
    draw = ImageDraw.Draw(mask)
    r = min(square_rgb.size) // 2 - 1
    cx = square_rgb.size[0] // 2
    cy = square_rgb.size[1] // 2
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    output.paste(square_rgb, mask=mask)
    return output


def _credential_background_reference(image_rgb: Image.Image) -> np.ndarray:
    arr = np.asarray(image_rgb.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    b = max(2, int(min(h, w) * 0.08))
    samples = np.concatenate(
        [
            arr[:b, :b].reshape(-1, 3),
            arr[:b, w - b : w].reshape(-1, 3),
            arr[h - b : h, :b].reshape(-1, 3),
            arr[h - b : h, w - b : w].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(samples, axis=0)


def _save_credential_debug_artifact(prefix: str, image: Image.Image) -> None:
    if not SETTINGS.credential_debug_mask:
        return
    debug_dir = Path(__file__).resolve().parent / "debug_masks"
    debug_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    image.save(debug_dir / f"{prefix}_{ts}.png")


def _credential_foreground_mask(image_rgb: Image.Image) -> np.ndarray:
    arr = np.asarray(image_rgb.convert("RGB"), dtype=np.float32)
    bg = _credential_background_reference(image_rgb)
    dist = np.sqrt(np.sum((arr - bg[None, None, :]) ** 2, axis=2))
    return dist > float(SETTINGS.credential_background_threshold)


def _normalize_credential_background(square_rgb: Image.Image, face: FaceBox) -> tuple[Image.Image, np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    """Separate foreground/background and force neutral uniform background conservatively."""
    arr = np.asarray(square_rgb.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    fg = _credential_foreground_mask(square_rgb)
    hair_guard = _credential_head_hair_protect_region(square_rgb.size, face)
    person_mask = fg | hair_guard

    if cv2 is not None:
        u8 = (person_mask.astype(np.uint8) * 255)
        u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)
        u8 = cv2.dilate(u8, np.ones((3, 3), dtype=np.uint8), iterations=1)
        person_mask = u8 > 0

    bg_ref = _credential_background_reference(square_rgb).astype(np.uint8)
    out = arr.copy()
    out[~person_mask] = bg_ref

    # Soft edge blend around person boundary to reduce halos.
    if cv2 is not None:
        pm = person_mask.astype(np.uint8) * 255
        feather = cv2.GaussianBlur(pm, (0, 0), sigmaX=2.0)
        alpha = np.clip(feather.astype(np.float32) / 255.0, 0.0, 1.0)[:, :, None]
        bg_img = np.zeros_like(out, dtype=np.uint8)
        bg_img[:, :, 0] = bg_ref[0]
        bg_img[:, :, 1] = bg_ref[1]
        bg_img[:, :, 2] = bg_ref[2]
        out = np.clip(alpha * arr.astype(np.float32) + (1.0 - alpha) * bg_img.astype(np.float32), 0, 255).astype(np.uint8)

    normalized = Image.fromarray(out, mode="RGB")
    metrics = {
        "foreground_area_ratio": float(person_mask.sum()) / float(person_mask.size),
        "background_area_ratio": float((~person_mask).sum()) / float(person_mask.size),
    }
    debug_masks = {
        "foreground": person_mask,
        "hair_guard": hair_guard,
    }
    return normalized, person_mask, metrics, debug_masks


def _credential_skin_mask(image_rgb: Image.Image) -> np.ndarray:
    if cv2 is None:
        return np.zeros((image_rgb.height, image_rgb.width), dtype=bool)
    bgr = cv2.cvtColor(np.asarray(image_rgb.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    return (y > 40) & (cr >= 133) & (cr <= 181) & (cb >= 78) & (cb <= 135)


def _credential_torso_region(size: tuple[int, int], face: FaceBox) -> np.ndarray:
    w, h = size
    region = np.zeros((h, w), dtype=np.uint8)
    cx = int(face.x + face.width * 0.5)
    # Start below neck to avoid regenerating skin/mandible transitions.
    neck_y = int(face.y + face.height * 1.05)
    top = max(0, neck_y)
    bottom = min(h, int(face.y + face.height * 1.95))
    half_w_top = max(8, int(face.width * 0.52))
    half_w_bottom = max(10, int(face.width * 1.18))

    pts = np.array(
        [
            [max(0, cx - half_w_top), top],
            [min(w - 1, cx + half_w_top), top],
            [min(w - 1, cx + half_w_bottom), bottom],
            [max(0, cx - half_w_bottom), bottom],
        ],
        dtype=np.int32,
    )
    if cv2 is not None:
        cv2.fillConvexPoly(region, pts, 255)
    else:
        img = Image.new("L", (w, h), 0)
        ImageDraw.Draw(img).polygon([tuple(map(int, p)) for p in pts], fill=255)
        region = np.asarray(img, dtype=np.uint8)
    return region > 0


def _credential_face_protect_region(size: tuple[int, int], face: FaceBox) -> np.ndarray:
    protect = face_ellipse_mask(
        size,
        face,
        width_ratio=0.55,
        height_ratio=0.70,
        center_y_ratio=0.56,
        feather_ratio=0.20,
    )
    return protect > 0.12


def _credential_head_hair_protect_region(size: tuple[int, int], face: FaceBox) -> np.ndarray:
    # Conservative protection around full head/hair to prevent color bleeding.
    protect = face_ellipse_mask(
        size,
        face,
        width_ratio=0.68,
        height_ratio=SETTINGS.credential_clothing_head_protect_ratio,
        center_y_ratio=0.50,
        feather_ratio=0.22,
    )
    return protect > 0.08


def _credential_neck_protect_region(size: tuple[int, int], face: FaceBox, skin_mask: np.ndarray) -> np.ndarray:
    w, h = size
    region = np.zeros((h, w), dtype=np.uint8)
    cx = int(face.x + face.width * 0.5)
    top = int(face.y + face.height * 0.72)
    bottom = int(face.y + face.height * (0.72 + SETTINGS.credential_clothing_neck_protect_ratio))
    half_w = max(8, int(face.width * 0.30))
    top = max(0, min(h - 1, top))
    bottom = max(top + 1, min(h, bottom))
    left = max(0, cx - half_w)
    right = min(w, cx + half_w)
    region[top:bottom, left:right] = 255
    neck_box = region > 0
    # Preserve only existing neck/skin pixels when available.
    return neck_box & skin_mask


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _strict_mask_composite(original: Image.Image, generated: Image.Image, mask_image: Image.Image) -> Image.Image:
    """Exact pixel preservation outside mask: final = original*(1-mask) + generated*mask."""
    orig = np.asarray(original.convert("RGB"), dtype=np.uint8)
    gen = np.asarray(generated.convert("RGB"), dtype=np.uint8)
    mask = np.asarray(mask_image.convert("L"), dtype=np.uint8) > 0
    out = orig.copy()
    out[mask] = gen[mask]
    return Image.fromarray(out, mode="RGB")


def _build_credential_clothing_mask(square_rgb: Image.Image, face: FaceBox) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    fg = _credential_foreground_mask(square_rgb)
    skin = _credential_skin_mask(square_rgb)
    torso = _credential_torso_region(square_rgb.size, face)
    face_core = _credential_face_protect_region(square_rgb.size, face)
    head_hair = _credential_head_hair_protect_region(square_rgb.size, face)
    neck = _credential_neck_protect_region(square_rgb.size, face, skin)
    protected = face_core | head_hair | neck

    target = torso & (~protected)
    target_pixels = int(target.sum())
    if target_pixels == 0:
        return np.zeros_like(target), {
            "clothing_coverage": 0.0,
            "clothing_missing_ratio": 1.0,
            "clothing_target_area_ratio": 0.0,
            "clothing_face_protected_area_ratio": float(protected.sum()) / float(target.size),
        }, {
            "target": target,
            "protected": protected,
        }

    clothing_present = target & fg & (~skin)
    coverage = float(clothing_present.sum()) / float(target_pixels)
    missing = target & (~clothing_present)

    if SETTINGS.credential_clothing_mask_padding > 0 and cv2 is not None and np.any(missing):
        pad = SETTINGS.credential_clothing_mask_padding
        kernel = np.ones((pad * 2 + 1, pad * 2 + 1), dtype=np.uint8)
        missing_u8 = (missing.astype(np.uint8) * 255)
        missing_u8 = cv2.dilate(missing_u8, kernel, iterations=1)
        missing = (missing_u8 > 0) & target

    # Hard-limit maximum generated area to keep generation minimal and conservative.
    max_area = int(missing.size * SETTINGS.credential_clothing_max_mask_area_ratio)
    if int(missing.sum()) > max_area > 0:
        ys, xs = np.where(missing)
        if ys.size > 0:
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            dist = (xs - cx) ** 2 + (ys - cy) ** 2
            keep_idx = np.argsort(dist)[:max_area]
            reduced = np.zeros_like(missing)
            reduced[ys[keep_idx], xs[keep_idx]] = True
            missing = reduced

    missing = missing & (~protected)

    metrics = {
        "clothing_coverage": coverage,
        "clothing_missing_ratio": max(0.0, 1.0 - coverage),
        "clothing_target_area_ratio": float(target_pixels) / float(target.size),
        "clothing_face_protected_area_ratio": float(protected.sum()) / float(target.size),
        "clothing_generated_area_ratio": float(missing.sum()) / float(target.size),
    }
    return missing, metrics, {
        "target": target,
        "protected": protected,
        "missing": missing,
        "face_core": face_core,
        "head_hair": head_hair,
        "neck": neck,
    }


def _credential_visual_quality_checks(square_rgb: Image.Image, face: FaceBox, fg_mask: np.ndarray) -> tuple[list[str], dict[str, float]]:
    """Visual checks unrelated to framing geometry: background residuals, halos and color artifacts."""
    reasons: list[str] = []
    metrics: dict[str, float] = {}

    arr = np.asarray(square_rgb.convert("RGB"), dtype=np.uint8)
    bg_ref = _credential_background_reference(square_rgb)

    head_hair = _credential_head_hair_protect_region(square_rgb.size, face)
    ring = np.zeros_like(head_hair)
    if cv2 is not None:
        dil = cv2.dilate((head_hair.astype(np.uint8) * 255), np.ones((17, 17), dtype=np.uint8), iterations=1) > 0
        ring = dil & (~head_hair)
    else:
        ring = ~head_hair

    # Evaluate residual background around head in places expected to be background.
    ring_bg = ring & (~fg_mask)
    if np.any(ring_bg):
        dist = np.sqrt(np.sum((arr.astype(np.float32) - bg_ref[None, None, :]) ** 2, axis=2))
        residual_ratio = float((dist[ring_bg] > max(18.0, float(SETTINGS.credential_background_threshold) * 1.2)).sum()) / float(ring_bg.sum())
        metrics["background_residual_ratio"] = residual_ratio
        if residual_ratio > SETTINGS.credential_artifact_threshold:
            reasons.append("background_residual_near_face")

    # Saturated blue/green artifact check in lower torso band.
    lower = np.zeros((arr.shape[0], arr.shape[1]), dtype=bool)
    y0 = int(face.y + face.height * 1.0)
    y0 = max(0, min(arr.shape[0], y0))
    lower[y0:, :] = True
    zone = lower & fg_mask
    if np.any(zone):
        r = arr[:, :, 0].astype(np.int16)
        g = arr[:, :, 1].astype(np.int16)
        b = arr[:, :, 2].astype(np.int16)
        sat = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
        bad = ((b > r + 20) | (g > r + 20)) & (sat > 24)
        ratio = float((bad & zone).sum()) / float(zone.sum())
        metrics["color_artifact_ratio"] = ratio
        if ratio > SETTINGS.credential_artifact_threshold:
            reasons.append("clothing_color_artifacts")

    return reasons, metrics


def _select_clothing_prompt(clothing_type: str) -> str | None:
    if clothing_type == "male_formal":
        return DEFAULT_CLOTHING_PROMPT_MALE
    if clothing_type == "female_formal":
        return DEFAULT_CLOTHING_PROMPT_FEMALE
    return None


def _run_credential_clothing_inpaint(
    base_rgb: Image.Image,
    clothing_mask: Image.Image,
    clothing_type: str,
    seed: int,
) -> Image.Image:
    prompt = _select_clothing_prompt(clothing_type)
    if prompt is None:
        raise RuntimeError("clothing_type nao definido para geracao.")

    work_size = working_size(base_rgb.size, min(SETTINGS.inpaint_resolution, SETTINGS.credential_output_size))
    source_work = base_rgb.resize(work_size, Image.Resampling.LANCZOS)
    mask_work = clothing_mask.resize(work_size, Image.Resampling.BILINEAR)

    pipeline = MODELS.get_inpaint()
    generator = MODELS.seed_generator(seed)
    result = pipeline(
        prompt=prompt,
        negative_prompt=DEFAULT_CLOTHING_NEGATIVE,
        image=source_work,
        mask_image=mask_work,
        strength=SETTINGS.credential_clothing_generation_strength,
        guidance_scale=SETTINGS.credential_clothing_guidance_scale,
        num_inference_steps=SETTINGS.credential_clothing_steps,
        generator=generator,
    )

    generated = result.images[0].convert("RGB")
    if generated.size != base_rgb.size:
        generated = generated.resize(base_rgb.size, Image.Resampling.LANCZOS)
    return generated


def apply_credential_clothing(
    square_rgb: Image.Image,
    face: FaceBox,
    *,
    clothing_type: str,
    seed: int,
) -> tuple[Image.Image, str, bool, list[str], dict[str, float]]:
    if not SETTINGS.credential_clothing_enabled or clothing_type == "disabled":
        return square_rgb, "not_required", False, [], {"clothing_coverage": 1.0, "clothing_missing_ratio": 0.0}

    missing, mask_metrics, mask_dbg = _build_credential_clothing_mask(square_rgb, face)
    coverage = float(mask_metrics.get("clothing_coverage", 0.0))
    target_pixels = int(mask_dbg["target"].sum()) if "target" in mask_dbg else 0

    if target_pixels <= 0:
        return square_rgb, "review", False, ["clothing_target_empty"], mask_metrics

    if coverage >= SETTINGS.credential_clothing_min_required:
        return square_rgb, "preserved", False, [], mask_metrics

    if clothing_type == "auto":
        return square_rgb, "review", False, ["clothing_type_undetermined"], mask_metrics

    if not np.any(missing):
        return square_rgb, "not_required", False, [], mask_metrics

    mask_img = Image.fromarray((missing.astype(np.uint8) * 255), mode="L")

    bbox = _mask_bbox(missing)
    if bbox is not None:
        mask_metrics["clothing_mask_x0"] = float(bbox[0])
        mask_metrics["clothing_mask_y0"] = float(bbox[1])
        mask_metrics["clothing_mask_x1"] = float(bbox[2])
        mask_metrics["clothing_mask_y1"] = float(bbox[3])

    if SETTINGS.credential_debug_mask:
        debug_dir = Path(__file__).resolve().parent / "debug_masks"
        debug_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        square_rgb.save(debug_dir / f"credential_original_{ts}.png")
        Image.fromarray((mask_dbg.get("target", np.zeros_like(missing)).astype(np.uint8) * 255), mode="L").save(
            debug_dir / f"credential_target_{ts}.png"
        )
        Image.fromarray((mask_dbg.get("protected", np.zeros_like(missing)).astype(np.uint8) * 255), mode="L").save(
            debug_dir / f"credential_protected_{ts}.png"
        )
        mask_img.save(debug_dir / f"credential_mask_{ts}.png")

    try:
        generated = _run_credential_clothing_inpaint(square_rgb, mask_img, clothing_type, seed)
        if SETTINGS.credential_debug_mask:
            generated.save(debug_dir / f"credential_generated_{ts}.png")
        composed = _strict_mask_composite(square_rgb, generated, mask_img)
        if SETTINGS.credential_debug_mask:
            composed.save(debug_dir / f"credential_composed_{ts}.png")

        # Safety gate: face/hair region must remain stable.
        before = np.asarray(square_rgb.convert("RGB"), dtype=np.float32)
        after = np.asarray(composed.convert("RGB"), dtype=np.float32)
        protected = mask_dbg.get("protected", np.zeros_like(missing))
        if np.any(protected):
            delta = np.abs(after - before).mean(axis=2)
            protected_delta = float(delta[protected].mean())
            if protected_delta > (SETTINGS.credential_artifact_threshold * 255.0):
                return square_rgb, "review", False, ["face_region_changed"], {
                    **mask_metrics,
                    "protected_delta": protected_delta,
                }

        return composed, "generated", True, [], mask_metrics
    except Exception:
        logger.exception("Falha na geracao de vestimenta para credential-process.")
        return square_rgb, "review", False, ["clothing_generation_failed"], mask_metrics


def _build_square_crop(
    source_rgb: Image.Image,
    eye_mid: tuple[float, float],
    person_bounds: tuple[float, float, float, float],
    face: FaceBox,
) -> tuple[Image.Image, dict[str, float]]:
    """Compute person-first square crop anchored by eye position and safety margins."""
    width, height = source_rgb.size
    eye_x, eye_y = eye_mid
    p_left, p_top, p_right, p_bottom = person_bounds

    # Approximate hair top from face with configurable head margin.
    head_top = max(0.0, min(p_top, face.y - 0.45 * face.height))
    shoulder_bottom = min(p_bottom, face.y + 1.28 * face.height)

    # Side implied by desired face scale and by geometric inclusion constraints.
    side_from_face = face.height / max(0.01, SETTINGS.credential_face_scale)
    side_from_horiz = max(
        (eye_x - p_left) / max(0.05, SETTINGS.credential_eye_target_x),
        (p_right - eye_x) / max(0.05, (1.0 - SETTINGS.credential_eye_target_x)),
    )
    side_from_vert = max(
        (eye_y - head_top) / max(0.05, SETTINGS.credential_eye_target_y - SETTINGS.credential_top_head_margin),
        (shoulder_bottom - eye_y) / max(0.05, (1.0 - SETTINGS.credential_eye_target_y - SETTINGS.credential_bottom_shoulder_margin)),
    )

    side = max(
        side_from_horiz,
        side_from_vert,
        side_from_face * SETTINGS.credential_min_side_from_face_ratio,
    )
    side = min(side, side_from_face * SETTINGS.credential_max_side_from_face_ratio)
    side = max(side, 128.0)

    max_native = max(width, height)
    if side > max_native * SETTINGS.credential_max_crop_ratio:
        side = max_native * SETTINGS.credential_max_crop_ratio

    left_eye_anchor = eye_x - SETTINGS.credential_eye_target_x * side
    face_center_x = face.x + face.width * 0.5
    left_face_anchor = face_center_x - 0.5 * side
    face_weight = SETTINGS.credential_face_center_weight
    left = (1.0 - face_weight) * left_eye_anchor + face_weight * left_face_anchor
    top = eye_y - SETTINGS.credential_eye_target_y * side
    right = left + side
    bottom = top + side

    pad_left = max(0.0, -left)
    pad_top = max(0.0, -top)
    pad_right = max(0.0, right - width)
    pad_bottom = max(0.0, bottom - height)

    if pad_left or pad_top or pad_right or pad_bottom:
        new_w = int(round(width + pad_left + pad_right))
        new_h = int(round(height + pad_top + pad_bottom))
        expanded = Image.new("RGB", (new_w, new_h), (238, 238, 238))
        expanded.paste(source_rgb, (int(round(pad_left)), int(round(pad_top))))
        source_rgb = expanded
        left += pad_left
        top += pad_top
        right += pad_left
        bottom += pad_top

    crop = source_rgb.crop((int(round(left)), int(round(top)), int(round(right)), int(round(bottom))))
    crop = crop.resize((SETTINGS.credential_output_size, SETTINGS.credential_output_size), Image.Resampling.LANCZOS)

    metrics = {
        "crop_side": float(side),
        "pad_left": float(pad_left),
        "pad_top": float(pad_top),
        "pad_right": float(pad_right),
        "pad_bottom": float(pad_bottom),
        "crop_left": float(left),
        "crop_top": float(top),
    }
    return crop, metrics


def _project_face_to_square(face: FaceBox, metrics: dict[str, float], output_size: int) -> FaceBox:
    side = max(1.0, metrics["crop_side"])
    left = metrics["crop_left"]
    top = metrics["crop_top"]

    x = int(round((face.x - left) / side * output_size))
    y = int(round((face.y - top) / side * output_size))
    w = int(round(face.width / side * output_size))
    h = int(round(face.height / side * output_size))

    x = max(0, min(output_size - 1, x))
    y = max(0, min(output_size - 1, y))
    w = max(16, min(output_size - x, w))
    h = max(16, min(output_size - y, h))
    return FaceBox(x=x, y=y, width=w, height=h)


def _validate_credential_frame(
    source_size: tuple[int, int],
    face: FaceBox,
    eyes: EyePair,
    person_bounds: tuple[float, float, float, float],
    metrics: dict[str, float],
) -> tuple[str, list[str], dict[str, float]]:
    reasons: list[str] = []
    side = max(1.0, metrics["crop_side"])
    crop_left = metrics["crop_left"]
    crop_top = metrics["crop_top"]

    face_cx = (face.x + face.width * 0.5 - crop_left) / side
    eye_mid_x = ((eyes.left[0] + eyes.right[0]) * 0.5 - crop_left) / side
    eye_mid_y = ((eyes.left[1] + eyes.right[1]) * 0.5 - crop_top) / side
    head_top = min(person_bounds[1], face.y - 0.45 * face.height)
    head_margin = (min(eyes.left[1], eyes.right[1]) - head_top) / side
    chin_y = (face.y + face.height - crop_top) / side
    shoulder_bottom = person_bounds[3]
    shoulder_vis = (shoulder_bottom - (face.y + face.height)) / side

    tol = SETTINGS.credential_validation_tolerance
    if eyes.source == "estimated":
        reasons.append("eyes_not_detected")
    if abs(eye_mid_x - SETTINGS.credential_eye_target_x) > tol:
        reasons.append("eyes_x_out_of_range")
    if abs(eye_mid_y - SETTINGS.credential_eye_target_y) > tol:
        reasons.append("eyes_y_out_of_range")
    if abs(face_cx - 0.5) > 0.14:
        reasons.append("face_not_centered")
    if head_margin < SETTINGS.credential_min_head_margin:
        reasons.append("insufficient_head_margin")
    if chin_y > 1.0 - 0.05:
        reasons.append("chin_too_close_to_bottom")
    if shoulder_vis < SETTINGS.credential_min_shoulder_visibility:
        reasons.append("insufficient_shoulder_visibility")

    metrics_out = {
        **metrics,
        "eye_mid_x_norm": float(eye_mid_x),
        "eye_mid_y_norm": float(eye_mid_y),
        "face_center_x_norm": float(face_cx),
        "head_margin_norm": float(head_margin),
        "chin_norm": float(chin_y),
        "shoulder_visibility_norm": float(shoulder_vis),
        "input_width": float(source_size[0]),
        "input_height": float(source_size[1]),
    }

    status = "approved" if not reasons else "review"
    return status, reasons, metrics_out


def process_credential_framing(
    source_image: Image.Image,
    eye_left: tuple[float, float] | None = None,
    eye_right: tuple[float, float] | None = None,
    *,
    clothing_type: str | None = None,
    seed: int = -1,
) -> CredentialFramingResult:
    """Person-first, landmark-anchored framing for credential profile photos."""
    source_rgb = _composite_on_neutral_background(source_image)
    alpha = source_image.getchannel("A") if source_image.mode == "RGBA" else None

    face: FaceBox | None = None
    if eye_left and eye_right:
        dx = abs(eye_right[0] - eye_left[0])
        eye_mid = ((eye_left[0] + eye_right[0]) * 0.5, (eye_left[1] + eye_right[1]) * 0.5)
        approx_w = max(64.0, dx / 0.36)
        approx_h = approx_w * 1.18
        face = FaceBox(
            x=int(round(eye_mid[0] - approx_w * 0.5)),
            y=int(round(eye_mid[1] - approx_h * 0.42)),
            width=int(round(approx_w)),
            height=int(round(approx_h)),
        )
    else:
        face = detect_primary_face(source_rgb)

    if face is None:
        resized = fit_image_for_model(source_rgb, SETTINGS.credential_output_size)
        return CredentialFramingResult(
            square=resized,
            round_image=_apply_round_mask(resized),
            status="review",
            reasons=["face_not_detected"],
            metrics={"input_width": float(source_rgb.width), "input_height": float(source_rgb.height)},
            clothing_status="review",
            clothing_generated=False,
        )

    if eye_left and eye_right:
        eyes = EyePair(left=eye_left, right=eye_right, source="manual")
    else:
        eyes = detect_eyes(source_rgb, face)
        if eyes is None:
            resized = fit_image_for_model(source_rgb, SETTINGS.credential_output_size)
            return CredentialFramingResult(
                square=resized,
                round_image=_apply_round_mask(resized),
                status="review",
                reasons=["eyes_not_detected"],
                metrics={"input_width": float(source_rgb.width), "input_height": float(source_rgb.height)},
                clothing_status="review",
                clothing_generated=False,
            )

    eye_mid = ((eyes.left[0] + eyes.right[0]) * 0.5, (eyes.left[1] + eyes.right[1]) * 0.5)
    person_bounds = estimate_person_bounds(source_rgb, face, alpha)
    square, metrics = _build_square_crop(source_rgb, eye_mid, person_bounds, face)

    face_square = detect_primary_face(square)
    if face_square is None:
        face_square = _project_face_to_square(face, metrics, SETTINGS.credential_output_size)
    clothing_choice = (clothing_type or SETTINGS.credential_clothing_type).strip().lower()
    if clothing_choice not in {"auto", "male_formal", "female_formal", "disabled"}:
        clothing_choice = SETTINGS.credential_clothing_type

    square, clothing_status, clothing_generated, clothing_reasons, clothing_metrics = apply_credential_clothing(
        square,
        face_square,
        clothing_type=clothing_choice,
        seed=seed,
    )

    round_image = _apply_round_mask(square)
    status, reasons, metrics = _validate_credential_frame(source_rgb.size, face, eyes, person_bounds, metrics)
    reasons = reasons + clothing_reasons
    if clothing_status == "review":
        status = "review"
    metrics.update(clothing_metrics)

    logger.info(
        "credential_frame status=%s clothing=%s generated=%s reasons=%s eye_source=%s crop_side=%.1f pad=(%.1f,%.1f,%.1f,%.1f)",
        status,
        clothing_status,
        clothing_generated,
        ",".join(reasons) if reasons else "-",
        eyes.source,
        metrics.get("crop_side", 0.0),
        metrics.get("pad_left", 0.0),
        metrics.get("pad_top", 0.0),
        metrics.get("pad_right", 0.0),
        metrics.get("pad_bottom", 0.0),
    )

    return CredentialFramingResult(
        square=square,
        round_image=round_image,
        status=status,
        reasons=reasons,
        metrics=metrics,
        clothing_status=clothing_status,
        clothing_generated=clothing_generated,
    )


# ============================================================================
# Mask operations
# ============================================================================

def prepare_inpaint_mask(
    mask_image: Image.Image,
    target_size: tuple[int, int],
    blur_radius: float,
) -> Image.Image:
    mask = mask_image.convert("L")

    if mask.size != target_size:
        mask = mask.resize(
            target_size,
            Image.Resampling.BILINEAR,
        )

    threshold = int(
        os.getenv("INPAINT_MASK_THRESHOLD", "16")
    )
    mask = mask.point(
        lambda value: 0 if value < threshold else value
    )

    if blur_radius > 0:
        mask = mask.filter(
            ImageFilter.GaussianBlur(radius=blur_radius)
        )

    return mask


def protect_face_identity_mask(
    mask_image: Image.Image,
    source_rgb: Image.Image,
) -> Image.Image:
    if not SETTINGS.face_protect:
        return mask_image

    face = detect_primary_face(source_rgb)
    if face is None:
        return mask_image

    mask = np.asarray(
        mask_image.convert("L"),
        dtype=np.float32,
    )

    protect = face_ellipse_mask(
        mask_image.size,
        face,
        width_ratio=0.34,
        height_ratio=0.40,
        center_y_ratio=0.62,
        feather_ratio=0.12,
    )

    atten = 1.0 - (
        protect * SETTINGS.face_protect_strength
    )
    result = np.clip(
        mask * atten,
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(result, mode="L")


def preserve_face_identity(
    source_rgb: Image.Image,
    generated_rgb: Image.Image,
    mask_image: Image.Image,
) -> Image.Image:
    if not SETTINGS.face_identity_lock:
        return generated_rgb

    face = detect_primary_face(source_rgb)
    if face is None:
        return generated_rgb

    source = np.asarray(
        source_rgb.convert("RGB"),
        dtype=np.float32,
    )
    generated = np.asarray(
        generated_rgb.convert("RGB"),
        dtype=np.float32,
    )
    mask = np.asarray(
        mask_image.convert("L"),
        dtype=np.uint8,
    )

    active = mask >= SETTINGS.face_lock_min_mask
    if not np.any(active):
        return generated_rgb

    core = face_ellipse_mask(
        mask_image.size,
        face,
        width_ratio=0.30,
        height_ratio=0.36,
        center_y_ratio=0.62,
        feather_ratio=0.16,
    )

    alpha = np.clip(
        core * SETTINGS.face_identity_lock_strength,
        0,
        1,
    )
    alpha = np.where(active, alpha, 0).astype(np.float32)
    alpha = alpha[:, :, None]

    output = (
        (1.0 - alpha) * generated
        + alpha * source
    )

    return Image.fromarray(
        np.clip(output, 0, 255).astype(np.uint8),
        mode="RGB",
    )


def mask_coverage(mask_image: Image.Image) -> float:
    values = np.asarray(
        mask_image,
        dtype=np.uint8,
    )

    if values.size == 0:
        return 0.0

    return float(values.mean() / 255.0)


def auto_repair_mask(
    alpha: Image.Image,
    touches: dict[str, bool],
) -> Image.Image:
    width, height = alpha.size
    min_side = min(width, height)

    edge_band = max(
        16,
        int(min_side * 0.11),
    )
    seam_blend = max(
        8,
        int(min_side * 0.02),
    )

    edge_mask = Image.new(
        "L",
        (width, height),
        0,
    )
    draw = ImageDraw.Draw(edge_mask)

    if touches["top"]:
        draw.rectangle(
            (0, 0, width, edge_band + seam_blend),
            fill=255,
        )

    if touches["left"]:
        draw.rectangle(
            (0, 0, edge_band + seam_blend, height),
            fill=255,
        )

    if touches["right"]:
        draw.rectangle(
            (
                width - edge_band - seam_blend,
                0,
                width,
                height,
            ),
            fill=255,
        )

    if touches["bottom"]:
        draw.rectangle(
            (
                0,
                height - edge_band - seam_blend,
                width,
                height,
            ),
            fill=255,
        )

    neighborhood = alpha.filter(
        ImageFilter.MaxFilter(size=31)
    )

    mask = ImageChops.multiply(
        edge_mask,
        neighborhood,
    )

    return mask.filter(
        ImageFilter.GaussianBlur(
            radius=max(3, seam_blend // 2)
        )
    )


def auto_repair_transparency_mask(
    alpha: Image.Image,
) -> Image.Image:
    min_side = min(alpha.size)

    neighborhood_size = max(
        31,
        int(min_side * 0.18),
    )
    if neighborhood_size % 2 == 0:
        neighborhood_size += 1

    neighborhood = alpha.filter(
        ImageFilter.MaxFilter(
            size=neighborhood_size
        )
    )

    transparent = ImageChops.invert(alpha)
    mask = ImageChops.multiply(
        transparent,
        neighborhood,
    )

    seam_blend = max(
        6,
        int(min_side * 0.02),
    )

    return mask.filter(
        ImageFilter.GaussianBlur(
            radius=seam_blend
        )
    )


# ============================================================================
# Outpaint
# ============================================================================

def outpaint_canvas(
    source: Image.Image,
    pads: dict[str, int],
    overlap: int,
) -> tuple[Image.Image, Image.Image]:
    top = pads["top"]
    left = pads["left"]
    right = pads["right"]
    bottom = pads["bottom"]

    stretched = np.pad(
        np.asarray(source),
        (
            (top, bottom),
            (left, right),
            (0, 0),
        ),
        mode="edge",
    )

    canvas = Image.fromarray(stretched).filter(
        ImageFilter.GaussianBlur(
            radius=max(8, min(source.size) // 20)
        )
    )
    canvas.paste(source, (left, top))

    keep = (
        left + (overlap if left else 0),
        top + (overlap if top else 0),
        left + source.width - (overlap if right else 0),
        top + source.height - (overlap if bottom else 0),
    )

    mask = Image.new(
        "L",
        canvas.size,
        255,
    )

    ImageDraw.Draw(mask).rectangle(
        (
            keep[0],
            keep[1],
            keep[2] - 1,
            keep[3] - 1,
        ),
        fill=0,
    )

    return canvas, mask


# ============================================================================
# Parameter tuning
# ============================================================================

def tune_enhance_params(
    strength: float,
    guidance_scale: float,
    steps: int,
) -> tuple[float, float, int]:
    if SETTINGS.device == "cpu" and SETTINGS.cpu_safe_mode:
        strength = min(strength, 0.4)
        guidance_scale = min(
            guidance_scale,
            SETTINGS.cpu_max_guidance,
        )
        steps = min(
            steps,
            SETTINGS.cpu_max_steps,
        )

    return strength, guidance_scale, steps


def tune_inpaint_params(
    strength: float,
    guidance_scale: float,
    steps: int,
) -> tuple[float, float, int]:
    if SETTINGS.device == "cpu" and SETTINGS.cpu_safe_mode:
        strength = min(strength, 0.75)
        guidance_scale = min(
            guidance_scale,
            SETTINGS.cpu_max_inpaint_guidance,
        )
        steps = min(
            steps,
            SETTINGS.cpu_max_inpaint_steps,
        )

    return strength, guidance_scale, steps


def refine_inpaint_for_realism(
    strength: float,
    guidance_scale: float,
    steps: int,
    coverage: float,
    mask_blur: float,
    blend_feather: float,
) -> tuple[float, float, int, float, float]:
    if coverage <= 0.08:
        strength = min(max(strength, 0.34), 0.5)
        guidance_scale = min(max(guidance_scale, 2.2), 3.4)
        steps = min(max(steps, 20), 30)
        mask_blur = max(mask_blur, 1.8)
        blend_feather = max(blend_feather, 1.5)

    elif coverage <= 0.20:
        strength = min(max(strength, 0.42), 0.6)
        guidance_scale = min(max(guidance_scale, 2.2), 3.6)
        steps = min(max(steps, 24), 34)
        mask_blur = max(mask_blur, 2.0)
        blend_feather = max(blend_feather, 1.8)

    else:
        strength = min(max(strength, 0.48), 0.64)
        guidance_scale = min(max(guidance_scale, 2.0), 3.4)
        steps = min(max(steps, 26), 38)
        mask_blur = max(mask_blur, 2.2)
        blend_feather = max(blend_feather, 2.0)

    return (
        strength,
        guidance_scale,
        steps,
        mask_blur,
        blend_feather,
    )


# ============================================================================
# Post-processing pipeline
# ============================================================================

def suppress_dark_artifacts(
    base_rgb: Image.Image,
    generated_rgb: Image.Image,
    mask_image: Image.Image,
) -> Image.Image:
    base = np.asarray(
        base_rgb.convert("RGB"),
        dtype=np.uint8,
    )
    generated = np.asarray(
        generated_rgb.convert("RGB"),
        dtype=np.uint8,
    )
    mask = np.asarray(
        mask_image.convert("L"),
        dtype=np.uint8,
    )

    luma = (
        0.2126 * generated[:, :, 0]
        + 0.7152 * generated[:, :, 1]
        + 0.0722 * generated[:, :, 2]
    )

    chroma_span = (
        generated.max(axis=2).astype(np.int16)
        - generated.min(axis=2).astype(np.int16)
    )

    dark_luma = int(
        os.getenv("INPAINT_DARK_LUMA", "28")
    )
    max_chroma = int(
        os.getenv("INPAINT_DARK_CHROMA_SPAN", "26")
    )
    min_mask = int(
        os.getenv("INPAINT_DARK_MIN_MASK", "92")
    )

    masked = mask >= min_mask
    artifact = (
        masked
        & (luma <= dark_luma)
        & (chroma_span <= max_chroma)
    )

    masked_pixels = int(masked.sum())
    artifact_pixels = int(artifact.sum())

    if not masked_pixels or not artifact_pixels:
        return generated_rgb

    ratio = artifact_pixels / masked_pixels
    max_ratio = float(
        os.getenv("INPAINT_DARK_MAX_RATIO", "0.65")
    )

    fixed = generated.copy()
    fixed[artifact] = base[artifact]

    if ratio > max_ratio:
        rescue = float(
            os.getenv(
                "INPAINT_DARK_RESCUE_BLEND",
                "0.45",
            )
        )
        rescue = max(0.0, min(1.0, rescue))

        blend = (
            rescue * fixed.astype(np.float32)
            + (1.0 - rescue) * base.astype(np.float32)
        ).astype(np.uint8)

        fixed[masked] = blend[masked]

    return Image.fromarray(
        fixed,
        mode="RGB",
    )


def harmonize_luma(
    base_rgb: Image.Image,
    generated_rgb: Image.Image,
    mask_image: Image.Image,
) -> Image.Image:
    base = np.asarray(
        base_rgb.convert("RGB"),
        dtype=np.uint8,
    )
    generated = np.asarray(
        generated_rgb.convert("RGB"),
        dtype=np.uint8,
    )
    mask = np.asarray(
        mask_image.convert("L"),
        dtype=np.uint8,
    )

    active = mask >= int(
        os.getenv(
            "INPAINT_HARMONIZE_MIN_MASK",
            "92",
        )
    )

    if not np.any(active):
        return generated_rgb

    base_luma = (
        0.2126 * base[:, :, 0]
        + 0.7152 * base[:, :, 1]
        + 0.0722 * base[:, :, 2]
    )
    generated_luma = (
        0.2126 * generated[:, :, 0]
        + 0.7152 * generated[:, :, 1]
        + 0.0722 * generated[:, :, 2]
    )

    source_mean = float(base_luma[active].mean())
    generated_mean = float(generated_luma[active].mean())

    if generated_mean <= 1.0:
        return generated_rgb

    gain = source_mean / generated_mean
    gain = max(
        float(os.getenv("INPAINT_HARMONIZE_MIN_GAIN", "0.86")),
        min(
            float(os.getenv("INPAINT_HARMONIZE_MAX_GAIN", "1.22")),
            gain,
        ),
    )

    if abs(gain - 1.0) < 0.03:
        return generated_rgb

    corrected = generated.astype(np.float32)
    corrected[active] *= gain

    return Image.fromarray(
        np.clip(corrected, 0, 255).astype(np.uint8),
        mode="RGB",
    )


def harmonize_color(
    base_rgb: Image.Image,
    generated_rgb: Image.Image,
    mask_image: Image.Image,
) -> Image.Image:
    if not SETTINGS.color_harmonize:
        return generated_rgb

    base = np.asarray(
        base_rgb.convert("RGB"),
        dtype=np.float32,
    )
    generated = np.asarray(
        generated_rgb.convert("RGB"),
        dtype=np.float32,
    )
    mask = np.asarray(
        mask_image.convert("L"),
        dtype=np.uint8,
    )

    active = mask >= int(
        os.getenv(
            "INPAINT_COLOR_MIN_MASK",
            "92",
        )
    )

    if not np.any(active):
        return generated_rgb

    strength = SETTINGS.color_harmonize_strength

    for channel in range(3):
        base_values = base[:, :, channel][active]
        generated_values = generated[:, :, channel][active]

        base_mean = float(base_values.mean())
        generated_mean = float(generated_values.mean())
        base_std = float(base_values.std())
        generated_std = float(generated_values.std())

        if generated_std < 1e-3:
            continue

        ratio = base_std / generated_std
        ratio = max(0.75, min(1.25, ratio))

        corrected = (
            generated[:, :, channel] - generated_mean
        ) * ratio + base_mean

        generated[:, :, channel][active] = (
            (1.0 - strength)
            * generated[:, :, channel][active]
            + strength * corrected[active]
        )

    return Image.fromarray(
        np.clip(generated, 0, 255).astype(np.uint8),
        mode="RGB",
    )


def sharpen_generated_region(
    base_rgb: Image.Image,
    generated_rgb: Image.Image,
    mask_image: Image.Image,
) -> Image.Image:
    if not SETTINGS.local_sharpen:
        return generated_rgb

    mask = np.asarray(
        mask_image.convert("L"),
        dtype=np.uint8,
    )
    active = mask >= int(
        os.getenv(
            "INPAINT_SHARPEN_MIN_MASK",
            "92",
        )
    )

    if not np.any(active):
        return generated_rgb

    generated_image = generated_rgb.convert("RGB")
    generated = np.asarray(
        generated_image,
        dtype=np.float32,
    )
    blurred = np.asarray(
        generated_image.filter(
            ImageFilter.GaussianBlur(
                radius=float(
                    os.getenv(
                        "INPAINT_SHARPEN_BLUR",
                        "0.65",
                    )
                )
            )
        ),
        dtype=np.float32,
    )

    amount = max(
        0.0,
        min(
            0.8,
            float(
                os.getenv(
                    "INPAINT_SHARPEN_AMOUNT",
                    "0.3",
                )
            ),
        ),
    )

    sharpened = np.clip(
        generated + amount * (generated - blurred),
        0,
        255,
    )

    output = generated.copy()
    output[active] = sharpened[active]

    return Image.fromarray(
        output.astype(np.uint8),
        mode="RGB",
    )


def enhance_detail_cv(
    generated_rgb: Image.Image,
    mask_image: Image.Image,
) -> Image.Image:
    if not SETTINGS.opencv_detail or cv2 is None:
        return generated_rgb

    mask = np.asarray(
        mask_image.convert("L"),
        dtype=np.uint8,
    )
    active = mask >= int(
        os.getenv(
            "INPAINT_OPENCV_MIN_MASK",
            "92",
        )
    )

    if not np.any(active):
        return generated_rgb

    try:
        generated = np.asarray(
            generated_rgb.convert("RGB"),
            dtype=np.uint8,
        )

        sigma_s = max(
            0.0,
            min(
                40.0,
                float(
                    os.getenv(
                        "INPAINT_OPENCV_SIGMA_S",
                        "8.0",
                    )
                ),
            ),
        )
        sigma_r = max(
            0.01,
            min(
                0.5,
                float(
                    os.getenv(
                        "INPAINT_OPENCV_SIGMA_R",
                        "0.16",
                    )
                ),
            ),
        )
        blend = max(
            0.0,
            min(
                0.7,
                float(
                    os.getenv(
                        "INPAINT_OPENCV_BLEND",
                        "0.26",
                    )
                ),
            ),
        )

        bgr = cv2.cvtColor(
            generated,
            cv2.COLOR_RGB2BGR,
        )
        detailed = cv2.detailEnhance(
            bgr,
            sigma_s=sigma_s,
            sigma_r=sigma_r,
        )
        detailed = cv2.cvtColor(
            detailed,
            cv2.COLOR_BGR2RGB,
        ).astype(np.float32)

        output = generated.astype(np.float32)
        output[active] = (
            (1.0 - blend) * output[active]
            + blend * detailed[active]
        )

        return Image.fromarray(
            np.clip(output, 0, 255).astype(np.uint8),
            mode="RGB",
        )
    except Exception:
        logger.exception("Falha no OpenCV detail enhancement.")
        return generated_rgb


def restore_face_gfpgan(
    generated_rgb: Image.Image,
    mask_image: Image.Image,
) -> Image.Image:
    if not SETTINGS.gfpgan_restore:
        return generated_rgb

    restorer = MODELS.get_gfpgan()
    if restorer is None or cv2 is None:
        return generated_rgb

    try:
        generated = np.asarray(
            generated_rgb.convert("RGB"),
            dtype=np.uint8,
        )
        mask = np.asarray(
            mask_image.convert("L"),
            dtype=np.uint8,
        )

        active = mask >= int(
            os.getenv(
                "INPAINT_GFPGAN_MIN_MASK",
                "56",
            )
        )

        if not np.any(active):
            return generated_rgb

        bgr = cv2.cvtColor(
            generated,
            cv2.COLOR_RGB2BGR,
        )

        _, _, restored_bgr = restorer.enhance(
            bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )

        if restored_bgr is None:
            return generated_rgb

        restored = cv2.cvtColor(
            restored_bgr,
            cv2.COLOR_BGR2RGB,
        ).astype(np.float32)

        alpha = np.zeros(
            mask.shape,
            dtype=np.float32,
        )
        alpha[active] = float(
            os.getenv(
                "INPAINT_GFPGAN_BLEND",
                "0.34",
            )
        )

        face = detect_primary_face(generated_rgb)
        if face is not None:
            focus = face_ellipse_mask(
                mask_image.size,
                face,
                width_ratio=0.42,
                height_ratio=0.46,
                center_y_ratio=0.60,
                feather_ratio=0.18,
            )
            alpha = np.maximum(
                alpha,
                focus * float(
                    os.getenv(
                        "INPAINT_GFPGAN_FACE_BOOST",
                        "0.5",
                    )
                ),
            )

        alpha = np.clip(alpha, 0.0, 0.75)
        alpha = alpha[:, :, None]

        output = (
            (1.0 - alpha) * generated
            + alpha * restored
        )

        return Image.fromarray(
            np.clip(output, 0, 255).astype(np.uint8),
            mode="RGB",
        )
    except Exception:
        logger.exception("Falha no GFPGAN.")
        return generated_rgb


def composite_inpaint_result(
    original_rgb: Image.Image,
    generated_rgb: Image.Image,
    mask_image: Image.Image,
    blend_feather: float,
) -> Image.Image:
    blend_mask = mask_image

    if blend_feather > 0:
        blend_mask = blend_mask.filter(
            ImageFilter.GaussianBlur(
                radius=blend_feather
            )
        )

    return Image.composite(
        generated_rgb.convert("RGB"),
        original_rgb.convert("RGB"),
        blend_mask,
    )


def run_postprocessing(
    base_rgb: Image.Image,
    generated_rgb: Image.Image,
    mask_image: Image.Image,
) -> Image.Image:
    """Quality pipeline.

    Ordem intencional:
    1. recuperar detalhes;
    2. restaurar rosto;
    3. harmonizar cor/luz;
    4. proteger identidade por último;
    5. limpar artefatos.
    """
    generated = sharpen_generated_region(
        base_rgb,
        generated_rgb,
        mask_image,
    )
    generated = restore_face_gfpgan(
        generated,
        mask_image,
    )
    generated = enhance_detail_cv(
        generated,
        mask_image,
    )
    generated = harmonize_color(
        base_rgb,
        generated,
        mask_image,
    )
    generated = harmonize_luma(
        base_rgb,
        generated,
        mask_image,
    )
    generated = preserve_face_identity(
        base_rgb,
        generated,
        mask_image,
    )
    generated = suppress_dark_artifacts(
        base_rgb,
        generated,
        mask_image,
    )

    return generated


# ============================================================================
# Inference helpers
# ============================================================================

def run_img2img(
    source: Image.Image,
    prompt: str,
    negative_prompt: str,
    strength: float,
    guidance_scale: float,
    steps: int,
    seed: int,
) -> Image.Image:
    start = time.perf_counter()

    strength, guidance_scale, steps = tune_enhance_params(
        strength,
        guidance_scale,
        steps,
    )

    source_work = fit_image_for_model(
        source,
        SETTINGS.img2img_resolution,
    )

    try:
        pipeline = MODELS.get_img2img()
        generator = MODELS.seed_generator(seed)

        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source_work,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        )

        output = result.images[0].resize(
            source.size,
            Image.Resampling.LANCZOS,
        )

        logger.info(
            "img2img concluido em %.2fs | size=%sx%s | steps=%s",
            time.perf_counter() - start,
            source.width,
            source.height,
            steps,
        )

        return output
    except RuntimeError as exc:
        logger.exception("Erro de inferencia img2img.")
        raise HTTPException(
            status_code=500,
            detail=f"Erro de inferencia: {exc}",
        ) from exc


def run_inpaint(
    source: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str,
    strength: float,
    guidance_scale: float,
    steps: int,
    seed: int,
) -> Image.Image:
    start = time.perf_counter()

    strength, guidance_scale, steps = tune_inpaint_params(
        strength,
        guidance_scale,
        steps,
    )

    work_size = working_size(
        source.size,
        SETTINGS.inpaint_resolution,
    )

    source_work = source.resize(
        work_size,
        Image.Resampling.LANCZOS,
    )
    mask_work = mask.resize(
        work_size,
        Image.Resampling.BILINEAR,
    )

    try:
        pipeline = MODELS.get_inpaint()
        generator = MODELS.seed_generator(seed)

        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source_work,
            mask_image=mask_work,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        )

        output = result.images[0].convert("RGB")

        if output.size != source.size:
            output = output.resize(
                source.size,
                Image.Resampling.LANCZOS,
            )

        logger.info(
            "inpaint concluido em %.2fs | size=%sx%s | steps=%s",
            time.perf_counter() - start,
            source.width,
            source.height,
            steps,
        )

        return run_postprocessing(
            source,
            output,
            mask,
        )
    except RuntimeError as exc:
        logger.exception("Erro de inferencia inpaint.")
        raise HTTPException(
            status_code=500,
            detail=f"Erro de inferencia no inpainting: {exc}",
        ) from exc


# ============================================================================
# Startup
# ============================================================================

def warmup_models_background() -> None:
    if not SETTINGS.warmup_on_start:
        return

    def warmup() -> None:
        try:
            MODELS.get_img2img()
            logger.info("Warmup img2img concluido.")
        except Exception:
            logger.exception("Warmup img2img falhou.")

    Thread(
        target=warmup,
        daemon=True,
    ).start()


@app.on_event("startup")
def on_startup() -> None:
    logger.info(
        "Portrait Studio iniciado | device=%s | max_quality=%s",
        SETTINGS.device,
        SETTINGS.max_quality,
    )
    warmup_models_background()


# ============================================================================
# Routes
# ============================================================================

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "portrait-studio-api",
    }


@app.get("/ready")
def ready() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "portrait-studio-api",
        "models": MODELS.status(),
    }


@app.get("/")
def root() -> Any:
    if FRONTEND_INDEX.exists():
        return FileResponse(str(FRONTEND_INDEX))

    return {
        "service": "portrait-studio-api",
        "version": "1.0.0",
        "ok": True,
        "health": "/health",
        "ready": "/ready",
        "enhance": "/enhance",
        "inpaint": "/inpaint",
        "auto-repair": "/auto-repair",
        "outpaint": "/outpaint",
        "credential-process": "/credential-process",
    }


@app.get("/config.js")
def frontend_config() -> Response:
    config_path = FRONTEND_DIR / "config.js"

    if config_path.exists():
        return FileResponse(
            str(config_path),
            media_type="application/javascript",
        )

    raise HTTPException(
        status_code=404,
        detail="config.js nao encontrado.",
    )


@app.post("/enhance")
def enhance(
    file: UploadFile = File(...),
    prompt: str = Form(DEFAULT_ENHANCE_PROMPT),
    negative_prompt: str = Form(DEFAULT_ENHANCE_NEGATIVE),
    strength: float = Form(0.33),
    guidance_scale: float = Form(4.9),
    num_inference_steps: int = Form(30),
    seed: int = Form(-1),
) -> Response:
    validate_strength(strength, 0.95)
    validate_steps(num_inference_steps, 160)
    validate_guidance(guidance_scale)

    source = load_upload(
        file,
        mode="RGB",
        label="file",
    )

    output = run_img2img(
        source=source,
        prompt=prompt,
        negative_prompt=negative_prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        steps=num_inference_steps,
        seed=seed,
    )

    return png_response(output)


@app.post("/inpaint")
def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(DEFAULT_INPAINT_PROMPT),
    negative_prompt: str = Form(DEFAULT_INPAINT_NEGATIVE),
    strength: float = Form(0.47),
    guidance_scale: float = Form(3.5),
    num_inference_steps: int = Form(38),
    seed: int = Form(-1),
    preserve_unmasked: bool = Form(True),
    mask_blur: float = Form(2.2),
    blend_feather: float = Form(1.8),
    realistic_mode: bool = Form(True),
) -> Response:
    validate_strength(strength)
    validate_steps(num_inference_steps)

    if mask_blur < 0 or mask_blur > 24:
        raise HTTPException(
            status_code=400,
            detail="mask_blur deve estar entre 0 e 24.",
        )

    if blend_feather < 0 or blend_feather > 24:
        raise HTTPException(
            status_code=400,
            detail="blend_feather deve estar entre 0 e 24.",
        )

    source = load_upload(
        image,
        mode="RGB",
        label="image",
    )
    mask_image = load_upload(
        mask,
        mode="L",
        label="mask",
    )

    mask_image = prepare_inpaint_mask(
        mask_image,
        source.size,
        blur_radius=mask_blur,
    )
    mask_image = protect_face_identity_mask(
        mask_image,
        source,
    )

    if mask_image.getbbox() is None:
        return png_response(source)

    if realistic_mode:
        coverage = mask_coverage(mask_image)

        (
            strength,
            guidance_scale,
            num_inference_steps,
            mask_blur,
            blend_feather,
        ) = refine_inpaint_for_realism(
            strength,
            guidance_scale,
            num_inference_steps,
            coverage,
            mask_blur,
            blend_feather,
        )

        mask_image = prepare_inpaint_mask(
            mask_image,
            source.size,
            blur_radius=mask_blur,
        )
        mask_image = protect_face_identity_mask(
            mask_image,
            source,
        )

    output_generated = run_inpaint(
        source=source,
        mask=mask_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        steps=num_inference_steps,
        seed=seed,
    )

    # Preserva pixels fora da máscara por padrão. A opção existe apenas para
    # compatibilidade com o frontend atual.
    if preserve_unmasked:
        output = composite_inpaint_result(
            source,
            output_generated,
            mask_image,
            blend_feather,
        )
    else:
        output = output_generated

    return png_response(output)


@app.post("/auto-repair")
def auto_repair(
    file: UploadFile = File(...),
    prompt: str = Form(DEFAULT_REPAIR_PROMPT),
    negative_prompt: str = Form(DEFAULT_REPAIR_NEGATIVE),
    strength: float = Form(0.56),
    guidance_scale: float = Form(3.5),
    num_inference_steps: int = Form(38),
    seed: int = Form(-1),
) -> Response:
    validate_strength(strength)
    validate_steps(num_inference_steps)
    validate_guidance(guidance_scale)

    source_data = read_upload_bytes(
        file,
        "file",
    )
    source_rgba = load_image_from_bytes(
        source_data,
        mode="RGBA",
        label="file",
    )

    alpha = source_rgba.getchannel("A")
    bounds = alpha.getbbox()

    if not bounds:
        return png_response(source_rgba)

    left, top, right, bottom = bounds
    width, height = source_rgba.size
    margin = max(
        2,
        int(min(width, height) * 0.015),
    )

    touches = {
        "top": top <= margin,
        "left": left <= margin,
        "right": right >= width - margin,
        "bottom": bottom >= height - margin,
    }

    base_rgb = Image.new(
        "RGB",
        source_rgba.size,
        (235, 235, 235),
    )
    base_rgb.paste(
        source_rgba.convert("RGB"),
        mask=alpha,
    )

    if any(touches.values()):
        mask_image = auto_repair_mask(
            alpha,
            touches,
        )
    elif SETTINGS.auto_repair_inner_transparency:
        mask_image = auto_repair_transparency_mask(alpha)
    else:
        return png_response(source_rgba)

    mask_image = protect_face_identity_mask(
        mask_image,
        base_rgb,
    )

    if mask_image.getbbox() is None:
        return png_response(source_rgba)

    output_rgb = run_inpaint(
        source=base_rgb,
        mask=mask_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        steps=num_inference_steps,
        seed=seed,
    )

    expanded_alpha = ImageChops.lighter(
        alpha,
        mask_image.point(
            lambda value: int(
                min(255, value * 1.25)
            )
        ),
    )

    output_rgba = output_rgb.convert("RGBA")
    output_rgba.putalpha(expanded_alpha)

    return png_response(output_rgba)


@app.post("/outpaint")
def outpaint(
    file: UploadFile = File(...),
    pad_top: int = Form(0),
    pad_left: int = Form(0),
    pad_right: int = Form(0),
    pad_bottom: int = Form(0),
    prompt: str = Form(DEFAULT_OUTPAINT_PROMPT),
    negative_prompt: str = Form(DEFAULT_OUTPAINT_NEGATIVE),
    guidance_scale: float = Form(3.3),
    num_inference_steps: int = Form(46),
    seed: int = Form(-1),
) -> Response:
    validate_steps(num_inference_steps)
    validate_guidance(guidance_scale)

    source = load_upload(
        file,
        mode="RGB",
        label="file",
    )

    pads = {
        "top": pad_top,
        "left": pad_left,
        "right": pad_right,
        "bottom": pad_bottom,
    }

    max_pad = max(source.size)

    if any(
        value < 0 or value > max_pad
        for value in pads.values()
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cada pad deve estar entre 0 e "
                f"{max_pad} px."
            ),
        )

    if not any(pads.values()):
        return png_response(source)

    _, guidance_scale, num_inference_steps = tune_inpaint_params(
        0.99,
        guidance_scale,
        num_inference_steps,
    )

    overlap = max(
        8,
        int(min(source.size) * 0.02),
    )

    canvas, mask = outpaint_canvas(
        source,
        pads,
        overlap,
    )

    mask = protect_face_identity_mask(
        mask,
        source,
    )

    # Outpaint usa o mesmo modelo de inpainting, mas com a máscara criada
    # a partir das áreas adicionadas ao canvas.
    work_size = working_size(
        canvas.size,
        int(
            os.getenv(
                "OUTPAINT_RESOLUTION",
                "256"
                if SETTINGS.device == "cpu"
                else (
                    "1280"
                    if SETTINGS.max_quality
                    else "1024"
                ),
            )
        ),
    )

    try:
        pipeline = MODELS.get_inpaint()
        generator = MODELS.seed_generator(seed)

        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=canvas.resize(
                work_size,
                Image.Resampling.LANCZOS,
            ),
            mask_image=mask.resize(
                work_size,
                Image.Resampling.BILINEAR,
            ),
            width=work_size[0],
            height=work_size[1],
            strength=0.8,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        generated = result.images[0].convert("RGB")
    except RuntimeError as exc:
        logger.exception("Erro de inferencia outpaint.")
        raise HTTPException(
            status_code=500,
            detail=f"Erro de inferencia no outpainting: {exc}",
        ) from exc

    generated = generated.resize(
        canvas.size,
        Image.Resampling.LANCZOS,
    )

    generated = run_postprocessing(
        canvas,
        generated,
        mask,
    )

    blend_mask = mask.filter(
        ImageFilter.GaussianBlur(
            radius=overlap / 2
        )
    )

    final = Image.composite(
        generated,
        canvas,
        blend_mask,
    )

    return png_response(final)


@app.post("/credential-process")
def credential_process(
    file: UploadFile = File(...),
    shape: str = Form("round"),
    clothing_type: str = Form("auto"),
    seed: int = Form(-1),
    eye_left_x: float | None = Form(None),
    eye_left_y: float | None = Form(None),
    eye_right_x: float | None = Form(None),
    eye_right_y: float | None = Form(None),
) -> Response:
    """Person-first credential framing endpoint (non-generative by default)."""
    source_data = read_upload_bytes(file, "file")

    with Image.open(io.BytesIO(source_data)) as image:
        source = image.convert("RGBA") if "A" in image.getbands() else image.convert("RGB")

    eye_left: tuple[float, float] | None = None
    eye_right: tuple[float, float] | None = None
    if None not in (eye_left_x, eye_left_y, eye_right_x, eye_right_y):
        eye_left = (float(eye_left_x), float(eye_left_y))
        eye_right = (float(eye_right_x), float(eye_right_y))

    result = process_credential_framing(
        source,
        eye_left=eye_left,
        eye_right=eye_right,
        clothing_type=clothing_type,
        seed=seed,
    )

    output = result.round_image if shape.strip().lower() == "round" else result.square
    response = png_response(output)
    response.headers["X-Credential-Status"] = result.status
    response.headers["X-Credential-Reasons"] = ",".join(result.reasons) if result.reasons else ""
    response.headers["X-Clothing-Status"] = result.clothing_status
    response.headers["X-Clothing-Generated"] = "true" if result.clothing_generated else "false"
    if "clothing_generated_area_ratio" in result.metrics:
        response.headers["X-Clothing-Mask-Area"] = f"{result.metrics['clothing_generated_area_ratio']:.6f}"
    if all(k in result.metrics for k in ("clothing_mask_x0", "clothing_mask_y0", "clothing_mask_x1", "clothing_mask_y1")):
        response.headers["X-Clothing-Mask-BBox"] = (
            f"{int(result.metrics['clothing_mask_x0'])},"
            f"{int(result.metrics['clothing_mask_y0'])},"
            f"{int(result.metrics['clothing_mask_x1'])},"
            f"{int(result.metrics['clothing_mask_y1'])}"
        )
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
