import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from diffusers import QwenImage21Pipeline
from huggingface_hub import InferenceClient
from huggingface_hub.inference._providers import get_provider_helper


EDIT_PROMPT = (
    "Professional corporate portrait. Keep the same person and facial identity. "
    "Formal business attire, clean grooming, realistic skin texture, neutral expression. "
    "Studio soft lighting, high sharpness on eyes, natural colors. "
    "Replace background with clean light-neutral office backdrop, no distracting objects, "
    "head-and-shoulders framing suitable for profile photo."
)

TRANSPARENT_PROMPT = (
    "RGBA image with transparency. Keep the same person and identity. "
    "Corporate headshot cutout, front-facing, formal business outfit, clean edge matte, "
    "natural skin tones, high detail in eyes and hair, no background, fully transparent outside subject, "
    "no extra text, no watermark."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic batch corporate portrait enhancement with Qwen Image 2.1."
    )
    parser.add_argument("--input-dir", default="input", help="Input directory for batch processing.")
    parser.add_argument("--output-dir", default="output", help="Output directory.")
    parser.add_argument("--input", default="input.png", help="Single input image for --mode single.")
    parser.add_argument(
        "--mode",
        choices=["batch", "single"],
        default="batch",
        help="Use batch mode by default."
    )
    parser.add_argument("--steps", type=int, default=40, help="Inference steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image-2.1",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--hf-api-model",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Fallback model id used only in hf-api mode when --model is unsupported for image-to-image.",
    )
    parser.add_argument(
        "--skip-transparent",
        action="store_true",
        help="Generate only the corporate edited image.",
    )
    return parser.parse_args()


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponivel")


def detect_mode() -> str:
    return "local-cuda" if torch.cuda.is_available() else "hf-api"


def create_local_pipeline(model: str) -> QwenImage21Pipeline:
    return QwenImage21Pipeline.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda")


def create_hf_client() -> InferenceClient:
    token = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return InferenceClient(token=token)


def pick_hf_api_model(preferred_model: str, fallback_model: str) -> str:
    try:
        get_provider_helper(None, task="image-to-image", model=preferred_model)
        return preferred_model
    except Exception:
        return fallback_model


def list_input_images(input_dir: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".webp"}
    return [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in extensions]


def process_image(
    pipe: QwenImage21Pipeline | None,
    client: InferenceClient | None,
    run_mode: str,
    model_id: str,
    image_path: Path,
    output_dir: Path,
    steps: int,
    seed: int,
    skip_transparent: bool,
) -> None:
    source = Image.open(image_path).convert("RGBA")
    base = image_path.stem

    if run_mode == "local-cuda":
        assert pipe is not None
        edited = pipe(
            prompt=EDIT_PROMPT,
            image=source,
            num_inference_steps=steps,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
    else:
        assert client is not None
        edited = client.image_to_image(
            image=source,
            prompt=EDIT_PROMPT,
            model=model_id,
        )
    edited_path = output_dir / f"{base}_corporativo.png"
    edited.save(edited_path)

    if skip_transparent:
        print(f"OK: {image_path.name} -> {edited_path.name}")
        return

    if run_mode == "local-cuda":
        assert pipe is not None
        transparent = pipe(
            prompt=TRANSPARENT_PROMPT,
            image=edited,
            num_inference_steps=steps,
            generator=torch.Generator("cuda").manual_seed(seed + 1),
        ).images[0]
    else:
        assert client is not None
        transparent = client.image_to_image(
            image=edited,
            prompt=TRANSPARENT_PROMPT,
            model=model_id,
        )
    transparent_path = output_dir / f"{base}_transparente.png"
    transparent.save(transparent_path)
    print(f"OK: {image_path.name} -> {edited_path.name}, {transparent_path.name}")


def main() -> None:
    args = parse_args()
    run_mode = detect_mode()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe: QwenImage21Pipeline | None = None
    client: InferenceClient | None = None
    model_for_run = args.model
    if run_mode == "local-cuda":
        pipe = create_local_pipeline(args.model)
        print("Modo: local-cuda")
    else:
        client = create_hf_client()
        model_for_run = pick_hf_api_model(args.model, args.hf_api_model)
        print("Modo: hf-api (CUDA nao encontrada)")
        print("Dica: defina HF_TOKEN para maior estabilidade de uso da API.")
        if model_for_run != args.model:
            print(
                "Aviso: modelo",
                args.model,
                "nao compativel com image-to-image no hf-api. Usando fallback:",
                model_for_run,
            )

    if args.mode == "single":
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {input_path}")
        process_image(
            pipe=pipe,
            client=client,
            run_mode=run_mode,
            model_id=model_for_run,
            image_path=input_path,
            output_dir=output_dir,
            steps=args.steps,
            seed=args.seed,
            skip_transparent=args.skip_transparent,
        )
        print(f"Concluido. Saidas em: {output_dir.resolve()}")
        return

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Pasta de entrada nao encontrada: {input_dir}")

    images = list_input_images(input_dir)
    if not images:
        raise RuntimeError("Nenhuma imagem encontrada na pasta de entrada.")

    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] Processando {image_path.name}...")
        process_image(
            pipe=pipe,
            client=client,
            run_mode=run_mode,
            model_id=model_for_run,
            image_path=image_path,
            output_dir=output_dir,
            steps=args.steps,
            seed=args.seed + index,
            skip_transparent=args.skip_transparent,
        )

    print(f"Concluido. {len(images)} arquivo(s) processado(s). Saidas em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
