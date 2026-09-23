import argparse
from pathlib import Path

from PIL import Image

from pipeline import (
    DEFAULT_MODEL,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    device_of,
    get_pipeline,
    run_inpaint,
)


def main():
    parser = argparse.ArgumentParser(description="Local inpainting proof-of-concept.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--mask", required=True, help="Input mask path (white=inpaint, black=keep).")
    parser.add_argument("--output", default="out_filled.png", help="Output path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=None, help="Fixed seed for reproducible runs.")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    mask = Image.open(args.mask)

    pipe = get_pipeline(args.model)
    print(f"Running on {device_of(pipe)} with model {args.model}")

    result = run_inpaint(
        image,
        mask,
        prompt=args.prompt,
        negative_prompt=args.negative,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        model_id=args.model,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    print(f"Saved: {output} ({result.width}x{result.height})")


if __name__ == "__main__":
    main()
