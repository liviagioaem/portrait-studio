import argparse
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Teste simples do endpoint /enhance")
    parser.add_argument("image", help="Caminho da imagem de entrada")
    parser.add_argument("--api", default="http://localhost:8000", help="Base URL da API")
    parser.add_argument("--out", default="enhance_test_output.png", help="Arquivo de saida")
    parser.add_argument(
        "--prompt",
        default=(
            "professional studio headshot of the same person, preserve identity and facial geometry, "
            "natural skin texture with realistic pores, neutral white balance, soft even lighting, "
            "sharp eyes, clean edges, high realism"
        ),
    )
    parser.add_argument(
        "--negative",
        default=(
            "cartoon, anime, painting, cgi, plastic skin, waxy skin, over-smoothing, lowres, blurry, "
            "noise, jpeg artifacts, deformed face, asymmetrical eyes, crossed eyes, extra eyes, extra limbs, "
            "duplicate person, text, watermark, logo"
        ),
    )
    parser.add_argument("--strength", type=float, default=0.45)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Imagem nao encontrada: {image_path}")

    url = args.api.rstrip("/") + "/enhance"
    data = {
        "prompt": args.prompt,
        "negative_prompt": args.negative,
        "strength": str(args.strength),
        "guidance_scale": str(args.guidance),
        "num_inference_steps": str(args.steps),
        "seed": str(args.seed),
    }

    with image_path.open("rb") as fp:
        files = {"file": (image_path.name, fp, "image/png")}
        response = requests.post(url, data=data, files=files, timeout=600)

    if response.status_code != 200:
        raise SystemExit(f"Falha {response.status_code}: {response.text}")

    out_path = Path(args.out)
    out_path.write_bytes(response.content)
    print(f"OK: imagem salva em {out_path.resolve()}")


if __name__ == "__main__":
    main()
