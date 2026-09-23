# AI Prototype: Generative Fill (Hugging Face Diffusers)

Este prototipo roda separado do Portrait Studio para validar inpainting com imagem + mascara.

A logica de inpainting fica em `pipeline.py`, compartilhada pelo script CLI e pela API.
Requer Python 3.10 ou superior.

## 1) Criar ambiente Python (Windows PowerShell)

```powershell
cd c:\Work\portrait-studio\ai-prototype
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Teste rapido local (arquivo para arquivo)

Coloque arquivos em `samples/`:

- `input.png` (foto original)
- `mask.png` (branco = preencher, preto = preservar)

Rode:

```powershell
python .\inpaint_test.py --image .\samples\input.png --mask .\samples\mask.png --output .\samples\out_filled.png --seed 1234
```

## 3) Rodar API local (para integrar depois ao frontend)

```powershell
uvicorn server:app --host 127.0.0.1 --port 8001
```

Healthcheck:

- `GET http://127.0.0.1:8001/health`
- `POST http://127.0.0.1:8001/warmup` (carrega o modelo antes do primeiro uso real)

Endpoint:

- `POST http://127.0.0.1:8001/inpaint`
- form-data:
  - `image` (file)
  - `mask` (file)
  - `prompt` (text, opcional)
  - `negative_prompt` (text, opcional)
  - `num_inference_steps` (int, opcional, padrao 30)
  - `guidance_scale` (float, opcional, padrao 7.0)
  - `seed` (int, opcional; `-1` ou ausente = aleatorio)

## Como a precisao e preservada

- A difusao roda entre 512 e 768 px no lado maior (faixa de treino do SD 1.5) e o
  resultado volta para a resolucao original.
- A mascara e binarizada (branco preenche, preto preserva).
- A area preservada e recomposta a partir do original, com suavizacao so na emenda.
  Sem isso o round-trip do VAE reduziria o detalhe do rosto inteiro.
- `--seed` torna a execucao reproduzivel.

## Notas importantes

- Em CPU vai funcionar, mas pode ser lento.
- Em GPU NVIDIA (CUDA) o desempenho melhora bastante.
- A qualidade depende da mascara e da quantidade de area faltante.
- Reconstrucao de cabelo/cabeca e uma estimativa plausivel, nao recuperacao factual.
