# HF Model Bridge

Projeto para processar imagem com modelo do Hugging Face em dois modos:

- remoto (API do Hugging Face)
- local (modelo rodando na sua maquina via Python)

## Arquivos mantidos

- `hf-server.js`: ponte local que recebe imagem e chama o modelo no Hugging Face.
- `package.json` e `package-lock.json`: scripts e dependencias Node.
- `.env`: variaveis de ambiente locais.

## Configuracao

No `.env`, use:

```env
HF_API_KEY=hf_seu_token
HF_MODEL=black-forest-labs/FLUX.2-klein-base-9b-fp8
PORT=8787
HF_INFERENCE_MODE=auto
HF_LOCAL_ENDPOINT=http://127.0.0.1:7860
```

Valores para `HF_INFERENCE_MODE`:

- `local`: usa apenas o servidor Python local
- `remote`: usa apenas a API Hugging Face
- `auto`: tenta local primeiro, e cai para remoto se falhar

## Uso

```bash
npm install
npm run start:hf
```

Health check:

```bash
curl http://localhost:8787/health
```

## Modo local (modelo baixado)

1. Instale dependencias Python:

```bash
py -m pip install -r requirements-local.txt
```

1. Suba o servidor local Python:

```bash
py -m uvicorn hf-local-server:app --host 127.0.0.1 --port 7860
```

1. Com `HF_INFERENCE_MODE=local` (ou `auto`) no `.env`, rode o bridge Node:

```bash
npm run start:hf
```

4. Envie imagem para:

```text
POST http://localhost:8787/api/hf/enhance
```

Campos aceitos no form-data:

- `image` (obrigatorio)
- `model` (opcional)
- `prompt` (opcional)

## GitHub Pages + Render

Para funcionar no GitHub Pages, mantenha o frontend estatico no Pages e rode o backend em um host Node publico.

1. Suba este repositorio no GitHub.
2. No Render, crie o servico usando o arquivo [render.yaml](render.yaml).
3. No painel do Render, configure o segredo HF_API_KEY.
4. Aguarde o deploy e copie a URL publica do backend.
5. Abra seu GitHub Pages com os parametros hfEndpoint e hfModel:

https://SEU-USUARIO.github.io/SEU-REPO/?hfEndpoint=https://SUA-URL.onrender.com&hfModel=black-forest-labs/FLUX.2-klein-base-9b-fp8

Teste rapido do backend publico:

https://SUA-URL.onrender.com/health
