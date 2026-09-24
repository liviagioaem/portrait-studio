# HF Model Bridge (Minimo)

Projeto reduzido para manter apenas o necessario para usar um modelo novo do Hugging Face via API.

## Arquivos mantidos

- `hf-server.js`: ponte local que recebe imagem e chama o modelo no Hugging Face.
- `package.json` e `package-lock.json`: scripts e dependencias Node.
- `.env`: variaveis de ambiente locais.

## Configuracao

No `.env`, use:

```env
HF_API_KEY=hf_seu_token
HF_MODEL=caidas/swin2SR-classical-sr-x2-64
PORT=8787
```

## Uso

```bash
npm install
npm run start:hf
```

Health check:

```bash
curl http://localhost:8787/health
```
