# Portrait Studio

Ferramenta web local para preparar retratos com fundo transparente em formato padrão, com revisão visual e exportação em lote.

## O que a ferramenta faz

- Importa múltiplas imagens (JPG, PNG, WebP).
- Detecta olhos automaticamente (com opção de ajuste manual).
- Remove fundo localmente no navegador.
- Padroniza enquadramento para uso profissional.
- Gera saídas quadrada e redonda por foto.
- Permite aprovação/rejeição por item.
- Exporta PNGs aprovados em ZIP.

## Arquitetura

- Frontend puro em um único arquivo: index.html
- Sem backend
- Execução local no browser
- Modelos carregados por CDN (MediaPipe)

## Integrando Stable Diffusion (Hugging Face)

Stable Diffusion nao deve rodar direto no navegador deste projeto por custo de GPU/memoria.
O caminho recomendado e manter o frontend atual e adicionar uma API Python para inferencia.

### 1) Backend local (ja incluído em `backend/`)

Arquivos criados:

- `backend/app.py` (API FastAPI com endpoint `/enhance`)
- `backend/requirements.txt`
- `backend/.env.example`

### 2) Criar ambiente e instalar dependencias

No terminal, na raiz do projeto:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3) Configurar token e modelos

Copie `backend/.env.example` para `.env` e ajuste se necessario.

- `HF_TOKEN`: token do Hugging Face (somente se o modelo exigir)
- `IMG2IMG_MODEL_ID`: por padrao `runwayml/stable-diffusion-v1-5` (gratuito/open-source)
- `IMG2IMG_RESOLUTION`: recomendado `768` (CPU mais rapido com `512`)
- `INPAINT_MODEL_ID`: por padrao `runwayml/stable-diffusion-inpainting`
- `INPAINT_RESOLUTION`: recomendado `512` para comecar
- `LOW_VRAM`: `1` para descarregar partes do modelo na CPU (mais lento, usa menos VRAM)
- `DEVICE`: `cuda` (recomendado) ou `cpu` (muito lento)

### 4) Iniciar API

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

Teste rapido:

```powershell
curl http://localhost:8000/health
```

### 5) Chamar o endpoint no frontend

Exemplo minimo (JavaScript) para enviar uma imagem ao Stable Diffusion image-to-image e receber PNG processado:

```javascript
async function enhanceWithFlux(file) {
  const form = new FormData();
  form.append("file", file);
  form.append(
    "prompt",
    "professional portrait, realistic skin, soft studio light",
  );
  form.append("negative_prompt", "artifacts, blur, distorted face, watermark");
  form.append("strength", "0.35");
  form.append("guidance_scale", "3.5");
  form.append("num_inference_steps", "30");

  const res = await fetch("http://localhost:8000/enhance", {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Falha no image-to-image: ${res.status} ${errText}`);
  }

  const blob = await res.blob();
  return URL.createObjectURL(blob);
}
```

Depois, use a URL retornada para mostrar preview ou substituir a imagem do card antes da exportacao final.

### 6) Inpainting (reconstruir parte da foto com mascara)

O backend agora expoe `POST /inpaint` com dois arquivos:

- `image`: imagem original
- `mask`: mascara em tons de cinza

Regra da mascara:

- branco (255) = area a reconstruir
- preto (0) = area preservada

Refinamentos do endpoint `/inpaint`:

- `preserve_unmasked=true` (padrao): preserva pixels fora da mascara sem alteracao.
- `mask_blur=2.4` (padrao): suaviza borda da mascara antes da inferencia.
- `blend_feather=2.0` (padrao): suaviza a costura no compositing final.
- `realistic_mode=true` (padrao): autoajusta `strength`, `guidance` e `steps` pelo tamanho da area mascarada para resultado mais natural.

Exemplo minimo (JavaScript):

```javascript
async function inpaintWithMask(imageFile, maskFile) {
  const form = new FormData();
  form.append("image", imageFile);
  form.append("mask", maskFile);
  form.append(
    "prompt",
    "natural skin texture, restore missing facial details, keep identity",
  );
  form.append(
    "negative_prompt",
    "artifacts, blur, deformed eyes, extra mouth, text, watermark",
  );
  form.append("strength", "0.52");
  form.append("guidance_scale", "3.2");
  form.append("num_inference_steps", "18");
  form.append("preserve_unmasked", "true");
  form.append("mask_blur", "2.4");
  form.append("blend_feather", "2.0");
  form.append("realistic_mode", "true");

  const res = await fetch("http://localhost:8000/inpaint", {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Falha no inpainting: ${res.status} ${errText}`);
  }

  const blob = await res.blob();
  return URL.createObjectURL(blob);
}
```

Dica: para retrato, comece com `strength` entre `0.6` e `0.8`; valores muito baixos quase nao alteram a area, e valores muito altos podem descaracterizar o rosto.

### 7) Completar cabeca/ombros cortados (automatico)

O backend expoe `POST /outpaint`, que expande a **foto original** nos lados indicados e usa o modelo de inpainting do Hugging Face para desenhar so a area nova. Os pixels originais nao sao alterados.

Parametros: `file` (imagem) e `pad_top`, `pad_left`, `pad_right`, `pad_bottom` (pixels a acrescentar em cada lado).

Como o frontend (`index.html`) usa:

1. remove o fundo da foto original e calcula o enquadramento
2. se a pessoa encosta numa borda da foto **e** o retrato exportado precisa de area alem dessa borda, pede o outpainting so desses lados
3. remove o fundo de novo na foto completada e gera as saidas quadrada e redonda

So roda no processamento final (a pre-visualizacao do alinhamento continua sem IA).
Se o backend estiver offline, o app continua normalmente sem interromper o lote.

O endpoint antigo `POST /auto-repair` continua disponivel, mas o frontend nao o usa mais.

## Deploy no Render (passo a passo)

### 1) Subir o repositorio no GitHub

Garanta que a pasta `backend/` esteja versionada com `app.py` e `requirements.txt`.

### 2) Criar o servico Web Service no Render

No painel do Render:

1. New + > Web Service
2. Conecte seu repositorio
3. Root Directory: `backend`
4. Runtime: `Python 3`

### 3) Definir Build e Start

- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### 4) Configurar variaveis de ambiente

Use estes valores iniciais:

- `DEVICE=cpu` (primeiro deploy para validar; depois troque para `cuda` se plano suportar GPU)
- `IMG2IMG_MODEL_ID=runwayml/stable-diffusion-v1-5`
- `IMG2IMG_RESOLUTION=512` (CPU mais estavel)
- `INPAINT_MODEL_ID=runwayml/stable-diffusion-inpainting`
- `INPAINT_RESOLUTION=512`
- `LOW_VRAM=1`
- `HF_HOME=/opt/render/project/.cache/huggingface`
- `TRANSFORMERS_CACHE=/opt/render/project/.cache/huggingface`
- `HF_TOKEN=` (somente se precisar)

### 5) Liberar CORS e testar

Depois de deploy, teste:

```bash
curl https://SUA-API.onrender.com/health
```

Se responder `{"ok": true}`, API esta pronta.

### 6) Apontar frontend para a API do Render

No arquivo `config.js`, troque:

- de: `http://localhost:8000`
- para: `https://SUA-API.onrender.com`

Alternativa recomendada (ja implementada):

- local: usa `http://localhost:8000`
- producao: usa `${window.location.origin}/api`
- override manual: `localStorage.setItem("portrait_api_base", "https://SUA-API.onrender.com")`

### 7) Ajustes de performance

- CPU: mantenha imagens em 512-768 px no lado maior.
- GPU: pode subir para 768-1024 px e mais passos.
- Se houver erro de memoria, reduza `IMG2IMG_RESOLUTION`, `INPAINT_RESOLUTION` e `num_inference_steps`.

### 8) Teste rapido do endpoint enhance

Foi adicionado o script `backend/test_enhance.py` para validar a API com uma imagem real.

Exemplo local:

```bash
cd backend
python test_enhance.py ./sua-imagem.png --api http://localhost:8000 --out ./resultado.png
```

Exemplo com API no Render:

```bash
cd backend
python test_enhance.py ./sua-imagem.png --api https://SUA-API.onrender.com --out ./resultado.png
```

## Plano Render (custo x desempenho)

Perfil 1 - Validacao barata (CPU):

- `DEVICE=cpu`
- `IMG2IMG_RESOLUTION=512`
- `INPAINT_RESOLUTION=512`
- `num_inference_steps=20-30`
- Uso: prova de conceito, baixo volume, latencia maior.

Perfil 2 - Producao inicial (GPU pequena):

- `DEVICE=cuda`
- `LOW_VRAM=1`
- `IMG2IMG_RESOLUTION=768`
- `INPAINT_RESOLUTION=512`
- `num_inference_steps=24-32`
- Uso: qualidade boa com custo controlado.

Perfil 3 - Qualidade alta (GPU maior):

- `DEVICE=cuda`
- `LOW_VRAM=0`
- `IMG2IMG_RESOLUTION=1024`
- `INPAINT_RESOLUTION=768-1024`
- `num_inference_steps=30-40`
- Uso: melhor qualidade e throughput, custo mais alto.

## Uso sem backend local (somente abrir e usar)

Para que outras pessoas usem IA sem rodar nada local, publique o backend em um servidor e aponte o frontend para essa URL.

### 1) Publicar backend em servidor

Suba a pasta `backend/` em um host Python (Render, Railway, Fly.io, VM, etc.).

Comando de start sugerido:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

Variaveis de ambiente no servidor:

- `HF_TOKEN`
- `IMG2IMG_MODEL_ID`
- `INPAINT_MODEL_ID`
- `DEVICE` (`cuda` recomendado)

### 2) Configurar o frontend para API remota

O `index.html` usa `window.PORTRAIT_API_BASE` ou `localStorage.portrait_api_base`.

Opcao A (recomendada, fixa no HTML):

```html
<script>
  window.PORTRAIT_API_BASE = "https://SUA-API-IA.com";
</script>
```

Opcao B (sem editar arquivo, via console do navegador):

```javascript
localStorage.setItem("portrait_api_base", "https://SUA-API-IA.com");
```

Com isso, o usuario final so abre o app e usa, sem backend local.

### Checklist de publicacao (time usando sem setup)

1. Publicar API de IA (backend) em um host com GPU quando possivel.
2. Configurar variaveis do backend no host: `HF_TOKEN`, `DEVICE`, `INPAINT_MODEL_ID`, `IMG2IMG_MODEL_ID`.
3. Obter a URL final da API publicada (exemplo: `https://portrait-api.seudominio.com`).
4. Editar `config.js` na raiz do projeto e trocar `window.PORTRAIT_API_BASE` para a URL final.
5. Publicar frontend estatico (GitHub Pages/Netlify/Vercel).
6. Abrir a URL publica do frontend e testar envio de imagem + auto-reparo.
7. Compartilhar apenas a URL do frontend com o time (sem passos de backend local).

### Observacoes importantes

- Stable Diffusion tambem e pesado: use GPU NVIDIA para desempenho melhor.
- Em CPU funciona, mas costuma ser inviavel para lote.
- Se sua meta principal e melhorar nitidez/upscale fiel ao original, combine este fluxo com um modelo dedicado de super-resolucao.

## Requisitos

- Navegador moderno (Chrome/Edge recomendados)
- Internet na primeira execução (para baixar modelos)

## Como usar

1. Abra o arquivo index.html no navegador.
2. Clique em Adicionar fotos ou arraste arquivos.
3. Aguarde o processamento automático.
4. Revise cada foto e ajuste olhos quando necessário.
5. Defina Nome e Sobrenome por foto (usados no nome do arquivo final).
6. Marque fotos aprovadas.
7. Clique em Baixar aprovadas para gerar o ZIP.

## Convenção de nome dos arquivos

Os arquivos exportados seguem este padrão:

001_perfil_nome_sobrenome_quadrada.png
001_perfil_nome_sobrenome_redonda.png

Onde:

- perfil\_ é fixo
- nome e sobrenome são editáveis no card da foto

## Ajustes rápidos de enquadramento

No arquivo index.html, os principais controles ficam em constantes:

- TOP_MARGIN_MM: margem superior
- STANDARD_FRAMING.eyeRatio: zoom relativo do rosto
- STANDARD_FRAMING.bodyPercentile: ancoragem de corpo
- eyeLineRatio (em getSettings): altura da linha dos olhos no quadro

## Deploy

### Opção 1: GitHub Pages

1. Suba o projeto para um repositório no GitHub.
2. Vá em Settings > Pages.
3. Source: Deploy from a branch.
4. Branch: main / root.
5. Aguarde o link público.

### Opção 2: Netlify Drop

1. Acesse app.netlify.com/drop.
2. Arraste a pasta do projeto.
3. Use o link gerado.

## Limitações conhecidas

- Fotos em grupo podem falhar por restrição de uma face.
- Qualidade do recorte depende da qualidade da imagem de origem.
- Modelos podem demorar na primeira carga.

## Boas práticas para melhores resultados

- Use fotos com boa iluminação frontal.
- Evite fotos muito comprimidas/baixa resolução.
- Prefira fundo original limpo quando possível.
- Revise o alinhamento dos olhos antes da exportação final.

## Privacidade

As imagens são processadas localmente no navegador do usuário.
