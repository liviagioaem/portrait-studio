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

### Compatibilidade com GitHub Pages

- O app principal em `index.html` funciona normalmente no GitHub Pages (site estatico).
- O script Python `qwen_image_edit.py` nao roda no GitHub Pages; ele e apenas para uso local.
- A ponte Node (`hf-server.js`) pode ser local ou publica. Para uso por outras pessoas no GitHub Pages, hospede a ponte em um endpoint publico HTTPS.

### Endpoint publico (recomendado para GitHub Pages)

1. Suba o backend [hf-server.js](hf-server.js) em um host Node (Render, Railway, Fly.io).
1. Defina variaveis de ambiente no host.

- `HF_API_KEY` (ou `POLLINATIONS_API_KEY`)
- `HF_MODEL` (opcional)
- `PORT` (normalmente automatico)

1. Aponte o frontend para esse endpoint publico.

Sem editar codigo, use query string no link do GitHub Pages:

```text
https://SEU-USUARIO.github.io/SEU-REPO/?hfEndpoint=https://seu-backend.onrender.com&hfModel=caidas/swin2SR-classical-sr-x2-64
```

Opcional: definir variaveis globais antes do script no HTML:

```html
<script>
  window.PORTRAIT_HF_ENDPOINT = "https://seu-backend.onrender.com";
  window.PORTRAIT_HF_MODEL = "caidas/swin2SR-classical-sr-x2-64";
</script>
```

Teste rapido do backend publico:

```bash
curl https://seu-backend.onrender.com/health
```

#### Deploy rapido no Render

1. Suba este repositorio no GitHub.
1. No Render, clique em New + e escolha Blueprint.
1. Selecione o repositorio. O arquivo [render.yaml](render.yaml) sera detectado automaticamente.
1. Em Environment Variables, preencha `HF_API_KEY` com seu token Hugging Face.
1. Aguarde o deploy e copie a URL publica do servico.
1. Use essa URL no GitHub Pages via querystring:

```text
https://SEU-USUARIO.github.io/SEU-REPO/?hfEndpoint=https://SUA-URL.onrender.com&hfModel=caidas/swin2SR-classical-sr-x2-64
```

## Requisitos

- Navegador moderno (Chrome/Edge recomendados)
- Internet na primeira execução (para baixar modelos)

### Opcional: melhoria com Hugging Face

- Node.js 18+ para rodar o servidor local de ponte da API
- Token Hugging Face em `HF_API_KEY` (ou `POLLINATIONS_API_KEY`)

## Como usar

1. Abra o arquivo index.html no navegador.
2. Clique em Adicionar fotos ou arraste arquivos.
3. Aguarde o processamento automático.
4. Revise cada foto e ajuste olhos quando necessário.
5. Defina Nome e Sobrenome por foto (usados no nome do arquivo final).
6. Marque fotos aprovadas.
7. Clique em Baixar aprovadas para gerar o ZIP.

## Melhorar fotos com Hugging Face (opcional)

### Modo automatico (sem botao/UI)

Use o script Python para processar tudo em lote automaticamente, sem interacao manual:

1. Crie a pasta `input` na raiz do projeto e coloque as fotos nela.
1. Execute:

```powershell
.\.venv\Scripts\python.exe qwen_image_edit.py --mode batch
```

1. As imagens processadas serao gravadas na pasta `output` com sufixos:

- `_corporativo.png`
- `_transparente.png`

Opcional:

- `--skip-transparent` para gerar somente a versao corporativa.

Importante:

- Esse modo automatico nao existe dentro do GitHub Pages, pois depende de Python local.

1. Instale dependências:

```bash
npm install
```

1. Configure sua chave em variável de ambiente:

PowerShell:

```powershell
$env:HF_API_KEY="hf_seu_token"
```

CMD:

```cmd
set HF_API_KEY=hf_seu_token
```

Sem chave API:

- Voce pode iniciar o servidor sem `HF_API_KEY`.
- Nesse caso ele tenta acesso anonimo ao Hugging Face (pode funcionar com limite baixo, lentidao ou bloqueio por rate limit).

1. Suba o servidor local:

```bash
npm run start:hf
```

1. Se quiser usar pela interface web, abra o `index.html` e processe normalmente. A melhoria por API ocorre automaticamente quando o endpoint estiver ativo.

Observacoes:

- A chave fica apenas no servidor local (nao no navegador).
- Se o servidor ou a API cair, o app volta automaticamente para o fluxo local sem interromper o lote.

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
