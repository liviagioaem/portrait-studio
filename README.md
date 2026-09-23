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
- Modelos carregados por CDN (MediaPipe, ONNX Runtime Web) e pelo Hugging Face

## Requisitos

- Navegador moderno (Chrome/Edge recomendados)
- Internet na primeira execução, para baixar os modelos
- **Servir a página por `http://localhost`** — ver abaixo
- Cerca de 30 MB livres em cache do navegador

### Por que servir por localhost

A Cache API, que guarda os pesos do modelo entre sessões, exige contexto seguro, e
abrir o `index.html` direto do disco (`file://`) não garante isso — sem ela os 26 MB
são baixados de novo toda sessão. Servir por localhost também habilita WebGPU, que
acelera o recorte quando a máquina tem GPU; sem ele roda em CPU, o que é viável com o
MODNet.

```powershell
cd c:\Work\portrait-studio
python -m http.server 8000
```

Depois abra `http://localhost:8000/index.html`.

## Como usar

1. Abra a página (ver acima).
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

## Preenchimento IA (beta)

O frontend pode chamar uma API local de inpainting no lugar do preenchimento simples
de bordas. É controlado pela constante `PIPELINE` no `index.html`, e vem desligado.

### Como testar

1. Inicie a API Python em `ai-prototype/`:

   ```powershell
   cd c:\Work\portrait-studio\ai-prototype
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   uvicorn server:app --host 127.0.0.1 --port 8001
   ```

2. No `index.html`, ajuste `PIPELINE.generativeFill` para `true`
   (`PIPELINE.edgeFill` já vem `true` e é pré-requisito).
3. Abra o `index.html` e processe as fotos.

Se a API local nao estiver ativa, o app continua com o preenchimento simples.

## Recorte

O recorte é feito pelo **MODNet** (`Xenova/modnet`, Apache 2.0), um modelo de matting
de retrato rodando no navegador via ONNX Runtime Web. Ele produz alfa suave com
detalhe de fio de cabelo, muito acima da máscara grosseira do MediaPipe Selfie
Segmentation, que era usado antes e ficou apenas como fallback.

- Pesos: 26 MB, baixados na primeira execução e guardados na Cache API.
- Entrada: proporção preservada, lado menor em 512 px, ambos os lados múltiplos de 32.
- Normalização para `[-1,1]` — **não** é ImageNet.
- A saída já é um alfa em `[0,1]`. Não aplique sigmoide: isso achataria tudo para
  cinza uniforme.

### Por que 512 e não mais

Medido em uma foto real: rodando com lado menor em 1024, o dobro da resolução de
treino, a rede inventa manchas grandes de fundo ao redor do cabelo. Em 512 elas
desaparecem. Não aumente `shortestEdge` sem verificar o recorte de novo.

`pruneMatteIslands` ainda remove ilhas soltas de fundo mal classificado, binarizando
em 128 para que alfa fraco não conecte uma ilha ao corpo e a proteja.

### Por que não BiRefNet

O BiRefNet_lite (MIT, 224 MB) é melhor no papel, mas tem dimensão fixa em 1024×1024 e
**estoura a heap do WASM**: a inferência aborta lançando um número cru em vez de um
`Error`, o que aparecia como uma falha silenciosa e caía no fallback. Só funcionaria
com WebGPU disponível. O MODNet roda em CPU em qualquer máquina.

Sobre licença: MODNet e BiRefNet permitem uso comercial. O RMBG da BRIA, o modelo mais
citado para remoção de fundo, é restrito a uso não comercial e foi descartado.

### Identificação de ser humano

São dois modelos com papéis distintos, e isso é proposital:

- O **FaceMesh** detecta o rosto. É ele que prova que existe um ser humano na foto, e
  o processamento automático exige exatamente uma face.
- O **MODNet** faz o recorte, mas separa o objeto que lê como sujeito do retrato —
  ele recortaria um cachorro com a mesma competência.

`validateHumanMatte` amarra os dois: verifica que o alfa preservou a região do rosto
detectado. Se o recorte tiver pegado outra coisa, a foto falha com mensagem explícita
em vez de exportar um PNG errado.

## Remoção de elementos estranhos

Fotos vindas de serviços de foto 3x4 costumam trazer réguas, linhas-guia, cotas
("30mm") e marcas d'água impressas na imagem.

> **As duas passagens estão desligadas.** Com o MODNet separando a pessoa, régua e
> marca d'água não entram no recorte, e estas heurísticas só arriscam comer detalhe de
> cabelo. Devem ser apagadas assim que o recorte novo for validado.

São duas passagens independentes, ligadas na constante `PIPELINE` do `index.html`:

1. **`strayRemoval`** (desligada) — uma abertura morfológica
   identifica filamentos mais finos que o corpo, e só apaga os que também são
   **pequenos**. É o duplo critério que importa: uma régua é fina e pequena; cabelo é
   fino mas faz parte de um componente grande, então sobrevive. Sem isso, uma linha
   sobrando na base deslocaria o enquadramento inteiro.
2. **`overlayRemoval`** (desligada) — regiões pequenas e finas cuja cor destoa muito
   da cor local do tecido são cobertas com a própria cor do tecido. É heurística e
   pode atingir gola, costura ou estampa; ligue com cautela. Quando age, a foto é
   marcada para revisão manual.

A região do rosto e acima dela nunca é tocada por nenhuma das duas.

Os limiares ficam na constante `FOREIGN_CLEANUP`, no `index.html`.

## Precisão e qualidade de imagem

O pipeline foi ajustado para preservar o máximo de detalhe do original:

- Orientação EXIF respeitada na importação.
- Redução de escala progressiva (em passos de 2x) na importação e na renderização
  final, evitando o serrilhado do reescalonamento direto do canvas.
- Limpeza de halo: pixels semitransparentes da borda do recorte são repintados com a
  cor dos vizinhos opacos, removendo o resíduo do fundo antigo no cabelo.
- O preenchimento de bordas não desfoca mais a área da foto original; só o anel
  estendido é suavizado.

A restauração (denoise leve + nitidez sutil) roda na resolução de origem, de
propósito: o denoise é uma janela 3×3 sobre a grade de pixels, então aplicá-lo depois
da redução para 1181px o deixaria cerca de duas vezes mais forte sobre o rosto.
A distância interocular vem do ponto médio dos cantos dos olhos — `STANDARD_FRAMING.eyeRatio`
está calibrado para essa medida e não deve ser trocada sem recalibrar o zoom.

## Boas práticas para melhores resultados

- Use fotos com boa iluminação frontal.
- Evite fotos muito comprimidas/baixa resolução.
- Prefira fundo original limpo quando possível.
- Revise o alinhamento dos olhos antes da exportação final.

## Privacidade

As imagens são processadas localmente no navegador do usuário.
