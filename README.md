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
