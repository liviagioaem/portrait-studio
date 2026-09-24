const express = require("express");
const multer = require("multer");

require("dotenv").config();

const app = express();
const upload = multer({ limits: { fileSize: 25 * 1024 * 1024 } });

const PORT = Number(process.env.PORT || 8787);
const HF_API_KEY =
  process.env.HF_API_KEY ||
  process.env.HF_TOKEN ||
  process.env.POLLINATIONS_API_KEY ||
  "";
const HF_MODEL_DEFAULT =
  process.env.HF_MODEL || "caidas/swin2SR-classical-sr-x2-64";
const HF_INFERENCE_MODE = (process.env.HF_INFERENCE_MODE || "auto").toLowerCase();
const HF_LOCAL_ENDPOINT =
  process.env.HF_LOCAL_ENDPOINT || "http://127.0.0.1:7860";
const HF_API_BASES = (
  process.env.HF_API_BASES ||
  "https://api-inference.huggingface.co,https://router.huggingface.co/hf-inference"
)
  .split(",")
  .map((v) => v.trim().replace(/\/+$/, ""))
  .filter(Boolean);

app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type,Authorization");
  if (req.method === "OPTIONS") return res.status(204).end();
  next();
});

app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    model: HF_MODEL_DEFAULT,
    mode: HF_INFERENCE_MODE,
    localEndpoint: HF_LOCAL_ENDPOINT,
    hfApiBases: HF_API_BASES,
  });
});

async function callLocalInference({ imageBuffer, model, prompt }) {
  const form = new FormData();
  form.append("image", new Blob([imageBuffer], { type: "image/png" }), "input.png");
  form.append("model", model);
  if (prompt) form.append("prompt", prompt);

  const response = await fetch(`${HF_LOCAL_ENDPOINT}/enhance`, {
    method: "POST",
    body: form,
  });

  const contentType = response.headers.get("content-type") || "";
  if (response.ok) {
    return {
      contentType: contentType.startsWith("image/") ? contentType : "image/png",
      data: Buffer.from(await response.arrayBuffer()),
    };
  }

  const raw = await response.text();
  throw new Error(`Local inference HTTP ${response.status}: ${raw.slice(0, 300)}`);
}

async function callHfInference({ imageBuffer, model, apiKey }) {
  const maxRetries = 4;

  let lastError = "Modelo indisponivel no momento.";

  for (const baseUrl of HF_API_BASES) {
    const endpoint = `${baseUrl}/models/${encodeURIComponent(model)}`;

    for (let attempt = 0; attempt < maxRetries; attempt += 1) {
      const headers = {
        "Content-Type": "application/octet-stream",
        Accept: "image/png",
      };
      if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers,
          body: imageBuffer,
        });

        const contentType = response.headers.get("content-type") || "";

        if (response.ok) {
          return {
            contentType: contentType.startsWith("image/")
              ? contentType
              : "image/png",
            data: Buffer.from(await response.arrayBuffer()),
          };
        }

        let errorMessage = `Hugging Face retornou HTTP ${response.status}.`;
        try {
          const body = await response.json();
          if (body?.error) errorMessage = body.error;

          // Model cold start: retry using estimated wait time when available.
          if (response.status === 503 && attempt < maxRetries - 1) {
            const estimated = Number(body?.estimated_time);
            const waitMs = Number.isFinite(estimated)
              ? Math.ceil(estimated * 1000)
              : 3000;
            await new Promise((resolve) =>
              setTimeout(resolve, Math.min(waitMs, 15000)),
            );
            continue;
          }
        } catch (_error) {
          const raw = await response.text();
          if (raw) errorMessage = raw.slice(0, 300);
        }

        lastError = `${errorMessage} (endpoint: ${baseUrl})`;
        break;
      } catch (error) {
        const cause = error?.cause?.message || error?.message || String(error);
        lastError = `fetch failed | cause: ${cause} (endpoint: ${baseUrl})`;
        break;
      }
    }
  }

  throw new Error(lastError);
}

app.post("/api/hf/enhance", upload.single("image"), async (req, res) => {
  try {
    if (!req.file?.buffer) {
      return res
        .status(400)
        .json({ error: "Envie a imagem no campo 'image'." });
    }

    const model = (req.body?.model || "").trim() || HF_MODEL_DEFAULT;
    const prompt = (req.body?.prompt || "").trim() || "";

    let enhanced;
    if (HF_INFERENCE_MODE === "local") {
      enhanced = await callLocalInference({
        imageBuffer: req.file.buffer,
        model,
        prompt,
      });
    } else if (HF_INFERENCE_MODE === "remote") {
      enhanced = await callHfInference({
        imageBuffer: req.file.buffer,
        model,
        apiKey: HF_API_KEY,
      });
    } else {
      try {
        enhanced = await callLocalInference({
          imageBuffer: req.file.buffer,
          model,
          prompt,
        });
      } catch (_localError) {
        enhanced = await callHfInference({
          imageBuffer: req.file.buffer,
          model,
          apiKey: HF_API_KEY,
        });
      }
    }

    res.setHeader("Content-Type", enhanced.contentType);
    return res.status(200).send(enhanced.data);
  } catch (error) {
    const rootCause = error?.cause?.message
      ? ` | cause: ${error.cause.message}`
      : "";
    return res
      .status(502)
      .json({
        error: (error.message || "Falha na melhoria por IA.") + rootCause,
      });
  }
});

app.listen(PORT, () => {
  console.log(`HF bridge online em http://localhost:${PORT}`);
});
