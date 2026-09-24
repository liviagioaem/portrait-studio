const model = process.env.HF_MODEL || "caidas/swin2SR-classical-sr-x2-64";
const baseUrl =
  process.env.HF_ENDPOINT || `http://localhost:${process.env.PORT || 8787}`;

async function main() {
  const pngBase64 =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WlAbW8AAAAASUVORK5CYII=";
  const imageBuffer = Buffer.from(pngBase64, "base64");

  const form = new FormData();
  form.append(
    "image",
    new Blob([imageBuffer], { type: "image/png" }),
    "tiny.png",
  );
  form.append("model", model);

  const response = await fetch(`${baseUrl}/api/hf/enhance`, {
    method: "POST",
    body: form,
  });

  const contentType = response.headers.get("content-type") || "";

  if (!response.ok) {
    const body = await response.text();
    throw new Error(
      `HF enhance failed: HTTP ${response.status} | ${body.slice(0, 300)}`,
    );
  }

  if (!contentType.startsWith("image/")) {
    const body = await response.text();
    throw new Error(
      `Expected image response, got '${contentType}' | body=${body.slice(0, 300)}`,
    );
  }

  const out = Buffer.from(await response.arrayBuffer());
  if (!out.length) {
    throw new Error("Received empty image payload from HF bridge.");
  }

  console.log(
    `Smoke test passed. content-type=${contentType} bytes=${out.length}`,
  );
}

main().catch((error) => {
  console.error(error.message || error);
  process.exit(1);
});
