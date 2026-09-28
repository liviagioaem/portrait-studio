// Resolve base da API com prioridade:
// 1) localStorage.portrait_api_base
// 2) window.PORTRAIT_API_BASE (se ja definido antes deste script)
// 3) localhost -> http://localhost:8000
// 4) producao -> mesmo host em /api (via proxy/rewrite)
(function resolvePortraitApiBase() {
  const fromStorage = localStorage.getItem("portrait_api_base");
  if (fromStorage && fromStorage.trim()) {
    window.PORTRAIT_API_BASE = fromStorage.trim().replace(/\/$/, "");
    return;
  }

  if (window.PORTRAIT_API_BASE && String(window.PORTRAIT_API_BASE).trim()) {
    window.PORTRAIT_API_BASE = String(window.PORTRAIT_API_BASE)
      .trim()
      .replace(/\/$/, "");
    return;
  }

  const host = window.location.hostname;
  const isLocal = host === "localhost" || host === "127.0.0.1";

  window.PORTRAIT_API_BASE = isLocal
    ? "http://localhost:8000"
    : `${window.location.origin}/api`;
})();
