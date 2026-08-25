"""FastAPI wrapper around the same functions the CLI uses. No logic lives here."""
from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from . import registry, optimizations
from .hardware import detect
from .ollama import OllamaError
from .recommender import recommend
from .vram import estimate

app = FastAPI(title="Herd", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                   allow_methods=["*"], allow_headers=["*"])

UI_DIST = Path(__file__).parent.parent / "ui" / "dist"


@app.get("/api/detect")
def api_detect():
    hw = detect()
    return hw.to_dict()


@app.get("/api/recommend")
def api_recommend(context: int | None = None, kv_quant: str = "fp16", limit: int = 12):
    try:
        return recommend(detect(), context, kv_quant, limit)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/optimize")
def api_optimize(model: str | None = None, quant: str = "Q4",
                 context: int | None = None, kv_quant: str = "fp16"):
    hw = detect()
    fit = m = None
    if model:
        m = registry.find_model(model)
        if not m:
            raise HTTPException(status_code=404, detail=f"Unknown model {model!r}")
        fit = estimate(m, quant, hw, context, kv_quant=kv_quant)
    return {"hardware": hw.to_dict(), "fit": fit.to_dict() if fit else None,
            "optimizations": [t.to_dict() for t in optimizations.advise(hw, fit, m)]}


@app.get("/api/models")
def api_models():
    return {"models": registry.models(), "defaults": registry.model_defaults()}


_orchestrator = None


def _orch():
    """One orchestrator for the process — building it probes NVML and Ollama."""
    global _orchestrator
    if _orchestrator is None:
        from .orchestrator import Orchestrator
        _orchestrator = Orchestrator()
    return _orchestrator


@app.get("/api/status")
def api_status():
    try:
        return _orch().status()
    except (OllamaError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.websocket("/ws/vram")
async def ws_vram(ws: WebSocket):
    """Push scheduler state every two seconds until the client goes away."""
    await ws.accept()
    try:
        while True:
            try:
                payload = await asyncio.to_thread(lambda: _orch().status())
            except (OllamaError, RuntimeError) as e:
                payload = {"error": str(e)}
            await ws.send_json(payload)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        return
    except RuntimeError:
        return                      # socket closed mid-send


@app.get("/api/health")
def api_health():
    return {"ok": True, "ui_built": UI_DIST.exists()}


if UI_DIST.exists():
    app.mount("/assets", StaticFiles(directory=UI_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = UI_DIST / path
        return FileResponse(f if f.is_file() else UI_DIST / "index.html")
else:
    @app.get("/")
    def not_built():
        return HTMLResponse(
            "<body style='font:16px/1.6 system-ui;max-width:38rem;margin:15vh auto;padding:2rem'>"
            "<h1>Herd</h1><p>The API is up, but the UI has not been built yet.</p>"
            "<pre style='background:#f4f4f5;padding:1rem;border-radius:.5rem'>"
            "cd ui &amp;&amp; npm install &amp;&amp; npm run build</pre>"
            "<p>Or run <code>npm run dev</code> in <code>ui/</code> for the dev server "
            "on :5173.</p><p>API: <a href='/api/recommend'>/api/recommend</a></p></body>",
            status_code=200)
