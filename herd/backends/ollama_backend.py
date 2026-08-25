"""Ollama: a llama.cpp server with a model manager bolted on. GGUF only."""
from __future__ import annotations

from typing import Iterator

from .. import ollama
from .base import Backend, BackendSpec, Unavailable

SPEC = BackendSpec(
    id="ollama",
    label="Ollama",
    ref_key="ollama",
    install="https://ollama.com/download",
    # A CUDA context plus Ollama's own per-model scratch buffers.
    runtime_overhead_gb=0.5,
    can_evict=True,
    notes="Easiest path. Serves GGUF, manages downloads, evicts on a timer we override.",
)


class OllamaBackend(Backend):
    spec = SPEC

    def available(self) -> tuple[bool, str]:
        try:
            return True, f"Ollama {ollama.version()} at {ollama.host()}"
        except ollama.OllamaError as e:
            return False, str(e)

    def list_models(self) -> list[str]:
        try:
            return [m.get("name") or m.get("model", "") for m in ollama.installed()]
        except ollama.OllamaError:
            return []

    def resident(self) -> list[dict]:
        try:
            return [{"id": m.get("name") or m.get("model"),
                     "size_gb": (m.get("size_vram") or m.get("size") or 0) / (1024 ** 3)}
                    for m in ollama.loaded()]
        except ollama.OllamaError:
            return []

    def ensure(self, model_id: str) -> None:
        if model_id in self.list_models():
            return
        raise Unavailable(f"{model_id!r} is not pulled.", f"ollama pull {model_id}")

    def load(self, model_id: str) -> None:
        list(ollama.generate(model_id, "", keep_alive="30m"))

    def unload(self, model_id: str) -> None:
        ollama.unload(model_id)

    def generate(self, model_id: str, prompt: str, system: str | None = None,
                 options: dict | None = None) -> Iterator[dict]:
        yield from ollama.generate(model_id, prompt, system, options, keep_alive="30m")
