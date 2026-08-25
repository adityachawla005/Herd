"""Any server that speaks the OpenAI API: vLLM, llama.cpp's llama-server, LM Studio, TGI.

One implementation covers all of them, which is why they aren't four backends.

The important difference from Ollama: these servers own their model. vLLM pre-allocates
gpu_memory_utilization (0.9 by default) of the entire card when it starts and serves one
model for its lifetime — there is no load, no evict, and its VRAM figure is the
reservation rather than the model. The scheduler needs to know that or its accounting is
fiction, hence can_evict=False and owns_card=True.
"""
from __future__ import annotations

import json
import os
import time
from typing import Iterator

import httpx

from .base import Backend, BackendSpec, Unavailable

SPEC = BackendSpec(
    id="openai",
    label="OpenAI-compatible server",
    ref_key="hf_repo",
    install="Start one: `vllm serve <repo>` · `llama-server -m model.gguf` · LM Studio",
    runtime_overhead_gb=0.0,        # the server already accounted for it
    can_evict=False,
    owns_card=True,
    reports_vram=False,
    bytes_scale={"Q4": 1.12},       # AWQ/GPTQ 4-bit carries scales and zero-points
    quants=("Q4", "FP16"),
    notes="vLLM, llama-server, LM Studio, TGI. Highest throughput; not schedulable.",
)

DEFAULT_BASE = "http://127.0.0.1:8000/v1"


class OpenAICompatBackend(Backend):
    spec = SPEC

    def __init__(self, base_url: str | None = None, api_key: str = "not-needed"):
        self.base = (base_url or os.environ.get("HERD_OPENAI_BASE") or DEFAULT_BASE).rstrip("/")
        self.api_key = api_key

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def available(self) -> tuple[bool, str]:
        try:
            r = httpx.get(f"{self.base}/models", headers=self._headers, timeout=3)
            r.raise_for_status()
            served = [m["id"] for m in r.json().get("data", [])]
            return True, f"{self.base} serving {', '.join(served) or 'nothing'}"
        except Exception:
            return False, f"no OpenAI-compatible server at {self.base}"

    def list_models(self) -> list[str]:
        try:
            r = httpx.get(f"{self.base}/models", headers=self._headers, timeout=3)
            return [m["id"] for m in r.json().get("data", [])]
        except Exception:
            return []

    def resident(self) -> list[dict]:
        """Whatever the server has is resident — it loaded it at startup, not on demand.
        size_gb is unknown from here; the caller should fall back to the fit estimate."""
        return [{"id": m, "size_gb": 0.0, "estimated": True} for m in self.list_models()]

    def ensure(self, model_id: str) -> None:
        if model_id not in self.list_models():
            raise Unavailable(
                f"{self.base} is not serving {model_id!r}.",
                f"Restart the server with it: vllm serve {model_id}")

    def load(self, model_id: str) -> None:
        self.ensure(model_id)          # nothing to do: the server loaded it at startup

    def unload(self, model_id: str) -> None:
        raise Unavailable(
            f"{self.spec.label} cannot unload {model_id!r} — the server owns its model "
            "for its lifetime.",
            "Stop the server to free the VRAM, or use Ollama for models you want scheduled.")

    def generate(self, model_id: str, prompt: str, system: str | None = None,
                 options: dict | None = None) -> Iterator[dict]:
        opts = options or {}
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        body = {"model": model_id, "messages": messages, "stream": True,
                "max_tokens": opts.get("num_predict", 512),
                "temperature": opts.get("temperature", 0.7),
                "stream_options": {"include_usage": True}}

        started = time.perf_counter()
        count = 0
        usage = {}
        try:
            with httpx.stream("POST", f"{self.base}/chat/completions", json=body,
                              headers=self._headers, timeout=None) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        piece = choice.get("delta", {}).get("content") or ""
                        if piece:
                            count += 1
                            yield {"response": piece}
        except httpx.ConnectError:
            raise Unavailable(f"Lost the connection to {self.base}.", self.spec.install)
        elapsed = time.perf_counter() - started
        yield {"done": True,
               "eval_count": usage.get("completion_tokens", count),
               "eval_duration": int(elapsed * 1e9),
               "prompt_eval_count": usage.get("prompt_tokens")}
