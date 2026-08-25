"""Thin client for the Ollama REST API. No SDK — it's five endpoints."""
from __future__ import annotations

import json
import os
from typing import Iterator

import httpx

from .errors import HerdError

DEFAULT_HOST = "http://127.0.0.1:11434"
INSTALL_HINT = ("Install it from https://ollama.com/download, then start it with "
                "`ollama serve` (it usually runs as a service already).")


class OllamaError(HerdError):
    """Ollama is not running, has no such model, or refused the request."""


def host() -> str:
    h = os.environ.get("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")
    return h if "://" in h else f"http://{h}"


def _request(method: str, path: str, **kw) -> httpx.Response:
    try:
        r = httpx.request(method, host() + path, timeout=kw.pop("timeout", 30), **kw)
        r.raise_for_status()
        return r
    except httpx.ConnectError:
        raise OllamaError(f"Ollama is not reachable at {host()}.", INSTALL_HINT)
    except httpx.TimeoutException:
        raise OllamaError(f"Ollama timed out at {host()}{path}.",
                          "It may be loading a model off disk — retry in a moment.")
    except httpx.HTTPStatusError as e:
        detail = e.response.text.strip()[:200]
        raise OllamaError(f"Ollama returned {e.response.status_code} for {path}: {detail}")


def version() -> str:
    return _request("GET", "/api/version", timeout=5).json().get("version", "unknown")


def installed() -> list[dict]:
    """Models pulled onto this machine."""
    return _request("GET", "/api/tags", timeout=10).json().get("models", [])


def loaded() -> list[dict]:
    """Models currently resident in VRAM, with their size and expiry."""
    return _request("GET", "/api/ps", timeout=10).json().get("models", [])


def generate(model: str, prompt: str, system: str | None = None,
             options: dict | None = None, keep_alive: str | None = None) -> Iterator[dict]:
    """Stream /api/generate. Yields the raw chunks; the last one carries the timings."""
    body = {"model": model, "prompt": prompt, "stream": True}
    if system:
        body["system"] = system
    if options:
        body["options"] = options
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
    try:
        with httpx.stream("POST", host() + "/api/generate", json=body, timeout=None) as r:
            if r.status_code == 404:
                r.read()
                raise OllamaError(f"Ollama has no model named {model!r}.",
                                  f"Pull it first: ollama pull {model}")
            r.raise_for_status()
            for line in r.iter_lines():
                if line.strip():
                    yield json.loads(line)
    except httpx.ConnectError:
        raise OllamaError(f"Ollama is not reachable at {host()}.", INSTALL_HINT)
    except httpx.HTTPStatusError as e:
        raise OllamaError(f"Ollama returned {e.response.status_code} while generating.")


def unload(model: str) -> None:
    """Evict a model from VRAM by asking for zero tokens with keep_alive=0."""
    _request("POST", "/api/generate",
             json={"model": model, "prompt": "", "keep_alive": 0, "stream": False}, timeout=60)
