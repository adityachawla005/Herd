"""Backend registry. `herd backends` shows which of these the machine can actually use."""
from __future__ import annotations

from .base import Backend, BackendError, BackendSpec, Unavailable
from .hf import HFBackend
from .llamacpp import LlamaCppBackend
from .ollama_backend import OllamaBackend
from .openai_compat import OpenAICompatBackend

# Order is preference: the easiest working option first.
BACKENDS: dict[str, type] = {
    "ollama": OllamaBackend,
    "llamacpp": LlamaCppBackend,
    "hf": HFBackend,
    "openai": OpenAICompatBackend,
}

DEFAULT = "ollama"
_cache: dict[str, Backend] = {}


def get(name: str = DEFAULT, **kw) -> Backend:
    if name not in BACKENDS:
        raise BackendError(f"Unknown backend {name!r}.",
                           f"Available: {', '.join(BACKENDS)}")
    if kw:
        return BACKENDS[name](**kw)
    if name not in _cache:
        _cache[name] = BACKENDS[name]()
    return _cache[name]


def spec(name: str) -> BackendSpec:
    return BACKENDS[name].spec


def survey() -> list[dict]:
    """Every backend with its capabilities and whether it works here."""
    out = []
    for name in BACKENDS:
        b = get(name)
        try:
            ok, why = b.available()
        except Exception as e:                    # a backend must never break the survey
            ok, why = False, f"probe failed: {str(e).splitlines()[0][:80]}"
        s = b.spec
        out.append({
            "id": s.id, "label": s.label, "available": ok, "detail": why,
            "ref_key": s.ref_key, "install": s.install, "notes": s.notes,
            "can_evict": s.can_evict, "owns_card": s.owns_card,
            "reports_vram": s.reports_vram, "quants": list(s.quants),
            "runtime_overhead_gb": s.runtime_overhead_gb,
            "bytes_scale": s.bytes_scale,
            "models": b.list_models() if ok else [],
        })
    return out


def first_available() -> str | None:
    return next((b["id"] for b in survey() if b["available"]), None)
