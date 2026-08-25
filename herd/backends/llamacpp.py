"""llama.cpp in-process, via llama-cpp-python. GGUF without an Ollama daemon."""
from __future__ import annotations

import importlib.util
import time
from typing import Iterator

from .base import Backend, BackendSpec, Unavailable

SPEC = BackendSpec(
    id="llamacpp",
    label="llama.cpp",
    ref_key="gguf_repo",
    install="uv pip install llama-cpp-python   # build with -DGGML_CUDA=on for GPU",
    runtime_overhead_gb=0.4,          # leaner than Ollama: no daemon, no model manager
    can_evict=True,
    notes="Direct control over -ngl, KV cache type and tensor placement. GGUF only.",
)


class LlamaCppBackend(Backend):
    spec = SPEC

    def __init__(self, n_gpu_layers: int = -1, n_ctx: int = 4096):
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self._loaded: dict[str, tuple] = {}

    def available(self) -> tuple[bool, str]:
        if not importlib.util.find_spec("llama_cpp"):
            return False, f"llama-cpp-python is not installed — {self.spec.install}"
        try:
            import llama_cpp
            return True, f"llama-cpp-python {getattr(llama_cpp, '__version__', '?')}"
        except Exception as e:                  # a broken CUDA build imports and then dies
            return False, f"llama_cpp fails to import: {str(e).splitlines()[0][:80]}"

    def _check(self):
        ok, why = self.available()
        if not ok:
            raise Unavailable(f"llama.cpp backend unavailable: {why}", self.spec.install)

    def list_models(self) -> list[str]:
        if not importlib.util.find_spec("huggingface_hub"):
            return []
        try:
            from huggingface_hub import scan_cache_dir
            return sorted({r.repo_id for r in scan_cache_dir().repos
                           if r.repo_type == "model"
                           and any(f.file_name.endswith(".gguf")
                                   for rev in r.revisions for f in rev.files)})
        except Exception:
            return []

    def resident(self) -> list[dict]:
        return [{"id": ref, "size_gb": size} for ref, (_, size) in self._loaded.items()]

    def ensure(self, model_id: str) -> None:
        self._check()
        if "/" in model_id and ":" not in model_id:
            raise Unavailable(
                f"{model_id!r} needs a GGUF filename: 'repo:filename.gguf'.",
                "Pick a quant from the repo's file list, e.g. "
                "'bartowski/Llama-3-8B-Instruct-GGUF:*Q4_K_M.gguf'")

    def load(self, model_id: str) -> None:
        """model_id is a local .gguf path, or 'repo_id:filename-pattern'."""
        self._check()
        if model_id in self._loaded:
            return
        from llama_cpp import Llama
        common = dict(n_gpu_layers=self.n_gpu_layers, n_ctx=self.n_ctx, verbose=False)
        if ":" in model_id and not model_id.endswith(".gguf"):
            repo, filename = model_id.split(":", 1)
            llm = Llama.from_pretrained(repo_id=repo, filename=filename, **common)
        else:
            llm = Llama(model_path=model_id, **common)
        # llama.cpp reports its own allocation; fall back to the file on disk.
        size = getattr(llm, "_model", None)
        size_gb = (getattr(size, "size", 0) or 0) / (1024 ** 3)
        self._loaded[model_id] = (llm, size_gb)

    def unload(self, model_id: str) -> None:
        self._loaded.pop(model_id, None)

    def generate(self, model_id: str, prompt: str, system: str | None = None,
                 options: dict | None = None) -> Iterator[dict]:
        if model_id not in self._loaded:
            self.load(model_id)
        llm, _ = self._loaded[model_id]
        opts = options or {}
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})

        started = time.perf_counter()
        count = 0
        stream = llm.create_chat_completion(
            messages=messages, stream=True,
            max_tokens=opts.get("num_predict", 512),
            temperature=opts.get("temperature", 0.7))
        for chunk in stream:
            piece = chunk["choices"][0].get("delta", {}).get("content", "")
            if piece:
                count += 1
                yield {"response": piece}
        elapsed = time.perf_counter() - started
        yield {"done": True, "eval_count": count, "eval_duration": int(elapsed * 1e9)}
