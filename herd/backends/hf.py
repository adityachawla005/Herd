"""HuggingFace Transformers, in-process.

Reaches the whole Hub, including models nobody has converted to GGUF. The price is
memory: native checkpoints are FP16/BF16, so a Llama-3-8B is ~16GB instead of the
~4.5GB it costs as Q4 GGUF. bitsandbytes brings that down, but NF4 is not free —
4-bit weights plus absmax scales land nearer 0.58 bytes/param than 0.5.

Every torch import is deferred: importing this module must stay cheap on a machine
that has no ML stack installed.
"""
from __future__ import annotations

import importlib.util
import threading
from typing import Iterator

from .base import Backend, BackendSpec, Unavailable

SPEC = BackendSpec(
    id="hf",
    label="HF Transformers",
    ref_key="hf_repo",
    install="uv pip install 'herd[hf]'   # torch, transformers, accelerate, bitsandbytes",
    # A torch CUDA context is heavier than llama.cpp's, and autograd buffers linger.
    runtime_overhead_gb=0.9,
    can_evict=True,
    bytes_scale={"Q4": 1.15, "Q8": 1.06},   # bitsandbytes NF4 / LLM.int8 overhead
    quants=("Q4", "Q8", "FP16"),            # no Q5 equivalent in bitsandbytes
    notes="Anything on the Hub, in its native format. Heaviest on VRAM and load time.",
)

REQUIRED = ("torch", "transformers", "accelerate")


class HFBackend(Backend):
    spec = SPEC

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._loaded: dict[str, tuple] = {}       # repo -> (model, tokenizer, size_gb)

    def available(self) -> tuple[bool, str]:
        missing = [m for m in REQUIRED if not importlib.util.find_spec(m)]
        if missing:
            return False, f"missing {', '.join(missing)} — {self.spec.install}"
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            return False, "torch is installed but sees no CUDA device"
        return True, f"torch {torch.__version__} on {self.device}"

    def _check(self):
        ok, why = self.available()
        if not ok:
            raise Unavailable(f"HF Transformers backend unavailable: {why}", self.spec.install)

    def list_models(self) -> list[str]:
        """Repos already in the local Hub cache — no network."""
        if not importlib.util.find_spec("huggingface_hub"):
            return []
        try:
            from huggingface_hub import scan_cache_dir
            return [r.repo_id for r in scan_cache_dir().repos if r.repo_type == "model"]
        except Exception:
            return []

    def resident(self) -> list[dict]:
        return [{"id": repo, "size_gb": size} for repo, (_, _, size) in self._loaded.items()]

    def ensure(self, model_id: str) -> None:
        self._check()
        if model_id in self.list_models():
            return
        from huggingface_hub import snapshot_download
        snapshot_download(model_id)

    def load(self, model_id: str, quant: str = "Q4") -> None:
        self._check()
        if model_id in self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: dict = {"device_map": self.device, "dtype": torch.float16}
        if quant in ("Q4", "Q8"):
            if not importlib.util.find_spec("bitsandbytes"):
                raise Unavailable(
                    f"{quant} needs bitsandbytes, which is not installed.",
                    "uv pip install bitsandbytes   (or load this model at FP16)")
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = (
                BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16,
                                   bnb_4bit_use_double_quant=True)
                if quant == "Q4" else BitsAndBytesConfig(load_in_8bit=True))

        before = torch.cuda.memory_allocated() if self.device == "cuda" else 0
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        tok = AutoTokenizer.from_pretrained(model_id)
        after = torch.cuda.memory_allocated() if self.device == "cuda" else 0
        self._loaded[model_id] = (model, tok, (after - before) / (1024 ** 3))

    def unload(self, model_id: str) -> None:
        if model_id not in self._loaded:
            return
        import gc
        import torch
        del self._loaded[model_id]
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def generate(self, model_id: str, prompt: str, system: str | None = None,
                 options: dict | None = None) -> Iterator[dict]:
        import time
        from transformers import TextIteratorStreamer

        if model_id not in self._loaded:
            self.load(model_id, (options or {}).get("quant", "Q4"))
        model, tok, _ = self._loaded[model_id]

        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:                       # base models have no chat template
            text = (f"{system}\n\n" if system else "") + prompt

        inputs = tok(text, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        opts = options or {}
        kwargs = dict(**inputs, streamer=streamer,
                      max_new_tokens=opts.get("num_predict", 512),
                      do_sample=opts.get("temperature", 0.7) > 0,
                      temperature=max(opts.get("temperature", 0.7), 1e-5))
        thread = threading.Thread(target=model.generate, kwargs=kwargs, daemon=True)

        started = time.perf_counter()
        thread.start()
        count = 0
        for piece in streamer:
            if piece:
                count += 1
                yield {"response": piece}
        thread.join()
        elapsed = time.perf_counter() - started
        yield {"done": True, "eval_count": count,
               "eval_duration": int(elapsed * 1e9),
               "prompt_eval_count": int(inputs["input_ids"].shape[-1])}
