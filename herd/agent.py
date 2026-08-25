"""Phase 2: pick the right local model for a task and stream the answer.

Selection reuses the Phase 1 fit math — a model is only a candidate if this machine
can actually run it, and only a choice if it is already pulled.
"""
from __future__ import annotations

import re
import time
from typing import Iterator

from . import backends, calibration, ollama, registry
from .errors import HerdError
from .hardware import detect, Hardware
from .recommender import recommend
from .vram import Fit

SYSTEM_PROMPTS = {
    "code": "You are a precise coding assistant. Give working code first, then a short "
            "explanation. Prefer the standard library. No filler.",
    "fast": "Answer in one or two sentences. No preamble.",
    "reasoning": "Think the problem through step by step, then state your conclusion clearly.",
    "chat": "You are a helpful, direct assistant. Be concise and concrete.",
}

# Cheap keyword classifier. Phase 3 replaces this with the router agent.
TASK_PATTERNS = [
    ("code", r"\b(code|function|refactor|bug|compile|api|regex|sql|script|class|"
             r"python|go|rust|java|typescript|javascript|c\+\+|implement|debug|stack trace)\b"),
    ("reasoning", r"\b(why|analyz|reason|prove|compare|trade[- ]?off|design|architect|"
                  r"evaluate|implications|strategy)\b"),
    ("fast", r"\b(what is|define|list|name the|convert|translate|tl;?dr|summar)\b"),
]


def classify(task: str) -> str:
    t = task.lower()
    for kind, pattern in TASK_PATTERNS:
        if re.search(pattern, t):
            return kind
    return "chat"


def installed_tags(backend: str = "ollama") -> dict[str, str]:
    """Map both a model's full id and its base name to the id the backend accepts."""
    index = {}
    for model_id in backends.get(backend).list_models():
        if model_id:
            index[model_id] = model_id
            index.setdefault(model_id.split(":")[0], model_id)
    return index


def candidates(hw: Hardware, task_type: str, context: int | None = None,
               backend: str = "ollama") -> list[Fit]:
    """Runnable models for this task on this backend, best first."""
    from .vram import all_fits
    usable = registry.model_defaults().get("usable_tok_s", 10)
    fits = [f for f in all_fits(hw, context, backend=backend)
            if f.tier != "no" and f.tokens_per_sec >= usable]
    matching = [f for f in fits if task_type in f.use_cases] or fits
    # Best quality that still runs; ties break toward the faster one.
    best: dict[str, Fit] = {}
    for f in matching:
        cur = best.get(f.model)
        if cur is None or (f.quality, f.tokens_per_sec) > (cur.quality, cur.tokens_per_sec):
            best[f.model] = f
    return sorted(best.values(), key=lambda f: (-f.quality, -f.tokens_per_sec))


def select(task: str, task_type: str | None = None, model: str | None = None,
           context: int | None = None, backend: str | None = None) -> dict:
    """Decide what to run. Raises HerdError if nothing usable is present."""
    hw = detect()
    backend = backend or backends.first_available()
    if not backend:
        raise HerdError(
            "No inference backend is available on this machine.",
            "Install one: Ollama (https://ollama.com/download), llama-cpp-python, "
            "or transformers. `herd backends` shows the full list.")
    spec = backends.spec(backend)
    have = installed_tags(backend)
    task_type = task_type or classify(task)

    if model:
        tag = have.get(model) or have.get(model.split(":")[0])
        if not tag:
            raise HerdError(
                f"{model!r} is not available on the {spec.label} backend.",
                _fetch_hint(backend, model) + (
                    f"   (present: {', '.join(sorted(set(have.values()))[:4])})"
                    if have else "   (nothing installed yet)"))
        reg = next((m for m in registry.models()
                    if m.get(spec.ref_key, "").split(":")[0] == tag.split(":")[0]), None)
        return {"tag": tag, "model": reg["name"] if reg else tag, "task_type": task_type,
                "reason": "you asked for it", "fit": None, "backend": backend,
                "hardware": hw.to_dict()}

    ranked = candidates(hw, task_type, context, backend)
    if not have:
        top = ranked[0] if ranked else None
        raise HerdError(
            f"No models are available on the {spec.label} backend yet.",
            f"Start with the best fit for this machine: {_fetch_hint(backend, top.model_ref)}"
            if top else "Nothing in the registry fits this machine — try a smaller model.")

    for i, fit in enumerate(ranked):
        tag = have.get(fit.model_ref) or have.get(fit.model_ref.split(":")[0])
        if not tag:
            continue
        why = f"best {task_type} model that fits — {fit.total_gb:.1f}GB, ~{fit.tokens_per_sec:.0f} tok/s"
        if fit.tier == "offload":
            why += f", {fit.gpu_layers}/{fit.total_layers} layers on GPU"
        if i > 0:
            why += f" (top pick {ranked[0].model} isn't pulled)"
        return {"tag": tag, "model": fit.model, "task_type": task_type, "reason": why,
                "fit": fit.to_dict(), "backend": backend, "hardware": hw.to_dict()}

    # Something is installed, but nothing the registry knows how to size.
    tag = sorted(set(have.values()))[0]
    return {"tag": tag, "model": tag, "task_type": task_type, "fit": None, "backend": backend,
            "reason": "no registry model is both runnable and present — using what's installed",
            "hardware": hw.to_dict()}


def _fetch_hint(backend: str, model_ref: str) -> str:
    """The command that would make this model available on this backend."""
    return {"ollama": f"ollama pull {model_ref}",
            "hf": f"huggingface-cli download {model_ref}",
            "llamacpp": f"huggingface-cli download {model_ref} --include '*Q4_K_M.gguf'",
            "openai": f"restart the server with it: vllm serve {model_ref}",
            }.get(backend, f"make {model_ref} available to the {backend} backend")


def run(task: str, task_type: str | None = None, model: str | None = None,
        file: str | None = None, context: int | None = None,
        keep_alive: str | None = None, backend: str | None = None) -> Iterator[dict]:
    """Yield NDJSON-shaped events: selection, then tokens, then stats."""
    prompt = task
    if file:
        try:
            content = open(file, encoding="utf-8", errors="replace").read()
        except OSError as e:
            raise ollama.OllamaError(f"Could not read {file}: {e.strerror}")
        prompt = f"{task}\n\n--- {file} ---\n{content}"

    choice = select(task, task_type, model, context)
    yield {"event": "selection", **choice}

    options = {}
    if context:
        options["num_ctx"] = context

    started = time.perf_counter()
    first_token_at = None
    tokens = 0
    impl = backends.get(choice["backend"])
    for chunk in impl.generate(choice["tag"], prompt,
                               SYSTEM_PROMPTS.get(choice["task_type"]), options):
        text = chunk.get("response", "")
        if text:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            tokens += 1
            yield {"event": "token", "text": text}
        if chunk.get("done"):
            eval_count = chunk.get("eval_count", tokens)
            eval_ns = chunk.get("eval_duration") or 0
            yield {"event": "done", "stats": {
                "tokens": eval_count,
                "tokens_per_sec": round(eval_count / (eval_ns / 1e9), 1) if eval_ns else None,
                "prompt_tokens": chunk.get("prompt_eval_count"),
                "load_ms": round((chunk.get("load_duration") or 0) / 1e6),
                "ttft_ms": round((first_token_at - started) * 1000) if first_token_at else None,
                "total_ms": round((time.perf_counter() - started) * 1000),
            }}


BENCH_PROMPT = "Write a numbered list of twelve short facts about the ocean."


def _empty_card(hw):
    """A copy of the machine with no models resident, for eligibility decisions."""
    from dataclasses import replace
    return replace(hw, gpus=[replace(g, vram_available_gb=g.vram_total_gb) for g in hw.gpus])


def _measure(tag: str, tokens: int) -> dict | None:
    """One timed generation. Returns None if the model never became fully resident."""
    measured = None
    for chunk in ollama.generate(tag, BENCH_PROMPT,
                                 options={"temperature": 0, "num_predict": tokens},
                                 keep_alive="5m"):
        if chunk.get("done"):
            count, ns = chunk.get("eval_count", 0), chunk.get("eval_duration") or 0
            measured = count / (ns / 1e9) if ns else None
    if not measured:
        return None
    for m in ollama.loaded():
        if (m.get("name") or m.get("model")) == tag:
            vram = (m.get("size_vram") or 0) / (1024 ** 3)
            full = bool(m.get("size_vram") and m["size_vram"] >= (m.get("size") or 0) * 0.98)
            return {"model": tag, "tok_s": round(measured, 2), "resident_gb": round(vram, 3),
                    "fully_on_gpu": full}
    return None


def bench(model: str | None = None, tokens: int = 128) -> dict:
    """Measure real decode speed and fit this machine's speed model.

    One model can only solve for the fixed per-token cost against an assumed bandwidth.
    Two differently sized models solve for both, and — more usefully — reveal when the
    machine does not behave like bandwidth-plus-a-constant at all. In that case nothing
    is stored: a confident wrong number is worse than an honest "unknown".
    """
    hw = detect()
    have = installed_tags()
    if not have:
        raise HerdError("No models installed to benchmark with.",
                        "ollama pull qwen2.5:0.5b   (~400MB, enough to calibrate)")

    if model:
        tag = have.get(model) or have.get(model.split(":")[0])
        if not tag:
            raise HerdError(f"{model!r} is not pulled.", f"ollama pull {model}")
        targets = [tag]
    else:
        # Every installed model that fits entirely in VRAM, smallest first. Use-case
        # tags are irrelevant here — different sizes are what make the fit possible.
        # Size against an empty card: each model is measured alone, so what is resident
        # right now must not decide what is eligible.
        from .vram import all_fits
        runnable = [f for f in all_fits(_empty_card(hw)) if f.fits_vram and f.quant == "Q4"]
        targets = []
        for f in sorted(runnable, key=lambda f: f.total_gb):
            tag = have.get(f.model_ref) or have.get(f.model_ref.split(":")[0])
            if tag and tag not in targets:
                targets.append(tag)
        targets = targets[:3] or [sorted(set(have.values()))[0]]

    samples = []
    for tag in targets:
        # Measure each model on an otherwise empty card, or the second one lands
        # half on the CPU and times the PCIe bus instead.
        for other in list(ollama.loaded()):
            name = other.get("name") or other.get("model")
            if name and name != tag:
                try:
                    ollama.unload(name)
                except HerdError:
                    pass
        s = _measure(tag, tokens)
        if s:
            samples.append(s)
    if not samples:
        raise HerdError("Ollama returned no usable timing data.",
                        "Check that the model loads: ollama run " + targets[0])

    device = hw.gpu.name if hw.gpu else "cpu"
    on_gpu = [s for s in samples if s["fully_on_gpu"]] or samples
    fitted = calibration.fit([(s["resident_gb"], 1 / s["tok_s"]) for s in on_gpu])

    result = {"device": device, "samples": samples, "table_bandwidth_gbs": hw.bandwidth_gbs,
              "driver_degraded": bool(hw.driver.get("degraded")),
              "fit": {k: (round(v, 2) if isinstance(v, float) else v)
                      for k, v in fitted.items() if k != "samples"}}

    if fitted.get("ok"):
        bw = fitted["effective_bandwidth_gbs"]
        overhead = fitted["token_overhead_ms"]
        # A fit far below the card's rated bandwidth is a finding, not a calibration.
        if bw < hw.bandwidth_gbs * 0.25:
            reason = (f"measured throughput implies {bw:.0f} GB/s against a rated "
                      f"{hw.bandwidth_gbs:.0f} GB/s — too large a gap to be quantization "
                      "overhead; the GPU is throttled, shared, or not doing the work")
            result["warning"] = reason.capitalize() + ". Not stored."
            result["rejected_to"] = str(calibration.reject(device, reason, samples))
        else:
            result["token_overhead_ms"] = round(overhead, 1)
            result["effective_bandwidth_gbs"] = round(bw, 1)
            result["saved_to"] = str(calibration.save(device, overhead, samples[0], bw))
    elif len(on_gpu) < 2:
        # Single sample: solve the fixed cost against the table bandwidth, as before.
        s = on_gpu[0]
        overhead = calibration.solve(s["tok_s"], s["resident_gb"], hw.bandwidth_gbs)
        result["token_overhead_ms"] = round(overhead, 1)
        result["saved_to"] = str(calibration.save(device, overhead, s))
        result["warning"] = ("Only one model was benchmarked, so this assumes the rated "
                             "bandwidth is real. Pull a second, differently sized model and "
                             "re-run to measure both terms.")
    else:
        reason = fitted.get("reason", "the measurements are inconsistent")
        result["warning"] = (f"{reason}. Nothing was stored — estimates stay on the "
                             "rated-bandwidth model, which will be optimistic.")
        result["rejected_to"] = str(calibration.reject(device, reason, samples))
    return result
