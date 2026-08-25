"""Turns raw fit math into an opinionated shortlist plus a local/cloud split."""
from __future__ import annotations

from . import backends, calibration, registry, optimizations
from .hardware import Hardware
from .vram import Fit, all_fits

QUANT_ORDER = ["FP16", "Q8", "Q5", "Q4"]


def _best_per_model(fits: list[Fit], prefer: str = "highest") -> list[Fit]:
    """One row per model: the highest quant that still runs, or — for models that
    don't fit at all — the smallest, since 'not even at Q4' is the useful fact."""
    best: dict[str, Fit] = {}
    for f in fits:
        cur = best.get(f.model)
        if cur is None:
            best[f.model] = f
            continue
        better = (QUANT_ORDER.index(f.quant) < QUANT_ORDER.index(cur.quant)
                  if prefer == "highest" else
                  QUANT_ORDER.index(f.quant) > QUANT_ORDER.index(cur.quant))
        if better:
            best[f.model] = f
    return list(best.values())


def _cap_per_model(fits: list[Fit], n: int = 2) -> list[Fit]:
    """Keep at most n quants of any one model, best first — otherwise four quants of
    a single small model crowd every other option out of the list."""
    count: dict[str, int] = {}
    out = []
    for f in fits:
        if count.get(f.model, 0) < n:
            count[f.model] = count.get(f.model, 0) + 1
            out.append(f)
    return out


def _pick(fits: list[Fit], use_case: str, min_tok_s: float) -> Fit | None:
    cands = [f for f in fits if use_case in f.use_cases and f.tokens_per_sec >= min_tok_s]
    if not cands:
        return None
    return max(cands, key=lambda f: (round(f.quality, 1), f.tokens_per_sec))


def hybrid_plan(hw: Hardware, great: list[Fit], offload: list[Fit]) -> dict:
    d = registry.model_defaults()
    usable = d.get("usable_tok_s", 10)
    mix = d.get("workload_mix", {})
    pool = great + [f for f in offload if f.tokens_per_sec >= usable]

    # Prefer a genuinely fast router, but a slow one still beats paying for the cloud
    # to classify a task — so fall back rather than leaving the slot empty.
    router = (_pick(pool, "fast", usable * 3) or _pick(pool, "chat", usable * 3)
              or _pick(pool, "fast", usable) or _pick(pool, "chat", usable))
    workhorse = _pick(pool, "chat", usable)
    coder = _pick(pool, "code", usable)
    reasoner = _pick(pool, "reasoning", usable)

    assignments, covered = [], 0.0
    for use_case, share in mix.items():
        f = {"chat": workhorse, "code": coder, "fast": router, "reasoning": reasoner}.get(use_case)
        # Reasoning on a small local model is the one that genuinely wants a bigger brain.
        if f and (use_case != "reasoning" or f.quality >= 7.5):
            covered += share
            assignments.append({"use_case": use_case, "share": share, "model": f.model,
                                "quant": f.quant, "ollama": f.ollama,
                                "tokens_per_sec": round(f.tokens_per_sec, 1)})
        else:
            assignments.append({"use_case": use_case, "share": share, "model": None})

    per_call = (d["typical_call_tokens_in"] * d["cloud_usd_per_mtok_in"]
                + d["typical_call_tokens_out"] * d["cloud_usd_per_mtok_out"]) / 1e6
    ctx_ceiling = max((f.context for f in great), default=0)

    to_cloud = [a["use_case"] for a in assignments if not a["model"]]
    to_cloud.append(f"context beyond {ctx_ceiling} tokens" if ctx_ceiling else "everything")
    return {
        "local": [a for a in assignments if a["model"]],
        "cloud": to_cloud,
        "local_coverage": round(covered, 2),
        "savings_per_1000_calls_usd": round(per_call * 1000 * covered, 2),
        "cloud_cost_per_1000_calls_usd": round(per_call * 1000, 2),
        "router": router.model if router else None,
        "workhorse": workhorse.model if workhorse else None,
    }


def recommend(hw: Hardware, context: int | None = None, kv_quant: str = "fp16",
              limit: int = 8, backend: str | None = None) -> dict:
    if hw.gpus and hw.vram_total_gb <= 0:
        raise RuntimeError(
            f"{hw.gpus[0].name} was found but reports 0GB of VRAM. The driver is probably "
            "half-installed — check `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD) directly.")

    backend = backend or backends.first_available() or backends.DEFAULT
    fits = all_fits(hw, context, kv_quant, backend=backend)
    great = _cap_per_model(sorted([f for f in fits if f.tier == "great"],
                                  key=lambda f: (-f.quality, -f.tokens_per_sec)))
    offload = sorted(_best_per_model([f for f in fits if f.tier == "offload"]),
                     key=lambda f: -f.tokens_per_sec)
    wont = sorted(_best_per_model([f for f in fits if f.tier == "no"], prefer="lowest"),
                  key=lambda f: f.total_gb)

    # Models already listed as fitting don't need a slower duplicate row below.
    fitting = {f.model for f in great}
    offload = [f for f in offload if f.model not in fitting]
    wont = [f for f in wont if f.model not in fitting and f.model not in {o.model for o in offload}]

    top = great[0] if great else (offload[0] if offload else None)
    tips = optimizations.advise(hw, top, backend=backend)
    cal = calibration.status(hw.gpu.name if hw.gpu else "cpu")

    return {
        "hardware": hw.to_dict(),
        "context": context or registry.model_defaults().get("context_length", 4096),
        "kv_quant": kv_quant,
        "backend": backend,
        "backends": backends.survey(),
        "runs_great": [f.to_dict() for f in great[:limit]],
        "runs_offload": [f.to_dict() for f in offload[:limit]],
        "wont_fit": [f.to_dict() for f in wont[:limit]],
        "hybrid": hybrid_plan(hw, great, offload),
        "optimizations": [t.to_dict() for t in tips],
        "calibration": cal,
        "notes": hw.notes,
    }
