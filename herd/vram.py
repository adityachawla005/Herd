"""The VRAM math engine: what fits, how it fits, how fast it runs.

Named vram.py rather than math.py so nothing in the package ever shadows stdlib math.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

from . import backends, registry
from .calibration import bandwidth_gbs as measured_bandwidth, overhead_ms
from .hardware import Hardware

GB = 1024 ** 3

QUANT_BYTES = {"Q4": 0.5, "Q5": 0.625, "Q8": 1.0, "FP16": 2.0}
# Quality cost vs FP16, rough perplexity-delta consensus from llama.cpp benchmarks.
QUANT_QUALITY_LOSS = {"Q4": 0.03, "Q5": 0.012, "Q8": 0.002, "FP16": 0.0}
KV_BYTES = {"fp16": 2.0, "q8": 1.0, "q4": 0.5}
RUNTIME_OVERHEAD_GB = 0.5     # CUDA context + kernels + scratch (the Ollama figure)


@dataclass
class Fit:
    model: str
    quant: str
    params_b: float
    arch: str
    context: int
    weight_gb: float
    kv_gb: float
    overhead_gb: float
    total_gb: float
    active_gb: float          # bytes actually read per token (MoE < total)
    fits_vram: bool
    fits_offload: bool
    tokens_per_sec: float
    gpu_layers: int
    total_layers: int
    tier: str                 # great | offload | no
    quality: float
    backend: str = "ollama"
    model_ref: str = ""
    use_cases: list[str] = field(default_factory=list)
    ollama: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def fully_offloaded(self) -> bool:
        return self.gpu_layers >= self.total_layers

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("weight_gb", "kv_gb", "total_gb", "active_gb"):
            d[k] = round(d[k], 2)
        d["tokens_per_sec"] = round(d["tokens_per_sec"], 1)
        return d


def kv_cache_gb(model: dict, context: int, kv_bytes: float, batch: int = 1) -> float:
    """2 (K and V) * layers * kv_heads * head_dim * context * batch * bytes.

    Uses n_kv_heads, not n_heads: with grouped-query attention (Llama-3, Qwen-2.5,
    Mistral) the KV cache is 4-8x smaller than the naive head count suggests.
    """
    return (2 * model["layers"] * model.get("n_kv_heads", model["n_heads"])
            * model["head_dim"] * context * batch * kv_bytes) / GB


def estimate(model: dict, quant: str, hw: Hardware, context: int | None = None,
             batch: int = 1, kv_quant: str = "fp16", backend: str = "ollama") -> Fit:
    """Memory and speed for one (model, quant, backend) on this machine.

    The backend matters: "4-bit" costs different amounts under GGUF, bitsandbytes NF4
    and AWQ, and a torch CUDA context is heavier than llama.cpp's.
    """
    if quant not in QUANT_BYTES:
        raise ValueError(f"Unknown quant {quant!r}; expected one of {list(QUANT_BYTES)}")
    spec = backends.spec(backend)
    defaults = registry.model_defaults()
    context = context or defaults.get("context_length", 4096)
    bpp = QUANT_BYTES[quant] * spec.scale(quant)
    overhead = spec.runtime_overhead_gb

    weight_gb = model["params_b"] * 1e9 * bpp / GB
    kv_gb = kv_cache_gb(model, context, KV_BYTES[kv_quant], batch)
    total_gb = weight_gb + kv_gb + overhead

    notes = []
    if model.get("note"):
        notes.append(model["note"])

    # MoE: every expert must be resident, but only `experts_per_token` are read per
    # token — so it costs like a 47B model and runs like a 13B one.
    active_ratio = model.get("active_params_b", model["params_b"]) / model["params_b"]
    active_gb = weight_gb * active_ratio + kv_gb + overhead
    if model["arch"] == "moe":
        notes.append(
            f"MoE: {model['params_b']:.0f}B total weights must all be resident "
            f"({total_gb:.1f}GB), but only {model.get('active_params_b'):.0f}B are read per "
            f"token — speed of a {model.get('active_params_b'):.0f}B model at the memory cost of "
            f"a {model['params_b']:.0f}B one. Offloading idle experts to RAM costs less "
            "than offloading dense layers.")

    if context > model.get("max_context", context):
        notes.append(f"Asked for {context} ctx but the model was trained to "
                     f"{model['max_context']}; quality degrades past that without RoPE scaling.")

    vram = hw.vram_available_gb
    ram = hw.ram_available_gb
    fits_vram = total_gb <= vram
    fits_offload = total_gb <= vram + ram * 0.5

    layers = model["layers"]
    gpu_layers = _layers_on_gpu(weight_gb, layers, vram, kv_gb, overhead)
    tok_s = _speed(hw, weight_gb, active_ratio, kv_gb, gpu_layers, layers, overhead)
    if not spec.can_evict:
        notes.append(f"{spec.label} loads its model at server start and holds it — the "
                     "scheduler can measure it but cannot evict it.")

    tier = "great" if fits_vram else ("offload" if fits_offload else "no")
    return Fit(
        model=model["name"], quant=quant, params_b=model["params_b"], arch=model["arch"],
        context=context, weight_gb=weight_gb, kv_gb=kv_gb, overhead_gb=overhead,
        total_gb=total_gb, active_gb=active_gb, fits_vram=fits_vram, fits_offload=fits_offload,
        tokens_per_sec=tok_s, gpu_layers=gpu_layers, total_layers=layers, tier=tier,
        quality=model.get("quality", 5.0) * (1 - QUANT_QUALITY_LOSS[quant] * 4),
        use_cases=model.get("use_cases", []), ollama=model.get("ollama", ""),
        backend=backend, model_ref=model.get(spec.ref_key, ""), notes=notes,
    )


def _layers_on_gpu(weight_gb: float, layers: int, vram: float, kv_gb: float,
                   overhead: float = RUNTIME_OVERHEAD_GB) -> int:
    """How many transformer blocks fit — this is llama.cpp's -ngl / Ollama's num_gpu."""
    if vram <= 0:
        return 0
    budget = vram - overhead - kv_gb
    if budget <= 0:
        return 0
    per_layer = weight_gb / layers
    return max(0, min(layers, int(budget / per_layer)))


def _speed(hw: Hardware, weight_gb: float, active_ratio: float, kv_gb: float,
           gpu_layers: int, layers: int, overhead: float = RUNTIME_OVERHEAD_GB) -> float:
    """1 / (bytes read per token / bandwidth + fixed per-token overhead).

    Two buses when part of the model lives in RAM — a single layer streamed over
    PCIe costs more than all the GPU-resident ones combined. The fixed term is what
    `herd bench` measures; without it, small models look several times faster than
    they run.
    """
    g = registry.gpus()
    ram_bw = g["cpu_ram_bandwidth_gbs"]
    device = hw.gpu.name if hw.gpu else "cpu"
    fixed = overhead_ms(device)[0] / 1000
    if not hw.gpus:
        read = weight_gb * active_ratio + kv_gb
        return 1 / (read / ram_bw + fixed) if read else 0.0

    gpu_bw = measured_bandwidth(device) or hw.bandwidth_gbs
    frac = gpu_layers / layers if layers else 0
    on_gpu = weight_gb * active_ratio * frac + (kv_gb if frac else 0) + overhead * frac
    on_cpu = weight_gb * active_ratio * (1 - frac) + (0 if frac else kv_gb)
    # Unified memory has no PCIe hop; discrete GPUs stream offloaded weights over it.
    cpu_bw = ram_bw if (hw.gpu and hw.gpu.unified) else min(ram_bw, g["pcie_gbs"])
    seconds = on_gpu / gpu_bw + (on_cpu / cpu_bw if on_cpu else 0) + fixed
    return 1 / seconds if seconds > 0 else 0.0


def max_context(model: dict, quant: str, hw: Hardware, kv_quant: str = "fp16",
                backend: str = "ollama") -> int:
    """Largest power-of-two context that still fits entirely in VRAM. 0 = weights alone don't."""
    spec = backends.spec(backend)
    weight_gb = model["params_b"] * 1e9 * QUANT_BYTES[quant] * spec.scale(quant) / GB
    budget = hw.vram_available_gb - weight_gb - spec.runtime_overhead_gb
    if budget <= 0:
        return 0
    per_token = kv_cache_gb(model, 1, KV_BYTES[kv_quant])
    ctx = int(budget / per_token) if per_token else 0
    cap = model.get("max_context", 8192)
    best = 0
    n = 512
    while n <= min(ctx, cap):
        best, n = n, n * 2
    return best


def all_fits(hw: Hardware, context: int | None = None, kv_quant: str = "fp16",
             batch: int = 1, backend: str = "ollama") -> list[Fit]:
    """Every (model, quant) this backend can actually serve.

    A model with no reference for the backend is skipped — Ollama can't run a repo it
    has no tag for, and bitsandbytes has no Q5.
    """
    spec = backends.spec(backend)
    out = []
    for m in registry.models():
        if not m.get(spec.ref_key):
            continue
        for q in m.get("quants", list(QUANT_BYTES)):
            if q in QUANT_BYTES and spec.supports(q):
                out.append(estimate(m, q, hw, context, batch, kv_quant, backend))
    return out
