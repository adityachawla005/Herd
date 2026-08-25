"""What to actually DO with the hardware: runtime, quant format, memory and speed levers.

Every tip is computed from the detected machine and the chosen model, not boilerplate
advice — if a lever doesn't apply here, it isn't emitted.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from . import backends, calibration
from .hardware import Hardware
from .vram import Fit, KV_BYTES, RUNTIME_OVERHEAD_GB, kv_cache_gb, max_context
from . import registry


@dataclass
class Tip:
    id: str
    title: str
    detail: str
    impact: str = ""
    command: str = ""
    tags: tuple = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d


# --- runtime / format choice ---------------------------------------------

def runtime_advice(hw: Hardware, fit: Fit | None = None) -> list[Tip]:
    tips = []
    gpu = hw.gpu
    vendor = gpu.vendor if gpu else "none"
    partial = fit is not None and not fit.fits_vram and fit.gpu_layers > 0

    if vendor == "apple":
        tips.append(Tip(
            "runtime.mlx", "Run MLX for Apple Silicon, GGUF as the fallback",
            "MLX is built for unified memory and beats llama.cpp Metal by roughly 20-30% on "
            "decode for the same weights. Keep a GGUF copy for anything MLX has no conversion "
            "of. Ollama uses llama.cpp/Metal under the hood, so it lands in the middle.",
            impact="~+25% tok/s vs GGUF/Metal",
            command="pip install mlx-lm && mlx_lm.generate --model mlx-community/<model>-4bit",
            tags=("runtime", "apple")))
    elif vendor == "nvidia" and fit is not None and fit.fits_vram:
        tips.append(Tip(
            "runtime.vllm", "Fully resident on an NVIDIA card — vLLM beats GGUF here",
            "Once every layer is in VRAM, llama.cpp's portability stops paying for itself. "
            "vLLM with an AWQ or GPTQ 4-bit checkpoint uses paged attention and continuous "
            "batching: similar single-stream speed, several times the throughput under "
            "concurrency. Stay on GGUF/Ollama if you want one-command model management.",
            impact="2-4x throughput when >1 request is in flight",
            command="uv pip install vllm && vllm serve <hf-repo>-AWQ --max-model-len 4096",
            tags=("runtime", "nvidia", "throughput")))
    elif partial:
        tips.append(Tip(
            "runtime.llamacpp", "Partial offload — llama.cpp/GGUF is the only good option",
            "vLLM and ExLlama need the whole model in VRAM; they will refuse this. llama.cpp "
            "splits by layer across GPU and CPU, so it degrades instead of failing. Ollama "
            "wraps the same engine and picks the layer count itself.",
            impact=f"runs at all vs OOM ({fit.gpu_layers}/{fit.total_layers} layers on GPU)",
            tags=("runtime", "offload")))
    if not gpu:
        tips.append(Tip(
            "runtime.cpu", "CPU-only: llama.cpp, and pin threads to physical cores",
            f"This box has {hw.cpu_cores} physical cores and {hw.cpu_threads} threads. "
            "Hyperthreads add contention rather than throughput on a memory-bound workload, "
            f"so cap threads at {hw.cpu_cores}. Decode is bound by RAM bandwidth, not FLOPs — "
            "dual-channel DDR4 puts a hard ceiling near 5-10 tok/s on a 7B Q4.",
            impact="~+15% vs oversubscribed threads",
            command=f"llama-cli -m model.gguf -t {hw.cpu_cores} --no-mmap",
            tags=("runtime", "cpu")))
    return tips


def quant_format_advice(hw: Hardware, fit: Fit | None = None) -> list[Tip]:
    tips = [Tip(
        "quant.kquants", "Prefer Q4_K_M GGUF, and imatrix builds of it",
        "Bit-width alone doesn't describe a quant. K-quants (Q4_K_M, Q5_K_M) keep attention "
        "and embedding tensors at higher precision and cost ~2% more disk than the legacy "
        "Q4_0 for a large quality gain. Importance-matrix (imatrix) builds calibrate on real "
        "text and gain again at the same size — Bartowski's and Unsloth's repos publish them.",
        impact="noticeably better output at identical VRAM",
        tags=("quant", "quality"))]

    if fit is not None and not fit.fits_vram and fit.quant == "Q4":
        tips.append(Tip(
            "quant.iq", "Below 4 bits, switch from K-quants to IQ quants",
            "IQ4_XS / IQ3_M use codebook quantization and hold quality far better than Q3_K "
            "at the same size. IQ4_XS lands ~9% under Q4_K_M — often the difference between "
            "full offload and streaming layers over PCIe. They cost a little CPU on prompt "
            "processing, which is irrelevant if the alternative is not fitting.",
            impact=f"~{fit.weight_gb * 0.09:.1f}GB smaller than Q4_K_M",
            tags=("quant", "memory")))

    gpu = hw.gpu
    if gpu and gpu.vendor == "nvidia":
        name = gpu.name.lower()
        if any(k in name for k in ("gtx 10", "gtx 16", "rtx 20", "titan v")):
            tips.append(Tip(
                "quant.arch", f"{gpu.name} is pre-Ampere — no bf16, no fast fp8",
                "Pascal and Turing cards have no bf16 path and no FP8 tensor cores. Stay on "
                "fp16 or integer quants; anything advertising bf16 or FP8 will fall back to "
                "an emulated slow path. llama.cpp's fp16 kernels are fine here.",
                impact="avoids a silent 3-5x slowdown",
                tags=("quant", "hardware")))
    return tips


# --- memory levers, all computed -----------------------------------------

def memory_levers(hw: Hardware, fit: Fit, model: dict) -> list[Tip]:
    tips = []
    vram = hw.vram_available_gb

    kv_q8 = kv_cache_gb(model, fit.context, KV_BYTES["q8"])
    saved = fit.kv_gb - kv_q8
    if saved > 0.05:
        tips.append(Tip(
            "mem.kvquant", "Quantize the KV cache to q8_0",
            f"The cache is {fit.kv_gb:.2f}GB at fp16 for {fit.context} tokens. q8_0 halves it "
            "with a quality cost that benchmarks put near zero — q4_0 halves it again but is "
            "visible on long contexts. In Ollama this needs flash attention switched on first.",
            impact=f"-{saved:.2f}GB VRAM ({saved / max(vram, 0.01) * 100:.0f}% of what's free)",
            command="OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve",
            tags=("memory", "kv")))

    tips.append(Tip(
        "mem.flashattn", "Turn on flash attention",
        "Fuses the attention kernels so the N-squared score matrix is never written to memory. "
        "Saves memory that grows with context and speeds up prompt processing. It is also the "
        "gate for KV cache quantization in Ollama. Off by default in older Ollama builds.",
        impact="-5-15% VRAM, faster prefill",
        command="export OLLAMA_FLASH_ATTENTION=1",
        tags=("memory", "speed")))

    if not fit.fits_vram:
        half = kv_cache_gb(model, fit.context // 2, KV_BYTES["fp16"])
        tips.append(Tip(
            "mem.context", f"Drop the context from {fit.context} to {fit.context // 2}",
            "Context is the cheapest thing to give up when you're short. Most chat turns never "
            f"approach {fit.context} tokens, and Ollama allocates the full window up front.",
            impact=f"-{fit.kv_gb - half:.2f}GB VRAM",
            command=f"ollama run {fit.ollama} --ctx-size {fit.context // 2}",
            tags=("memory", "context")))
        if fit.gpu_layers > 0:
            tips.append(Tip(
                "mem.ngl", f"Pin exactly {fit.gpu_layers} of {fit.total_layers} layers to the GPU",
                "Let the loader guess and it will either leave VRAM on the table or overshoot "
                "into the driver's shared-memory fallback, which is slower than plain CPU. "
                f"Each layer here is about {fit.weight_gb / fit.total_layers * 1024:.0f}MB.",
                impact=f"~{fit.tokens_per_sec:.0f} tok/s instead of thrashing",
                command=f"llama-cli -m model.gguf -ngl {fit.gpu_layers}   "
                        f"# Ollama: PARAMETER num_gpu {fit.gpu_layers}",
                tags=("memory", "offload")))

    if fit.arch == "moe":
        tips.append(Tip(
            "mem.moeoffload", "Keep attention on the GPU, push experts to CPU",
            "MoE offloads better than dense models: only two experts per layer are read per "
            "token, so idle experts sitting in RAM cost far less than a dense layer would. "
            "llama.cpp exposes this directly with a tensor-override regex.",
            impact="fits in a fraction of the VRAM at a modest speed cost",
            command='llama-cli -m mixtral.gguf -ngl 99 -ot "\\.ffn_.*_exps\\.=CPU"',
            tags=("memory", "moe")))

    if fit.fits_vram:
        mc = max_context(model, fit.quant, hw)
        if mc > fit.context:
            tips.append(Tip(
                "mem.headroom", f"Room to open the context up to {mc} tokens",
                f"{vram - fit.total_gb:.1f}GB of VRAM is unused at {fit.context} ctx. Spend it on "
                "context rather than leaving it idle — or on keeping a second small model warm.",
                impact=f"{mc // fit.context}x the context, same card",
                command=f"ollama run {fit.ollama} --ctx-size {mc}",
                tags=("memory", "context")))
    return tips


def speed_levers(hw: Hardware, fit: Fit) -> list[Tip]:
    tips = []
    if fit.fits_vram and fit.params_b >= 7:
        tips.append(Tip(
            "speed.specdec", "Speculative decoding with a small draft model",
            "A 1B draft proposes several tokens, the real model verifies them in one forward "
            "pass. Output is identical to running the big model alone — this is not an "
            "approximation. Gains are biggest on code and structured text, where the draft "
            "guesses right most of the time. Costs the draft model's VRAM.",
            impact="1.5-2.5x tok/s on code, ~1.3x on prose",
            command="llama-server -m target.gguf -md draft-1b.gguf --draft-max 8",
            tags=("speed", "advanced")))
    tips.append(Tip(
        "speed.keepalive", "Stop paying the reload tax",
        "Ollama unloads a model 5 minutes after the last request; the next call then eats a "
        "multi-second cold load off disk. Raise it for models you use all day, and drop it to "
        "0 for ones you want evicted immediately to free VRAM.",
        impact="removes multi-second first-token stalls",
        command="export OLLAMA_KEEP_ALIVE=30m",
        tags=("speed", "ops")))
    tips.append(Tip(
        "speed.prefix", "Reuse the prompt prefix",
        "Prefill is compute-bound and scales with prompt length. Keeping the system prompt and "
        "any fixed context byte-identical across calls lets the KV cache be reused, so only the "
        "new tokens get processed. Changing one character at the front invalidates all of it.",
        impact="near-zero prefill on repeat calls",
        tags=("speed", "prompting")))
    if hw.gpu and hw.gpu.vendor == "nvidia" and hw.vram_total_gb - hw.vram_available_gb > 0.5:
        used = hw.vram_total_gb - hw.vram_available_gb
        tips.append(Tip(
            "speed.reclaim", f"{used:.1f}GB of VRAM is already taken by other processes",
            "Desktop compositors, browsers and editors hold VRAM. On a small card that is the "
            "difference between full offload and streaming over PCIe. A headless session or a "
            "closed browser is the cheapest upgrade available.",
            impact=f"up to +{used:.1f}GB usable",
            command="nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv",
            tags=("ops", "memory")))
    return tips


def quality_levers(hw: Hardware, fit: Fit) -> list[Tip]:
    tips = []
    if fit.quant == "Q4" and fit.total_gb * 1.25 <= hw.vram_available_gb:
        tips.append(Tip(
            "quality.upquant", "There is room for Q5 or Q8 — take it",
            "Quantization loss is not linear. Q8 is within noise of fp16, Q5_K_M is close, and "
            "Q4 is where degradation starts showing on reasoning and long-form output. If a "
            "higher quant still fits entirely in VRAM, it is free quality.",
            impact="measurably better output, same speed class",
            tags=("quality", "quant")))
    tips.append(Tip(
        "quality.grammar", "Constrain structured output with a grammar, don't parse and pray",
        "Small local models are much worse at 'reply with only JSON' than frontier models. "
        "GBNF grammars in llama.cpp and the format parameter in Ollama constrain sampling at "
        "the token level, so invalid output is impossible rather than unlikely.",
        impact="eliminates a whole class of retry logic",
        command='curl localhost:11434/api/generate -d \'{"model":"...","format":"json"}\'',
        tags=("quality", "prompting")))
    return tips


def calibration_advice(hw: Hardware) -> list[Tip]:
    device = hw.gpu.name if hw.gpu else "cpu"
    st = calibration.status(device)
    ov = st["token_overhead_ms"]
    if st["measured"]:
        return []
    if st["unreliable"]:
        return [Tip(
            "calibrate.rejected", "This machine does not match the speed model",
            "A calibration run was attempted and its result rejected: " + st["reason"] +
            ". The tok/s figures below come from rated bandwidth and will be optimistic. "
            "Common causes: a throttled or power-limited laptop GPU, another process "
            "sharing the card, or a driver that cannot report its own state.",
            impact="treat every speed here as an upper bound",
            command="herd bench --json   # to see the raw samples",
            tags=("accuracy", "limits"))]
    return [Tip(
        "calibrate.bench", "Calibrate the speed estimates against this machine",
        "Every tok/s figure here assumes decode is purely memory-bound. That holds for "
        "large models and breaks badly for small ones, where fixed per-token cost — kernel "
        "launches, sampling, sync — dominates. One 128-token run measures the real fixed "
        f"cost for this GPU and stores it; until then everything uses a generic {ov:.0f}ms.",
        impact="turns rough guesses into machine-specific numbers",
        command="herd bench",
        tags=("accuracy", "ops"))]


def backend_advice(backend: str) -> list[Tip]:
    """Levers that only exist on the backend actually in use."""
    spec = backends.spec(backend)
    tips = []
    if backend == "hf":
        tips.append(Tip(
            "backend.bnb", "Native checkpoints are FP16 — quantize on load or pay 4x",
            "transformers loads safetensors at their stored precision, so a Llama-3-8B is "
            "~16GB before anything else. BitsAndBytesConfig(load_in_4bit=True, "
            "bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True) brings that to about "
            "5.7GB — still above the 4.7GB the same model costs as GGUF Q4, because NF4 "
            "stores absmax scales alongside the weights.",
            impact="~4x less VRAM than the default FP16 load",
            command="AutoModelForCausalLM.from_pretrained(repo, quantization_config=bnb_cfg)",
            tags=("memory", "quant", "hf")))
        tips.append(Tip(
            "backend.sdpa", "Use SDPA or FlashAttention-2 instead of the eager path",
            "transformers defaults to an attention implementation that materializes the score "
            "matrix. attn_implementation='sdpa' uses PyTorch's fused kernels; 'flash_attention_2' "
            "is faster still on Ampere and newer, but needs the flash-attn package and will not "
            "build on pre-Ampere cards.",
            impact="lower memory at long context, faster prefill",
            command="from_pretrained(repo, attn_implementation='sdpa')",
            tags=("speed", "memory", "hf")))
    elif backend == "llamacpp":
        tips.append(Tip(
            "backend.ngl", "You get the raw knobs — use them",
            "Going direct to llama.cpp means -ngl, --cache-type-k/v, -ot for tensor placement "
            "and --split-mode are all yours, rather than inferred by a wrapper. That is the "
            "whole reason to skip Ollama; if you aren't setting them, Ollama is doing the same "
            "work with better ergonomics.",
            impact="full control over placement and cache precision",
            command="llama-cli -m model.gguf -ngl 33 --cache-type-k q8_0 --cache-type-v q8_0",
            tags=("runtime", "memory", "llamacpp")))
    elif backend == "openai":
        tips.append(Tip(
            "backend.util", "vLLM reserves the card up front — tune the fraction, not the model",
            "gpu_memory_utilization defaults to 0.90 of total VRAM, claimed at startup and used "
            "for the KV cache pool regardless of model size. On a shared desktop GPU that "
            "starves the compositor; lower it. It also means nothing else can be scheduled "
            "beside it, and per-model eviction does not exist.",
            impact="stops vLLM from taking the whole card",
            command="vllm serve <repo> --gpu-memory-utilization 0.7 --max-model-len 4096",
            tags=("memory", "ops", "vllm")))
        tips.append(Tip(
            "backend.batching", "Continuous batching is the reason to run a server",
            "PagedAttention plus continuous batching is where vLLM pays off: single-stream "
            "speed is comparable to llama.cpp, but concurrent requests scale several times "
            "better. If you only ever send one request at a time, you are carrying the "
            "complexity for nothing.",
            impact="several times the throughput under concurrency",
            tags=("speed", "throughput")))
    if not spec.can_evict:
        tips.append(Tip(
            "backend.noevict", f"{spec.label} cannot be scheduled per model",
            "It loads one model at server start and holds it for its lifetime, so Herd's "
            "LRU eviction has nothing to call. Use it for a single hot model, and keep Ollama "
            "or the in-process backends for agents that need to swap.",
            impact="scheduler reports usage but cannot free it",
            tags=("scheduler", "limits")))
    return tips


def advise(hw: Hardware, fit: Fit | None = None, model: dict | None = None,
           backend: str = "ollama") -> list[Tip]:
    tips = (calibration_advice(hw) + backend_advice(backend)
            + runtime_advice(hw, fit) + quant_format_advice(hw, fit))
    if fit is not None:
        model = model or registry.find_model(fit.model) or {}
        if model:
            tips += memory_levers(hw, fit, model)
        tips += speed_levers(hw, fit) + quality_levers(hw, fit)
    return tips
