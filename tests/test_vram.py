"""Self-check for the VRAM math. Run: .venv/bin/python tests/test_vram.py"""
import sys, os, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# Never read the developer's real calibration file — these checks must be reproducible.
os.environ["HERD_CALIBRATION"] = str(pathlib.Path(tempfile.gettempdir()) / "herd-test-cal.json")

from herd import registry
from herd.hardware import Hardware, GPU
from herd.vram import estimate, kv_cache_gb, max_context, QUANT_BYTES
from herd import calibration
from herd.recommender import recommend

GB = 1024 ** 3


def hw(vram=8.0, ram=32.0, bw=272.0):
    return Hardware(gpus=[GPU("RTX 4060", "nvidia", "cuda", vram, vram, bw)],
                    cpu_cores=8, cpu_threads=16, ram_total_gb=ram, ram_available_gb=ram)


def m(name):
    got = registry.find_model(name)
    assert got, name
    return got


def test_weight_math():
    f = estimate(m("Llama-3-8B"), "Q4", hw())
    assert abs(f.weight_gb - 8.03e9 * 0.5 / GB) < 1e-6
    assert abs(f.total_gb - (f.weight_gb + f.kv_gb + 0.5)) < 1e-9
    # Doubling the bit width doubles the weights, exactly.
    assert abs(estimate(m("Llama-3-8B"), "Q8", hw()).weight_gb - 2 * f.weight_gb) < 1e-6


def test_kv_uses_gqa_not_head_count():
    llama = m("Llama-3-8B")            # 32 heads, 8 KV heads
    expect = 2 * 32 * 8 * 128 * 4096 * 2 / GB
    assert abs(kv_cache_gb(llama, 4096, 2.0) - expect) < 1e-9
    # CodeLlama has no GQA, so its cache is 4x Llama-3's despite the same shape.
    assert kv_cache_gb(m("CodeLlama-7B"), 4096, 2.0) / kv_cache_gb(llama, 4096, 2.0) == 4
    # KV scales linearly in context and in cache precision.
    assert abs(kv_cache_gb(llama, 8192, 2.0) - 2 * expect) < 1e-9
    assert abs(kv_cache_gb(llama, 4096, 1.0) - expect / 2) < 1e-9


def test_tiers_and_boundaries():
    big = estimate(m("Llama-3-70B"), "Q4", hw(vram=8, ram=32))
    assert big.tier == "no" and not big.fits_vram
    small = estimate(m("Phi-3-mini"), "Q4", hw(vram=8))
    assert small.tier == "great" and small.fits_vram and small.gpu_layers == small.total_layers
    # Same model, tiny card: offload, and only some layers land on the GPU.
    mid = estimate(m("Llama-3-8B"), "Q4", hw(vram=3.0, ram=32))
    assert mid.tier == "offload" and 0 < mid.gpu_layers < mid.total_layers
    # More VRAM never means fewer layers on the GPU.
    layers = [estimate(m("Llama-3-8B"), "Q4", hw(vram=v)).gpu_layers for v in (1, 2, 4, 6, 8)]
    assert layers == sorted(layers)


def test_speed_ordering():
    # Fully resident beats partial offload beats none, on identical weights.
    fast = estimate(m("Llama-3-8B"), "Q4", hw(vram=16)).tokens_per_sec
    slow = estimate(m("Llama-3-8B"), "Q4", hw(vram=3)).tokens_per_sec
    none = estimate(m("Llama-3-8B"), "Q4", Hardware(ram_available_gb=32)).tokens_per_sec
    assert fast > slow > 0 and none > 0 and fast > none
    # Resident case matches memory time plus the fixed per-token cost.
    f = estimate(m("Llama-3-8B"), "Q4", hw(vram=16, bw=272))
    expect = 1 / (f.total_gb / 272 + calibration.DEFAULT_OVERHEAD_MS / 1000)
    assert abs(f.tokens_per_sec - expect) / expect < 0.02
    # A heavier quant of the same model is always slower.
    assert (estimate(m("Llama-3-8B"), "Q4", hw(vram=24)).tokens_per_sec >
            estimate(m("Llama-3-8B"), "Q8", hw(vram=24)).tokens_per_sec)


def test_moe_is_flagged_and_cheap_per_token():
    f = estimate(m("Mixtral-8x7B"), "Q4", hw(vram=48, ram=64))
    assert f.arch == "moe"
    # All experts resident, but only the active slice is read per token.
    assert f.active_gb < f.total_gb
    assert any("MoE" in n for n in f.notes)
    # It should decode faster than a dense model of the same total size would.
    dense = estimate(m("Llama-3-70B"), "Q4", hw(vram=48, ram=64))
    assert f.tokens_per_sec > dense.tokens_per_sec


def test_max_context_fits_and_is_capped():
    model = m("Llama-3-8B")
    ctx = max_context(model, "Q4", hw(vram=8))
    assert ctx > 0 and ctx & (ctx - 1) == 0                      # power of two
    assert ctx <= model["max_context"]
    assert estimate(model, "Q4", hw(vram=8), context=ctx).fits_vram
    # Weights alone over budget -> no context fits at all.
    assert max_context(m("Llama-3-70B"), "Q4", hw(vram=8)) == 0


def test_bandwidth_lookup_prefers_longest_match():
    assert registry.bandwidth_for("NVIDIA GeForce RTX 4060 Ti", "nvidia") == (288.0, True)
    assert registry.bandwidth_for("NVIDIA GeForce RTX 4060", "nvidia") == (272.0, True)
    bw, known = registry.bandwidth_for("Some Unreleased GPU", "nvidia")
    assert not known and bw > 0


def test_recommend_partitions_every_model():
    r = recommend(hw(vram=8, ram=32), limit=50)
    seen = {f["model"] for k in ("runs_great", "runs_offload", "wont_fit") for f in r[k]}
    assert seen == {mm["name"] for mm in registry.models()}, seen
    assert r["hybrid"]["workhorse"] and 0 <= r["hybrid"]["local_coverage"] <= 1
    assert r["optimizations"]
    # A machine with nothing free should still answer, with everything in won't-fit.
    tiny = recommend(hw(vram=0.2, ram=0.2), limit=50)
    assert not tiny["runs_great"] and tiny["wont_fit"]


def test_calibration_round_trip():
    # 19.4 tok/s on 1.3GB resident at 128 GB/s -> ~41ms of fixed per-token cost.
    ov = calibration.solve(19.4, 1.3, 128.0)
    assert 35 < ov < 45
    # A model whose bytes alone already exceed the measured time has no fixed cost left.
    assert calibration.solve(2.0, 100.0, 128.0) == 0.0
    # The fixed cost distorts small models far more than large ones — that asymmetry
    # is the whole reason the term exists.
    def gap(name):
        f = estimate(m(name), "Q4", hw(vram=80, bw=1008))
        return (1008 / f.total_gb) / f.tokens_per_sec        # pure-bandwidth / calibrated
    assert gap("Qwen-2.5-0.5B") > 3 * gap("Llama-3-70B") > 1.0


def test_fit_solves_bandwidth_and_fixed_cost():
    # Two points on a clean machine: 1GB at 20ms, 3GB at 50ms -> 66.7 GB/s, 5ms fixed.
    f = calibration.fit([(1.0, 0.020), (3.0, 0.050)])
    assert f["ok"] and abs(f["effective_bandwidth_gbs"] - 66.7) < 0.5
    assert abs(f["token_overhead_ms"] - 5.0) < 0.5


def test_fit_refuses_physically_impossible_results():
    # Real measurements from a GTX 1650 whose driver reports ERR!: the implied fixed
    # cost is negative, so the model does not describe the machine.
    bad = calibration.fit([(1.211, 1 / 19.7), (3.083, 1 / 4.2)])
    assert not bad["ok"] and bad["token_overhead_ms"] < 0 and bad["reason"]
    # One sample cannot separate bandwidth from fixed cost at all.
    assert not calibration.fit([(1.2, 0.05)])["ok"]
    # A bigger model measuring faster is nonsense, not a negative slope to encode.
    assert not calibration.fit([(1.0, 0.05), (3.0, 0.02)])["ok"]


def test_rejected_calibration_is_not_used_as_a_measurement():
    dev = "Test GPU"
    calibration._load.cache_clear() if hasattr(calibration._load, "cache_clear") else None
    calibration.reject(dev, "throttled", [{"model": "x"}])
    st = calibration.status(dev)
    assert st["unreliable"] and not st["measured"] and st["reason"] == "throttled"
    # Falling back to the default keeps estimates optimistic but never silently wrong.
    assert calibration.overhead_ms(dev) == (calibration.DEFAULT_OVERHEAD_MS, False)
    assert calibration.bandwidth_gbs(dev) is None


def test_quant_table_matches_spec():
    assert QUANT_BYTES == {"Q4": 0.5, "Q5": 0.625, "Q8": 1.0, "FP16": 2.0}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")
