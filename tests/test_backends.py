"""Self-check for the backend abstraction. Run: .venv/bin/python tests/test_backends.py

Nothing here needs a backend installed — the point is that the math and the scheduler
behave correctly per backend, including the ones this machine can't run.
"""
import sys, os, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ["HERD_CALIBRATION"] = str(pathlib.Path(tempfile.gettempdir()) / "herd-test-cal.json")

from herd import backends, registry, scheduler
from herd.backends.base import BackendSpec, Unavailable
from herd.backends.openai_compat import OpenAICompatBackend
from herd.errors import HerdError
from herd.hardware import Hardware, GPU
from herd.scheduler import ModelScheduler
from herd.vram import estimate, all_fits


def hw(vram=24.0, ram=64.0):
    return Hardware(gpus=[GPU("Fake GPU", "nvidia", "cuda", vram, vram, 500.0)],
                    cpu_cores=8, cpu_threads=16, ram_total_gb=ram, ram_available_gb=ram)


def llama():
    return registry.find_model("Llama-3-8B")


def test_survey_never_raises_and_is_complete():
    rows = backends.survey()
    assert {r["id"] for r in rows} == set(backends.BACKENDS)
    for r in rows:
        assert isinstance(r["available"], bool) and r["detail"]
        assert r["quants"] and r["ref_key"] and r["install"]


def test_four_bit_costs_differ_by_backend():
    """GGUF Q4, bitsandbytes NF4 and AWQ 4-bit are all 'Q4' and none cost the same."""
    sizes = {b: estimate(llama(), "Q4", hw(), backend=b).weight_gb
             for b in ("ollama", "llamacpp", "hf", "openai")}
    assert sizes["ollama"] == sizes["llamacpp"]            # both plain GGUF
    assert sizes["hf"] > sizes["openai"] > sizes["ollama"]  # NF4 > AWQ > GGUF
    # FP16 is FP16 everywhere — no scale factor applies.
    fp16 = {b: estimate(llama(), "FP16", hw(), backend=b).weight_gb
            for b in ("ollama", "hf", "openai")}
    assert len(set(round(v, 6) for v in fp16.values())) == 1
    assert 14.5 < fp16["hf"] < 15.5                        # the ~16GB 8B everyone quotes


def test_runtime_overhead_is_per_backend():
    ollama_fit = estimate(llama(), "Q4", hw(), backend="ollama")
    hf_fit = estimate(llama(), "Q4", hw(), backend="hf")
    assert hf_fit.overhead_gb > ollama_fit.overhead_gb     # torch context is heavier
    assert estimate(llama(), "Q4", hw(), backend="openai").overhead_gb == 0.0


def test_all_fits_respects_backend_capabilities():
    # bitsandbytes has no Q5 equivalent, so it must not be offered.
    assert not [f for f in all_fits(hw(), backend="hf") if f.quant == "Q5"]
    assert [f for f in all_fits(hw(), backend="ollama") if f.quant == "Q5"]
    # Each fit carries the id its own backend accepts.
    for f in all_fits(hw(), backend="hf"):
        assert "/" in f.model_ref, f.model_ref            # an HF repo, not an Ollama tag
    for f in all_fits(hw(), backend="ollama"):
        assert "/" not in f.model_ref, f.model_ref


def test_model_without_a_reference_is_skipped():
    """A registry entry missing this backend's field isn't offered on it."""
    original = registry.models
    fake = [dict(m) for m in original()][:1]
    fake[0].pop("hf_repo", None)
    registry.models = lambda: fake
    try:
        assert all_fits(hw(), backend="hf") == []
        assert all_fits(hw(), backend="ollama")            # still has its ollama tag
    finally:
        registry.models = original


# --- scheduling a backend that cannot evict -------------------------------

class FrozenBackend:
    """A vLLM-shaped backend: one model, loaded at startup, never evictable."""
    spec = BackendSpec(id="frozen", label="Frozen server", ref_key="hf_repo",
                       install="n/a", can_evict=False, owns_card=True, reports_vram=False)

    def __init__(self):
        self.unload_calls = 0

    def available(self): return True, "fake"
    def list_models(self): return ["served/model"]
    def resident(self): return [{"id": "served/model", "size_gb": 3.0}]
    def ensure(self, m): pass
    def load(self, m): pass

    def unload(self, m):
        self.unload_calls += 1
        raise Unavailable("cannot unload")

    def generate(self, *a, **kw): return iter([{"done": True}])


def test_non_evictable_backend_is_never_evicted():
    b = FrozenBackend()
    events = []
    s = ModelScheduler(hw(vram=8), budget_gb=4.0, backend=b, on_event=events.append)
    assert s.used_gb == 3.0
    # Under pressure it must report, not thrash — and never call unload.
    assert s.evict("served/model") is False
    assert b.unload_calls == 0
    assert any(e["event"] == "evict_unsupported" for e in events)
    assert s.make_room(10.0) < 10.0
    assert s.status()["backend"]["can_evict"] is False


def test_status_reports_backend_capabilities():
    b = FrozenBackend()
    st = ModelScheduler(hw(), budget_gb=4.0, backend=b).status()
    assert st["backend"]["owns_card"] is True
    assert st["backend"]["reports_vram"] is False


def test_openai_backend_refuses_to_unload_with_a_hint():
    b = OpenAICompatBackend(base_url="http://127.0.0.1:9/v1")
    ok, why = b.available()
    assert not ok and "127.0.0.1:9" in why                # unreachable, reported cleanly
    assert b.list_models() == []
    try:
        b.unload("anything")
        assert False, "should have refused"
    except HerdError as e:
        assert "cannot unload" in str(e) and e.hint


def test_unknown_backend_names_are_rejected():
    try:
        backends.get("tensorrt")
        assert False, "should have raised"
    except HerdError as e:
        assert "ollama" in e.hint


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")
