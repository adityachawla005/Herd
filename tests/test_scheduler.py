"""Self-check for the VRAM scheduler's policy. Run: .venv/bin/python tests/test_scheduler.py

Ollama is stubbed: the policy (what is resident, what gets evicted, in what order) is
the logic worth testing, and it should not need gigabytes of downloads to verify.
"""
import sys, os, pathlib, tempfile, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ["HERD_CALIBRATION"] = str(pathlib.Path(tempfile.gettempdir()) / "herd-test-cal.json")

from herd import ollama, scheduler
from herd.hardware import Hardware, GPU
from herd.scheduler import ModelScheduler

# Disk size is what the scheduler predicts from; VRAM size is what it actually costs
# once loaded (weights plus KV cache), which is why the two tables differ.
DISK = {"small:a": 0.8, "small:b": 0.8, "big:c": 2.0}
VRAM = {"small:a": 1.0, "small:b": 1.0, "big:c": 2.5}
BUDGET = 3.5


class FakeOllama:
    """Stands in for a running Ollama with a fixed catalogue."""

    def __init__(self):
        self.vram = {}
        self.unloads = []

    def loaded(self):
        return [{"name": t, "size_vram": int(g * 1024 ** 3), "size": int(g * 1024 ** 3)}
                for t, g in self.vram.items()]

    def installed(self):
        return [{"name": t, "size": int(g * 1024 ** 3)} for t, g in DISK.items()]

    def generate(self, tag, prompt, *a, **kw):
        self.vram[tag] = VRAM[tag]
        return iter([{"response": "", "done": True}])

    def unload(self, tag):
        self.unloads.append(tag)
        self.vram.pop(tag, None)


def setup():
    fake = FakeOllama()
    for name in ("loaded", "installed", "generate", "unload"):
        setattr(scheduler.ollama, name, getattr(fake, name))
    hw = Hardware(gpus=[GPU("Fake GPU", "nvidia", "cuda", 4.0, 4.0, 300.0)],
                  ram_total_gb=16, ram_available_gb=16)
    return fake, hw


def sched(fake, hw, **kw):
    # budget_gb pins the arithmetic so the test doesn't depend on a real card's free VRAM.
    return ModelScheduler(hw, budget_gb=BUDGET, **kw)


def test_cache_hit_avoids_reload():
    fake, hw = setup()
    events = []
    s = sched(fake, hw, on_event=events.append)
    s.acquire("small:a", "agent1")
    s.release("small:a", "agent1")
    s.acquire("small:a", "agent2")
    kinds = [e["event"] for e in events]
    assert kinds == ["load", "cache_hit"], kinds
    assert s.status()["used_gb"] == 1.0


def test_evicts_lru_when_full():
    fake, hw = setup()
    events = []
    s = sched(fake, hw, on_event=events.append)
    s.acquire("small:a", "a"); s.release("small:a", "a")
    time.sleep(0.01)
    s.acquire("small:b", "b"); s.release("small:b", "b")
    assert s.used_gb == 2.0
    # big:c needs 2.5 of a 3.5 budget: evicting small:a alone makes exactly enough room,
    # so the LRU victim goes and the newer model stays.
    s.acquire("big:c", "c")
    assert fake.unloads == ["small:a"], fake.unloads
    assert set(s.resident) == {"small:b", "big:c"}
    assert [e["event"] for e in events if e["event"] == "evict"]


def test_pinned_and_busy_models_survive_pressure():
    fake, hw = setup()
    s = sched(fake, hw, pinned=("small:a",))
    s.acquire("small:a", "router"); s.release("small:a", "router")
    time.sleep(0.01)
    s.acquire("small:b", "worker")          # held, never released
    s.acquire("big:c", "other")
    # Neither the pinned router nor the in-use worker may be evicted, so the load
    # proceeds over budget rather than stealing a model out from under an agent.
    assert "small:a" in s.resident and "small:b" in s.resident
    assert fake.unloads == [], fake.unloads


def test_pressure_is_reported_when_room_cannot_be_made():
    fake, hw = setup()
    events = []
    s = sched(fake, hw, pinned=("small:a", "small:b"), on_event=events.append)
    s.acquire("small:a"); s.acquire("small:b")
    s.acquire("big:c")
    assert any(e["event"] == "pressure" for e in events)


def test_refresh_notices_models_ollama_dropped():
    fake, hw = setup()
    s = sched(fake, hw)
    s.acquire("small:a", "a")
    fake.vram.clear()                        # Ollama's own idle timer fired
    s.refresh()
    assert s.resident == {}
    assert s.status()["used_gb"] == 0


def test_status_reports_utilization():
    fake, hw = setup()
    s = sched(fake, hw)
    s.acquire("small:a", "a")
    st = s.status()
    assert st["budget_gb"] == BUDGET and st["used_gb"] == 1.0
    assert abs(st["utilization"] - 1 / BUDGET) < 0.01
    assert st["loaded"][0]["tag"] == "small:a" and st["loaded"][0]["holders"] == ["a"]


def test_unknown_tag_falls_back_to_disk_size():
    fake, hw = setup()
    s = sched(fake, hw)
    # Not in models.json, so the estimate comes from the installed size plus headroom.
    assert abs(s.predict_size_gb("big:c") - DISK["big:c"] * 1.25) < 0.01


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")
