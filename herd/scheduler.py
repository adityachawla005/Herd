"""Phase 3: VRAM-aware model scheduler.

Ollama will happily load a second model on top of a full card and let the driver
thrash. This layer decides what is resident: it tracks footprints and last-use, and
evicts the least-recently-used model before loading one that would not fit.

The backend's own report is the source of truth for what is resident — the scheduler
never trusts its own bookkeeping over the runtime's. Backends that cannot evict (a vLLM
server owns its model for its lifetime) are tracked but never chosen as victims, and
say so rather than silently failing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Callable

from . import backends, ollama, registry
from .backends.base import Backend
from .errors import HerdError
from .hardware import detect, Hardware
from .vram import estimate

SAFETY_GB = 0.3          # leave the driver a little room


@dataclass
class Resident:
    tag: str
    size_gb: float
    last_used: float
    loaded_at: float
    holders: set = field(default_factory=set)
    pinned: bool = False

    @property
    def busy(self) -> bool:
        return bool(self.holders)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["holders"] = sorted(self.holders)
        d["idle_s"] = round(time.time() - self.last_used, 1)
        d["size_gb"] = round(self.size_gb, 2)
        return d


class SchedulerError(RuntimeError):
    pass


class ModelScheduler:
    def __init__(self, hw: Hardware | None = None, pinned: tuple[str, ...] = (),
                 on_event: Callable[[dict], None] | None = None, budget_gb: float | None = None,
                 backend: "str | Backend" = "ollama"):
        self.hw = hw or detect()
        self.backend = backends.get(backend) if isinstance(backend, str) else backend
        self.pinned = set(pinned)
        self.resident: dict[str, Resident] = {}
        self._on_event = on_event or (lambda e: None)
        self._budget = budget_gb
        self._sizes: dict[str, float] = {}
        self.refresh()

    # --- state ------------------------------------------------------------

    def refresh(self) -> None:
        """Reconcile with the backend. Models it dropped on its own disappear here too."""
        try:
            live = {m["id"]: m for m in self.backend.resident()}
        except HerdError:
            live = {}
        now = time.time()
        for tag, m in live.items():
            # A backend that can't measure per-model VRAM (vLLM) reports 0; estimate instead.
            size = m.get("size_gb") or self.predict_size_gb(tag)
            if tag in self.resident:
                self.resident[tag].size_gb = size or self.resident[tag].size_gb
            else:
                self.resident[tag] = Resident(tag, size, now, now, pinned=tag in self.pinned)
        for tag in list(self.resident):
            if tag not in live:
                del self.resident[tag]

    @property
    def budget_gb(self) -> float:
        """VRAM this scheduler may spend, excluding whatever else owns the card."""
        if self._budget is not None:
            return self._budget
        total = self.hw.vram_total_gb
        if total <= 0:
            return self.hw.ram_available_gb * 0.5      # CPU-only: RAM is the budget
        detected = detect()
        ours = sum(r.size_gb for r in self.resident.values())
        foreign = max(0.0, total - detected.vram_available_gb - ours)
        return max(0.0, total - foreign - SAFETY_GB)

    @property
    def used_gb(self) -> float:
        return sum(r.size_gb for r in self.resident.values())

    def predict_size_gb(self, tag: str) -> float:
        """Footprint before loading: registry math if we know the model, disk size if not."""
        if tag in self._sizes:
            return self._sizes[tag]
        ref_key = self.backend.spec.ref_key
        base = tag.split(":")[0]
        model = next((m for m in registry.models()
                      if m.get(ref_key) and m[ref_key].split(":")[0] == base), None)
        size = None
        if model:
            size = estimate(model, "Q4", self.hw, backend=self.backend.spec.id).total_gb
        elif self.backend.spec.id == "ollama":
            try:
                for m in ollama.installed():
                    if (m.get("name") or m.get("model")) == tag:
                        # Weights on disk plus room for cache and runtime.
                        size = (m.get("size") or 0) / (1024 ** 3) * 1.25
            except ollama.OllamaError:
                pass
        self._sizes[tag] = size or 2.0
        return self._sizes[tag]

    # --- policy -----------------------------------------------------------

    def _evictable(self) -> list[Resident]:
        if not self.backend.spec.can_evict:
            return []
        return sorted((r for r in self.resident.values() if not r.pinned and not r.busy),
                      key=lambda r: r.last_used)

    def evict(self, tag: str) -> bool:
        r = self.resident.get(tag)
        if not r or r.pinned or r.busy:
            return False
        if not self.backend.spec.can_evict:
            self._emit("evict_unsupported", tag=tag, backend=self.backend.spec.id)
            return False
        try:
            self.backend.unload(tag)
        except HerdError as e:
            self._emit("evict_failed", tag=tag, error=str(e))
            return False
        del self.resident[tag]
        self._emit("evict", tag=tag, freed_gb=round(r.size_gb, 2),
                   idle_s=round(time.time() - r.last_used, 1))
        return True

    def make_room(self, need_gb: float) -> float:
        """Evict LRU models until need_gb fits. Returns the free space achieved."""
        budget = self.budget_gb
        while budget - self.used_gb < need_gb:
            victims = self._evictable()
            if not victims:
                break
            if not self.evict(victims[0].tag):
                break
        return budget - self.used_gb

    def acquire(self, tag: str, agent: str = "") -> Resident:
        """Ensure tag is resident and marked in use. Loads or evicts as needed."""
        self.refresh()
        if tag in self.resident:
            r = self.resident[tag]
            r.last_used = time.time()
            r.holders.add(agent or "anon")
            r.pinned = r.pinned or tag in self.pinned
            self._emit("cache_hit", tag=tag, agent=agent, size_gb=round(r.size_gb, 2))
            return r

        need = self.predict_size_gb(tag)
        free = self.budget_gb - self.used_gb
        if free < need:
            free = self.make_room(need)
        if free < need:
            self._emit("pressure", tag=tag, need_gb=round(need, 2), free_gb=round(free, 2))

        started = time.perf_counter()
        self.backend.load(tag)
        self.refresh()
        r = self.resident.get(tag)
        if r is None:
            raise SchedulerError(
                f"{tag} did not stay resident after loading — it is probably too large for "
                f"{self.budget_gb:.1f}GB of usable VRAM and {self.backend.spec.label} fell "
                "back to CPU.")
        r.holders.add(agent or "anon")
        r.last_used = time.time()
        r.pinned = tag in self.pinned
        self._emit("load", tag=tag, agent=agent, size_gb=round(r.size_gb, 2),
                   load_ms=round((time.perf_counter() - started) * 1000))
        return r

    def release(self, tag: str, agent: str = "") -> None:
        r = self.resident.get(tag)
        if r:
            r.holders.discard(agent or "anon")
            r.last_used = time.time()

    def status(self) -> dict:
        self.refresh()
        budget = self.budget_gb
        return {
            "vram_total_gb": round(self.hw.vram_total_gb, 2),
            "budget_gb": round(budget, 2),
            "used_gb": round(self.used_gb, 2),
            "free_gb": round(budget - self.used_gb, 2),
            "utilization": round(self.used_gb / budget, 3) if budget > 0 else 0.0,
            "loaded": [r.to_dict() for r in sorted(self.resident.values(),
                                                   key=lambda r: -r.last_used)],
            "pinned": sorted(self.pinned),
            "backend": {"id": self.backend.spec.id, "label": self.backend.spec.label,
                        "can_evict": self.backend.spec.can_evict,
                        "owns_card": self.backend.spec.owns_card,
                        "reports_vram": self.backend.spec.reports_vram},
        }

    def _emit(self, event: str, **kw) -> None:
        self._on_event({"event": event, "t": time.time(), **kw})
