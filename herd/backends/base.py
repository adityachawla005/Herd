"""The Backend contract.

A backend is anything that can hold a model in memory and stream tokens out of it.
They differ in ways the scheduler has to know about, so capability flags are part of
the contract rather than something to discover by calling a method and catching:

  can_evict   Ollama and in-process backends can drop one model and keep others.
              A vLLM server cannot — it owns its model for the process lifetime.
  owns_card   vLLM pre-allocates gpu_memory_utilization (0.9 by default) of the whole
              GPU up front. Nothing else can be scheduled beside it, and its reported
              usage is the reservation, not the model.
  reports_vram Whether resident() returns real footprints or estimates.

Precision names are shared (Q4/Q5/Q8/FP16) but what they cost is not. GGUF Q4_K_M,
bitsandbytes NF4 and AWQ 4-bit are all "4-bit" and all land on different numbers, so
each backend carries a scale factor over the nominal bytes-per-param table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol

from ..errors import HerdError


@dataclass(frozen=True)
class BackendSpec:
    id: str
    label: str
    ref_key: str                  # which registry field holds this backend's model id
    install: str
    runtime_overhead_gb: float = 0.5
    can_evict: bool = True
    owns_card: bool = False
    reports_vram: bool = True
    # Real cost relative to the nominal bytes-per-param table, per precision.
    bytes_scale: dict = field(default_factory=dict)
    quants: tuple = ("Q4", "Q5", "Q8", "FP16")
    notes: str = ""

    def scale(self, quant: str) -> float:
        return self.bytes_scale.get(quant, 1.0)

    def supports(self, quant: str) -> bool:
        return quant in self.quants


class Backend(Protocol):
    """What the scheduler and orchestrator are allowed to assume."""

    spec: BackendSpec

    def available(self) -> tuple[bool, str]:
        """(usable here, why not). Must never raise."""

    def list_models(self) -> list[str]:
        """Model ids already present locally."""

    def resident(self) -> list[dict]:
        """[{'id':..., 'size_gb':...}] currently occupying VRAM."""

    def ensure(self, model_id: str) -> None:
        """Fetch the model if it isn't local. May be slow."""

    def load(self, model_id: str) -> None: ...

    def unload(self, model_id: str) -> None: ...

    def generate(self, model_id: str, prompt: str, system: str | None = None,
                 options: dict | None = None) -> Iterator[dict]:
        """Yield {'response': str} chunks, then one {'done': True, ...} with timings."""


class BackendError(HerdError):
    """A backend could not do what was asked, with a hint for fixing it."""


class Unavailable(BackendError):
    """Raised when a backend is asked to work but isn't installed or running."""
