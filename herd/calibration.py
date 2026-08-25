"""Measured per-token overhead, so estimates match this machine instead of theory.

Decode time per token is roughly (bytes read / bandwidth) + a fixed cost that has
nothing to do with memory: kernel launches, sampling, sync. On big models the first
term dominates and the spec's bandwidth formula is fine. On small ones the fixed cost
dominates — a GTX 1650 spends ~40ms/token there, which makes a pure bandwidth estimate
4x optimistic. `herd bench` measures the real number and stores it here.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

DEFAULT_OVERHEAD_MS = 10.0


def path() -> Path:
    base = os.environ.get("HERD_CALIBRATION")
    if base:
        return Path(base)
    cfg = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(cfg) / "herd" / "calibration.json"


def _load() -> dict:
    try:
        return json.loads(path().read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def overhead_ms(device: str) -> tuple[float, bool]:
    """(ms per token, measured). Falls back to a generic default."""
    entry = _load().get(device)
    if entry and isinstance(entry.get("token_overhead_ms"), (int, float)):
        return float(entry["token_overhead_ms"]), True
    return DEFAULT_OVERHEAD_MS, False


def bandwidth_gbs(device: str) -> float | None:
    """Measured effective bandwidth, if a multi-point fit produced a credible one.

    A measurement beats a lookup table: the table says what the silicon can do, this
    says what it did.
    """
    entry = _load().get(device) or {}
    bw = entry.get("effective_bandwidth_gbs")
    return float(bw) if isinstance(bw, (int, float)) and bw > 0 else None


def reject(device: str, reason: str, samples: list) -> Path:
    """Record that calibration was attempted and the result was not credible.

    Clears any earlier stored value: a number we have just proved does not generalize
    is worse than no number, and this stops the CLI from telling you to run a
    calibration it is going to refuse again.
    """
    data = _load()
    data[device] = {"unreliable": True, "reason": reason, "samples": samples,
                    "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n")
    return p


def status(device: str) -> dict:
    entry = _load().get(device) or {}
    return {"measured": isinstance(entry.get("token_overhead_ms"), (int, float)),
            "unreliable": bool(entry.get("unreliable")),
            "reason": entry.get("reason", ""),
            "token_overhead_ms": entry.get("token_overhead_ms", DEFAULT_OVERHEAD_MS),
            "effective_bandwidth_gbs": entry.get("effective_bandwidth_gbs")}


def save(device: str, overhead: float, sample: dict,
         effective_bandwidth: float | None = None) -> Path:
    data = _load()
    data[device] = {"token_overhead_ms": round(overhead, 2),
                    "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "sample": sample}
    if effective_bandwidth:
        data[device]["effective_bandwidth_gbs"] = round(effective_bandwidth, 1)
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n")
    return p


def solve(measured_tok_s: float, resident_gb: float, bandwidth_gbs: float) -> float:
    """Back out the fixed cost: 1/tok_s = bytes/bandwidth + overhead."""
    memory_s = resident_gb / bandwidth_gbs if bandwidth_gbs else 0
    return max(0.0, (1 / measured_tok_s - memory_s) * 1000)


def fit(samples: list[tuple[float, float]]) -> dict:
    """Least-squares fit of seconds_per_token = gb / bandwidth + fixed.

    One sample can only solve for the fixed cost given an assumed bandwidth; two or
    more solve for both, which is the only way to tell a slow card from a heavy
    constant. Returns the fit plus whether it is physically credible — a negative
    fixed cost means the linear model does not describe this machine, and encoding it
    anyway would produce confident nonsense.
    """
    pts = sorted({(round(gb, 3), spt) for gb, spt in samples if gb > 0 and spt > 0})
    if len(pts) < 2:
        return {"ok": False, "reason": "need two differently sized models to fit both terms",
                "samples": pts}
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    denom = n * sxx - sx * sx
    if denom == 0:
        return {"ok": False, "reason": "all samples are the same size", "samples": pts}
    slope = (n * sxy - sx * sy) / denom            # seconds per GB
    intercept = (sy - slope * sx) / n              # seconds of fixed cost
    if slope <= 0:
        return {"ok": False, "reason": "larger models measured faster than smaller ones",
                "samples": pts}
    out = {"ok": intercept >= 0, "effective_bandwidth_gbs": 1 / slope,
           "token_overhead_ms": intercept * 1000, "samples": pts}
    if intercept < 0:
        out["reason"] = ("the fit implies a negative fixed cost, so decode time here is not "
                         "bandwidth + constant — something else is throttling this machine")
    return out
