"""Loads the JSON registries. Override paths with HERD_MODELS / HERD_GPUS."""
import json, os
from pathlib import Path
from functools import lru_cache

HERE = Path(__file__).parent


@lru_cache(maxsize=None)
def _load(name: str, env: str) -> dict:
    path = Path(os.environ.get(env) or HERE / name)
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"Registry not found: {path}. Set ${env} to point at it.")
    except json.JSONDecodeError as e:
        raise SystemExit(f"Registry {path} is not valid JSON: {e}")


def models() -> list[dict]:
    return _load("models.json", "HERD_MODELS")["models"]


def model_defaults() -> dict:
    return _load("models.json", "HERD_MODELS").get("defaults", {})


def find_model(name: str) -> dict | None:
    n = name.lower()
    for m in models():
        if n in (m["name"].lower(), m.get("ollama", "").lower()) or n == m["name"].lower().replace("-", ""):
            return m
    return None


def gpus() -> dict:
    return _load("gpus.json", "HERD_GPUS")


def bandwidth_for(device_name: str, vendor: str) -> tuple[float, bool]:
    """(GB/s, is_exact). Longest matching key wins so 'rtx 4060 ti' beats 'rtx 4060'."""
    g = gpus()
    name = device_name.lower()
    best = None
    for key, bw in g["gpus"].items():
        if key in name and (best is None or len(key) > len(best[0])):
            best = (key, bw)
    if best:
        return float(best[1]), True
    return float(g["fallback_gbs"].get(vendor, 200)), False
