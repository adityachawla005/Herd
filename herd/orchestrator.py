"""Phase 3: route a task to a specialist agent, on a VRAM budget.

The router is a small model that stays pinned in VRAM. It classifies the task and
scores its difficulty; the orchestrator then picks a specialist, asks the scheduler
for that model (loading and evicting as needed), and streams the answer. When the
task is hard and nothing local runs it fast enough, it goes to the cloud instead.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterator

import httpx
import yaml

from . import backends, calibration, ollama, registry
from .errors import HerdError
from .hardware import detect
from .scheduler import ModelScheduler
from .vram import all_fits

CONFIG = Path(__file__).parent / "agents.yaml"
STATS = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")) / "herd" / "stats.json"


def load_config(path: str | None = None) -> dict:
    p = Path(path or os.environ.get("HERD_AGENTS") or CONFIG)
    try:
        cfg = yaml.safe_load(p.read_text()) or {}
    except FileNotFoundError:
        raise RuntimeError(f"No agent config at {p}. Copy herd/agents.yaml there, or set $HERD_AGENTS.")
    except yaml.YAMLError as e:
        raise RuntimeError(f"{p} is not valid YAML: {e}")
    if not cfg.get("agents"):
        raise RuntimeError(f"{p} defines no agents.")
    return cfg


class Orchestrator:
    def __init__(self, config: dict | None = None, on_event=None):
        self.cfg = config or load_config()
        self.agents = self.cfg["agents"]
        self.routing = self.cfg.get("routing", {})
        self.hw = detect()
        self.events: list[dict] = []
        self._sink = on_event or (lambda e: None)
        self.default_backend = (self.cfg.get("backend")
                                or backends.first_available() or backends.DEFAULT)
        # One scheduler per backend: each tracks its own residency and eviction rules.
        self._schedulers: dict[str, ModelScheduler] = {}
        for name, spec in self.agents.items():
            if spec.get("pinned"):
                backend, tag, _ = self.resolve(name)
                if tag and not tag.startswith("cloud/"):
                    self._sched(backend, pin=tag)

    # --- model resolution -------------------------------------------------

    def _sched(self, backend: str, pin: str | None = None) -> ModelScheduler:
        """The scheduler for one backend, created on first use."""
        s = self._schedulers.get(backend)
        if s is None:
            s = ModelScheduler(self.hw, pinned=(pin,) if pin else (),
                               on_event=self._emit_raw, backend=backend)
            self._schedulers[backend] = s
        elif pin:
            s.pinned.add(pin)
        return s

    def agent_backend(self, agent: str) -> str:
        return self.agents[agent].get("backend") or self.default_backend

    def _installed(self, backend: str) -> dict[str, str]:
        try:
            index = {}
            for model_id in backends.get(backend).list_models():
                if model_id:
                    index[model_id] = model_id
                    index.setdefault(model_id.split(":")[0], model_id)
            return index
        except HerdError:
            return {}

    def resolve(self, agent: str) -> tuple[str, str | None, str]:
        """(backend, model id, why). A 'cloud/...' id means off-machine."""
        spec = self.agents[agent]
        backend = self.agent_backend(agent)
        have = self._installed(backend)
        for key in ("model", "fallback"):
            want = spec.get(key)
            if not want:
                continue
            if want.startswith("cloud/"):
                return backend, want, f"{key}: routed to {want.split('/', 1)[1]}"
            tag = have.get(want) or have.get(want.split(":")[0])
            if tag:
                return backend, tag, ("preferred model" if key == "model"
                                      else f"fallback — {spec['model']} is not available")
        return (backend, None,
                f"neither {spec.get('model')} nor {spec.get('fallback')} is on the "
                f"{backends.spec(backend).label} backend")

    def _any_local(self, agent: str) -> tuple[str, str, str]:
        """Last resort for --local: the best model that is both pulled and runnable,
        whatever the agent asked for. The role prompt still applies."""
        from .agent import candidates
        # Any backend will do here — prefer the agent's own, then whatever else works.
        order = [self.agent_backend(agent)] + [b["id"] for b in backends.survey() if b["available"]]
        for backend in dict.fromkeys(order):
            have = self._installed(backend)
            if not have:
                continue
            for fit in candidates(self.hw, agent if agent in ("code", "fast") else "chat", backend=backend):
                tag = have.get(fit.model_ref) or have.get(fit.model_ref.split(":")[0])
                if tag:
                    return backend, tag, (f"--local: {self.agents[agent].get('model')} isn't "
                                          f"available, running the {agent} role on {tag}")
            tag = sorted(set(have.values()))[0]
            return backend, tag, f"--local: nothing ideal is available, falling back to {tag}"
        spec = self.agents[agent]
        raise HerdError(
            "--local was requested but no local models are available on any backend.\n"
            f"  ollama pull {spec.get('model')}   (or edit herd/agents.yaml)")

    def local_speed(self, tag: str, backend: str | None = None) -> float:
        """Estimated tok/s for a model id on a backend, using the calibrated fit math."""
        backend = backend or self.default_backend
        base = tag.split(":")[0]
        for f in all_fits(self.hw, backend=backend):
            if f.model_ref.split(":")[0] == base and f.quant == "Q4":
                return f.tokens_per_sec
        return 0.0

    # --- routing ----------------------------------------------------------

    def route(self, task: str) -> dict:
        """Ask the pinned router model which specialist should take this."""
        specialists = [a for a in self.agents if a != "router"]
        backend, tag, why = self.resolve("router")
        fallback = {"agent": self._keyword_route(task, specialists),
                    "complexity": 5, "reason": "keyword fallback", "router_model": None}
        if not tag or tag.startswith("cloud/"):
            return fallback

        sched = self._sched(backend, pin=tag)
        sched.acquire(tag, "router")
        prompt = (f"Task: {task}\n\nSpecialists: {', '.join(specialists)}\n"
                  "Reply with JSON only.")
        try:
            out = "".join(c.get("response", "") for c in backends.get(backend).generate(
                tag, prompt, self.agents["router"]["role"],
                options={"temperature": 0, "num_predict": 80, "format": "json"}))
            parsed = json.loads(out[out.find("{"):out.rfind("}") + 1])
            agent = str(parsed.get("agent", "")).strip().lower()
            if agent not in specialists:
                agent = self._keyword_route(task, specialists)
            return {"agent": agent,
                    "complexity": max(1, min(10, int(float(parsed.get("complexity", 5))))),
                    "reason": str(parsed.get("reason", ""))[:120], "router_model": tag}
        except (json.JSONDecodeError, ValueError, TypeError, IndexError):
            fallback["reason"] = "router returned unusable JSON — fell back to keywords"
            fallback["router_model"] = tag
            return fallback
        except HerdError as e:
            fallback["reason"] = f"router unavailable ({e})"
            return fallback
        finally:
            sched.release(tag, "router")

    @staticmethod
    def _keyword_route(task: str, specialists: list[str]) -> str:
        from .agent import classify
        kind = classify(task)
        for want in {"code": ["coder"], "reasoning": ["reasoner"],
                     "fast": ["summarizer"], "chat": []}.get(kind, []):
            if want in specialists:
                return want
        return specialists[0] if specialists else "reasoner"

    def should_use_cloud(self, decision: dict, tag: str | None,
                         backend: str | None = None) -> tuple[bool, str]:
        threshold = self.routing.get("complexity_threshold", 7)
        min_tok_s = self.routing.get("min_tok_s", 8)
        if tag and tag.startswith("cloud/"):
            return True, f"the agent's only available model is {tag}"
        if not tag:
            return True, "no local model for this agent is pulled"
        speed = self.local_speed(tag, backend)
        if decision["complexity"] >= threshold and 0 < speed < min_tok_s:
            return True, (f"complexity {decision['complexity']}/10 and the best local fit runs "
                          f"at ~{speed:.0f} tok/s (floor is {min_tok_s})")
        return False, ""

    # --- execution --------------------------------------------------------

    def run(self, task: str, agent: str | None = None, force_local: bool = False) -> Iterator[dict]:
        started = time.perf_counter()
        if agent and agent not in self.agents:
            raise RuntimeError(f"No agent named {agent!r}. Defined: {', '.join(self.agents)}")

        decision = ({"agent": agent, "complexity": 5, "reason": "you named the agent",
                     "router_model": None} if agent else self.route(task))
        for e in self._drain():                       # whatever loading the router needed
            yield e
        yield {"event": "route", **decision}

        backend, tag, why = self.resolve(decision["agent"])
        use_cloud, cloud_reason = self.should_use_cloud(decision, tag, backend)
        if use_cloud and force_local:
            if not tag or tag.startswith("cloud/"):
                backend, tag, why = self._any_local(decision["agent"])
            else:
                why += " (--local: staying on this machine despite the routing rule)"
            use_cloud = False

        if use_cloud:
            yield {"event": "dispatch", "agent": decision["agent"], "destination": "cloud",
                   "model": self.routing.get("cloud", {}).get("model"), "reason": cloud_reason}
            yield from self._run_cloud(task, decision, started)
            return

        yield {"event": "dispatch", "agent": decision["agent"], "destination": "local",
               "model": tag, "backend": backend, "reason": why}
        yield from self._run_local(task, decision, tag, backend, started)

    def _run_local(self, task: str, decision: dict, tag: str, backend: str,
                   started: float) -> Iterator[dict]:
        name = decision["agent"]
        sched = self._sched(backend)
        before = tag in sched.resident
        sched.acquire(tag, name)
        for e in self._drain():
            yield e
        try:
            tokens = 0
            for chunk in backends.get(backend).generate(tag, task,
                                                        self.agents[name].get("role")):
                if text := chunk.get("response", ""):
                    tokens += 1
                    yield {"event": "token", "text": text}
                if chunk.get("done"):
                    ns = chunk.get("eval_duration") or 0
                    count = chunk.get("eval_count", tokens)
                    stats = {"tokens": count,
                             "tokens_per_sec": round(count / (ns / 1e9), 1) if ns else None,
                             "cache_hit": before,
                             "total_ms": round((time.perf_counter() - started) * 1000)}
                    yield {"event": "done", "destination": "local", "agent": name, "model": tag,
                           "backend": backend, "stats": stats,
                           "totals": self._record("local", count,
                                                  detail={"agent": name, "model": tag, **stats})}
        finally:
            sched.release(tag, name)

    def _run_cloud(self, task: str, decision: dict, started: float) -> Iterator[dict]:
        cloud = self.routing.get("cloud") or {}
        key = os.environ.get(cloud.get("api_key_env", ""), "")
        if not key:
            raise RuntimeError(
                f"This task should go to the cloud ({cloud.get('model')}), but "
                f"${cloud.get('api_key_env')} is not set.\n"
                "  Set it, or re-run with --local to force the local model anyway.")
        role = self.agents[decision["agent"]].get("role", "")
        text, usage = _cloud_call(cloud, key, role, task)
        yield {"event": "token", "text": text}
        cost = (usage.get("in", 0) * cloud.get("usd_per_mtok_in", 0)
                + usage.get("out", 0) * cloud.get("usd_per_mtok_out", 0)) / 1e6
        yield {"event": "done", "destination": "cloud", "agent": decision["agent"],
               "model": cloud.get("model"),
               "stats": {"tokens": usage.get("out"), "cost_usd": round(cost, 5),
                         "total_ms": round((time.perf_counter() - started) * 1000)},
               "totals": self._record("cloud", usage.get("out", 0), cost,
                                      detail={"agent": decision["agent"], "cost_usd": round(cost, 5),
                                              "model": cloud.get("model")})}

    # --- bookkeeping ------------------------------------------------------

    def _emit_raw(self, event: dict) -> None:
        """Scheduler events are queued, not pushed — run() interleaves them with tokens
        so the caller sees one ordered stream instead of two."""
        self.events.append(event)
        self._sink(event)

    def _drain(self) -> list[dict]:
        out, self.events = self.events, []
        return out

    def _record(self, destination: str, tokens: int, cost: float = 0.0,
                detail: dict | None = None) -> dict:
        """Running local-vs-cloud tally, so the savings claim is measured, not assumed."""
        try:
            totals = json.loads(STATS.read_text())
        except (OSError, json.JSONDecodeError):
            totals = {"local_tasks": 0, "cloud_tasks": 0, "local_tokens": 0,
                      "cloud_tokens": 0, "cloud_spend_usd": 0.0}
        totals[f"{destination}_tasks"] = totals.get(f"{destination}_tasks", 0) + 1
        totals[f"{destination}_tokens"] = totals.get(f"{destination}_tokens", 0) + (tokens or 0)
        totals["cloud_spend_usd"] = round(totals.get("cloud_spend_usd", 0.0) + cost, 5)

        d = registry.model_defaults()
        per_call = (d["typical_call_tokens_in"] * d["cloud_usd_per_mtok_in"]
                    + d["typical_call_tokens_out"] * d["cloud_usd_per_mtok_out"]) / 1e6
        totals["estimated_saved_usd"] = round(totals["local_tasks"] * per_call, 4)
        # A short history so the dashboard can show what was routed where.
        recent = totals.get("recent", [])
        recent.insert(0, {"t": time.time(), "destination": destination, **(detail or {})})
        totals["recent"] = recent[:20]
        try:
            STATS.parent.mkdir(parents=True, exist_ok=True)
            STATS.write_text(json.dumps(totals, indent=2) + "\n")
        except OSError:
            pass
        return totals

    def status(self) -> dict:
        agents = {}
        for name in self.agents:
            backend, tag, why = self.resolve(name)
            sched = self._schedulers.get(backend)
            cloud = bool(tag and tag.startswith("cloud/"))
            agents[name] = {"tag": tag, "backend": backend, "why": why,
                            "pinned": bool(self.agents[name].get("pinned")),
                            "resident": bool(sched and tag in sched.resident) and not cloud,
                            "est_tok_s": (round(self.local_speed(tag, backend), 1)
                                          if tag and not cloud else None)}
        try:
            totals = json.loads(STATS.read_text())
        except (OSError, json.JSONDecodeError):
            totals = {}
        device = self.hw.gpu.name if self.hw.gpu else "cpu"
        # Ensure the default backend has a scheduler so status is never empty.
        self._sched(self.default_backend)
        per_backend = {b: s.status() for b, s in self._schedulers.items()}
        primary = per_backend.get(self.default_backend) or next(iter(per_backend.values()))
        return {"vram": primary, "vram_by_backend": per_backend, "agents": agents,
                "totals": totals, "backend": self.default_backend,
                "backends": backends.survey(),
                "calibrated": calibration.status(device)["measured"],
                "ollama": {"host": ollama.host()}}


def _cloud_call(cloud: dict, key: str, system: str, task: str) -> tuple[str, dict]:
    """One non-streaming call. Cloud latency is small next to a cold local load."""
    provider = cloud.get("provider", "anthropic")
    try:
        if provider == "anthropic":
            r = httpx.post("https://api.anthropic.com/v1/messages", timeout=120,
                           headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                    "content-type": "application/json"},
                           json={"model": cloud["model"], "max_tokens": 2048, "system": system,
                                 "messages": [{"role": "user", "content": task}]})
            r.raise_for_status()
            data = r.json()
            text = "".join(b.get("text", "") for b in data.get("content", []))
            u = data.get("usage", {})
            return text, {"in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0)}
        r = httpx.post("https://api.openai.com/v1/chat/completions", timeout=120,
                       headers={"Authorization": f"Bearer {key}"},
                       json={"model": cloud["model"],
                             "messages": [{"role": "system", "content": system},
                                          {"role": "user", "content": task}]})
        r.raise_for_status()
        data = r.json()
        u = data.get("usage", {})
        return (data["choices"][0]["message"]["content"],
                {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)})
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"{provider} returned {e.response.status_code}: "
                           f"{e.response.text.strip()[:200]}")
    except httpx.RequestError as e:
        raise RuntimeError(f"Could not reach {provider}: {e}")
