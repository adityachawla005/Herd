"""JSON backend for the Go CLI. Every command prints one JSON object on stdout.

Errors leave as {"error": ..., "hint": ...} with exit code 1 so the Go side can render
them without parsing tracebacks. Human formatting lives in the Go binary, not here.
"""
from __future__ import annotations

import argparse
import json
import sys


def _emit(obj: dict) -> None:
    json.dump(obj, sys.stdout, default=str)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _fail(msg: str, hint: str = "") -> int:
    _emit({"error": msg, "hint": hint})
    return 1


def cmd_detect(args) -> int:
    from .hardware import detect
    _emit(detect().to_dict())
    return 0


def cmd_recommend(args) -> int:
    from .hardware import detect
    from .recommender import recommend
    _emit(recommend(detect(), args.context, args.kv_quant, args.limit, args.backend))
    return 0


def cmd_optimize(args) -> int:
    from .hardware import detect
    from . import registry, optimizations
    from .vram import estimate
    hw = detect()
    fit = None
    model = None
    if args.model:
        model = registry.find_model(args.model)
        if not model:
            names = ", ".join(m["name"] for m in registry.models())
            return _fail(f"Unknown model {args.model!r}.", f"Registry has: {names}")
        fit = estimate(model, args.quant, hw, args.context, kv_quant=args.kv_quant,
                       backend=args.backend or "ollama")
    tips = optimizations.advise(hw, fit, model, args.backend or "ollama")
    _emit({"hardware": hw.to_dict(), "fit": fit.to_dict() if fit else None,
           "optimizations": [t.to_dict() for t in tips], "notes": hw.notes})
    return 0


def cmd_backends(args) -> int:
    from . import backends
    _emit({"backends": backends.survey(), "default": backends.first_available()})
    return 0


def cmd_models(args) -> int:
    from . import registry
    _emit({"models": registry.models(), "defaults": registry.model_defaults()})
    return 0


def cmd_run(args) -> int:
    """Streams NDJSON: one selection event, then tokens, then stats."""
    from . import agent
    from .errors import HerdError
    try:
        for event in agent.run(args.task, args.task_type, args.model, args.file,
                               args.context, args.keep_alive, args.backend):
            _emit(event)
    except HerdError as e:
        return _fail(str(e), e.hint)
    return 0


def cmd_agent(args) -> int:
    """Streams NDJSON: routing decision, scheduler events, tokens, then totals."""
    from .orchestrator import Orchestrator
    from .errors import HerdError
    try:
        orch = Orchestrator()
        for event in orch.run(args.task, args.agent, args.local):
            _emit(event)
    except HerdError as e:
        return _fail(str(e), e.hint)
    except RuntimeError as e:
        return _fail(str(e))
    return 0


def cmd_status(args) -> int:
    from .orchestrator import Orchestrator
    from .errors import HerdError
    try:
        _emit(Orchestrator().status())
    except HerdError as e:
        return _fail(str(e), e.hint)
    except RuntimeError as e:
        return _fail(str(e))
    return 0


def cmd_bench(args) -> int:
    from . import agent
    from .errors import HerdError
    try:
        _emit(agent.bench(args.model, args.tokens))
    except HerdError as e:
        return _fail(str(e), e.hint)
    return 0


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        return _fail("uvicorn is not installed.", "Run: uv sync   (or: uv pip install -e .)")
    from .server import app
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="herd-backend", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--context", type=int, default=None)
        sp.add_argument("--kv-quant", default="fp16", choices=["fp16", "q8", "q4"])
        sp.add_argument("--backend", default=None,
                        choices=["ollama", "llamacpp", "hf", "openai"])
        return sp

    sub.add_parser("detect").set_defaults(fn=cmd_detect)
    r = common(sub.add_parser("recommend"))
    r.add_argument("--limit", type=int, default=8)
    r.set_defaults(fn=cmd_recommend)
    o = common(sub.add_parser("optimize"))
    o.add_argument("--model", default=None)
    o.add_argument("--quant", default="Q4", choices=["Q4", "Q5", "Q8", "FP16"])
    o.set_defaults(fn=cmd_optimize)
    sub.add_parser("models").set_defaults(fn=cmd_models)
    sub.add_parser("backends").set_defaults(fn=cmd_backends)
    rn = common(sub.add_parser("run"))
    rn.add_argument("task")
    rn.add_argument("--task-type", default=None, choices=["chat", "code", "fast", "reasoning"])
    rn.add_argument("--model", default=None)
    rn.add_argument("--file", default=None)
    rn.add_argument("--keep-alive", default=None)
    rn.set_defaults(fn=cmd_run)
    ag = sub.add_parser("agent")
    ag.add_argument("task")
    ag.add_argument("--agent", default=None)
    ag.add_argument("--local", action="store_true")
    ag.add_argument("--backend", default=None,
                    choices=["ollama", "llamacpp", "hf", "openai"])
    ag.set_defaults(fn=cmd_agent)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    b = sub.add_parser("bench")
    b.add_argument("--model", default=None)
    b.add_argument("--tokens", type=int, default=128)
    b.set_defaults(fn=cmd_bench)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8787)
    s.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        return 130
    except Exception as e:                          # never let a traceback reach the CLI
        import traceback
        if "--traceback" in sys.argv:
            traceback.print_exc()
        return _fail(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
