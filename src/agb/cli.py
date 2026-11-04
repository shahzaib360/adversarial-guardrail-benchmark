"""Command-line interface: ``agb {data,prepare,plan,score,collect,analyze,ci-settings}``."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

from .config import load_config


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _csv(value: str | None) -> list[str] | None:
    items = [v.strip() for v in (value or "").split(",") if v.strip()]
    return items or None


def cmd_data(a) -> None:
    from collections import Counter

    from .data import build_items

    cfg = load_config(a.config, a.profile)
    for track, tcfg in cfg["tracks"].items():
        items = build_items(track, tcfg, seed=int(cfg["seed"]))
        print(f"{track}: {len(items)} items", dict(Counter((i.source, i.label) for i in items)))


def cmd_prepare(a) -> None:
    from .pipeline import prepare

    cfg = load_config(a.config, a.profile)
    prepare(cfg, a.out, neural=not a.no_neural, only_transforms=_csv(a.transforms), log=_log)


def cmd_plan_neural(a) -> None:
    from .pipeline import plan_neural

    cfg = load_config(a.config, a.profile)
    print(json.dumps(plan_neural(cfg, chars_per_shard=a.chars_per_shard), separators=(",", ":")))


def cmd_neural(a) -> None:
    from .pipeline import prepare_neural

    cfg = load_config(a.config, a.profile)
    prepare_neural(cfg, a.track, a.transform, a.shard, a.nshards, a.out, log=_log)


def cmd_merge(a) -> None:
    from .pipeline import merge_inputs

    cfg = load_config(a.config, a.profile)
    merge_inputs(cfg, a.base, a.parts, a.out, log=_log)


def cmd_plan(a) -> None:
    from .pipeline import plan

    cfg = load_config(a.config, a.profile)
    p = plan(
        cfg, models=_csv(a.models), include_optional=a.include_optional, check_access=not a.no_access_check, log=_log
    )
    if a.out_json:
        pathlib.Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out_json).write_text(json.dumps(p, indent=2))
    print(json.dumps(p["matrix"], separators=(",", ":")))


def cmd_score(a) -> None:
    from .scoring import run_scoring

    cfg = load_config(a.config, a.profile)
    mcfg = cfg["models"][a.model]
    out = a.out or f"scores/{a.model}.shard-{a.shard}.jsonl"
    meta = run_scoring(
        a.inputs,
        a.model,
        mcfg,
        out,
        shard=a.shard,
        nshards=a.nshards,
        max_seconds=a.max_seconds,
        revision=a.revision or None,
        log=_log,
    )
    _log(json.dumps({k: v for k, v in meta.items() if k != "classifier"}))


def cmd_collect(a) -> None:
    from .pipeline import collect

    collect(a.inputs, a.scores_dir, a.out, plan_path=a.plan, log=_log)


def cmd_analyze(a) -> None:
    from .report import analyze

    cfg = load_config(a.config, a.profile)
    analyze(a.raw, a.out, cfg, log=_log)


def cmd_readme(a) -> None:
    from .report import update_readme

    cfg = load_config(a.config, a.profile)
    update_readme(a.readme, a.results, cfg)
    _log(f"[readme] updated the results block in {a.readme}")


def cmd_ci_settings(a) -> None:
    """Resolve benchmark settings for GitHub Actions (workflow_dispatch inputs or .github/benchmark-run.yaml)."""
    import yaml

    settings = {"profile": "smoke", "models": "", "include_optional": "false", "push_results": "true"}
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    event = json.loads(pathlib.Path(event_path).read_text()) if event_path and os.path.exists(event_path) else {}
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        for k, v in (event.get("inputs") or {}).items():
            if k in settings and v is not None:
                settings[k] = str(v).lower() if isinstance(v, bool) else str(v)
    elif os.path.exists(a.run_file):
        data = yaml.safe_load(pathlib.Path(a.run_file).read_text()) or {}
        for k, v in data.items():
            if k in settings and v is not None:
                if isinstance(v, bool):
                    v = str(v).lower()
                elif isinstance(v, list):
                    v = ",".join(map(str, v))
                settings[k] = str(v)
    for k, v in settings.items():
        print(f"{k}={v}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="agb", description="Adversarial guardrail benchmark")
    ap.add_argument("--config", default=None, help="path to benchmark.yaml (default: configs/benchmark.yaml)")
    ap.add_argument("--profile", default="full", help="config profile: smoke | full")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("data", help="download sources and print item counts").set_defaults(fn=cmd_data)

    p = sub.add_parser("prepare", help="build items and apply all transforms -> inputs.jsonl")
    p.add_argument("--out", default="work/inputs.jsonl")
    p.add_argument("--no-neural", action="store_true", help="skip paraphrase/translation transforms")
    p.add_argument("--transforms", default=None, help="comma-separated subset of transforms")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("plan-neural", help="emit the GitHub Actions matrix of neural-transform shards")
    p.add_argument("--chars-per-shard", type=int, default=40_000)
    p.set_defaults(fn=cmd_plan_neural)

    p = sub.add_parser("neural", help="apply one neural transform to one shard of a track's items")
    p.add_argument("--track", required=True)
    p.add_argument("--transform", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_neural)

    p = sub.add_parser("merge", help="append neural-transform shards to the deterministic inputs")
    p.add_argument("--base", default="work/inputs.jsonl")
    p.add_argument("--parts", default="neural")
    p.add_argument("--out", default="work/inputs.jsonl")
    p.set_defaults(fn=cmd_merge)

    p = sub.add_parser("plan", help="emit the GitHub Actions matrix of model shards")
    p.add_argument("--models", default=None, help="comma-separated model names (default: all non-optional)")
    p.add_argument("--include-optional", action="store_true")
    p.add_argument("--no-access-check", action="store_true")
    p.add_argument("--out-json", default=None)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("score", help="score (a shard of) the inputs with one model")
    p.add_argument("--inputs", default="work/inputs.jsonl")
    p.add_argument("--model", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--out", default=None)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--revision", default=None)
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("collect", help="merge score shards into publishable raw files (no texts)")
    p.add_argument("--inputs", default="work/inputs.jsonl")
    p.add_argument("--scores-dir", default="scores")
    p.add_argument("--out", default="results/raw")
    p.add_argument("--plan", default=None)
    p.set_defaults(fn=cmd_collect)

    p = sub.add_parser("analyze", help="compute metrics, figures and REPORT.md from raw results")
    p.add_argument("--raw", default="results/raw")
    p.add_argument("--out", default="results")
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("readme", help="refresh the results block of README.md from results/")
    p.add_argument("--readme", default="README.md")
    p.add_argument("--results", default="results")
    p.set_defaults(fn=cmd_readme)

    p = sub.add_parser("ci-settings", help="(CI) resolve workflow settings as key=value lines")
    p.add_argument("--run-file", default=".github/benchmark-run.yaml")
    p.set_defaults(fn=cmd_ci_settings)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
