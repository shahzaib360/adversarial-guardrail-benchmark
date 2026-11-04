"""Load ``configs/benchmark.yaml`` and apply a profile (``smoke`` / ``full``)."""

from __future__ import annotations

import copy
import pathlib

import yaml

DEFAULT_CONFIG = pathlib.Path(__file__).resolve().parents[2] / "configs" / "benchmark.yaml"


def load_config(path: str | pathlib.Path | None = None, profile: str | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["profile"] = profile or "full"
    prof = (cfg.get("profiles") or {}).get(cfg["profile"])
    if prof is None:
        raise ValueError(f"unknown profile {cfg['profile']!r}; choose from {list(cfg.get('profiles', {}))}")
    if "n_per_source" in prof:
        for track in cfg["tracks"].values():
            for src in track["sources"].values():
                src["n"] = int(prof["n_per_source"])
    if "shards" in prof:
        for m in cfg["models"].values():
            m["shards"] = int(prof["shards"])
    if "transforms" in prof:
        keep = set(prof["transforms"])
        for track in cfg["tracks"].values():
            track["transforms"] = [t for t in track["transforms"] if t in keep]
    for name, m in cfg["models"].items():
        m.setdefault("label", name)
        m.setdefault("shards", 1)
        m.setdefault("canon", True)
        m.setdefault("variants", {"default": None})
        m.setdefault("primary_track", m["tracks"][0])
    return cfg


def primary_variant(model_cfg: dict) -> str:
    return next(iter(model_cfg["variants"]))
