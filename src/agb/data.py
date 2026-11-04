"""Benchmark items: loading, filtering, de-duplication and deterministic sampling.

Two tracks are built from public data:

* ``harmful`` (content-safety track, the Llama Guard use case): positives are harmful
  requests, negatives are benign requests, including borderline-but-safe prompts that
  superficially resemble harmful ones.
* ``injection`` (prompt-attack track, the Prompt Guard use case): positives are
  in-the-wild jailbreak prompts and prompt-injection / hijacking attacks, negatives are
  regular user prompts (many of them role-play/persona prompts) and ordinary
  instructions.
"""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import asdict, dataclass

from .sources import fetch

csv.field_size_limit(1 << 30)


@dataclass
class Item:
    item_id: str
    track: str
    source: str
    label: int  # 1 = harmful / attack, 0 = benign
    category: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- loaders
# Each loader returns a list of (raw_id, text, category) tuples.


def _read_csv(name: str) -> list[dict]:
    with open(fetch(name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_jsonl(name: str) -> list[dict]:
    with open(fetch(name), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_harmbench_standard() -> list[tuple[str, str, str]]:
    rows = _read_csv("harmbench")
    return [
        (r["BehaviorID"], r["Behavior"].strip(), r["SemanticCategory"])
        for r in rows
        if r["FunctionalCategory"] == "standard"
    ]


def load_xstest_safe() -> list[tuple[str, str, str]]:
    rows = _read_csv("xstest")
    return [(r["id"], r["prompt"].strip(), r["type"]) for r in rows if r["label"] == "safe"]


def load_xstest_unsafe() -> list[tuple[str, str, str]]:
    rows = _read_csv("xstest")
    return [(r["id"], r["prompt"].strip(), r["type"]) for r in rows if r["label"] == "unsafe"]


def load_alpaca() -> list[tuple[str, str, str]]:
    with open(fetch("alpaca"), encoding="utf-8") as f:
        rows = json.load(f)
    # Only self-contained instructions (no separate "input" field).
    return [(str(i), r["instruction"].strip(), "instruction") for i, r in enumerate(rows) if not r["input"].strip()]


def load_jailbreakhub_jailbreak() -> list[tuple[str, str, str]]:
    rows = _read_csv("jailbreakhub_jailbreak")
    return [(str(i), r["prompt"].strip(), r["platform"]) for i, r in enumerate(rows)]


def load_jailbreakhub_regular() -> list[tuple[str, str, str]]:
    jailbreaks = {_norm(r["prompt"]) for r in _read_csv("jailbreakhub_jailbreak")}
    rows = _read_csv("jailbreakhub_regular")
    return [
        (str(i), r["prompt"].strip(), r["platform"]) for i, r in enumerate(rows) if _norm(r["prompt"]) not in jailbreaks
    ]


_WORD = re.compile(r"[A-Za-z]{3,}")


def load_tensortrust() -> list[tuple[str, str, str]]:
    out = []
    for name, cat in (("tensortrust_hijacking", "hijacking"), ("tensortrust_extraction", "extraction")):
        for r in _read_jsonl(name):
            text = r["attack"].strip()
            # Drop pure token-repetition noise: keep attacks that contain real instructions.
            if len({w.lower() for w in _WORD.findall(text)}) < 5:
                continue
            out.append((f"{cat}:{r['sample_id']}", text, cat))
    return out


LOADERS = {
    "harmbench_standard": load_harmbench_standard,
    "xstest_safe": load_xstest_safe,
    "xstest_unsafe": load_xstest_unsafe,
    "alpaca": load_alpaca,
    "jailbreakhub": load_jailbreakhub_jailbreak,
    "jailbreakhub_regular": load_jailbreakhub_regular,
    "tensortrust": load_tensortrust,
}


# --------------------------------------------------------------------------- sampling


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def build_items(track: str, track_cfg: dict, seed: int, n_override: int | None = None) -> list[Item]:
    """Load every source of a track, filter by length, de-duplicate, and sample."""
    min_chars = int(track_cfg.get("min_chars", 1))
    max_chars = int(track_cfg.get("max_chars", 10_000))
    items: list[Item] = []
    seen: set[str] = set()
    for source, scfg in track_cfg["sources"].items():
        loader = LOADERS[scfg.get("loader", source)]
        label = int(scfg["label"])
        n = int(n_override if n_override is not None else scfg["n"])
        candidates = []
        local_seen: set[str] = set()
        for raw_id, text, category in sorted(loader(), key=lambda t: t[0]):
            key = _norm(text)
            if not (min_chars <= len(text) <= max_chars) or key in local_seen or key in seen:
                continue
            local_seen.add(key)
            candidates.append((raw_id, text, category))
        rng = random.Random(f"{seed}:{track}:{source}")
        if n < len(candidates):
            candidates = rng.sample(candidates, n)
        else:
            rng.shuffle(candidates)
        for raw_id, text, category in candidates:
            seen.add(_norm(text))
            items.append(Item(f"{source}:{raw_id}", track, source, label, category, text))
    return items
