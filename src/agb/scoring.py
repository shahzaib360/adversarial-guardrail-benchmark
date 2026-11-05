"""Run one classifier over (a shard of) the prepared inputs.

Identical texts are scored once: many transforms leave benign/clean texts unchanged
after canonicalization, and the canonicalized view is only an extra text to score.
Results are appended to a JSONL file as they are produced, so an interrupted run can
be resumed, and a soft time budget lets CI jobs stop cleanly before their hard timeout.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time

from .classifiers import build_classifier


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_samples(path: str | pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def subsample_items(samples: list[dict], per_source: int) -> set[str]:
    """Deterministic subset of item ids: the first ``per_source`` items of each (track, source) by id hash."""
    groups: dict[tuple[str, str], set[str]] = {}
    for s in samples:
        groups.setdefault((s["track"], s["source"]), set()).add(s["item_id"])
    keep: set[str] = set()
    for ids in groups.values():
        keep.update(sorted(ids, key=lambda i: hashlib.sha1(i.encode()).hexdigest())[:per_source])
    return keep


def texts_for_model(samples: list[dict], model_cfg: dict) -> dict[str, str]:
    tracks = set(model_cfg.get("tracks", []))
    limit = model_cfg.get("max_items_per_source")
    keep = subsample_items(samples, int(limit)) if limit else None
    texts: dict[str, str] = {}
    for s in samples:
        if s["track"] not in tracks or (keep is not None and s["item_id"] not in keep):
            continue
        texts[sha1(s["text"])] = s["text"]
        if model_cfg.get("canon", True) and s.get("canon_text"):
            texts[sha1(s["canon_text"])] = s["canon_text"]
    return texts


def run_scoring(
    inputs: str | pathlib.Path,
    model_name: str,
    model_cfg: dict,
    out_path: str | pathlib.Path,
    shard: int = 0,
    nshards: int = 1,
    max_seconds: float | None = None,
    chunk: int = 64,
    revision: str | None = None,
    log=print,
) -> dict:
    samples = load_samples(inputs)
    texts = texts_for_model(samples, model_cfg)
    keys = sorted(texts)[shard::nshards]
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["h"] for line in f if line.strip()}
    todo = [k for k in keys if k not in done]
    # Group texts of similar length together (helps padding efficiency for batched models).
    todo.sort(key=lambda k: len(texts[k]))
    log(f"[score] {model_name} shard {shard}/{nshards}: {len(keys)} texts, {len(done)} cached, {len(todo)} to do")

    t_load = time.time()
    clf = build_classifier(model_name, model_cfg, revision=revision, log=log)
    load_s = time.time() - t_load
    probe = getattr(clf, "generate_probe", None)
    if callable(probe):
        log(f"[score] probe output for a benign greeting: {probe('Hello, how are you today?')!r}")

    t0 = time.time()
    n_new, stopped_early = 0, False
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), chunk):
            batch = todo[i : i + chunk]
            t_b = time.time()
            res = clf.score_texts([texts[k] for k in batch])
            per_ms = 1000 * (time.time() - t_b) / max(1, len(batch))
            for k, r in zip(batch, res):
                row = {"h": k, "s": {v: round(float(x), 6) for v, x in r["scores"].items()}, "o": r.get("out", "")}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n_new += len(batch)
            elapsed = time.time() - t0
            log(f"[score] {model_name}: {len(done) + n_new}/{len(keys)} ({per_ms:.0f} ms/text, {elapsed:.0f}s)")
            if max_seconds and elapsed > max_seconds:
                stopped_early = True
                log(f"[score] time budget of {max_seconds:.0f}s reached; stopping early")
                break

    meta = {
        "model": model_name,
        "shard": shard,
        "nshards": nshards,
        "n_texts": len(keys),
        "n_scored": len(done) + n_new,
        "complete": not stopped_early and len(done) + n_new >= len(keys),
        "seconds_scoring": round(time.time() - t0, 1),
        "seconds_loading": round(load_s, 1),
        "ms_per_text": round(1000 * (time.time() - t0) / max(1, n_new), 1),
        "continuous": bool(getattr(clf, "continuous", True)),
        "classifier": clf.info(),
    }
    with open(out_path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    clf.close()
    return meta
