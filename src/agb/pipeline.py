"""prepare -> plan -> score (see scoring.py) -> collect.

* ``prepare``  builds items, applies every transform, adds the canonicalized view and
               writes ``inputs.jsonl`` (contains texts; a CI-internal artifact, not published).
               In CI the slow neural transforms are split off: ``prepare --no-neural`` +
               ``plan_neural`` -> ``prepare_neural`` shards in parallel -> ``merge_inputs``.
* ``plan``     decides which models/shards to run (skipping gated models without access)
               and emits a GitHub Actions matrix.
* ``collect``  merges score shards into publishable files that contain *no texts*:
               ``samples.csv.gz`` (metadata + text hashes) and ``scores/<model>.csv.gz``.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import pathlib
import time
from collections import Counter

from .canon import canonicalize
from .data import build_items
from .transforms import REGISTRY, build_samples


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------ prepare
def _neural_backend(cfg: dict, log=print):
    try:
        from .transforms.neural import NeuralTransforms
    except ImportError as e:
        log(f"[prepare] neural transforms unavailable ({e}); install the 'ml' extra to enable them")
        return None
    n = cfg.get("neural", {})
    return NeuralTransforms(
        paraphrase_model=n.get("paraphrase_model", "humarin/chatgpt_paraphraser_on_T5_base"),
        translation_model=n.get("translation_model", "facebook/nllb-200-distilled-600M"),
        similarity_model=n.get("similarity_model", "sentence-transformers/LaBSE"),
        paraphrase_min_sim=float(n.get("paraphrase_min_sim", 0.80)),
        sentence_min_sim=float(n.get("sentence_min_sim", 0.75)),
        translation_min_sim=float(n.get("translation_min_sim", 0.70)),
        log=log,
    )


def _write_samples(samples, out: pathlib.Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict() if hasattr(s, "to_dict") else s, ensure_ascii=False) + "\n")


def _summary(cfg: dict, rows: list[dict]) -> dict:
    out: dict = {"profile": cfg.get("profile"), "seed": cfg["seed"], "tracks": {}, "n_samples": len(rows)}
    for track in cfg["tracks"]:
        tr = [r for r in rows if r["track"] == track]
        clean = [r for r in tr if r["transform"] == "none"]
        out["tracks"][track] = {
            "items": dict(Counter(r["source"] for r in clean)),
            "transforms": dict(Counter(r["transform"] for r in tr)),
            "invalid": dict(Counter(r["transform"] for r in tr if not r["valid"])),
        }
    return out


def prepare(cfg: dict, out: str | pathlib.Path, neural: bool = True, only_transforms=None, log=print) -> dict:
    """Build every sample of every track (optionally without the neural transforms)."""
    t0 = time.time()
    needs_neural = any(REGISTRY[t].fn is None for tr in cfg["tracks"].values() for t in tr["transforms"])
    backend = _neural_backend(cfg, log) if neural and needs_neural else None
    all_samples = []
    for track, tcfg in cfg["tracks"].items():
        items = build_items(track, tcfg, seed=int(cfg["seed"]))
        names = [t for t in tcfg["transforms"] if not only_transforms or t in only_transforms]
        log(f"[prepare] {track}: {len(items)} items x {len(names)} transforms")
        samples = build_samples(items, names, seed=int(cfg["seed"]), neural=backend, log=log)
        for s in samples:
            s.canon_text = canonicalize(s.text)
        all_samples.extend(samples)
    out = pathlib.Path(out)
    _write_samples(all_samples, out)
    summary = _summary(cfg, [s.to_dict() for s in all_samples])
    summary["seconds"] = round(time.time() - t0, 1)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"[prepare] wrote {len(all_samples)} samples to {out} in {summary['seconds']}s")
    return summary


def plan_neural(cfg: dict, chars_per_shard: int = 40_000) -> dict:
    """GitHub matrix for the neural transforms, sharded by the amount of text to process."""
    include = []
    for track, tcfg in cfg["tracks"].items():
        neural = [t for t in tcfg["transforms"] if REGISTRY[t].fn is None]
        if not neural:
            continue
        chars = sum(len(i.text) for i in build_items(track, tcfg, seed=int(cfg["seed"])))
        for t in neural:
            n = max(1, math.ceil(chars / chars_per_shard))
            include += [{"track": track, "transform": t, "shard": k, "nshards": n} for k in range(n)]
    return {"include": include}


def prepare_neural(
    cfg: dict, track: str, transform: str, shard: int, nshards: int, out: str | pathlib.Path, log=print
) -> int:
    """Apply one neural transform to one shard of a track's items."""
    backend = _neural_backend(cfg, log)
    if backend is None:
        raise RuntimeError("neural transforms need the 'ml' extra (torch, transformers, sentence-transformers)")
    items = build_items(track, cfg["tracks"][track], seed=int(cfg["seed"]))[shard::nshards]
    t0 = time.time()
    samples = build_samples(items, [transform], seed=int(cfg["seed"]), neural=backend, log=log)
    for s in samples:
        s.canon_text = canonicalize(s.text)
    _write_samples(samples, pathlib.Path(out))
    log(
        f"[neural] {track}/{transform} shard {shard}/{nshards}: {len(samples)} samples "
        f"({sum(not s.valid for s in samples)} invalid) in {time.time() - t0:.0f}s"
    )
    return len(samples)


def merge_inputs(
    cfg: dict, base: str | pathlib.Path, parts_dir: str | pathlib.Path, out: str | pathlib.Path, log=print
) -> dict:
    """Append neural-transform shards to the deterministic inputs (de-duplicated by uid)."""
    rows: dict[str, dict] = {}
    with open(base, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[r["uid"]] = r
    n_base = len(rows)
    parts = sorted(pathlib.Path(parts_dir).rglob("*.jsonl")) if pathlib.Path(parts_dir).exists() else []
    for p in parts:
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows[r["uid"]] = r
    out = pathlib.Path(out)
    _write_samples(list(rows.values()), out)
    summary = _summary(cfg, list(rows.values()))
    summary["neural_parts"] = [p.name for p in parts]
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"[merge] {n_base} deterministic + {len(rows) - n_base} neural samples from {len(parts)} part(s) -> {out}")
    return summary


# ------------------------------------------------------------------------------ plan
def _hf_access(model_id: str) -> tuple[bool, str | None, str]:
    """(accessible, resolved revision sha, reason)."""
    try:
        from huggingface_hub import HfApi, hf_hub_download

        info = HfApi().model_info(model_id)
        hf_hub_download(model_id, "config.json", revision=info.sha)
        return True, info.sha, "ok"
    except Exception as e:  # noqa: BLE001 - gated / missing / network
        return False, None, f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"


def plan(
    cfg: dict, models: list[str] | None = None, include_optional: bool = False, check_access: bool = True, log=print
) -> dict:
    selected, skipped, revisions = [], {}, {}
    for name, m in cfg["models"].items():
        if models and name not in models:
            continue
        if m.get("optional") and not include_optional and not (models and name in models):
            continue
        if check_access and m["backend"].startswith("hf_"):
            ok, sha, reason = _hf_access(m["model_id"])
            if not ok:
                hint = (
                    " (gated: accept the license on Hugging Face and add an HF_TOKEN secret)" if m.get("gated") else ""
                )
                skipped[name] = reason + hint
                log(f"[plan] skipping {name}: {reason}{hint}")
                continue
            revisions[name] = sha
        selected.append(name)
    include = []
    for name in selected:
        m = cfg["models"][name]
        n = int(m.get("shards", 1))
        for k in range(n):
            include.append(
                {
                    "model": name,
                    "backend": m["backend"],
                    "model_id": m.get("model_id", ""),
                    "shard": k,
                    "nshards": n,
                    "revision": revisions.get(name) or "",
                }
            )
    return {"matrix": {"include": include}, "selected": selected, "skipped": skipped, "revisions": revisions}


# ------------------------------------------------------------------------------ collect
def collect(
    inputs: str | pathlib.Path,
    scores_dir: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    plan_path: str | pathlib.Path | None = None,
    log=print,
) -> dict:
    out_dir = pathlib.Path(out_dir)
    (out_dir / "scores").mkdir(parents=True, exist_ok=True)

    # samples.csv.gz: everything except the texts themselves.
    n = 0
    with (
        open(inputs, encoding="utf-8") as fin,
        gzip.open(out_dir / "samples.csv.gz", "wt", newline="", encoding="utf-8") as fout,
    ):
        w = csv.writer(fout)
        w.writerow(
            [
                "uid",
                "item_id",
                "track",
                "source",
                "label",
                "category",
                "transform",
                "family",
                "valid",
                "h",
                "hc",
                "n_chars",
                "sim",
            ]
        )
        for line in fin:
            s = json.loads(line)
            canon = s.get("canon_text") or s["text"]
            w.writerow(
                [
                    s["uid"],
                    s["item_id"],
                    s["track"],
                    s["source"],
                    s["label"],
                    s["category"],
                    s["transform"],
                    s["family"],
                    int(bool(s["valid"])),
                    _sha1(s["text"]),
                    _sha1(canon),
                    len(s["text"]),
                    (s.get("info") or {}).get("sim", ""),
                ]
            )
            n += 1

    runs: dict = {}
    scores_dir = pathlib.Path(scores_dir)
    by_model: dict[str, list[pathlib.Path]] = {}
    for p in sorted(scores_dir.rglob("*.jsonl")):
        by_model.setdefault(p.name.split(".shard-")[0], []).append(p)
    for model, files in by_model.items():
        rows: dict[str, dict] = {}
        metas = []
        for p in files:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        rows[r["h"]] = r
            mp = p.with_suffix(".meta.json")
            if mp.exists():
                metas.append(json.loads(mp.read_text()))
        variants = sorted({v for r in rows.values() for v in r["s"]})
        with gzip.open(out_dir / "scores" / f"{model}.csv.gz", "wt", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["h", *[f"s_{v}" for v in variants], "out"])
            for h in sorted(rows):
                r = rows[h]
                w.writerow([h, *[r["s"].get(v, "") for v in variants], r.get("o", "")])
        runs[model] = {
            "n_scored": len(rows),
            "shards_found": sorted(m["shard"] for m in metas),
            "nshards": metas[0]["nshards"] if metas else None,
            "complete": bool(metas)
            and all(m["complete"] for m in metas)
            and len(metas) == (metas[0]["nshards"] if metas else 0),
            "ms_per_text": round(sum(m["ms_per_text"] for m in metas) / len(metas), 1) if metas else None,
            "continuous": all(m.get("continuous", True) for m in metas) if metas else None,
            "classifier": metas[0]["classifier"] if metas else {},
        }
        log(f"[collect] {model}: {len(rows)} scored texts from {len(files)} file(s)")

    manifest = {
        "n_samples": n,
        "runs": runs,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "github_run": os.environ.get("GITHUB_RUN_ID"),
    }
    if plan_path and pathlib.Path(plan_path).exists():
        p = json.loads(pathlib.Path(plan_path).read_text())
        manifest["skipped"] = p.get("skipped", {})
        manifest["revisions"] = p.get("revisions", {})
    summary_path = pathlib.Path(inputs).with_suffix(".summary.json")
    if summary_path.exists():
        manifest["prepare"] = json.loads(summary_path.read_text())
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest
