"""prepare -> plan -> score (see scoring.py) -> collect.

* ``prepare``  builds items, applies every transform, adds the canonicalized view and
               writes ``inputs.jsonl`` (contains texts; a CI-internal artifact, not published).
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
def prepare(cfg: dict, out: str | pathlib.Path, neural: bool = True, only_transforms=None, log=print) -> dict:
    t0 = time.time()
    backend = None
    needs_neural = any(REGISTRY[t].fn is None for tr in cfg["tracks"].values() for t in tr["transforms"])
    if neural and needs_neural:
        try:
            from .transforms.neural import NeuralTransforms

            n = cfg.get("neural", {})
            backend = NeuralTransforms(
                paraphrase_model=n.get("paraphrase_model", "humarin/chatgpt_paraphraser_on_T5_base"),
                translation_model=n.get("translation_model", "facebook/nllb-200-distilled-600M"),
                similarity_model=n.get("similarity_model", "sentence-transformers/LaBSE"),
                paraphrase_min_sim=float(n.get("paraphrase_min_sim", 0.80)),
                sentence_min_sim=float(n.get("sentence_min_sim", 0.75)),
                translation_min_sim=float(n.get("translation_min_sim", 0.70)),
                log=log,
            )
        except ImportError as e:
            log(f"[prepare] neural transforms unavailable ({e}); install the 'ml' extra to enable them")

    all_samples = []
    summary: dict = {"profile": cfg.get("profile"), "seed": cfg["seed"], "tracks": {}}
    for track, tcfg in cfg["tracks"].items():
        items = build_items(track, tcfg, seed=int(cfg["seed"]))
        names = [t for t in tcfg["transforms"] if not only_transforms or t in only_transforms]
        log(f"[prepare] {track}: {len(items)} items x {len(names)} transforms")
        samples = build_samples(items, names, seed=int(cfg["seed"]), neural=backend, log=log)
        for s in samples:
            s.canon_text = canonicalize(s.text)
        all_samples.extend(samples)
        summary["tracks"][track] = {
            "items": dict(Counter(i.source for i in items)),
            "transforms": sorted({s.transform for s in samples}),
            "invalid": dict(Counter(s.transform for s in samples if not s.valid)),
        }

    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    summary["n_samples"] = len(all_samples)
    summary["seconds"] = round(time.time() - t0, 1)
    with open(out.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"[prepare] wrote {len(all_samples)} samples to {out} in {summary['seconds']}s")
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
