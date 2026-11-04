"""Turn raw results (samples + scores, no texts) into metric tables, figures and REPORT.md."""

from __future__ import annotations

import json
import math
import pathlib

import pandas as pd

from .config import primary_variant
from .metrics import attach_scores, per_source, per_transform, summary, threshold_at_fpr
from .transforms import REGISTRY


# ------------------------------------------------------------------------------ compute
def compute(raw_dir: str | pathlib.Path, cfg: dict) -> dict[str, pd.DataFrame]:
    raw_dir = pathlib.Path(raw_dir)
    samples = pd.read_csv(raw_dir / "samples.csv.gz", dtype={"item_id": str, "category": str})
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    thr = float(cfg.get("threshold", 0.5))
    cal_fpr = float(cfg.get("calibration_fpr", 0.05))
    pt_rows, sm_rows, ps_rows = [], [], []
    for model, mcfg in cfg["models"].items():
        path = raw_dir / "scores" / f"{model}.csv.gz"
        if not path.exists():
            continue
        scores = pd.read_csv(path, dtype={"h": str, "out": str})
        run = manifest.get("runs", {}).get(model, {})
        continuous = bool(run.get("continuous", True))
        variants = [v for v in mcfg["variants"] if f"s_{v}" in scores.columns]
        for variant in variants:
            for track in mcfg["tracks"]:
                tdf = samples[samples["track"] == track]
                if tdf.empty:
                    continue
                df = attach_scores(tdf, scores, variant)
                attacks = [t for t in cfg["tracks"][track]["transforms"] if t != "none"]
                for canon in (False, True):
                    col = "score_canon" if canon else "score"
                    clean_neg = df[(df["transform"] == "none") & (df["label"] == 0) & df[col].notna()][col]
                    cal = threshold_at_fpr(clean_neg, cal_fpr) if continuous and len(clean_neg) else None
                    key = {"model": model, "variant": variant, "primary_variant": variant == primary_variant(mcfg),
                           "canon": canon, "track": track,
                           "primary_track": track == mcfg["primary_track"]}  # fmt: skip
                    pt = per_transform(df, col, thr, cal, continuous)
                    for c, v in reversed(list(key.items())):
                        pt.insert(0, c, v)
                    pt_rows.append(pt)
                    sm = summary(df, col, thr, attacks)
                    sm.update(key)
                    sm["cal_threshold"] = cal
                    sm["continuous"] = continuous
                    sm["complete"] = run.get("complete")
                    sm_rows.append(sm)
                    ps = per_source(df, col, thr)
                    for c, v in reversed(list(key.items())):
                        ps.insert(0, c, v)
                    ps_rows.append(ps)
    per_t = pd.concat(pt_rows, ignore_index=True) if pt_rows else pd.DataFrame()
    summ = pd.DataFrame(sm_rows)
    per_s = pd.concat(ps_rows, ignore_index=True) if ps_rows else pd.DataFrame()
    validity = (
        samples[samples["family"].isin(["paraphrase", "translation"])]
        .assign(sim=lambda d: pd.to_numeric(d["sim"], errors="coerce"))
        .groupby(["track", "transform"])
        .agg(n=("valid", "size"), valid=("valid", "mean"), mean_sim=("sim", "mean"))
        .reset_index()
    )
    return {"per_transform": per_t, "summary": summ, "per_source": per_s, "validity": validity,
            "manifest": manifest}  # fmt: skip


# ------------------------------------------------------------------------------ figures
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "nan": "#f0efec"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "nan": "#2c2c2a"},
}  # fmt: skip
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]  # fmt: skip
ORANGE = ["#fde7dc", "#fbd0bb", "#f8b896", "#f4a07a", "#f0885d", "#eb6834", "#d85a28", "#c04d20", "#a6421b",
          "#8c3716", "#722d12", "#5c240e"]  # fmt: skip


def _cmap(steps: list[str], theme: str):
    from matplotlib.colors import LinearSegmentedColormap

    # Light mode: near-zero is the lightest step (recedes into the light surface).
    # Dark mode: near-zero is the darkest step (recedes into the dark surface).
    seq = steps if theme == "light" else steps[::-1]
    return LinearSegmentedColormap.from_list("seq", seq)


def _style(ax, th):
    ax.set_facecolor(th["surface"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=th["ink2"], length=0, labelsize=8.5)


def heatmap_figure(per_t: pd.DataFrame, cfg: dict, track: str, path: pathlib.Path, theme: str) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    th = THEMES[theme]
    d = per_t[(per_t["track"] == track) & per_t["primary_variant"] & ~per_t["canon"]]
    models = [m for m, c in cfg["models"].items() if c["primary_track"] == track and m in set(d["model"])]
    transforms = [t for t in cfg["tracks"][track]["transforms"] if t != "none" and t in set(d["transform"])]
    if not models or not transforms:
        return False
    rows = ["none", *transforms]
    labels = [cfg["models"][m]["label"] for m in models]
    panel_w = max(2.4, 0.95 * len(models))
    fig, axes = plt.subplots(
        1, 2, figsize=(1.6 + 2 * panel_w, 1.9 + 0.27 * len(rows)), facecolor=th["surface"], gridspec_kw={"wspace": 0.12}
    )
    for ax, metric, steps, title in (
        (axes[0], "miss", BLUE, "Miss rate\nharmful / attack inputs"),
        (axes[1], "fpr", ORANGE, "False-positive rate\nbenign inputs, same transform"),
    ):
        mat = np.full((len(rows), len(models)), np.nan)
        for i, t in enumerate(rows):
            for j, m in enumerate(models):
                v = d[(d["transform"] == t) & (d["model"] == m)][metric]
                if len(v):
                    mat[i, j] = float(v.iloc[0])
        cmap = _cmap(steps, theme).with_extremes(bad=th["nan"])
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        _style(ax, th)
        ax.set_xticks(range(len(models)), labels, rotation=35, ha="right", rotation_mode="anchor", color=th["ink"])
        ax.set_yticks(
            range(len(rows)), [("clean" if r == "none" else r.replace("_", " ")) for r in rows] if ax is axes[0] else []
        )
        ax.set_title(title, color=th["ink"], fontsize=9.5, loc="left", pad=8, linespacing=1.3)
        # 2px surface gaps between cells, and a wider gap under the clean row
        ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color=th["surface"], linewidth=2)
        ax.tick_params(which="minor", length=0)
        ax.axhline(0.5, color=th["surface"], linewidth=5)
        cb = fig.colorbar(
            im,
            ax=ax,
            orientation="horizontal",
            fraction=0.02,
            pad=0.03 + 0.012 * max(map(len, labels)),
            aspect=25,
            shrink=0.8,
        )
        cb.set_ticks([0, 0.5, 1.0], labels=["0%", "50%", "100%"])
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=th["ink2"], labelsize=8, length=0)
    fig.suptitle(cfg["tracks"][track]["title"], color=th["ink"], fontsize=11, x=0.02, ha="left", fontweight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=th["surface"])
    plt.close(fig)
    return True


def canon_figure(summ: pd.DataFrame, cfg: dict, path: pathlib.Path, theme: str) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    th = THEMES[theme]
    light_c, dark_c = ("#86b6ef", "#184f95") if theme == "light" else ("#256abf", "#9ec5f4")
    d = summ[summ["primary_variant"] & summ["primary_track"]]
    order = [(m, c["primary_track"]) for m, c in cfg["models"].items() if m in set(d["model"])]
    if not order:
        return False
    names = [f"{cfg['models'][m]['label']}  ({t})" for m, t in order]
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 1.3 + 0.36 * len(order)), facecolor=th["surface"], sharey=True, gridspec_kw={"wspace": 0.08}
    )
    for ax, metric, title in (
        (axes[0], "best_of_n_evasion", "Best-of-N evasion\nharmful items missed by at least one attack"),
        (axes[1], "any_transform_fpr", "Any-transform FPR\nbenign items flagged under at least one transform"),
    ):
        _style(ax, th)
        for i, (m, _t) in enumerate(order):
            raw = d[(d["model"] == m) & ~d["canon"]][metric]
            can = d[(d["model"] == m) & d["canon"]][metric]
            if not len(raw) or not len(can):
                continue
            a, b = float(raw.iloc[0]), float(can.iloc[0])
            ax.plot([a, b], [i, i], color=th["axis"], linewidth=2, solid_capstyle="round", zorder=1)
            ax.scatter([a], [i], s=64, color=light_c, edgecolor=th["surface"], linewidth=2, zorder=2)
            ax.scatter([b], [i], s=64, color=dark_c, edgecolor=th["surface"], linewidth=2, zorder=3)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(len(order) - 0.5, -0.5)  # first model on top (shared axis: set once, no double inversion)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_yticks(range(len(order)), names, color=th["ink"])
        ax.grid(axis="x", color=th["grid"], linewidth=1)
        ax.set_axisbelow(True)
        ax.set_title(title, color=th["ink"], fontsize=9.5, loc="left", pad=8, linespacing=1.3)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=c,
            markeredgecolor=th["surface"],
            markeredgewidth=2,
            label=lab,
        )
        for c, lab in ((light_c, "raw input"), (dark_c, "with canonicalization"))
    ]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=2, frameon=False, fontsize=8.5)
    for txt in leg.get_texts():
        txt.set_color(th["ink2"])
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=th["surface"])
    plt.close(fig)
    return True


# ------------------------------------------------------------------------------ markdown
def pct(x, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{100 * x:.{digits}f}%"


def _md_table(header: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    align = align or ["l"] + ["r"] * (len(header) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _summary_rows(summ: pd.DataFrame, models: list[str], cfg: dict, canon: bool) -> list[list[str]]:
    rows = []
    for m in models:
        s = summ[(summ["model"] == m) & (summ["canon"] == canon) & summ["primary_variant"]]
        if s.empty:
            continue
        s = s.iloc[0]
        worst = f"{str(s.get('worst_transform', '–')).replace('_', ' ')} ({pct(s.get('worst_miss'))})"
        rows.append([cfg["models"][m]["label"], pct(s["clean_miss"]), pct(s["clean_fpr"]), pct(s["mean_attack_miss"]),
                     pct(s["mean_attack_fpr"]), worst, pct(s["best_of_n_evasion"]), pct(s["any_transform_fpr"])])  # fmt: skip
    return rows


def write_report(results: dict, cfg: dict, out_dir: pathlib.Path, figures: dict[str, str]) -> str:
    per_t, summ, per_s, validity, manifest = (
        results[k] for k in ("per_transform", "summary", "per_source", "validity", "manifest")
    )
    L: list[str] = []
    L.append("# Results\n")
    L.append(
        "_Auto-generated by `agb analyze` from the raw files in `results/raw/` (no prompt texts are "
        "stored there; texts are reproducible from the pinned sources + code)._\n"
    )
    prep = manifest.get("prepare", {})
    L.append(
        f"- Profile: **{prep.get('profile', '?')}** · samples: **{manifest.get('n_samples', '?')}** · "
        f"generated: {manifest.get('created', '?')}"
        + (f" · GitHub Actions run `{manifest['github_run']}`" if manifest.get("github_run") else "")
    )
    thr = cfg.get("threshold", 0.5)
    L.append(
        f"- Decision rule: flag when score ≥ {thr} (for 0/1 verdicts: the model's own verdict). "
        f'"Calibrated" = threshold set for ≤{pct(cfg.get("calibration_fpr", 0.05))} FPR on clean benign inputs.'
    )
    runs = manifest.get("runs", {})
    if runs:
        rows = []
        for m, r in runs.items():
            c = r.get("classifier", {})
            src = c.get("model_id") or c.get("model") or c.get("lexicon", "")
            rev = c.get("revision") or c.get("ollama_version") or ""
            rows.append(
                [
                    cfg["models"].get(m, {}).get("label", m),
                    f"`{src}`",
                    f"`{str(rev)[:12]}`" if rev else "",
                    str(r.get("n_scored")),
                    "yes" if r.get("complete") else "**partial**",
                    "yes" if r.get("continuous") else "no (0/1)",
                    f"{r.get('ms_per_text')}" if r.get("ms_per_text") is not None else "",
                ]
            )
        L.append("\n### Models run\n")
        L.append(
            _md_table(
                [
                    "Model",
                    "Weights",
                    "Revision / server",
                    "Texts scored",
                    "Complete",
                    "Continuous score",
                    "ms / text (CPU)",
                ],
                rows,
                ["l", "l", "l", "r", "l", "l", "r"],
            )
        )
    skipped = manifest.get("skipped") or {}
    if skipped:
        L.append("\n**Not run:** " + "; ".join(f"`{m}` — {why}" for m, why in skipped.items()))

    for track, tcfg in cfg["tracks"].items():
        models = [m for m, c in cfg["models"].items() if c["primary_track"] == track
                  and not summ.empty and m in set(summ["model"])]  # fmt: skip
        if not models:
            continue
        L.append(f"\n## {tcfg['title']}\n")
        srcs = ", ".join(f"`{s}` ({'harmful/attack' if v['label'] else 'benign'})" for s, v in tcfg["sources"].items())
        L.append(f"Sources: {srcs}.\n")
        if figures.get(track):
            L.append(f"![{track} heatmap]({figures[track]})\n")
        hdr = [
            "Model",
            "Clean miss",
            "Clean FPR",
            "Mean miss (attacks)",
            "Mean FPR (attacks)",
            "Worst attack (miss)",
            "Best-of-N evasion",
            "Any-transform FPR",
        ]
        L.append("### Summary\n")
        L.append(_md_table(hdr, _summary_rows(summ[summ["track"] == track], models, cfg, canon=False)))

        # per-transform miss / FPR grid
        d = per_t[(per_t["track"] == track) & per_t["primary_variant"] & ~per_t["canon"]]
        L.append("\n### Miss rate · FPR per transform\n")
        L.append(
            "Each cell: **miss rate** on harmful/attack inputs · FPR on benign inputs with the same "
            "transform. 95% Wilson intervals are in `metrics/per_transform.csv`.\n"
        )
        rows = []
        for t in tcfg["transforms"]:
            r = [f"{t.replace('_', ' ')}" + (" (clean)" if t == "none" else "")]
            for m in models:
                v = d[(d["model"] == m) & (d["transform"] == t)]
                r.append("–" if v.empty else f"**{pct(v['miss'].iloc[0])}** · {pct(v['fpr'].iloc[0])}")
            rows.append(r)
        L.append(_md_table(["Transform", *[cfg["models"][m]["label"] for m in models]], rows))

        # calibrated operating point
        cal_models = [
            m for m in models if not d[(d["model"] == m)].get("miss_cal", pd.Series(dtype=float)).isna().all()
        ]
        if cal_models and "miss_cal" in d.columns:
            L.append(f"\n### At a calibrated threshold (≤{pct(cfg.get('calibration_fpr', 0.05))} clean FPR)\n")
            rows = []
            for t in tcfg["transforms"]:
                r = [t.replace("_", " ")]
                for m in cal_models:
                    v = d[(d["model"] == m) & (d["transform"] == t)]
                    r.append(
                        "–"
                        if v.empty or pd.isna(v["miss_cal"].iloc[0])
                        else f"**{pct(v['miss_cal'].iloc[0])}** · {pct(v['fpr_cal'].iloc[0])}"
                    )
                rows.append(r)
            L.append(_md_table(["Transform", *[cfg["models"][m]["label"] for m in cal_models]], rows))

        # AUROC
        if d["auroc"].notna().any():
            L.append("\n### AUROC per transform (threshold-free separability)\n")
            rows = []
            for t in tcfg["transforms"]:
                r = [t.replace("_", " ")]
                for m in models:
                    v = d[(d["model"] == m) & (d["transform"] == t)]["auroc"]
                    r.append("–" if v.empty or pd.isna(v.iloc[0]) else f"{v.iloc[0]:.2f}")
                rows.append(r)
            L.append(_md_table(["Transform", *[cfg["models"][m]["label"] for m in models]], rows))

        # canonicalization
        L.append("\n### With input canonicalization (defense)\n")
        rows = []
        for m in models:
            a = summ[(summ["model"] == m) & (summ["track"] == track) & summ["primary_variant"] & ~summ["canon"]]
            b = summ[(summ["model"] == m) & (summ["track"] == track) & summ["primary_variant"] & summ["canon"]]
            if a.empty or b.empty:
                continue
            a, b = a.iloc[0], b.iloc[0]
            rows.append(
                [
                    cfg["models"][m]["label"],
                    f"{pct(a['mean_attack_miss'])} → {pct(b['mean_attack_miss'])}",
                    f"{pct(a['best_of_n_evasion'])} → {pct(b['best_of_n_evasion'])}",
                    f"{pct(a['clean_fpr'])} → {pct(b['clean_fpr'])}",
                    f"{pct(a['mean_attack_fpr'])} → {pct(b['mean_attack_fpr'])}",
                    f"{pct(a['any_transform_fpr'])} → {pct(b['any_transform_fpr'])}",
                ]
            )
        L.append(
            _md_table(
                [
                    "Model",
                    "Mean miss (attacks)",
                    "Best-of-N evasion",
                    "Clean FPR",
                    "Mean FPR (attacks)",
                    "Any-transform FPR",
                ],
                rows,
            )
        )

        # FPR by benign source
        ps = per_s[(per_s["track"] == track) & per_s["primary_variant"] & ~per_s["canon"] & (per_s["label"] == 0)]
        if not ps.empty:
            L.append("\n### False-positive rate by benign source\n")
            rows = []
            for m in models:
                for src in [s for s, v in tcfg["sources"].items() if not v["label"]]:
                    g = ps[(ps["model"] == m) & (ps["source"] == src)]
                    clean = g[g["transform"] == "none"]["flag_rate"]
                    att = g[g["transform"] != "none"]["flag_rate"]
                    worst = g[g["transform"] != "none"].sort_values("flag_rate").tail(1)
                    rows.append(
                        [
                            cfg["models"][m]["label"],
                            f"`{src}`",
                            pct(clean.iloc[0]) if len(clean) else "–",
                            pct(att.mean()) if len(att) else "–",
                            f"{worst['transform'].iloc[0].replace('_', ' ')} ({pct(worst['flag_rate'].iloc[0])})"
                            if len(worst)
                            else "–",
                        ]
                    )
            L.append(
                _md_table(
                    ["Model", "Benign source", "Clean FPR", "Mean FPR (attacks)", "Worst transform"],
                    rows,
                    ["l", "l", "r", "r", "l"],
                )
            )

        # secondary variants (e.g. Prompt Guard v1 'any', Qwen3Guard 'loose')
        sv = summ[(summ["track"] == track) & ~summ["primary_variant"] & ~summ["canon"] & summ["model"].isin(models)]
        if not sv.empty:
            L.append("\n### Alternative scoring rules\n")
            rows = []
            for _, s in sv.iterrows():
                rows.append(
                    [
                        f"{cfg['models'][s['model']]['label']} — `{s['variant']}`",
                        pct(s["clean_miss"]),
                        pct(s["clean_fpr"]),
                        pct(s["mean_attack_miss"]),
                        pct(s["mean_attack_fpr"]),
                        pct(s["best_of_n_evasion"]),
                        pct(s["any_transform_fpr"]),
                    ]
                )
            L.append(
                _md_table(
                    [
                        "Model / rule",
                        "Clean miss",
                        "Clean FPR",
                        "Mean miss (attacks)",
                        "Mean FPR (attacks)",
                        "Best-of-N evasion",
                        "Any-transform FPR",
                    ],
                    rows,
                )
            )

        # cross-track models (e.g. Prompt Guard scored on harmful requests)
        others = sorted(set(summ[(summ["track"] == track) & ~summ["primary_track"]]["model"]))
        if others:
            L.append("\n### Out-of-scope detectors on this track (for layered-defense context)\n")
            L.append(_md_table(hdr, _summary_rows(summ[summ["track"] == track], others, cfg, canon=False)))

    if not validity.empty:
        L.append("\n## Validity of semantic transforms\n")
        L.append("Paraphrases/translations below the LaBSE similarity threshold are excluded from all metrics.\n")
        rows = [
            [
                r["track"],
                r["transform"].replace("_", " "),
                str(int(r["n"])),
                pct(r["valid"]),
                f"{r['mean_sim']:.2f}" if pd.notna(r["mean_sim"]) else "–",
            ]
            for _, r in validity.iterrows()
        ]
        L.append(
            _md_table(["Track", "Transform", "Samples", "Valid", "Mean similarity"], rows, ["l", "l", "r", "r", "r"])
        )
    L.append("\n## Transform glossary\n")
    L.append(
        _md_table(
            ["Transform", "Family", "What it does"],
            [[n.replace("_", " "), s.family, s.description] for n, s in REGISTRY.items()],
            ["l", "l", "l"],
        )
    )
    text = "\n".join(L) + "\n"
    (out_dir / "REPORT.md").write_text(text, encoding="utf-8")
    return text


def analyze(raw_dir: str | pathlib.Path, out_dir: str | pathlib.Path, cfg: dict, log=print) -> dict:
    out_dir = pathlib.Path(out_dir)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    results = compute(raw_dir, cfg)
    for name in ("per_transform", "summary", "per_source", "validity"):
        df = results[name]
        df.to_csv(out_dir / "metrics" / f"{name}.csv", index=False, float_format="%.4f")
    figures: dict[str, str] = {}
    if not results["per_transform"].empty:
        for track in cfg["tracks"]:
            ok = False
            for theme in ("light", "dark"):
                ok = (
                    heatmap_figure(
                        results["per_transform"],
                        cfg,
                        track,
                        out_dir / "figures" / f"heatmap_{track}_{theme}.png",
                        theme,
                    )
                    or ok
                )
            if ok:
                figures[track] = f"figures/heatmap_{track}_light.png"
        for theme in ("light", "dark"):
            if canon_figure(results["summary"], cfg, out_dir / "figures" / f"canonicalization_{theme}.png", theme):
                figures["canon"] = "figures/canonicalization_light.png"
    write_report(results, cfg, out_dir, figures)
    log(f"[analyze] wrote metrics, {len(figures)} figure sets and REPORT.md to {out_dir}")
    return results


# ------------------------------------------------------------------------------ README block
README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _picture(results_rel: str, stem: str, alt: str, width: int = 860) -> str:
    return (
        "<picture>\n"
        f'  <source media="(prefers-color-scheme: dark)" srcset="{results_rel}/figures/{stem}_dark.png">\n'
        f'  <img alt="{alt}" src="{results_rel}/figures/{stem}_light.png" width="{width}">\n'
        "</picture>"
    )


def readme_block(results_dir: str | pathlib.Path, cfg: dict, results_rel: str = "results") -> str:
    """Markdown for the README results section, built from results/metrics/*.csv."""
    results_dir = pathlib.Path(results_dir)
    summ = pd.read_csv(results_dir / "metrics" / "summary.csv")
    manifest = json.loads((results_dir / "raw" / "manifest.json").read_text())
    for col in ("canon", "primary_variant", "primary_track"):
        summ[col] = summ[col].astype(str).str.lower() == "true"
    L = [README_START, ""]
    prep = manifest.get("prepare", {})
    L.append(
        f"_Profile **{prep.get('profile', '?')}**, {manifest.get('n_samples', '?')} transformed samples, generated "
        f"{manifest.get('created', '?')[:10]} on GitHub-hosted CPU runners. Full tables with 95% intervals: "
        f"[`{results_rel}/REPORT.md`]({results_rel}/REPORT.md)._\n"
    )
    hdr = ["Guard", "Clean miss", "Clean FPR", "Mean miss under attack", "Mean FPR under attack",
           "Best-of-N evasion", "Any-transform FPR"]  # fmt: skip
    for track, tcfg in cfg["tracks"].items():
        models = [m for m, c in cfg["models"].items() if c["primary_track"] == track and m in set(summ["model"])]
        if not models:
            continue
        L.append(f"### {tcfg['title']}\n")
        if (results_dir / "figures" / f"heatmap_{track}_light.png").exists():
            L.append(
                _picture(
                    results_rel, f"heatmap_{track}", f"Miss rate and false-positive rate per attack, {track} track"
                )
            )
            L.append("")
        rows = []
        for m in models:
            s = summ[(summ["model"] == m) & (summ["track"] == track) & summ["primary_variant"] & ~summ["canon"]]
            if s.empty:
                continue
            s = s.iloc[0]
            rows.append([cfg["models"][m]["label"], pct(s["clean_miss"]), pct(s["clean_fpr"]), pct(s["mean_attack_miss"]),
                         pct(s["mean_attack_fpr"]), pct(s["best_of_n_evasion"]), pct(s["any_transform_fpr"])])  # fmt: skip
        L.append(_md_table(hdr, rows))
        L.append("")
    if (results_dir / "figures" / "canonicalization_light.png").exists():
        L.append("### Input canonicalization: what it recovers, what it costs\n")
        L.append(
            _picture(
                results_rel,
                "canonicalization",
                "Best-of-N evasion and any-transform FPR with and without canonicalization",
            )
        )
        L.append("")
    skipped = manifest.get("skipped") or {}
    if skipped:
        L.append(
            "Not run in this report: " + ", ".join(f"`{m}`" for m in skipped) + " (see REPORT.md for the reason).\n"
        )
    L.append(README_END)
    return "\n".join(L)


def update_readme(readme: str | pathlib.Path, results_dir: str | pathlib.Path, cfg: dict) -> None:
    readme = pathlib.Path(readme)
    text = readme.read_text(encoding="utf-8")
    start, end = text.index(README_START), text.index(README_END) + len(README_END)
    readme.write_text(text[:start] + readme_block(results_dir, cfg) + text[end:], encoding="utf-8")
