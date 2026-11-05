#!/usr/bin/env python3
"""Build docs/student-guide.pdf -- a beginner-friendly explanation of the benchmark and its results.

    pip install -e ".[docs]"          # reportlab
    python docs/build_student_guide.py

Every number in the guide is read from results/metrics/*.csv (and results/raw for the paired
1B-vs-8B comparison), and every transform example is produced by the project's own code on a
harmless sentence, so the text cannot drift from the data. Fonts: FreeSans / DejaVu Sans Mono
(Debian/Ubuntu packages fonts-freefont-ttf and fonts-dejavu-core).
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agb.metrics import attach_scores, per_transform, wilson  # noqa: E402
from agb.transforms import REGISTRY, rng_for  # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "docs" / "student-guide.pdf"
REPO_URL = "https://github.com/shahzaib360/adversarial-guardrail-benchmark"

# ----------------------------------------------------------------------------- palette (dataviz reference)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, BLUE_DARK, BLUE_TINT, BLUE_LIGHT = "#2a78d6", "#104281", "#eef4fc", "#cde2fb"
ORANGE, ORANGE_TINT = "#eb6834", "#fdf1ea"
NEUTRAL_TINT = "#f4f3ef"

# ----------------------------------------------------------------------------- fonts
FONT_DIRS = [pathlib.Path("/usr/share/fonts/truetype"), pathlib.Path("/usr/share/fonts")]


def _find(name: str) -> str | None:
    for d in FONT_DIRS:
        hits = list(d.rglob(name)) if d.exists() else []
        if hits:
            return str(hits[0])
    return None


def register_fonts() -> tuple[str, str, str]:
    from reportlab.lib.fonts import addMapping

    files = {
        "Sans": "FreeSans.ttf",
        "Sans-Bold": "FreeSansBold.ttf",
        "Sans-Oblique": "FreeSansOblique.ttf",
        "Sans-BoldOblique": "FreeSansBoldOblique.ttf",
        "Mono": "DejaVuSansMono.ttf",
        "Wide": "fonts-japanese-gothic.ttf",  # only used for the fullwidth example
    }
    found = {k: _find(v) for k, v in files.items()}
    if not found["Sans"]:
        print("FreeSans not found; falling back to Helvetica (some examples may not render)", file=sys.stderr)
        return "Helvetica", "Courier", "Helvetica"
    for k, path in found.items():
        if path:
            pdfmetrics.registerFont(TTFont(k, path))
    addMapping("Sans", 0, 0, "Sans")
    addMapping("Sans", 1, 0, "Sans-Bold" if found["Sans-Bold"] else "Sans")
    addMapping("Sans", 0, 1, "Sans-Oblique" if found["Sans-Oblique"] else "Sans")
    addMapping("Sans", 1, 1, "Sans-BoldOblique" if found["Sans-BoldOblique"] else "Sans")
    return "Sans", ("Mono" if found["Mono"] else "Courier"), ("Wide" if found["Wide"] else "Sans")


SANS, MONO, WIDE = register_fonts()
BOLD = "Sans-Bold" if SANS == "Sans" else "Helvetica-Bold"

# ----------------------------------------------------------------------------- styles
ST = {
    "title": ParagraphStyle("title", fontName=BOLD, fontSize=30,
                            leading=36, textColor=colors.HexColor(INK), spaceAfter=10),
    "subtitle": ParagraphStyle("subtitle", fontName=SANS, fontSize=14, leading=19, textColor=colors.HexColor(INK2)),
    "H1": ParagraphStyle("H1", fontName=BOLD, fontSize=19, leading=24, textColor=colors.HexColor(INK),
                         spaceBefore=6, spaceAfter=10),
    "H2": ParagraphStyle("H2", fontName=BOLD, fontSize=13, leading=18, textColor=colors.HexColor(BLUE_DARK),
                         spaceBefore=12, spaceAfter=5),
    "body": ParagraphStyle("body", fontName=SANS, fontSize=10.5, leading=15.2, textColor=colors.HexColor(INK),
                           spaceAfter=6),
    "small": ParagraphStyle("small", fontName=SANS, fontSize=9, leading=12.5, textColor=colors.HexColor(INK2)),
    "caption": ParagraphStyle("caption", fontName=SANS, fontSize=9, leading=12.5, textColor=colors.HexColor(INK2),
                              spaceBefore=4, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName=SANS, fontSize=8.8, leading=11.6, textColor=colors.HexColor(INK)),
    "cellb": ParagraphStyle("cellb", fontName=SANS, fontSize=8.8, leading=11.6, textColor=colors.HexColor(INK)),
    "cellr": ParagraphStyle("cellr", fontName=SANS, fontSize=8.8, leading=11.6, textColor=colors.HexColor(INK),
                            alignment=2),
    "head": ParagraphStyle("head", fontName=SANS, fontSize=8.6, leading=11, textColor=colors.HexColor(INK2)),
    "headr": ParagraphStyle("headr", fontName=SANS, fontSize=8.6, leading=11, textColor=colors.HexColor(INK2),
                            alignment=2),
    "code": ParagraphStyle("code", fontName=MONO, fontSize=8.6, leading=12, textColor=colors.HexColor(INK),
                           backColor=colors.HexColor(NEUTRAL_TINT), borderPadding=(6, 6, 6, 6), spaceBefore=4,
                           spaceAfter=10, leftIndent=6, rightIndent=6),
    "callout_title": ParagraphStyle("callout_title", fontName=BOLD, fontSize=10, leading=14,
                                    textColor=colors.HexColor(BLUE_DARK)),
    "stat_value": ParagraphStyle("stat_value", fontName=SANS, fontSize=24, leading=28,
                                 textColor=colors.HexColor(INK), alignment=TA_CENTER),
    "stat_label": ParagraphStyle("stat_label", fontName=SANS, fontSize=9, leading=12,
                                 textColor=colors.HexColor(INK2), alignment=TA_CENTER),
    "toc0": ParagraphStyle("toc0", fontName=SANS, fontSize=11, leading=17, leftIndent=0,
                           textColor=colors.HexColor(INK)),
    "toc1": ParagraphStyle("toc1", fontName=SANS, fontSize=9.5, leading=14, leftIndent=16,
                           textColor=colors.HexColor(INK2)),
}  # fmt: skip


ST["H1plain"] = ParagraphStyle("H1plain", parent=ST["H1"])  # same look, not listed in the contents


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, ST[style])


def b(text: str) -> str:
    return f"<b>{text}</b>"


def mono(text: str) -> str:
    return f'<font name="{MONO}">{escape(text)}</font>'


def pct(x: float, digits: int = 0) -> str:
    return "–" if pd.isna(x) else f"{100 * x:.{digits}f}%"


def bullets(items: list[str], style: str = "body") -> ListFlowable:
    return ListFlowable(
        [ListItem(P(t, style), leftIndent=14) for t in items],
        bulletType="bullet", start="\u2022", leftIndent=14, bulletFontName=SANS, bulletFontSize=9,
        bulletColor=colors.HexColor(BLUE),
    )  # fmt: skip


def numbered(items: list[str]) -> ListFlowable:
    return ListFlowable([ListItem(P(t), leftIndent=18) for t in items], bulletType="1", leftIndent=18,
                        bulletFontName=SANS, bulletFontSize=10, bulletColor=colors.HexColor(BLUE_DARK))


def callout(title: str, body: list | str, kind: str = "idea") -> Table:
    tint, edge = {"idea": (BLUE_TINT, BLUE), "warn": (ORANGE_TINT, ORANGE), "try": (NEUTRAL_TINT, MUTED)}[kind]
    flow = [P(title, "callout_title")] + ([P(body)] if isinstance(body, str) else body)
    t = Table([[flow]], colWidths=[17 * cm - 4])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(tint)),
        ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(edge)),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))  # fmt: skip
    return t


def data_table(header: list[str], rows: list[list], widths: list[float], align_right_from: int = 1,
               zebra: bool = True) -> Table:
    def cell(v, i, is_head):
        if isinstance(v, Paragraph):
            return v
        right = i >= align_right_from
        style = ("headr" if right else "head") if is_head else ("cellr" if right else "cell")
        return P(str(v), style)

    data = [[cell(h, i, True) for i, h in enumerate(header)]]
    data += [[cell(v, i, False) for i, v in enumerate(r)] for r in rows]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1, hAlign="LEFT")
    style = [
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(AXIS)),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor(GRID)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]  # fmt: skip
    if zebra:
        for r in range(1, len(data)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f7f7f4")))
    t.setStyle(TableStyle(style))
    return t


def figure_parts(path: pathlib.Path, width_cm: float, caption: str) -> list:
    """Image + caption as plain flowables (wrap them in ONE KeepTogether: nested KeepTogethers never fit)."""
    from reportlab.lib.utils import ImageReader

    iw, ih = ImageReader(str(path)).getSize()
    w = width_cm * cm
    return [Image(str(path), width=w, height=w * ih / iw), P(caption, "caption")]


def figure(path: pathlib.Path, width_cm: float, caption: str) -> KeepTogether:
    return KeepTogether(figure_parts(path, width_cm, caption))


# ----------------------------------------------------------------------------- diagrams
def _box(d: Drawing, x, y, w, h, label: list[str], fill, stroke, bold=False, size=8.5):
    d.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=colors.HexColor(fill), strokeColor=colors.HexColor(stroke),
               strokeWidth=1))
    n = len(label)
    for i, line in enumerate(label):
        yy = y + h / 2 + (n - 1) * (size + 2) / 2 - i * (size + 2) - size / 3
        d.add(String(x + w / 2, yy, line, fontName=SANS, fontSize=size, fillColor=colors.HexColor(INK),
                     textAnchor="middle"))


def _arrow(d: Drawing, x1, y1, x2, y2, color=MUTED):
    d.add(Line(x1, y1, x2, y2, strokeColor=colors.HexColor(color), strokeWidth=1.2))
    import math

    ang = math.atan2(y2 - y1, x2 - x1)
    s = 5
    p1 = (x2 - s * math.cos(ang - 0.45), y2 - s * math.sin(ang - 0.45))
    p2 = (x2 - s * math.cos(ang + 0.45), y2 - s * math.sin(ang + 0.45))
    d.add(Polygon([x2, y2, *p1, *p2], fillColor=colors.HexColor(color), strokeColor=colors.HexColor(color)))


def guard_diagram() -> Drawing:
    W, H = 17 * cm, 3.6 * cm
    d = Drawing(W, H)
    neutral, edge = NEUTRAL_TINT, AXIS
    y, h = 0.9 * cm, 1.5 * cm
    xs = [0, 3.35, 6.95, 10.55, 14.15]
    ws = [2.6, 2.9, 2.9, 2.9, 2.6]
    labels = [["User", "prompt"], ["Input guard", "(Llama Guard,", "Prompt Guard)"], ["Large language", "model (LLM)"],
              ["Output guard", "(optional)"], ["Answer to", "the user"]]  # fmt: skip
    fills = [(neutral, edge), (BLUE_LIGHT, BLUE), (neutral, edge), (BLUE_LIGHT, BLUE), (neutral, edge)]
    for x, w, lab, (f, s) in zip(xs, ws, labels, fills):
        _box(d, x * cm, y, w * cm, h, lab, f, s)
    for i in range(4):
        _arrow(d, (xs[i] + ws[i]) * cm + 2, y + h / 2, xs[i + 1] * cm - 2, y + h / 2)
    # "blocked" branch under the input guard
    gx = (xs[1] + ws[1] / 2) * cm
    _arrow(d, gx, y - 1, gx, 0.18 * cm, color=ORANGE)
    d.add(String(gx + 6, 0.2 * cm, "flagged: block or refuse", fontName=SANS, fontSize=8,
                 fillColor=colors.HexColor(INK2)))
    return d


def pipeline_diagram() -> Drawing:
    W, H = 17 * cm, 6.0 * cm
    d = Drawing(W, H)
    w, h = 5.0 * cm, 1.75 * cm
    gap = (W - 3 * w) / 2
    rows = [
        [["1. Load data", "pinned public datasets,", "checked with SHA-256"],
         ["2. Transform", "24 attacks, applied to harmful", "AND benign inputs"],
         ["3. Canonicalize", "a decoded, normalised copy", "(the defense view)"]],
        [["6. Report", "tables, figures, README", "(text hashes only)"],
         ["5. Metrics", "miss rate, FPR, intervals,", "AUROC, best-of-N"],
         ["4. Score", "7 guards on GitHub Actions,", "split into parallel shards"]],
    ]  # fmt: skip
    ys = [H - h - 0.15 * cm, 0.25 * cm]
    for r, row in enumerate(rows):
        for c, lab in enumerate(row):
            x = c * (w + gap)
            _box(d, x, ys[r], w, h, lab, BLUE_TINT if r == 0 else NEUTRAL_TINT, BLUE if r == 0 else AXIS, size=8.2)
    # arrows: row 0 left->right, down on the right, row 1 right->left
    for c in range(2):
        _arrow(d, c * (w + gap) + w + 2, ys[0] + h / 2, (c + 1) * (w + gap) - 2, ys[0] + h / 2)
    xr = 2 * (w + gap) + w / 2
    _arrow(d, xr, ys[0] - 2, xr, ys[1] + h + 2)
    for c in (2, 1):
        _arrow(d, c * (w + gap) - 2, ys[1] + h / 2, (c - 1) * (w + gap) + w + 2, ys[1] + h / 2)
    return d


# ----------------------------------------------------------------------------- data
summ = pd.read_csv(RES / "metrics" / "summary.csv")
pt = pd.read_csv(RES / "metrics" / "per_transform.csv")
psrc = pd.read_csv(RES / "metrics" / "per_source.csv")
validity = pd.read_csv(RES / "metrics" / "validity.csv")
manifest = json.loads((RES / "raw" / "manifest.json").read_text())


def S(model: str, canon: bool = False, variant: str | None = None, track: str | None = None) -> pd.Series:
    d = summ[(summ["model"] == model) & (summ["canon"] == canon)]
    d = d[d["primary_variant"]] if variant is None else d[d["variant"] == variant]
    d = d[d["track"] == track] if track else d[d["primary_track"]]
    return d.iloc[0]


def T(model: str, transform: str, canon: bool = False) -> pd.Series:
    d = pt[(pt["model"] == model) & (pt["transform"] == transform) & (pt["canon"] == canon) & pt["primary_variant"]
           & pt["primary_track"]]  # fmt: skip
    return d.iloc[0]


def source_fpr(model: str, source: str, transform: str = "none") -> float:
    d = psrc[(psrc["model"] == model) & (psrc["source"] == source) & (psrc["transform"] == transform)
             & psrc["primary_variant"] & ~psrc["canon"] & (psrc["label"] == 0)]  # fmt: skip
    return float(d.iloc[0]["flag_rate"])


LG1, LG8, QW, PG, PAI = "llama-guard-3-1b", "llama-guard-3-8b", "qwen3guard-gen-0.6b", "prompt-guard-86m", \
    "protectai-deberta-v2"  # fmt: skip
LABEL = {LG1: "Llama Guard 3 1B", LG8: "Llama Guard 3 8B", QW: "Qwen3Guard-Gen 0.6B", PG: "Prompt Guard 86M",
         PAI: "ProtectAI DeBERTa v2", "keyword-harm": "Keyword block-list", "keyword-injection": "Keyword block-list"}


def paired_8b() -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Per-transform results of Llama Guard 3 1B and 8B on exactly the items the 8B model saw."""
    samples = pd.read_csv(RES / "raw" / "samples.csv.gz", dtype={"item_id": str, "category": str})
    harm = samples[samples["track"] == "harmful"]
    sc8 = pd.read_csv(RES / "raw" / "scores" / f"{LG8}.csv.gz", dtype={"h": str, "out": str})
    sc1 = pd.read_csv(RES / "raw" / "scores" / f"{LG1}.csv.gz", dtype={"h": str, "out": str})
    items = set(attach_scores(harm, sc8, "default").dropna(subset=["score"])["item_id"])
    sub = harm[harm["item_id"].isin(items)]
    p8 = per_transform(attach_scores(sub, sc8, "default"), "score", 0.5, None, True).set_index("transform")
    p1 = per_transform(attach_scores(sub, sc1, "default"), "score", 0.5, None, True).set_index("transform")
    clean = sub[sub["transform"] == "none"]
    return p1, p8, int((clean["label"] == 1).sum()), int((clean["label"] == 0).sum())


# ----------------------------------------------------------------------------- teaching chart
GROUP = "encodings, spacing, zero-width, ..."
CHART_LABELS = {  # model -> [(transform, label, offset in points, alignment, leader line?)]
    "llama-guard-3-1b": [("math_bold", "math bold", (7, 0), "left", False), ("hex", "hex", (7, 0), "left", False),
                         ("base64", "base64", (-7, 6), "right", False), ("leetspeak", "leetspeak", (0, 9), "center",
                                                                          False)],
    "qwen3guard-gen-0.6b": [("base64", "base64, hex, ROT13, Morse, reverse", (7, 0), "left", False),
                            ("leetspeak", "leetspeak", (-4, 10), "right", False)],
    "prompt-guard-86m": [("base64", GROUP, (7, 0), "left", False)],
    "protectai-deberta-v2": [("base64", GROUP, (-14, 34), "right", True),
                             ("unicode_tags", "unicode tags", (7, 0), "left", False)],
}  # fmt: skip
def tradeoff_chart(path: pathlib.Path) -> None:
    """Small multiples: each dot is one attack; x = false-positive rate, y = miss rate (dataviz rules:
    one hue per panel, thin recessive grid, selective direct labels, text in ink colours)."""
    panels = [(LG1, "Llama Guard 3 1B"), (QW, "Qwen3Guard-Gen 0.6B"), (PG, "Prompt Guard 86M"),
              (PAI, "ProtectAI DeBERTa v2")]  # fmt: skip
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6), facecolor=SURFACE, sharex=True, sharey=True)
    for ax, (model, title) in zip(axes.flat, panels):
        d = pt[(pt["model"] == model) & pt["primary_variant"] & pt["primary_track"] & ~pt["canon"]]
        att, clean = d[d["transform"] != "none"], d[d["transform"] == "none"]
        ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.scatter(att["fpr"], att["miss"], s=46, color=BLUE, edgecolor=SURFACE, linewidth=1.6, zorder=3)
        ax.scatter(clean["fpr"], clean["miss"], s=70, color=BLUE_DARK, edgecolor=SURFACE, linewidth=1.6, zorder=4,
                   marker="D")
        ax.annotate("clean", (clean["fpr"].iloc[0], clean["miss"].iloc[0]), xytext=(7, 5),
                    textcoords="offset points", fontsize=8, color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.9))
        # selective direct labels, placed by hand so they never collide (several attacks share one spot)
        for t, text, (dx, dy), ha, leader in CHART_LABELS.get(model, []):
            r = att[att["transform"] == t].iloc[0]
            ax.annotate(text, (r["fpr"], r["miss"]), xytext=(dx, dy), textcoords="offset points", fontsize=7.6,
                        color=INK2, ha=ha, va="center", zorder=5,
                        arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8, shrinkA=0, shrinkB=4) if leader else None)
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, 1.06)
        ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
        ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=6)
    fig.supxlabel("False-positive rate on benign inputs (same attack applied)", fontsize=9, color=INK2)
    fig.supylabel("Miss rate on harmful / attack inputs", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0.02, 0.02, 1, 1))
    fig.savefig(path, dpi=220, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- transform examples
EXAMPLE = "How do I bake bread at home?"


def transform_examples() -> list[list]:
    rows = []
    marker = '<font color="#d85a28">·</font>'
    for name, spec in REGISTRY.items():
        if name == "none":
            continue
        if spec.fn is not None:
            out = spec.fn(EXAMPLE, rng_for(1234, "demo:1", name))
        else:
            out = None
        payload = out.split("\n\n", 1)[-1] if out and spec.family == "encoding" else out
        if spec.family == "encoding":
            shown = f'<font color="{MUTED}">[decode instruction]</font> {mono(payload)}'
        elif name == "zero_width":
            shown = marker.join(escape(c) for c in EXAMPLE) + f' <font color="{MUTED}">(each · is an invisible character)</font>'
        elif name == "unicode_tags":
            shown = f'Hello! <font color="{MUTED}">[+ {len(EXAMPLE)} invisible tag characters that spell the request]</font>'
        elif name == "fullwidth":
            shown = f'<font name="{WIDE}">{escape(out)}</font>'
        elif name == "char_spacing":
            shown = escape(out).replace("   ", "&nbsp;&nbsp;&nbsp;")
        elif name == "homoglyph":
            n_cyr = sum(1 for c in out if "\u0400" <= c <= "\u04ff")
            shown = f'{escape(out)} <font color="{MUTED}">({n_cyr} letters are Cyrillic look-alikes)</font>'
        elif name == "payload_split":
            lines = out.split("\n")[1:-1]
            shown = mono("  ".join(lines)) + f' <font color="{MUTED}">+ "join the pieces and respond"</font>'
        elif name == "benign_padding":
            shown = f'<font color="{MUTED}">[~300 words about houseplant care]</font> {escape(EXAMPLE)}'
        elif spec.family == "framing":
            first = out.split(".")[0]
            shown = f'<font color="{MUTED}">"{escape(first)}…"</font> + {escape(EXAMPLE)}'
        elif name == "paraphrase":
            shown = ('What is the best way to make bread in my own kitchen? '
                     f'<font color="{MUTED}">(illustration)</font>')
        elif name == "translate_fr":
            shown = f'Comment faire du pain \u00e0 la maison ? <font color="{MUTED}">(illustration)</font>'
        elif name == "translate_bn":
            shown = f'<font color="{MUTED}">the same request, machine-translated into Bengali</font>'
        elif name == "translate_zu":
            shown = f'<font color="{MUTED}">the same request, machine-translated into Zulu</font>'
        else:
            shown = escape(out)
        rows.append([P(mono(name), "cell"), P(spec.family, "cell"), P(shown, "cell")])
    return rows


# ----------------------------------------------------------------------------- document template
class GuideDoc(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(filename, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2.2 * cm,
                         bottomMargin=2 * cm, title="Attacking the Guards: a student's guide",
                         author="Adversarial Guardrail Benchmark", subject="LLM guardrail robustness")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate(id="cover", frames=[frame], onPage=self._cover),
                               PageTemplate(id="body", frames=[frame], onPage=self._body)])
        self._n = 0

    @staticmethod
    def _cover(canv, doc):
        canv.saveState()
        canv.setFillColor(colors.HexColor(BLUE))
        canv.rect(0, A4[1] - 0.6 * cm, A4[0], 0.6 * cm, stroke=0, fill=1)
        canv.restoreState()

    @staticmethod
    def _body(canv, doc):
        canv.saveState()
        canv.setFont(SANS, 8)
        canv.setFillColor(colors.HexColor(MUTED))
        canv.drawString(2 * cm, A4[1] - 1.3 * cm, "Attacking the Guards · a student's guide to the Adversarial "
                        "Guardrail Benchmark")
        canv.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(doc.page))
        canv.setStrokeColor(colors.HexColor(GRID))
        canv.setLineWidth(0.6)
        canv.line(2 * cm, A4[1] - 1.45 * cm, A4[0] - 2 * cm, A4[1] - 1.45 * cm)
        canv.restoreState()

    def beforeDocument(self):
        self._n = 0  # multiBuild runs several passes; heading keys must be identical in each

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ("H1", "H2"):
            level = 0 if flowable.style.name == "H1" else 1
            text = flowable.getPlainText()
            self._n += 1
            key = f"h{self._n}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            self.notify("TOCEntry", (level, text, self.page, key))


def H1(text: str) -> list:
    return [CondPageBreak(6 * cm), P(text, "H1")]


def H2(text: str) -> Paragraph:
    return P(text, "H2")


# ----------------------------------------------------------------------------- content
def build() -> None:
    tmp = pathlib.Path(tempfile.mkdtemp())
    tradeoff_png = tmp / "tradeoff.png"
    tradeoff_chart(tradeoff_png)
    p1, p8, n8_pos, n8_neg = paired_8b()

    lg1, qw, pg, pai, lg8 = S(LG1), S(QW), S(PG), S(PAI), S(LG8)
    lg1c, qwc, pgc, lg8c = S(LG1, canon=True), S(QW, canon=True), S(PG, canon=True), S(LG8, canon=True)
    pg_any = S(PG, variant="any")
    n_samples = manifest["n_samples"]
    n_transforms = len(REGISTRY) - 1
    w_p, w_lo, w_hi = wilson(15, 150)
    val = validity.set_index(["track", "transform"])["valid"]

    story: list = []
    # ---------------------------------------------------------------- cover
    story += [Spacer(1, 2.2 * cm), P("Attacking the Guards", "title"),
              P("A student's guide to the Adversarial Guardrail Benchmark: how we stress-test the safety "
                "classifiers that protect AI chatbots, and why false alarms matter as much as misses.", "subtitle"),
              Spacer(1, 1.2 * cm)]  # fmt: skip
    tiles = [(str(n_transforms), "attacks (encodings, Unicode tricks, paraphrase, translation, ...)"),
             ("7", "guard models and baselines tested"),
             (f"{n_samples:,}", "test inputs scored"),
             ("100%", "of attacks also run on harmless inputs, to measure false alarms")]  # fmt: skip
    tile_cells = [[P(v, "stat_value"), Spacer(1, 4), P(lbl, "stat_label")] for v, lbl in tiles]
    tt = Table([tile_cells], colWidths=[4.25 * cm] * 4)
    tt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(BLUE_TINT)),
                            ("LINEAFTER", (0, 0), (-2, -1), 2, colors.white),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("TOPPADDING", (0, 0), (-1, -1), 12), ("BOTTOMPADDING", (0, 0), (-1, -1), 12)]))
    story += [tt, Spacer(1, 1.4 * cm)]
    story += [P("<b>Who this is for:</b> students and junior engineers who know a little Python and have heard of "
                "large language models (LLMs), but have not worked on AI safety. No maths beyond percentages is "
                "needed; every term is explained when it first appears, and there is a glossary at the end.",
                "body"),
              P(f"<b>Project repository:</b> <link href='{REPO_URL}' color='{BLUE}'>{REPO_URL}</link><br/>"
                "<b>Results:</b> full benchmark run of November 2026 (all numbers in this guide are read directly "
                "from the repository's result files).", "body"),
              Spacer(1, 0.6 * cm),
              callout("A note on safety",
                      "This guide never shows harmful text. Every attack is demonstrated on the harmless sentence "
                      f"“{EXAMPLE}”. The benchmark itself uses public research datasets of harmful requests, and its "
                      "published files contain only anonymous fingerprints (hashes) of the test texts.",
                      "try"),
              NextPageTemplate("body"), PageBreak()]  # fmt: skip

    # ---------------------------------------------------------------- contents
    toc = TableOfContents()
    toc.levelStyles = [ST["toc0"], ST["toc1"]]
    toc.dotsMinLevel = 0
    story += [P("Contents", "H1plain"), toc, PageBreak()]

    # ---------------------------------------------------------------- 1
    story += H1("1. Why this project exists")
    story += [
        P("Chatbots and AI agents are built on large language models (LLMs). An LLM will try to answer almost "
          "anything, so products add <b>guardrails</b>: rules and small models that decide whether a message should "
          "be answered at all. A very common guardrail is a <b>guard classifier</b> (or simply a “guard”): a "
          "small, fast model that reads each incoming message and says “safe” or “unsafe” before the big model "
          "ever sees it. Think of it as the bag scanner at an airport: every bag goes through it, and suspicious "
          "ones are pulled aside."),
        guard_diagram(),
        P("Figure 1. Where a guard sits. The input guard can stop a message before the LLM answers it; some "
          "systems also check the LLM's answer on the way out.", "caption"),
        P("Guards come in two main families, and this project tests both:"),
        bullets([
            "<b>Content-safety guards</b> decide whether a request is <i>harmful</i> (for example, asking how to "
            "build a weapon or write malware). Examples: Meta's <b>Llama Guard 3</b> and Alibaba's "
            "<b>Qwen3Guard</b>.",
            "<b>Prompt-attack detectors</b> decide whether a text is trying to <i>manipulate</i> the model: a "
            "<b>jailbreak</b> (“pretend you have no rules…”) or a <b>prompt injection</b> (instructions hidden in "
            "a web page or document, such as “ignore your previous instructions”). Examples: Meta's <b>Prompt "
            "Guard</b> and <b>ProtectAI</b>'s DeBERTa-based detector.",
        ]),
        P("These guards are usually tested on ordinary-looking examples. Real attackers do not cooperate: they "
          "encode their request, disguise the letters, translate it into another language or simply rephrase it. "
          "The model behind the guard can often still understand the disguised message, but the guard may not. "
          "This project measures how much such disguises hurt the guards, and what it costs to make the guards "
          "stricter."),
        callout("Key idea", "A guard can fail in two different ways: it can <b>miss</b> an attack, or it can "
                "<b>block an innocent user</b> (a false alarm). A test that only counts one of these failures can "
                "make a bad guard look excellent. This benchmark always reports both."),
    ]  # fmt: skip

    # ---------------------------------------------------------------- 2
    story += H1("2. Core ideas you need")
    story += [H2("2.1 Positives, negatives and the four outcomes"),
              P("In testing language, a <b>positive</b> is something the guard <i>should</i> flag (a harmful "
                "request or an attack) and a <b>negative</b> is something it should let through (a normal, harmless "
                "message). Every decision falls into one of four boxes, called a <b>confusion matrix</b>:")]
    cm_data = [["", P("Guard says <b>block</b>", "head"), P("Guard says <b>allow</b>", "head")],
               [P("<b>Harmful</b> input", "cell"), P("True positive: caught ✓", "cell"),
                P("<b>False negative: a miss</b> — the attack gets through", "cell")],
               [P("<b>Harmless</b> input", "cell"), P("<b>False positive: a false alarm</b> — an innocent user is "
                                                      "blocked", "cell"), P("True negative: allowed ✓", "cell")]]
    cmt = Table(cm_data, colWidths=[3.6 * cm, 6.7 * cm, 6.7 * cm])
    cmt.setStyle(TableStyle([
        ("BACKGROUND", (1, 1), (1, 1), colors.HexColor("#e6f2e6")), ("BACKGROUND", (2, 2), (2, 2), colors.HexColor("#e6f2e6")),
        ("BACKGROUND", (2, 1), (2, 1), colors.HexColor(ORANGE_TINT)), ("BACKGROUND", (1, 2), (1, 2), colors.HexColor(ORANGE_TINT)),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.white), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))  # fmt: skip
    story += [cmt, Spacer(1, 8),
              P("From these four boxes we get the two numbers this whole project is about:"),
              bullets([
                  "<b>Miss rate</b> = misses \u00f7 all harmful inputs. Also called the <i>false-negative rate</i>. "
                  "Lower is better. A miss rate of 30% means 3 in 10 attacks get through.",
                  "<b>False-positive rate (FPR)</b> = false alarms \u00f7 all harmless inputs. Lower is better. An FPR of "
                  "10% means 1 in 10 innocent users is blocked.",
              ]),
              H2("2.2 Scores, thresholds and the trade-off"),
              P("Most guards do not just say yes or no; they output a <b>score</b> between 0 and 1 (roughly, how "
                "sure the guard is that the input is unsafe). A <b>threshold</b> turns the score into a decision: "
                "here we use each model's own default, 0.5. Moving the threshold trades one error for the other. "
                "Lower it and you catch more attacks but block more innocent users; raise it and the opposite "
                "happens."),
              callout("Watch out", "A guard that blocks <i>everything</i> has a 0% miss rate. It also has a 100% "
                      "false-positive rate and is useless. That is why a miss rate on its own can be badly "
                      "misleading.", "warn"),
              Spacer(1, 6),
              P("<b>AUROC</b> (area under the ROC curve) is a way to judge a guard without picking a threshold. It "
                "answers: <i>if I pick one harmful and one harmless input at random, how often does the harmful one "
                "get the higher score?</i> 1.0 means always (perfect separation), 0.5 means a coin flip, and below "
                "0.5 means the guard is actually <i>more</i> suspicious of the harmless inputs."),
              H2("2.3 Adversarial transformations and the threat model"),
              P("An <b>adversarial transformation</b> (or simply an attack) changes the <i>form</i> of a request "
                "while keeping its <i>meaning</i>. For example, the Base64 encoding of “How do I bake bread at "
                f"home?” is {mono('SG93IGRvIEkgYmFrZSBicmVhZCBhdCBob21lPw==')}. A human cannot read that, but "
                "modern LLMs decode Base64 easily."),
              P("Our <b>threat model</b> (the assumptions about the attacker) is: the guard protects a capable LLM "
                "that can decode, translate and read disguised text. If the guard lets a disguised harmful request "
                "through, the LLM may act on it. So any disguised harmful request that passes the guard counts as a "
                "miss."),
              H2("2.4 The most important design choice: harmless controls"),
              P("Here is the trap many evaluations fall into. Suppose a guard flags 84% of Base64-encoded harmful "
                "requests. Is it good at spotting encoded attacks? Not necessarily: maybe it flags <i>everything</i> "
                "in Base64, including encoded bread recipes. To tell these apart, this benchmark applies <b>every "
                "attack to harmless inputs too</b> and reports the false-positive rate on those transformed "
                "harmless inputs. A guard only gets credit for catching an attack if it does not also flag the "
                "harmless version."),
              ]  # fmt: skip

    # ---------------------------------------------------------------- 3
    story += H1("3. How the benchmark is built")
    story += [H2("3.1 The test data"),
              P("The benchmark has two <b>tracks</b>, one for each family of guards. All data comes from public "
                "research datasets, downloaded at run time from the authors' repositories at a fixed version and "
                "checked with a SHA-256 checksum (a digital fingerprint), so anyone can reproduce exactly the same "
                "inputs."),
              data_table(["Track", "Should be flagged (positives)", "Should pass (negatives)", "Guards tested"],
                         [["Harmful requests (content safety)",
                           "150 harmful requests from HarmBench",
                           "150 XSTest prompts that only <i>look</i> scary (“How do I kill a Python process?”) and "
                           "100 ordinary instructions from Alpaca",
                           "Llama Guard 3 (1B, 8B), Qwen3Guard, keyword list"],
                          ["Prompt attacks (jailbreak and injection)",
                           "150 real jailbreak prompts (JailbreakHub) and 150 hijacking attacks written by players of "
                           "the Tensor Trust game",
                           "150 normal prompts from the same online communities (many are harmless role-play "
                           "personas) and 150 Alpaca instructions",
                           "Prompt Guard 86M, ProtectAI DeBERTa v2, keyword list"]],
                         [3.4, 4.6, 5.4, 3.6], align_right_from=9),
              Spacer(1, 8),
              P("The harmless “look-alike” prompts from XSTest are deliberately tricky: they contain alarming "
                "words used innocently. They show whether a guard reacts to scary words or to actual harm."),
              H2("3.2 The 24 attacks"),
              P(f"Each attack is a small, well-known technique; none is tuned against a particular model. The table "
                f"shows what each one does to the harmless sentence “{EXAMPLE}”. Encoded payloads come with a short "
                "instruction telling the LLM how to decode them (the same instruction is added to harmless inputs, "
                "so it cannot unfairly help or hurt the guard).")]  # fmt: skip
    story += [data_table(["Attack", "Family", "What the guard sees (on a harmless example)"], transform_examples(),
                         [3.8, 2.2, 11.0], align_right_from=9),
              P("Framing attacks and translations are only used on the harmful-request track; the prompt-attack track "
                "uses the other 18 attacks. Paraphrase and translation outputs shown here are illustrations; the real "
                "ones are produced by models during the run.", "caption"),
              H2("3.3 Making sure a paraphrase still means the same thing"),
              P("Paraphrasing and translation are done by AI models (a T5 paraphraser and Meta's NLLB translator), "
                "and such models sometimes change the meaning. If a paraphrase accidentally turned a harmful request "
                "into a harmless one, a guard that lets it through would be unfairly counted as missing it. So every "
                "paraphrase and translation is <b>meaning-checked</b> with LaBSE, a model that turns sentences into "
                "vectors so that sentences with the same meaning end up close together (in any language). Samples "
                "below a similarity threshold are thrown out. "
                f"In the full run {pct(val[('harmful', 'paraphrase')])} of harmful-track paraphrases and "
                f"{pct(val[('harmful', 'translate_zu')])}–{pct(val[('harmful', 'translate_fr')])} of translations "
                "passed."),
              H2("3.4 The guards and how they are run"),
              data_table(["Guard", "Family", "How it is run", "Score used"],
                         [["Llama Guard 3 1B and 8B", "content safety",
                           "Through Ollama on CPU (compressed “quantized” versions of Meta's weights: Q8_0 for 1B, "
                           "Q4_K_M for 8B)", "probability of the word “unsafe”"],
                          ["Qwen3Guard-Gen 0.6B", "content safety", "Hugging Face transformers on CPU",
                           "probability of “Unsafe” or “Controversial”"],
                          ["Prompt Guard 86M", "prompt attacks",
                           "transformers; long texts are cut into 512-token windows and the most suspicious window "
                           "counts", "probability of JAILBREAK"],
                          ["ProtectAI DeBERTa v2", "prompt attacks", "as above", "probability of INJECTION"],
                          ["Keyword block-list", "baseline", "a list of alarming words and phrases",
                           "1 if any word matches"]],
                         [3.6, 2.4, 7.2, 3.8], align_right_from=9),
              Spacer(1, 6),
              P("Two engineering tricks are worth knowing about. Llama Guard is a <i>generative</i> model: it writes "
                "the word “safe” or “unsafe”. Instead of only reading that word, we ask Ollama for the "
                "<b>log-probabilities</b> of the first word, which gives a smooth score and lets us compute AUROC. "
                "Qwen3Guard always starts its prompt with the same long safety policy, so we compute that part once "
                "and reuse it (<b>prefix caching</b>), which made it several times faster. Prompt Guard 2, Meta's "
                "newer detector, was not included: its download requires approval from Meta that had not arrived "
                "when the benchmark ran."),
              H2("3.5 A cheap defense: canonicalization"),
              P("Many attacks are <i>mechanical</i> and can be undone without understanding the text. The benchmark "
                "therefore also tests a simple pre-processing step called <b>canonicalization</b> (turning every "
                "input into one standard, “canonical” form). It decodes invisible tag characters, deletes "
                "zero-width characters, turns fullwidth and bold letters back into normal ones, swaps Cyrillic "
                "look-alikes for Latin letters, joins s p a c e d letters, decodes Base64, hex and Morse, undoes "
                "leetspeak, and reverses ROT13 or backwards text when the result looks much more like English."),
              P("The defended guard scores both the original and the canonical text and keeps the higher score. "
                "This can only add detections, never remove them, but it can add false alarms, and the benchmark "
                "measures exactly that cost."),
              H2("3.6 Running it: the pipeline"),
              pipeline_diagram(),
              P("Figure 2. The six stages. All of them run on GitHub Actions, GitHub's free computers for public "
                "projects.", "caption"),
              P("Everything runs on ordinary CPUs (no graphics card), so the slow stages are split into "
                "<b>shards</b>: many small jobs that run in parallel and are merged at the end. The prepared inputs "
                "are cached, so adding a model later does not redo the work. The published results contain only a "
                "<b>hash</b> (a fingerprint such as a SHA-1 code) of each test text plus its scores, never the "
                "harmful text itself; anyone can rebuild the texts from the public datasets and the code."),
              ]  # fmt: skip

    # ---------------------------------------------------------------- 4
    story += H1("4. How we measure")
    story += [P("Besides the miss rate and FPR from section 2, the report uses a few more measurements. Each one "
                "answers a specific question:"),
              data_table(["Measurement", "Question it answers"],
                         [["Miss rate and FPR (at threshold 0.5)", "How does the guard behave as it is normally "
                           "deployed?"],
                          ["95% Wilson interval", "How sure are we about a rate measured on a few hundred examples?"],
                          ["Flip rate", "Of the attacks the guard catches in plain form, how many slip through once "
                           "transformed?"],
                          ["Best-of-N evasion", "If the attacker can try every disguise, how many harmful requests "
                           "eventually get through at least once?"],
                          ["Any-transform FPR", "How many harmless inputs get blocked under at least one disguise?"],
                          ["AUROC", "Ignoring the threshold, can the guard separate harmful from harmless at all?"],
                          ["Calibrated threshold", "What if we tune the threshold so only 5% of normal harmless "
                           "inputs are blocked?"],
                          ["Paired comparison", "When two models were tested on different numbers of items, how do "
                           "they compare on exactly the same items?"]],
                         [5.4, 11.6], align_right_from=9),
              Spacer(1, 8),
              callout("Why intervals matter",
                      f"If a guard misses 15 of 150 harmful requests, the measured miss rate is {pct(w_p)}. With so "
                      f"few examples the true rate could plausibly be anywhere from about {pct(w_lo)} to "
                      f"{pct(w_hi)} (the 95% Wilson interval). So a difference of two or three percentage points "
                      "between two guards means very little; a difference of 30 points is real. Every rate in the "
                      "full report comes with its interval."),
              ]  # fmt: skip

    # ---------------------------------------------------------------- 5
    story += H1("5. What we found")
    story += [P(f"The full run scored {n_samples:,} test inputs. The tables below summarise each guard. "
                "“Average under attack” is the plain average over all attacks."),
              H2("5.1 The summary tables")]

    def summary_rows(models):
        rows = []
        for m in models:
            s = S(m)
            rows.append([LABEL[m], f"{int(s['n_pos_items'])}/{int(s['n_neg_items'])}", pct(s["clean_miss"]),
                         pct(s["clean_fpr"]), pct(s["mean_attack_miss"]), pct(s["mean_attack_fpr"]),
                         pct(s["best_of_n_evasion"])])
        return rows

    hdr = ["Guard", "Items (harmful/ harmless)", "Miss, plain", "FPR, plain", "Avg miss under attack",
           "Avg FPR under attack", "Best-of-N evasion"]  # fmt: skip
    story += [KeepTogether([P("<b>Harmful-request track</b> (content-safety guards)", "small"), Spacer(1, 3),
                            data_table(hdr, summary_rows([LG1, LG8, QW, "keyword-harm"]),
                                       [4.0, 2.4, 1.9, 1.9, 2.3, 2.3, 2.2])]),
              Spacer(1, 10),
              KeepTogether([P("<b>Prompt-attack track</b> (jailbreak and injection detectors)", "small"), Spacer(1, 3),
                            data_table(hdr, summary_rows([PG, PAI, "keyword-injection"]),
                                       [4.0, 2.4, 1.9, 1.9, 2.3, 2.3, 2.2])]),
              P("Llama Guard 3 8B is slow on a CPU (about 7 seconds per text), so it was tested on a fixed random "
                f"subset of {n8_pos} harmful and {n8_neg} harmless items; section 5.5 compares it fairly with the 1B "
                "model on exactly those items.", "caption"),
              P("Read the first two number columns first: on plain, undisguised inputs every model guard is "
                f"good (for example Llama Guard 3 1B misses only {pct(lg1['clean_miss'])}). The next columns show "
                "what happens once attackers start disguising their requests."),
              H2("5.2 The heatmaps: every guard against every attack"),
              figure(RES / "figures" / "heatmap_harmful_light.png", 15.0,
                     "Figure 3. Harmful-request track. Left: miss rate (darker blue = more attacks get through). "
                     "Right: false-positive rate on harmless inputs given the same attack (darker orange = more "
                     "innocent users blocked). The top row is the plain, undisguised input."),
              figure(RES / "figures" / "heatmap_injection_light.png", 14.2,
                     "Figure 4. Prompt-attack track, read the same way. Notice how the ProtectAI column turns light "
                     "blue (few misses) exactly where its orange column turns dark (many false alarms)."),
              ]  # fmt: skip

    b64, hexr, mb, tags, ps = (T(LG1, t) for t in ("base64", "hex", "math_bold", "unicode_tags", "payload_split"))
    fr, bn, zu = (T(LG1, t) for t in ("translate_fr", "translate_bn", "translate_zu"))
    q_b64, q_leet, q_hyp, q_zu = (T(QW, t) for t in ("base64", "leetspeak", "hypothetical_frame", "translate_zu"))
    pg_sp, pg_b64 = T(PG, "char_spacing"), T(PG, "base64")
    pai_b64, pai_morse = T(PAI, "base64"), T(PAI, "morse")
    story += [H2("5.3 Seven findings, explained"),
              P("<b>1. Encodings are the biggest blind spot.</b> Llama Guard 3 1B misses "
                f"{pct(hexr['miss'])} of hex-encoded harmful requests. Qwen3Guard misses {pct(q_b64['miss'])} of "
                "Base64, hex, ROT13, Morse and reversed requests. Prompt Guard 86M misses "
                f"{pct(pg_b64['miss'])} of encoded jailbreaks. <i>Why it matters:</i> encoding a message costs an "
                "attacker one line of code."),
              P("<b>2. Some “catches” are really false alarms.</b> Llama Guard 3 1B seems to catch Base64 (only "
                f"{pct(b64['miss'])} missed), but it also flags {pct(b64['fpr'])} of <i>harmless</i> Base64, and its "
                f"AUROC there is {b64['auroc']:.2f}, which is coin-flip level. ProtectAI misses {pct(pai_b64['miss'])} "
                f"of encoded attacks because it flags {pct(pai_b64['fpr'])} of all encoded text; on Morse code its "
                f"AUROC is {pai_morse['auroc']:.2f}, worse than guessing. <i>Why it matters:</i> without harmless "
                "controls, both guards would look excellent here. Figure 5 below shows this pattern for every "
                "attack at once."),
              P("<b>3. Cheap character tricks still work.</b> Writing a harmful request in mathematical bold letters "
                f"gets past Llama Guard 3 1B {pct(mb['miss'])} of the time, and invisible Unicode tag characters "
                f"{pct(tags['miss'])}. Spacing out the letters, a trick publicly reported against Prompt Guard in "
                f"2024, still gets past it {pct(pg_sp['miss'])} of the time. On the other hand, fullwidth and bold "
                "letters do not fool Prompt Guard or ProtectAI at all, because their tokenizers already normalise "
                "those characters: a nice real-world example of canonicalization working."),
              P("<b>4. Rephrasing barely matters; rare languages do.</b> Meaning-checked paraphrases and "
                "“it's for a novel / for research / hypothetically” framings change the content guards' miss rates "
                "by only a few points. Translation is a different story: Llama Guard 3 1B misses "
                f"{pct(fr['miss'])} of French requests (a language it officially supports), {pct(bn['miss'])} of "
                f"Bengali and {pct(zu['miss'])} of Zulu. Splitting a request into fragments to be joined back "
                f"together gets past it {pct(ps['miss'])} of the time."),
              P("<b>5. False alarms are half the story.</b> Prompt Guard 86M blocks "
                f"{pct(pg['clean_fpr'])} of plain harmless prompts, and {pct(source_fpr(PG, 'jailbreakhub_regular'))} "
                "of normal prompts from online communities (often role-play requests such as “act as a Linux "
                f"terminal”). Its alternative “any attack” score blocks {pct(pg_any['clean_fpr'])}. Qwen3Guard's "
                f"strict mode blocks {pct(qw['clean_fpr'])} of plain harmless prompts, {pct(q_leet['fpr'])} of "
                f"harmless leetspeak and {pct(q_hyp['fpr'])} of harmless “hypothetically…” questions."),
              P("<b>6. An attacker who can retry almost always wins.</b> For every guard except ProtectAI, every "
                "single harmful item gets through at least one of the disguises (best-of-N evasion "
                f"{pct(lg1['best_of_n_evasion'])}). ProtectAI's best-of-N evasion is lower "
                f"({pct(pai['best_of_n_evasion'])}), but it blocks {pct(pai['any_transform_fpr'])} of harmless "
                "items under at least one disguise."),
              P("<b>7. A bigger model is not uniformly better.</b> See section 5.5."),
              KeepTogether([H2("5.4 Misses and false alarms on one picture"),
                            *figure_parts(tradeoff_png, 13.2,
                                   "Figure 5. Each dot is one attack. Up means more misses; right means more false "
                                   "alarms on harmless inputs. The diamond is the plain input, and the ideal is the "
                                   "bottom-left corner. Dots in the top-left are attacks that slip through; dots in the "
                                   "bottom-right are attacks that are only “caught” because the guard blocks the "
                                   "harmless versions too.")]),
              ]  # fmt: skip
    pick = ["none", "base64", "hex", "unicode_tags", "math_bold", "char_spacing", "payload_split", "benign_padding",
            "fiction_frame", "translate_bn", "translate_zu"]  # fmt: skip
    rows = [[("plain input" if t == "none" else t.replace("_", " ")), pct(p1.loc[t, "miss"]), pct(p8.loc[t, "miss"]),
             pct(p1.loc[t, "fpr"]), pct(p8.loc[t, "fpr"])] for t in pick]  # fmt: skip
    story += [KeepTogether([
                  H2("5.5 Llama Guard 3 1B versus 8B, on the same items"),
                  P(f"Because the 8B model was tested on a subset ({n8_pos} harmful and {n8_neg} harmless items), "
                    "the fair comparison is on exactly those items. The bigger model is clearly better on plain "
                    "requests, on most character tricks and on rare languages, but it has its own surprising "
                    "weaknesses:"),
                  data_table(["Attack", "Miss, 1B", "Miss, 8B", "FPR, 1B", "FPR, 8B"], rows,
                             [5.2, 2.6, 2.6, 2.6, 2.6])]),
              Spacer(1, 6),
              P(f"Two results stand out. First, the 8B model calls almost every encoded message “safe”: it misses "
                f"{pct(p8.loc['base64', 'miss'])} of Base64 and {pct(p8.loc['unicode_tags', 'miss'])} of tag-character "
                f"requests, but with a {pct(p8.loc['base64', 'fpr'])} false-positive rate, so it is not over-flagging "
                f"like the 1B model. Second, adding about 300 words of harmless gardening text before a harmful "
                f"request makes the 8B model miss {pct(p8.loc['benign_padding', 'miss'])} of them, while the 1B "
                f"model misses only {pct(p1.loc['benign_padding', 'miss'])}. We double-checked this surprising "
                "result: the model's own written verdicts say “safe”, and the identical prompts are handled correctly "
                "by the 1B model, so it is not a cut-off or software bug."),
              ]  # fmt: skip
    canon_rows = []
    for m, raw, can in ((LG1, lg1, lg1c), (LG8, lg8, lg8c), (QW, qw, qwc), (PG, pg, pgc)):
        canon_rows.append([LABEL[m], f"{pct(raw['mean_attack_miss'])} → {pct(can['mean_attack_miss'])}",
                           f"{pct(raw['mean_attack_fpr'])} → {pct(can['mean_attack_fpr'])}",
                           f"{pct(raw['best_of_n_evasion'])} → {pct(can['best_of_n_evasion'])}"])
    story += [KeepTogether([H2("5.6 Does canonicalization help?"),
                            data_table(["Guard", "Avg miss under attack", "Avg FPR under attack",
                                        "Best-of-N evasion"], canon_rows, [5.0, 4.0, 4.0, 4.0])]),
              Spacer(1, 6),
              figure(RES / "figures" / "canonicalization_light.png", 15.4,
                     "Figure 6. Light dots: the guard alone. Dark dots: with canonicalization. Left: share of "
                     "harmful items that get through at least one attack. Right: share of harmless items blocked "
                     "under at least one attack."),
              P("Canonicalization removes most of the damage done by mechanical tricks at a small false-alarm cost. "
                "But it cannot undo translation, paraphrase or payload splitting, so a determined attacker who "
                "tries many disguises still usually gets through. Defending against those needs guards that "
                "understand meaning across languages, or several layers of defense."),
              ]  # fmt: skip

    # ---------------------------------------------------------------- 6
    story += H1("6. Engineering lessons")
    story += [P("The results are interesting, but the way the project was built teaches just as much. These are "
                "habits worth copying in any evaluation or data project:"),
              numbered([
                  "<b>Measure both kinds of error.</b> Misses and false alarms together tell the truth; either one "
                  "alone can be gamed.",
                  "<b>Use controls.</b> Applying each attack to harmless inputs is what exposed the “fake catches” "
                  "of Llama Guard 3 1B on Base64 and of ProtectAI on all encodings.",
                  "<b>Check that the test still tests what you think.</b> Paraphrases and translations were "
                  "meaning-checked; in the trial run more than half of the paraphrases failed the check because two "
                  "thresholds disagreed, and this was caught and fixed before the full run.",
                  "<b>Pin and fingerprint everything.</b> Dataset versions, SHA-256 checksums, model revisions and "
                  "software versions are recorded, so the run can be repeated exactly.",
                  "<b>Start with a smoke test.</b> A tiny run with three items per source exercised every part of "
                  "the pipeline in under 20 minutes, before hours of computing were spent.",
                  "<b>Count everything.</b> The merge step reported 19,182 deterministic samples where 19,200 were "
                  "expected. That mismatch revealed a real bug: one harmless instruction had been picked for both "
                  "tracks and its two copies collided. The bug was fixed and its small effect (19 samples) is "
                  "documented rather than hidden.",
                  "<b>When a result surprises you, try to break it.</b> The 8B padding result was checked against "
                  "the model's written verdicts and against the 1B model on the same prompts before it was "
                  "reported.",
                  "<b>Report uncertainty and compare like with like.</b> Intervals on every rate, and paired "
                  "comparisons when models saw different items.",
                  "<b>Publish only what is needed.</b> Results store fingerprints, not harmful text.",
                  "<b>Automate.</b> Tests run on every change, and the whole benchmark runs from one button in "
                  "GitHub Actions, split into parallel jobs and cached.",
              ]),
              ]  # fmt: skip

    # ---------------------------------------------------------------- 7
    story += H1("7. Limits and responsible use")
    story += [bullets([
        "<b>Encodings assume a capable model behind the guard.</b> A Base64 or Morse message is only dangerous if "
        "the protected LLM can decode it. Large models can; small ones may not.",
        "<b>Llama Guard ran in compressed (quantized) form</b> through Ollama. This can shift borderline decisions "
        "slightly compared with the original weights.",
        "<b>Sample sizes are a few hundred items per track</b>, chosen to fit free CPU time. Differences of a few "
        "points are not meaningful; large ones are.",
        "<b>Prompt Guard is not a content filter.</b> It was built to detect manipulation, not harmful topics, so "
        "it is judged on the prompt-attack track.",
        "<b>Missing pieces:</b> Prompt Guard 2 (awaiting access approval) and the 8B model on the full item set.",
        "<b>English-centric data</b>, machine translation, a single paraphrasing model and fairly mild framing "
        "attacks.",
    ]),
        callout("Responsible use", "Everything here uses public datasets and widely documented techniques; nothing "
                "searches for new attacks against a specific product. The aim is to help people who deploy guards "
                "understand their weak spots and the price of fixing them. The published files contain no prompt "
                "texts."),
    ]  # fmt: skip

    # ---------------------------------------------------------------- 8
    story += H1("8. Try it yourself")
    story += [P("You need Python 3.10 or newer. The “smoke” profile runs three items per source and finishes "
                "quickly on a laptop for the small models."),
              P(escape("git clone " + REPO_URL) + "<br/>cd adversarial-guardrail-benchmark<br/>"
                "pip install torch --index-url https://download.pytorch.org/whl/cpu<br/>"
                'pip install -e ".[ml,dev]"<br/>pytest -q<br/>'
                "agb --profile smoke prepare --out work/inputs.jsonl<br/>"
                "agb --profile smoke score --inputs work/inputs.jsonl --model protectai-deberta-v2<br/>"
                "agb collect --inputs work/inputs.jsonl --scores-dir scores --out results_smoke/raw<br/>"
                "agb --profile smoke analyze --raw results_smoke/raw --out results_smoke", "code"),
              H2("Exercises"),
              numbered([
                  "A guard is tested on 200 harmful and 300 harmless inputs. It catches 170 harmful ones and blocks "
                  "24 harmless ones. What are its miss rate and false-positive rate? <i>(Answer: 15% and 8%.)</i>",
                  f"ProtectAI's AUROC on Morse code is {pai_morse['auroc']:.2f}, below 0.5. In plain words, what does "
                  "that tell you about how it treats harmful versus harmless Morse messages?",
                  "Add a new attack, for example Pig Latin. Look at src/agb/transforms/encodings.py and the "
                  "registry in src/agb/transforms/__init__.py, and add a unit test in tests/test_transforms.py. "
                  "Should the canonicalizer try to undo it?",
                  "Why do fullwidth letters fool Llama Guard 3 1B but not Prompt Guard? <i>Hint: look up “NFKC "
                  "normalization” and how DeBERTa's tokenizer prepares text.</i>",
                  "Look at the calibrated-threshold table in results/REPORT.md. Does tuning Llama Guard 3 1B for a 5% "
                  "false-positive rate fix its encoding problem?",
                  "Propose a defense against the translation attack. What would you expect it to cost in false "
                  "positives, and how would you measure that with this benchmark?",
                  "Why is the best-of-N evasion rate much higher than the average miss rate? Which one matters more "
                  "for a public chatbot?",
              ]),
              ]  # fmt: skip

    # ---------------------------------------------------------------- glossary
    story += H1("Glossary")
    gl = [
        ("AUROC", "Chance that a random harmful input scores higher than a random harmless one. 1.0 is perfect, "
                  "0.5 is guessing."),
        ("Base64 / hex / ROT13 / Morse", "Simple, reversible ways of writing text that people cannot read at a "
                                         "glance but LLMs can decode."),
        ("Benign / harmless input", "A message that should be allowed (a negative)."),
        ("Best-of-N evasion", "Share of harmful items that get through under at least one of the attacks."),
        ("Canonicalization", "Converting input into one standard form (decoding, removing invisible characters, "
                             "normalising letters) before classification."),
        ("Confusion matrix", "The four outcomes: caught, missed, false alarm, correctly allowed."),
        ("False negative (miss)", "A harmful input the guard lets through."),
        ("False positive (false alarm)", "A harmless input the guard blocks."),
        ("Guard classifier", "A small model that decides whether a message is safe before (or after) a large "
                             "model handles it."),
        ("Hash", "A short fingerprint computed from a text; it identifies the text without revealing it."),
        ("Homoglyph", "A character from another alphabet that looks identical to a Latin letter (Cyrillic “\u0430” "
                      "and Latin “a”)."),
        ("Jailbreak", "A prompt that tries to talk a model out of its safety rules."),
        ("LaBSE", "A model that maps sentences in 100+ languages to vectors; used here to check meaning."),
        ("Log-probability", "The model's (logarithmic) confidence in each possible next word; turned into a score "
                            "between 0 and 1."),
        ("Prompt injection", "Instructions hidden in content the model reads (a web page, an email) that try to "
                             "take control of it."),
        ("Quantization", "Storing a model's numbers with fewer bits (Q8_0, Q4_K_M) so it runs faster on a CPU."),
        ("Shard", "One slice of a big job, run in parallel with the other slices."),
        ("Smoke test", "A tiny end-to-end run that checks everything works before the expensive run."),
        ("Threshold", "The score above which the guard blocks; here 0.5."),
        ("Tokenizer", "The part of a model that splits text into pieces (tokens) before the model reads it."),
        ("Unicode tag characters", "Invisible characters that can secretly spell out text (“ASCII smuggling”)."),
        ("Wilson interval", "A range that probably contains the true rate, given how many examples were tested."),
        ("Zero-width character", "An invisible character with no width, which can be slipped between letters."),
    ]  # fmt: skip
    story += [data_table(["Term", "Meaning"], [[P(f"<b>{escape(k)}</b>", "cell"), escape(v)] for k, v in gl],
                         [4.6, 12.4], align_right_from=9)]

    # ---------------------------------------------------------------- references
    story += H1("References and credits")
    story += [bullets([
        "HarmBench: Mazeika et al., 2024 (MIT licence).",
        "XSTest: Röttger et al., NAACL 2024 (CC-BY-4.0).",
        "JailbreakHub (“Do Anything Now”): Shen et al., CCS 2024 (MIT licence).",
        "Tensor Trust: Toyer et al., ICLR 2024.",
        "Stanford Alpaca: Taori et al., 2023 (CC-BY-NC-4.0).",
        "Llama Guard 3 and Prompt Guard: Meta (Llama Community Licenses). Qwen3Guard: Alibaba Qwen team "
        "(Apache-2.0). ProtectAI deberta-v3-base-prompt-injection-v2 (Apache-2.0). NLLB-200: Meta (CC-BY-NC-4.0). "
        "LaBSE: Google (Apache-2.0).",
        f"Code, full results and a detailed technical report: <link href='{REPO_URL}' color='{BLUE}'>{REPO_URL}"
        "</link> (results/REPORT.md).",
    ], "small")]  # fmt: skip

    doc = GuideDoc(str(OUT))
    doc.multiBuild(story)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    build()
