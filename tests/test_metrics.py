import math

import numpy as np
import pandas as pd
import pytest

from agb.metrics import auroc, per_transform, summary, threshold_at_fpr, wilson


def test_wilson_interval():
    p, lo, hi = wilson(0, 10)
    assert p == 0 and lo == 0 and hi == pytest.approx(0.2775, abs=1e-3)
    p, lo, hi = wilson(5, 10)
    assert p == 0.5 and lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert all(math.isnan(x) for x in wilson(0, 0))


def test_auroc():
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auroc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert auroc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert math.isnan(auroc([], [0.1]))


@pytest.mark.parametrize("target", [0.0, 0.05, 0.1, 0.5])
def test_threshold_at_fpr_never_exceeds_target(target):
    rng = np.random.default_rng(0)
    neg = np.round(rng.random(200), 2)  # many ties
    t = threshold_at_fpr(neg, target)
    assert (neg >= t).mean() <= target + 1e-12


def _frame():
    rows = []
    # 4 harmful items, 4 benign items, transforms: none, enc
    scores = {
        ("h1", "none"): 0.9, ("h1", "enc"): 0.1,   # caught clean, evades under enc
        ("h2", "none"): 0.8, ("h2", "enc"): 0.7,   # caught both
        ("h3", "none"): 0.2, ("h3", "enc"): 0.9,   # missed clean
        ("h4", "none"): 0.95, ("h4", "enc"): 0.6,  # caught both
        ("b1", "none"): 0.1, ("b1", "enc"): 0.9,   # FP under enc
        ("b2", "none"): 0.2, ("b2", "enc"): 0.1,
        ("b3", "none"): 0.6, ("b3", "enc"): 0.2,   # FP clean
        ("b4", "none"): 0.0, ("b4", "enc"): 0.0,
    }  # fmt: skip
    for (item, t), s in scores.items():
        rows.append(
            {
                "item_id": item,
                "transform": t,
                "family": "clean" if t == "none" else "encoding",
                "label": int(item.startswith("h")),
                "source": "x",
                "valid": 1,
                "score": s,
            }
        )
    return pd.DataFrame(rows)


def test_per_transform_and_summary():
    df = _frame()
    pt = per_transform(df, "score", 0.5, None, continuous=True).set_index("transform")
    assert pt.loc["none", "miss"] == 0.25 and pt.loc["none", "fpr"] == 0.25
    assert pt.loc["enc", "miss"] == 0.25 and pt.loc["enc", "fpr"] == 0.25
    # flip: of h1,h2,h4 (caught clean), only h1 evades under enc
    assert pt.loc["enc", "flip"] == pytest.approx(1 / 3)
    s = summary(df, "score", 0.5, ["enc"])
    assert s["best_of_n_evasion"] == 0.5  # h1 (via enc) and h3 (clean) evade
    assert s["any_transform_fpr"] == 0.5  # b1 (enc) and b3 (clean)
    assert s["worst_transform"] == "enc"


def test_invalid_samples_are_excluded():
    df = _frame()
    df.loc[(df["item_id"] == "h1") & (df["transform"] == "enc"), "valid"] = 0
    pt = per_transform(df, "score", 0.5, None, continuous=True).set_index("transform")
    assert pt.loc["enc", "n_pos"] == 3 and pt.loc["enc", "miss"] == 0.0
