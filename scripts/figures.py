"""Shared figure style and helpers for index.qmd, annex/annex.qmd and the processing notebooks.

Statistics read-out and paired-difference panels come from mindmap (`PairwiseResult.vs`,
`MetricSuite.stat.paired_differences`, `MetricSuite.plot.paired_differences`).

Imported with `from scripts.figures import *`; everything listed in `__all__` is available.
"""
import hashlib
import inspect
import json
import shutil
from pathlib import Path

import matplotlib.colors
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from mindmap.metric.results import ClusterResult, PairwiseResult

_RESULT_TYPES = {"ClusterResult": ClusterResult, "PairwiseResult": PairwiseResult}

__all__ = [
    "COLORS", "LINE_COLOR", "METHODS", "METHOD_LABELS", "FILL_ALPHA", "LINE_MATTE",
    "fmt_p", "fmt_stat", "fmt_num", "fmt_ci", "main_cluster", "cached", "margin_tests", "margin_verdicts",
    "paired_margin_plot", "mark_reference", "matte",
    "crop_white", "image_panel", "colorbar", "share_ylim", "column_headers",
]

# ---------------------------------------------------------------- style
COLORS = {
    "wcset":    "#2a78d6",  # blue
    "cset":     "#1baf7a",  # aqua
    "sloreta":  "#008300",  # green
    "dspm":     "#4a3aa7",  # violet
    "wmne":     "#e34948",  # red
    "beltrami": "#eda100",  # ochre - Studies II/III only
}
LINE_COLOR = "#898781"  # neutral gray for reference lines

# Study I methods. GBF (wCSET on the Laplace-Beltrami mesh eigenbasis; tag "beltrami") is computed for the
# clinical cohorts only (Studies II/III) and appended locally where those studies build their
# method order. MCE is excluded from every study: its L1 sparsity makes correlation-type metrics
# react to whether it fires at all rather than to the quality of its prior.
METHODS = ["wcset", "cset", "sloreta", "dspm", "wmne"]
METHOD_LABELS = {
    "wcset": "wCSET", "cset": "CSET", "sloreta": "sLORETA", "dspm": "dSPM", "wmne": "wMNE",
    "beltrami": "GBF",
}

FILL_ALPHA = 0.45  # shared fill opacity for boxes and violins
LINE_MATTE = 0.75  # line strength: lines stay opaque but softened toward white


# ---------------------------------------------------------------- statistics quoted in the prose
def fmt_p(p, thresh=0.001):
    """p-value for prose, e.g. '< 0.001' or '= 0.023'."""
    return f"< {thresh}" if p < thresh else f"= {p:.3f}"

def fmt_stat(value, digits=2):
    """Effect size (rank-biserial, Cohen's dz, etc.) for prose, e.g. '-0.48'."""
    return f"{value:.{digits}f}"

def fmt_num(value, digits=1):
    """Plain descriptive number (medians, mm, %, correlation units) for prose."""
    return f"{value:.{digits}f}"

def fmt_ci(row, name, digits=1, unit="", level=0.95):
    """Estimate with its CI from a `vs()` row, e.g. '-0.8 mm (95% CI -1.2 to -0.4)': a test's effect size
    (`<name>_ci_low`/`_ci_high` columns) or an `EstimateResult`'s `"estimate"` (`ci_low`/`ci_high`). `level` is the
    CI's coverage, as computed by the test (e.g. 0.99 for a Bonferroni-corrected family of five)."""
    low, high = (f"{name}_ci_low", f"{name}_ci_high") if f"{name}_ci_low" in row else ("ci_low", "ci_high")
    return f"{row[name]:.{digits}f}{unit} ({level * 100:g}% CI {row[low]:.{digits}f} to {row[high]:.{digits}f})"

def main_cluster(cluster_df):
    """Largest-mass cluster of one competitor's cluster table."""
    return cluster_df.loc[cluster_df["mass"].abs().idxmax()]

def cached(test, folder, *args, **kwargs):
    """`test(*args, **kwargs)` (a bound `suite.stat.*` method), reloaded from `folder` instead of rerun when
    it was saved by the very same call and still matches the suite's data (`suite.verify`). For the
    expensive cluster permutations, which index.qmd and the Annex both read."""
    folder = Path(folder)
    call = hashlib.sha1(repr((test.__name__, args, sorted(kwargs.items()))).encode()).hexdigest()
    suite = test.__self__._suite
    if (folder/"call.json").exists():
        saved = json.loads((folder/"call.json").read_text())
        if saved["call"] == call:
            result = _RESULT_TYPES[saved["type"]].load(folder)
            if suite.verify(result):
                return result
    result = test(*args, **kwargs)
    if folder.exists():
        shutil.rmtree(folder)
    result.save(folder)
    (folder/"call.json").write_text(json.dumps({"call": call, "type": type(result).__name__}))
    return result

# ---------------------------------------------------------------- tests against the margin (±δ)
# Stars in the paired-difference panels test a meaningful effect in either direction: ▲ = significantly
# better by more than +δ, ▼ = significantly worse by more than −δ (one-sided, Holm within each direction),
# ns = neither. Estimates, CIs, effect sizes and counts come from the unchanged two-sided test ("base").
# group_a - group_b vs. the offset, for "better"/"worse" read in vs(reference, higher_is_better)'s orientation
_MARGIN_SIDES = {
    True: {"better": (+1, "greater"), "worse": (-1, "less")},
    False: {"better": (-1, "less"), "worse": (+1, "greater")},
}


def margin_tests(test, *args, delta, reference, higher_is_better=True, **kwargs):
    """`test(*args, **kwargs)` (a bound `suite.stat.wilcoxon`/`.bootstrap` with `reference` as every pair's
    group_a) three times: "base" as given (two-sided vs. 0: estimates, CIs, effect sizes), and one-sided
    against the margin in each direction -- "better" (reference better by more than `delta`) and "worse"
    (worse by more than `delta`), "better"/"worse" read as in `result.vs(reference, higher_is_better)`.
    Returns {"base", "better", "worse", "delta", "reference", "higher_is_better"}."""
    base = test(*args, **kwargs)
    groups = set(base.table["group_a"])
    assert groups == {reference}, f"every comparison must have {reference!r} as group_a, got {groups}"
    side_kwargs = dict(kwargs)
    if "ci" in inspect.signature(test).parameters:
        side_kwargs["ci"] = False  # intervals come from the base test
    out = {"base": base, "delta": delta, "reference": reference, "higher_is_better": higher_is_better}
    for direction, (sign, alternative) in _MARGIN_SIDES[higher_is_better].items():
        result = test(*args, **{**side_kwargs, "offset": sign * delta, "alternative": alternative})
        read = result.vs(reference, higher_is_better=higher_is_better)
        # vs() re-expresses each test as "reference better by more than offset": check the orientation
        assert (read["alternative"] == ("greater" if direction == "better" else "less")).all(), (direction, read["alternative"])
        assert np.allclose(read["offset"].astype(float), delta if direction == "better" else -delta), (direction, read["offset"])
        out[direction] = result
    return out


def _stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def margin_verdicts(tests, alpha=0.05):
    """One row per competitor: `direction` ("better"/"worse"/None, reference vs. competitor), its corrected
    one-sided `p_corrected` (the smaller of the two when neither is significant), both p-values, and the
    `stars` label drawn in the figures ("▲ ***", "▼ *", "ns")."""
    ref, hib = tests["reference"], tests["higher_is_better"]
    better = tests["better"].vs(ref, higher_is_better=hib)["p_corrected"]
    worse = tests["worse"].vs(ref, higher_is_better=hib)["p_corrected"]
    rows = {}
    for c in better.index:
        pb, pw = float(better[c]), float(worse[c])
        assert not (pb < alpha and pw < alpha), (c, pb, pw)
        direction = "better" if pb < alpha else "worse" if pw < alpha else None
        p = pb if direction == "better" else pw if direction == "worse" else min(pb, pw)
        stars = "ns" if direction is None else ("▲ " if direction == "better" else "▼ ") + _stars(p)
        rows[c] = {"direction": direction, "p_corrected": p, "p_better": pb, "p_worse": pw, "stars": stars}
    return pd.DataFrame.from_dict(rows, orient="index")


def paired_margin_plot(suite, tests, order, ax, margin_label="δ", annotate=("rank_biserial", "counts"), **kwargs):
    """`suite.plot.paired_differences` of the base test, its first annotation line replaced by the margin
    verdict (`margin_verdicts`), and dashed lines at ±δ. Positive = reference better, as in the tests."""
    ref, hib, delta = tests["reference"], tests["higher_is_better"], tests["delta"]
    plot = suite.plot.paired_differences(
        tests["base"], reference=ref, higher_is_better=hib, order=order, annotate=("stars",) + tuple(annotate),
        show_offset=False, ax=ax, **kwargs,
    )
    verdicts = margin_verdicts(tests)
    n_lines = 1 + len(annotate)
    texts = plot.artists["text"]
    assert len(texts) == n_lines * len(order), (len(texts), n_lines, order)
    for i, c in enumerate(order):
        texts[i * n_lines].set_text(verdicts.loc[c, "stars"])
    for sign, name in [(+1, f"+{margin_label}"), (-1, f"−{margin_label}")]:
        ax.axhline(sign * delta, color="#0b0b0b", linestyle="--", linewidth=1, alpha=0.5, zorder=0.5)
        ax.annotate(name, xy=(0.0, sign * delta), xycoords=("axes fraction", "data"), xytext=(3, 2 * sign),
                    textcoords="offset points", fontsize=7, ha="left", va="bottom" if sign > 0 else "top", alpha=0.7)
    plot.data.attrs["margin_verdicts"] = verdicts
    return plot


# ---------------------------------------------------------------- labels and colors
def mark_reference(ax, reference, suffix=" (ref.)"):
    """Appends `suffix` to `reference`'s entry in `ax`'s already-built legend, in place."""
    for text in ax.get_legend().get_texts():
        if text.get_text() == reference:
            text.set_text(text.get_text() + suffix)
            break

def matte(color, strength=LINE_MATTE):
    """Opaque color blended toward white, matching the softened fills."""
    r, g, b = matplotlib.colors.to_rgb(color)
    return (1 - strength + strength * r, 1 - strength + strength * g, 1 - strength + strength * b)

# ---------------------------------------------------------------- images and layout
def crop_white(img, pad=4):
    """Trim the white margin around a rendered brain."""
    mask = img[..., :3].min(axis=2) < 0.98
    rows, cols = np.where(mask)
    return img[max(rows.min() - pad, 0):rows.max() + pad, max(cols.min() - pad, 0):cols.max() + pad]

def image_panel(ax, path, crop=True):
    """Show a rendered image on `ax`, white margin trimmed, without ticks or frame."""
    img = mpimg.imread(path)
    ax.imshow(crop_white(img) if crop else img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def colorbar(fig, rect, clim, cmap, label):
    """Horizontal colorbar at figure coordinates `rect` = [x0, y0, width, height]."""
    cax = fig.add_axes(rect)
    fig.colorbar(ScalarMappable(norm=matplotlib.colors.Normalize(*clim), cmap=cmap),
                 cax=cax, orientation="horizontal")
    cax.set_xlabel(label, fontsize=8)
    cax.tick_params(labelsize=7)
    return cax

def share_ylim(axes):
    """Give every axes in `axes` the union of their y-limits."""
    axes = list(axes)
    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo, hi)

def column_headers(fig, headers, y):
    """Bold header above each column: `headers` = [(top axes of the column, text), ...]."""
    for ax, text in headers:
        fig.text(ax.get_position().x0, y, text, fontsize=11, fontweight="bold", ha="left", va="bottom")
