"""Shared figure style and helpers for index.qmd, annex/annex.qmd and the processing notebooks.

Imported with `from scripts.figures import *`; everything listed in `__all__` is available.
"""
import matplotlib.colors
import matplotlib.image as mpimg
import numpy as np
from matplotlib.cm import ScalarMappable

__all__ = [
    "COLORS", "LINE_COLOR", "METHODS", "METHOD_LABELS", "VIOLIN_MIN_N", "FILL_ALPHA", "LINE_MATTE",
    "fmt_p", "fmt_stat", "fmt_num", "main_cluster", "competitor_row", "ref_effect", "stars_for",
    "mark_reference", "ref_labels", "matte", "soften_boxes", "paired_diff_panel", "mark_direction",
    "crop_white", "image_panel", "colorbar", "share_ylim", "column_headers", "method_boxes",
    "vs_annotations",
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

# Study I methods. Beltrami (wCSET on the Laplace-Beltrami mesh eigenbasis) is computed for the
# clinical cohorts only (Studies II/III) and appended locally where those studies build their
# method order. MCE is excluded from every study: its L1 sparsity makes correlation-type metrics
# react to whether it fires at all rather than to the quality of its prior.
METHODS = ["wcset", "cset", "sloreta", "dspm", "wmne"]
METHOD_LABELS = {
    "wcset": "wCSET", "cset": "CSET", "sloreta": "sLORETA", "dspm": "dSPM", "wmne": "wMNE",
    "beltrami": "Beltrami",
}

# Paired-difference panels: points below VIOLIN_MIN_N units, violins above.
VIOLIN_MIN_N = 30
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

def main_cluster(cluster_df):
    """Largest-mass cluster of one competitor's cluster table."""
    return cluster_df.loc[cluster_df["mass"].abs().idxmax()]

def competitor_row(result, competitor, reference="wcset"):
    """Result row comparing `reference` with `competitor`, in either stored order."""
    t = result.table
    row = t[(t["group_a"] == reference) & (t["group_b"] == competitor)]
    if row.empty:
        row = t[(t["group_a"] == competitor) & (t["group_b"] == reference)]
    return row.iloc[0]

def ref_effect(result, competitor, reference="wcset", higher_is_better=True):
    """Effect size signed so positive = `reference` better."""
    row = competitor_row(result, competitor, reference)
    es = row["effect_size"] if row["group_a"] == reference else -row["effect_size"]
    return es if higher_is_better else -es

def stars_for(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# ---------------------------------------------------------------- labels and colors
def mark_reference(ax, reference, suffix=" (ref.)"):
    """Appends `suffix` to `reference`'s entry in `ax`'s already-built legend, in place."""
    for text in ax.get_legend().get_texts():
        if text.get_text() == reference:
            text.set_text(text.get_text() + suffix)
            break

def ref_labels(names, reference, suffix=" (ref.)"):
    """Tick-label list with `suffix` appended to `reference`'s entry."""
    return [f"{n}{suffix}" if n == reference else n for n in names]

def matte(color, strength=LINE_MATTE):
    """Opaque color blended toward white, matching the softened fills."""
    r, g, b = matplotlib.colors.to_rgb(color)
    return (1 - strength + strength * r, 1 - strength + strength * g, 1 - strength + strength * b)

def soften_boxes(ax):
    """Apply FILL_ALPHA to box fills already drawn on `ax`, keeping edges opaque."""
    for patch in ax.patches:
        r, g, b, _ = patch.get_facecolor()
        patch.set_facecolor((r, g, b, FILL_ALPHA))


# ---------------------------------------------------------------- panels
def paired_diff_panel(ax, diffs, order, colors, annotations=None, seed=0):
    """Draw per-unit paired differences, one column per entry of `order`.

    diffs: {name: 1-D array}, already oriented so positive = reference better.
    annotations: {name: [line, ...]} drawn above each column, top line first.
    """
    ax.axhline(0, color=LINE_COLOR, linewidth=1, zorder=0)
    rng = np.random.default_rng(seed)
    for i, c in enumerate(order):
        y = np.asarray(diffs[c], dtype=float)
        if len(y) >= VIOLIN_MIN_N:
            vp = ax.violinplot([y], positions=[i], widths=0.7, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(colors[c])
                body.set_edgecolor(colors[c])
                body.set_alpha(FILL_ALPHA)
                body.set_zorder(1)
            q1, q3 = np.percentile(y, [25, 75])
            ax.plot([i, i], [q1, q3], color="#0b0b0b", linewidth=1.5, zorder=3)
        else:
            x = i + rng.uniform(-0.12, 0.12, size=len(y))
            ax.scatter(x, y, s=18, color=colors[c], alpha=0.8,
                       edgecolors="white", linewidths=0.6, zorder=2)
        med = np.median(y)
        ax.plot([i - 0.22, i + 0.22], [med, med], color="#0b0b0b", linewidth=2, zorder=4)

    ymax = max(np.max(diffs[c]) for c in order)
    ymin = min(np.min(diffs[c]) for c in order)
    span = (ymax - ymin) or 1.0
    n_lines = max((len(v) for v in (annotations or {}).values()), default=0)
    step = 0.075 * span
    for i, c in enumerate(order):
        for k, line in enumerate((annotations or {}).get(c, [])):
            ax.text(i, ymax + 0.02 * span + (n_lines - 1 - k) * step, line, ha="center", va="bottom",
                    fontsize=8 if k == 0 else 7, color="#222222" if k == 0 else "#555555")
    ax.set_ylim(ymin - 0.05 * span, ymax + (0.06 + n_lines * 0.075) * span)
    ax.set_xticks(range(len(order)))
    ax.spines[["top", "right"]].set_visible(False)

def mark_direction(ax, better="wCSET better", worse="other method better"):
    """Label which sign of the difference favors which side."""
    for text, dy, va in [(f"{better} ↑", 4, "bottom"), (f"{worse} ↓", -4, "top")]:
        ax.annotate(text, xy=(1.0, 0.0), xycoords=("axes fraction", "data"),
                    xytext=(4, dy), textcoords="offset points", fontsize=7,
                    color="#666666", ha="left", va=va)

def method_boxes(ax, wide, order, colors=COLORS, reference="wcset", chance=None, chance_value=None):
    """Box per method of the columns of `wide` (one row per pairing unit), outliers hidden.

    chance: optional (lo, hi) band shaded behind the boxes; chance_value: optional dotted line.
    """
    if chance is not None:
        ax.axhspan(*chance, color="#c3c2b7", alpha=0.35, linewidth=0, zorder=0)
    if chance_value is not None:
        ax.axhline(chance_value, color=LINE_COLOR, linestyle=":", linewidth=1, zorder=0)
    bp = ax.boxplot([wide[m].to_numpy(dtype=float) for m in order], positions=range(len(order)),
                    widths=0.55, patch_artist=True, showfliers=False)
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(colors[m])
        patch.set_edgecolor("#222222")
    soften_boxes(ax)
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color("#222222")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(ref_labels([METHOD_LABELS[m] for m in order], METHOD_LABELS[reference]),
                       rotation=45, ha="right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

def vs_annotations(result, diffs, competitors, reference="wcset", effect="dz"):
    """Lines above each paired-difference column: significance, effect size, win rate vs. 50%."""
    return {c: [stars_for(competitor_row(result, c, reference)["p_corrected"]),
                f"{effect}={ref_effect(result, c, reference):+.2f}",
                f"{(np.asarray(diffs[c]) > 0).mean() - 0.5:+.0%}"] for c in competitors}


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
