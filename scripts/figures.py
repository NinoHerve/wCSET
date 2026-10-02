"""Shared figure style and helpers for index.qmd, annex/annex.qmd and the processing notebooks.

Statistics read-out and paired-difference panels come from mindmap (`PairwiseResult.vs`,
`MetricSuite.stat.paired_differences`, `MetricSuite.plot.paired_differences`).

Imported with `from scripts.figures import *`; everything listed in `__all__` is available.
"""
import matplotlib.colors
import matplotlib.image as mpimg
import numpy as np
from matplotlib.cm import ScalarMappable

__all__ = [
    "COLORS", "LINE_COLOR", "METHODS", "METHOD_LABELS", "FILL_ALPHA", "LINE_MATTE",
    "fmt_p", "fmt_stat", "fmt_num", "main_cluster", "mark_reference", "matte",
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

# Study I methods. Beltrami (wCSET on the Laplace-Beltrami mesh eigenbasis) is computed for the
# clinical cohorts only (Studies II/III) and appended locally where those studies build their
# method order. MCE is excluded from every study: its L1 sparsity makes correlation-type metrics
# react to whether it fires at all rather than to the quality of its prior.
METHODS = ["wcset", "cset", "sloreta", "dspm", "wmne"]
METHOD_LABELS = {
    "wcset": "wCSET", "cset": "CSET", "sloreta": "sLORETA", "dspm": "dSPM", "wmne": "wMNE",
    "beltrami": "Beltrami",
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

def main_cluster(cluster_df):
    """Largest-mass cluster of one competitor's cluster table."""
    return cluster_df.loc[cluster_df["mass"].abs().idxmax()]

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
