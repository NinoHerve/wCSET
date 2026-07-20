"""
Utilities to run processing steps in loops with error handling.
"""

import logging
import traceback
from functools import wraps
from pathlib import Path

import numpy as np

# Root logger: only WARNING+ reaches stdout. Propagated records from loggers
# we don't control (e.g. pyxu, which logs one INFO line per solver iteration
# under a logger named after its temp workdir, not under "pyxu") skip the
# ancestor loggers' level checks entirely, so the level must live on the
# handler itself, not just on the root logger via basicConfig(level=...).
_root_handler = logging.StreamHandler()
_root_handler.setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING, handlers=[_root_handler], force=True)

# Dedicated logger for this project's own messages. propagate=False keeps it
# independent of the root handler above, so our INFO messages always show.
logger = logging.getLogger("wcset")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(_handler)

def loop(subjects):
    """Allows to loop processing step through a list of subjects."""
    def decorator(func):
        """Decorator for processing function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper for looping"""
            for sub in subjects:
                try:
                    func(sub, *args, **kwargs)
                    logger.info("SUCCESS: sub-%s", sub)

                except Exception as _:
                    logger.error("FAILED: sub-%s", sub)
                    logger.error(traceback.format_exc())
        return wrapper
    return decorator


def check_files(files: list[Path], silent: bool = False):
    """Check if files exist. Returns 1 at least 1 file does."""

    for f in files:
        # exception (stc files)
        if f.name.endswith("stc"):
            f = f.parent/f"{f.name}-lh.stc"
        if f.exists():
            if not silent:
                logger.info("Following file already exists: %s", str(f))
            return True
        
        # create parent directory if doesn't exist
        f.parent.mkdir(parents=True, exist_ok=True)

    return False


def resample_pr_curve(precision, recall, n_points=100):
    """
    Resamples a precision-recall curve onto a fixed, common recall grid.

    A raw curve (e.g. from `sklearn.metrics.precision_recall_curve`) carries as
    many points as there are unique prediction values, which gets unwieldy when
    pooling many runs and makes curves from different runs impossible to average
    directly (their recall values don't line up). Interpolating onto a shared
    grid solves both: it bounds the point count and gives every run the same
    x-axis so per-run curves can be averaged into a summary curve with a CI.

    Uses the standard "interpolated precision" (max precision achievable at any
    recall >= r), which also removes the sawtooth wiggle of the raw curve.

    Parameters
    ----------
    precision, recall: array-like
        Raw precision-recall curve, e.g. from `mm.core.calc_performance_curves`.

    n_points: int
        Number of points of the resampled curve.

    Returns
    -------
    precision_grid, recall_grid: np.ndarray of shape (n_points,)
    """
    order = np.argsort(recall)
    recall_sorted = np.asarray(recall)[order]
    precision_sorted = np.asarray(precision)[order]

    # best precision achievable at any recall >= r
    envelope = np.maximum.accumulate(precision_sorted[::-1])[::-1]

    recall_grid = np.linspace(0, 1, n_points)
    precision_grid = np.interp(recall_grid, recall_sorted, envelope)

    return precision_grid, recall_grid
