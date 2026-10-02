"""
Module Name: review_resection.py
Description: Open freeview with a subject's pre-op T1w, post-op T1w and
RAMPS resection mask for visual QC.

Author: Nino Herve
Created: 2026-07-10
License: MIT
"""

import argparse
import subprocess

from data_manager import SpikeManager
from utils import logger


def review_resection(dm, sub):
    """Open freeview on the pre-op T1w, post-op T1w and resection mask for one subject."""
    sub = sub.removeprefix("sub-")

    pre_t1 = dm.get_path(sub, "T1w", session="preop", res="cmp")
    post_t1 = dm.get_path(sub, "T1w", session="postop", space="individual", desc="brain")

    output_dir = dm.roots["mm"].parent / "ramps" / f"sub-{sub}"
    fmask = output_dir / "RAMPS_Resection_Mask_Output" / "RAMP_The_resection_mask_in_PRE.nii.gz"

    missing = [f for f in (pre_t1, post_t1, fmask) if not f.exists()]
    if missing:
        for f in missing:
            logger.warning("sub-%s: missing %s", sub, f)
        return

    subprocess.Popen([
        "freeview",
        "-v", str(pre_t1), str(post_t1), f"{fmask}:colormap=heat:opacity=0.5",
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subjects", nargs="+", help="Subject IDs, e.g. pat01 or sub-pat01")
    args = parser.parse_args()

    dm = SpikeManager("processing/spike.config.yml")

    for sub in args.subjects:
        review_resection(dm, sub)
