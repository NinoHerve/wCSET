"""
Standalone launcher for `mindmap`'s interactive harmonics viewer.

Run as a separate process (e.g. `!python -m scripts.view_harmonics --desc fused`
from a notebook cell, with the repo root as the working directory) so PyVista
opens a real OS window instead of Jupyter's inline/embedded renderer, which
swallows the arrow-key events the viewer relies on to scroll through
harmonics. Must be run with `-m` (not as a plain script path) so that
`scripts.data_manager` resolves as an importable package.

Pass several `--desc` values to compare conditions side by side, each in its
own linked panel, scrolled through together with one shared slider/keyboard
(several `mm.viz.BrainPlot.from_source_space` joined with `|`). A single
`--desc` falls back to a plain single-panel `BrainPlot`.
"""

import argparse

import mindmap as mm
from scripts.data_manager import SimfManager

VALID_DESCS = ["original", "fused", "fused-bin", "geom-atlas", "geom-atlas-bin", "beltrami"]


def load_eig(dm, subject, desc):
    """Load harmonics for one condition; 'original' means the true connectome (no desc entity)."""
    return dm.load(subject, "eig", **({} if desc == "original" else {"desc": desc}))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="processing/simf.config.yml", help="Path to configuration YAML file.")
    parser.add_argument("--subject", default="10", help="Subject ID.")
    parser.add_argument(
        "--desc",
        nargs="+",
        default=["original"],
        choices=VALID_DESCS,
        metavar="DESC",
        help=f"One or more conditions to view (choices: {', '.join(VALID_DESCS)}). "
             "Several values open one linked panel per condition.",
    )
    args = parser.parse_args()

    dm = SimfManager(args.config)
    src = dm.load(args.subject, "fwd", desc="sim")["src"]

    if len(args.desc) == 1:
        eig = load_eig(dm, args.subject, args.desc[0])
        mm.viz.BrainPlot.from_source_space(src, values=eig["vec"], states=eig["val"]).show()
    else:
        panel = None
        for desc in args.desc:
            eig = load_eig(dm, args.subject, desc)
            brain = mm.viz.BrainPlot.from_source_space(
                src, values=eig["vec"], states=eig["val"], title=desc,
            )
            panel = brain if panel is None else panel | brain
        panel.show()


if __name__ == "__main__":
    main()
