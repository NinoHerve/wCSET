"""Simulate the tuning spikes, with exactly the parameters of the paper's simulation.

Same simulator, head model, connectome, spike, epoch, propagation, noise and SNR settings as
`processing/processing_simf.qmd` (`simulation.py`). Only three things differ:

- regions: the 50 tuning regions in `results/simf/tuning/tuning-regions.csv`, none of which is in
  Study I's probe set (so tuning and evaluation do not overlap);
- repetitions: 1 per region by default (`--repetitions`), and a different base seed;
- output: a separate folder (`--out`), never the paper's simulation folder. The script refuses to
  write into an existing folder.

Usage (mindmap env):
    python notebooks/simulate_tuning_spikes.py --out /media/nino/DATA/SIMF/derivatives/simulations-tuning
    python notebooks/simulate_tuning_spikes.py --out <scratch dir> --n-sources 1   # timing test
    # power-analysis pilot (notebooks/solve_pilot.py, notebooks/power_analysis.ipynb): 4 new runs per region
    python notebooks/simulate_tuning_spikes.py --repetitions 4 --seed 2027 \
        --out /media/nino/DATA/SIMF/derivatives/simulations-pilot
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

import mindmap as mm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.data_manager import SimfManager

PAPER_SIM = Path("/media/nino/DATA/SIMF/derivatives/simulations")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path, help="new output folder (must not exist)")
    parser.add_argument("--n-sources", type=int, default=None, help="only the first N regions (timing test)")
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repetitions", type=int, default=1, help="runs per region (power pilot: 4)")
    args = parser.parse_args()

    out = args.out.resolve()
    if out == PAPER_SIM.resolve() or PAPER_SIM.resolve() in out.parents:
        sys.exit(f"Refusing to write into the paper's simulation folder: {out}")
    if out.exists():
        sys.exit(f"Output folder already exists, refusing to overwrite: {out}")

    dm = SimfManager(str(PROJECT / "processing" / "simf.config.yml"))
    dm.add_root("sim", out)          # redirect every simulation output path to the new folder
    sub = "10"

    # --- identical to processing_simf.qmd, simulation.py -------------------------------------
    fwd = dm.load(sub, "fwd", desc="sim")
    con = dm.load(sub, "con")
    eeg = dm.load(sub, "ave-temp")
    parc = dm.load(sub, "dseg", space="fs", segmentation="multiconn2022", scale="4")
    ras2tkras = dm.load(sub, "ras2tkras")

    spike = mm.analysis.simulation.SpikeSignal(width=0.05, onset=0.5, repeat=1, scale=1e-8, event_id=1)
    head_model = mm.analysis.simulation.HeadModel(
        sensors=eeg, forward=fwd, connectivity_weights=con > 0, connectivity_lengths=con,
        parcellation=parc, ras2tkras=ras2tkras,
    )
    epoch_config = mm.analysis.simulation.EpochConfig(
        tmin=-0.45, tmax=0.45, baseline=(-0.45, -0.2), signal_window=(-0.1, 0.45),
    )
    prop_config = mm.analysis.simulation.PropagationConfig(
        conduction_speed=4.5, u_fiber_speed=2.5, u_fiber_max_length=35,
        nodal_delay=7, spatial_decay=0.5, jitter=1,
    )
    sim = mm.analysis.simulation.Simulator(head_model, epoch_config)
    noise_config = mm.analysis.simulation.NoiseConfig()
    snr_config = mm.analysis.simulation.SNRConfig(method="peak", spike_window=(-0.03, 0.03))
    # ------------------------------------------------------------------------------------------

    regions = pd.read_csv(PROJECT / "results/simf/tuning/tuning-regions.csv")["region"].tolist()
    probe_set = set(pd.read_csv(PROJECT / "results/simf/intermediate/probe-set.csv")["region"])
    assert not set(regions) & probe_set, "tuning regions overlap Study I's probe set"
    if args.n_sources:
        regions = regions[: args.n_sources]
    sources = [mm.analysis.simulation.Source([br], spike) for br in regions]

    t0 = time.time()
    mm.analysis.simulation.generate_simulation_dataset(
        dm=dm, subject=sub, simulator=sim, sources=sources, prop_config=prop_config,
        repetitions=args.repetitions, snr_targets=[5.0], snr_config=snr_config, noise_config=noise_config,
        base_seed=args.seed, entities={"task": "spk"}, n_jobs=args.n_jobs,
    )
    print(f"simulated {len(sources)} regions x {args.repetitions} runs into {out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
