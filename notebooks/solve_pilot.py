"""Solve and score the power-analysis pilot spikes, with exactly Study I's solvers and metrics.

The pilot spikes come from `simulate_tuning_spikes.py --repetitions 4 --seed 2027` (50 tuning regions, none in
Study I's probe set, 4 runs each). Each run is solved with the 5 Study I methods, set up as in
`processing/processing_simf.qmd`'s `solve_inverse.py`, and scored as in its `comparison.py`: spatial
correlation over time against the noisy ground truth, and localization error at 25% of the peak (qrp).
STCs are not saved; only the metrics are, as a MetricSuite with the same metric names and coordinates as
`results/simf/flagship`, so `notebooks/power_analysis.ipynb` can run the exact tests of `index.qmd`.

Usage (mindmap env):
    python notebooks/solve_pilot.py --check-parity       # flagship run 1 must reproduce its saved STCs/metrics
    python notebooks/solve_pilot.py --sim <scratch>/pilot-timing --out <scratch>/pilot-timing-out --n-runs 1
    nohup python notebooks/solve_pilot.py > results/simf/power_pilot.log 2>&1 &
"""
import argparse
import ast
import shutil
import sys
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd

import mindmap as mm
from mindmap.analysis.functional.common.metric import find_peak_localization_error
from mindmap.analysis.functional.inverse import InverseSimTask, run_parallel_pipeline

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.data_manager import SimfManager

PAPER_SIM = Path("/media/nino/DATA/SIMF/derivatives/simulations")
PILOT_SIM = Path("/media/nino/DATA/SIMF/derivatives/simulations-pilot")
OUT = PROJECT / "results/simf/power_pilot"
SUB = "10"
METHODS = ["wcset", "cset", "sloreta", "dspm", "wmne"]  # Study I's methods (scripts/figures.py)
CROP = [-0.02, 0.15]  # s, solve_inverse.py's crop


# --- identical to solve_inverse.py (processing_simf.qmd), except that every method is always solved -----------
def preprocess_crop(ctx):
    # crop for solving; ctx["ave"] itself stays uncropped for the SNR estimate
    return {"data": ctx["ave"].copy().crop(tmin=CROP[0], tmax=CROP[1])}


def make_reconstruct(fwd, eig):
    mne_methods = {"sloreta": "sLORETA", "dspm": "dSPM"}  # depth=None, no depth weighting

    def reconstruct(ctx):
        data, ncov = ctx["data"], ctx["ncov"]
        snr, _ = mm.analysis.eeg.estimate_evoked_snr(
            evoked=ctx["ave"], baseline=ctx["simulator"].epoch_config.baseline, noise_cov=ncov
        )
        lambda2_auto = max(0.01, 1.0 / (snr.max() ** 2))

        solutions = {}
        inv = mm.analysis.inverse.wCSET(
            forward=fwd, noise_cov=ncov, depth=0.2, eigenvectors=eig["vec"], eigenvalues=eig["val"],
            harm=5, harmonic_method="lowpass",
        )
        solutions["wcset"] = inv.solve(data, lambda2_auto)
        inv = mm.analysis.inverse.CSET(forward=fwd, noise_cov=ncov, depth=0.2, eigenvectors=eig["vec"])
        solutions["cset"] = inv.solve(data, lambda2=0.1)
        inv_mne_op = mne.minimum_norm.make_inverse_operator(
            info=data.info, forward=fwd, noise_cov=ncov, fixed=True, depth=None, verbose="warning",
        )
        for m, mne_name in mne_methods.items():
            solutions[m] = mne.minimum_norm.apply_inverse(data, inv_mne_op, lambda2_auto, method=mne_name,
                                                          verbose="warning")
        inv_wmne_op = mne.minimum_norm.make_inverse_operator(
            info=data.info, forward=fwd, noise_cov=ncov, fixed=True, depth=2, verbose="warning",
        )
        solutions["wmne"] = mne.minimum_norm.apply_inverse(data, inv_wmne_op, lambda2_auto, method="MNE",
                                                           verbose="warning")
        return {"solutions": solutions}

    return reconstruct
# ----------------------------------------------------------------------------------------------------------------


# --- identical to comparison.py's scoring (processing_simf.qmd) -------------------------------------------------
def score(ctx, dipole_positions):
    """Spatial correlation over time vs. the noisy ground truth, qrp/pp times, localization error at qrp."""
    solutions = ctx["solutions"]
    stc0 = next(iter(solutions.values()))
    ground = ctx["gold-noisy"].copy().crop(tmin=stc0.times[0], tmax=stc0.times[-1])
    data = ctx["ave"].copy().crop(tmin=stc0.times[0], tmax=stc0.times[-1])
    hm = ctx["simulator"].hm
    true_source_mask = ctx["src"].select_targets(hm.src, hm.parc, hm.ras2tkras)

    corr = {m: np.asarray(mm.analysis.inverse.compute_stc_correlation(ground, stc, mode="spatial", method="pearson"))
            for m, stc in solutions.items()}
    phase_time, qrp_idx = {}, None
    for phase_name, ratio in [("qrp", 0.25), ("pp", 1.0)]:
        idx = mm.analysis.eeg.get_rising_phase(data, ratio=ratio, as_idx=True)
        phase_time[phase_name] = data.times[idx].item()
        if phase_name == "qrp":
            qrp_idx = idx
    loc = {m: find_peak_localization_error(stc.data[:, qrp_idx], true_source_mask, dipole_positions)["distance_mm"]
           for m, stc in solutions.items()}
    return {
        "corr": np.stack([np.ravel(corr[m]) for m in METHODS]),
        "loc": np.array([float(np.ravel(loc[m])[0]) for m in METHODS]),
        "times": np.asarray(data.times),
        "qrp_time": phase_time["qrp"],
        "pp_time": phase_time["pp"],
    }
# ----------------------------------------------------------------------------------------------------------------


def make_task(dm, run):
    sim = dm.roots["sim"] / f"sub-{SUB}"
    return InverseSimTask(
        run=int(run),
        simulator_dir=sim / "simulator",
        fave=dm.get_path(SUB, "ave", task="spk", run=run),
        fcov=dm.get_path(SUB, "cov", task="spk", run=run),
        fgold=dm.get_path(SUB, "gold", task="spk", run=run),
        fsource=sim / "sources" / f"run-{int(run):03d}_source.json",
        fgoldnoisy=dm.get_path(SUB, "gold", task="spk", run=run, desc="noisy"),
    )


def read_index(dm):
    index = pd.read_csv(dm.roots["sim"] / f"sub-{SUB}" / "simulation_index.tsv", sep="\t")
    index["region"] = index["source_index_list"].apply(lambda s: ast.literal_eval(s)[0])
    return index


def check_parity(dm, fwd, eig, dipole_positions, run=1):
    """Re-solve and re-score one flagship run with the functions above; compare with the paper's saved STCs and
    results/simf/flagship. Run in-process with the same load -> preprocess -> process order as
    run_parallel_pipeline."""
    region = read_index(dm).set_index("run").loc[run, "region"]
    task = make_task(dm, run)
    ctx = {"task": task, **task.load()}
    ctx.update(preprocess_crop(ctx))
    ctx.update(make_reconstruct(fwd, eig)(ctx))
    ok = True
    for m in METHODS:
        saved = mne.read_source_estimate(str(dm.get_path(SUB, "stc", task="spk", run=run, desc=m)))
        new = ctx["solutions"][m].data
        # CSET's proximal-gradient solver stops at a relative change of 1e-4 (solvers.py), and repeated solves of
        # the same data differ at that level (~0.3% of the maximum here); its metrics are checked below.
        tol = (1e-2 if m == "cset" else 1e-4) * np.abs(saved.data).max()
        diff = np.abs(new - saved.data).max()
        print(f"  STC {m:8s} max |diff| = {diff:.3g} (tol {tol:.3g})")
        ok &= new.shape == saved.data.shape and diff <= tol

    s = score(ctx, dipole_positions)
    flagship = mm.metric.MetricSuite.load(PROJECT / "results/simf/flagship")
    loc_ref = flagship.to_df(metrics="localization_error", filters={"run": run}).set_index("Method")["Value"]
    corr_ref = flagship.to_df(metrics="correlation", filters={"region": region})
    for k, m in enumerate(METHODS):
        loc_ok = np.isclose(s["loc"][k], loc_ref[m], atol=1e-6)
        # correlation is stored per region (one row per rep, no run id): run 1 must match one of its region's reps
        curves = corr_ref[corr_ref["Method"] == m].pivot_table(index="rep", columns="times", values="Value")
        corr_diff = np.abs(curves.to_numpy() - s["corr"][k]).max(axis=1).min()
        print(f"  {m:8s} loc {s['loc'][k]:.2f} vs {loc_ref[m]:.2f} mm | best corr-curve max |diff| {corr_diff:.2g}")
        ok &= bool(loc_ok) and corr_diff < 1e-3
    print("PARITY OK" if ok else "PARITY FAILED")
    return ok


def assemble(index, ckpt, out, sim_root):
    """Checkpoints -> one MetricSuite with results/simf/flagship's metric names and coordinates."""
    runs = [int(r) for r in index["run"] if (ckpt / f"run-{int(r):03d}.npz").exists()]
    res = {r: np.load(ckpt / f"run-{r:03d}.npz") for r in runs}
    run2region = dict(zip(index["run"].astype(int), index["region"]))
    regions = np.array([run2region[r] for r in runs])
    times = res[runs[0]]["times"]

    suite = mm.metric.MetricSuite(f"sub-{SUB} power-analysis pilot")
    for k, m in enumerate(METHODS):
        for region in np.unique(regions):
            idx = [r for r, reg in zip(runs, regions) if reg == region]
            suite.register("correlation", np.stack([res[r]["corr"][k] for r in idx]), Method=m, times=times,
                           region=region)
        suite.register("localization_error", np.array([res[r]["loc"][k] for r in runs]), Method=m,
                       run=np.array(runs), region=regions, phase="qrp")
    suite.register("qrp_time", np.array([float(res[r]["qrp_time"]) for r in runs]), run=np.array(runs))
    suite.register("pp_time", np.array([float(res[r]["pp_time"]) for r in runs]), run=np.array(runs))
    suite.metadata["units"] = {"correlation": "r", "localization_error": "mm", "qrp_time": "s", "pp_time": "s"}
    suite.metadata["pilot"] = {"simulations": str(sim_root), "n_runs": len(runs),
                               "n_regions": int(len(np.unique(regions))), "methods": METHODS, "crop": CROP}
    suite.save(out, overwrite=True)
    print(f"saved {len(runs)} runs ({len(np.unique(regions))} regions) to {out}")
    return len(runs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=Path, default=PILOT_SIM, help="pilot simulation folder")
    parser.add_argument("--out", type=Path, default=OUT, help="output MetricSuite folder")
    parser.add_argument("--n-runs", type=int, default=None, help="only the first N runs (timing test)")
    parser.add_argument("--n-jobs", type=int, default=25)
    parser.add_argument("--check-parity", action="store_true", help="flagship run 1 vs. its saved STCs/metrics")
    parser.add_argument("--keep-checkpoints", action="store_true")
    args = parser.parse_args()
    mne.set_log_level("error")

    dm = SimfManager(str(PROJECT / "processing" / "simf.config.yml"))
    fwd = dm.load(SUB, "fwd", desc="inv")
    eig = dm.load(SUB, "eig")
    dipole_positions = mm.anatomy.source_space.get_dipole_positions(fwd["src"]) * 1000

    if args.check_parity:
        sys.exit(0 if check_parity(dm, fwd, eig, dipole_positions) else 1)

    sim_root = args.sim.resolve()
    if sim_root == PAPER_SIM.resolve():
        sys.exit("Refusing to score the paper's own simulations as a pilot.")
    dm.add_root("sim", sim_root)  # redirect every simulation path to the pilot folder
    index = read_index(dm)
    if args.n_runs:
        index = index.iloc[: args.n_runs]
    probe_set = set(pd.read_csv(PROJECT / "results/simf/intermediate/probe-set.csv")["region"])
    assert not set(index["region"]) & probe_set, "pilot regions overlap Study I's probe set"

    # outside args.out: suite.save(overwrite=True) may clear that folder
    ckpt = args.out.parent / f"_{args.out.name}_checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    todo = [r for r in index["run"] if not (ckpt / f"run-{int(r):03d}.npz").exists()]
    print(f"{len(index) - len(todo)}/{len(index)} runs already scored, {len(todo)} left.", flush=True)

    def postprocess(ctx):
        s = score(ctx, dipole_positions)
        np.savez(ckpt / f"run-{ctx['task'].run:03d}.npz", **s)
        return ctx["task"].run

    t0, batch = time.time(), 2 * args.n_jobs  # batches only to log progress; each run is checkpointed
    for i in range(0, len(todo), batch):
        tasks = [make_task(dm, r) for r in todo[i:i + batch]]
        run_parallel_pipeline(tasks, preprocess_crop, make_reconstruct(fwd, eig), postprocess,
                              n_jobs=args.n_jobs, n_threads_per_job=1)
        done = min(i + batch, len(todo))
        el = time.time() - t0
        print(f"[{done}/{len(todo)}] {el / 60:.1f} min elapsed, {el / done:.1f} s/run, "
              f"~{el / done * (len(todo) - done) / 60:.0f} min left", flush=True)

    n = assemble(index, ckpt, args.out, sim_root)
    if n == len(index) and not args.keep_checkpoints:
        shutil.rmtree(ckpt)


if __name__ == "__main__":
    main()
