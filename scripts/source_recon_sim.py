"""
Module Name: source_recon_sim.py
Description: Script to run source reconstruction of the simulation dataset
in parallel.

This script runs source reconstruction of all files in the simulation folder
in parallel to greatly improve speed.

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import mindmap as mm
from data_manager import SimfManager


if __name__ == "__main__":
    # 1. Define paths and methods
    dm = SimfManager("simf.config.yml")
    sub = "10"

    AVE_DIR = dm.get_path(sub, "ave", task="spk", run="1").parent
    STC_DIR = dm.get_path(sub, "stc", task="spk", run="1", desc="wcset").parent
    methods = ["dspm", "sloreta", "cset", "wcset"]
    lambda2 = [0.01, 0.01, 0.1, 0.1]

    STC_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Gather all files
    all_files = [
        f for f in AVE_DIR.glob("*.fif")
        if "template" not in f.name
    ]
    total_dataset_size = len(all_files)

    # 3. Filter out what is already done
    print("Checking disk for existing results...")
    files_ave = mm.analysis.inverse.filter_remaining_files(all_files, methods)

    already_done = total_dataset_size - len(files_ave)
    print(f"Dataset Status: {already_done}/{total_dataset_size} files already completed.")
    print(f"Files left to process: {len(files_ave)}")

    if len(files_ave) == 0:
        print("All files processed! Exiting.")
        exit()
    
    # Generate matching covariance file paths
    files_cov = [Path(str(f).replace("ave", "cov")) for f in files_ave]

    for fcov in files_cov:
        if not fcov.exists():
            raise FileNotFoundError(f"Missing expected covariance file: {fcov}")

    # 4. Load static variables
    forward = dm.load(sub, "fwd", desc="inv")
    eig = dm.load(sub, "eig")

    # 5. Initialize process factory
    process_one_file = mm.analysis.inverse.make_process_sample(
        methods = methods,
        lambda2 = lambda2,
        forward = forward,
        eig = eig,
        crop = [-0.01, 0.15],
    )

    # 6. Launch Parallel Processing with a tqdm Progress Bar
    print("Spawning parallele workers across available cores...")

    results = Parallel(n_jobs=-1)(
        delayed(process_one_file)(fave, fcov) 
        for fave, fcov in tqdm(
            zip(files_ave, files_cov),
            total=len(files_ave),
            desc="Reconstructing Sources",
        )
    )

    print("\nProcessing cycle complete!")