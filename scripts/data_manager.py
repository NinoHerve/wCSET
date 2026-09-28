"""
Module Name: dataset_readers.py
Description: Classes to read data from specific dataset.

This module provides dataset readers for specific datasets. Every reader is build upon the 
DatasetReader class which manages dataset meta-information. Every class indicates how to access
a specific data by id. 

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""

from pathlib import Path
import glob

import numpy as np
import scipy as sp
import pandas as pd

import mne
import nibabel as nib

import mindmap as mm


class SimfManager(mm.io.DatasetManager):
    """Manages the SIMF dataset."""
    def __init__(self, config):
        """SIMF configuration."""
        super().__init__(config)

        # add new roots
        self.add_root("sim", self.config["DIR_SIM"])
        self.add_root("cmp", self.config["DIR_CMP"])

        # add datatypes
        self.add_datatype("ave-temp", "mm", self.build_simf_path, lambda f: mne.read_evokeds(f, verbose="warning")[0])
        self.add_datatype("eeg-trans", "sim", self.build_simf_path, load_identity_trans)
        self.add_datatype("eeg", "sim", self.build_simf_path, lambda f: mne.io.read_raw_brainvision(f, verbose="warning"))
        self.add_datatype("gold", "sim", self.build_simf_path, mne.read_source_estimate)
        self.add_datatype("stc", "sim", self.build_simf_path, mne.read_source_estimate)
        self.add_datatype("cov", "sim", self.build_simf_path, self._load_cov)
        self.add_datatype("ave", "sim", self.build_simf_path, lambda f: mne.read_evokeds(f, verbose="warning")[0])
        self.add_datatype("tractogram", "cmp", self.build_simf_path, nib.streamlines.load)
        self.add_datatype("t12diff", "cmp", self.build_simf_path, self._load_t12diff)

    def build_simf_path(self, root, subject, suffix, **kwargs):
        """Special SIMF paths."""
        path = None

        if suffix == "ave-temp":
            path = root/f"sub-{subject}"/"ave"/f"sub-{subject}_desc-template-coreg_ave.fif"

        elif suffix == "eeg-trans":
            path = Path(".")

        elif suffix in ["eeg", "gold", "cov", "ave", "stc"]:
            task = kwargs.get("task")
            run = kwargs.get("run")
            if task is None or run is None:
                raise ValueError(f"{suffix} data requires task and run entities.")

            folder = root/f"sub-{subject}"/suffix
            desc = kwargs.get("desc")
            # "gold"/"ave" carry an optional desc (e.g. desc="noisy" for gold,
            # desc="clean" for ave) to distinguish the noise variant saved
            # alongside the default file, rather than living under their own
            # datatype - see mindmap's generate_simulation_dataset/_run_one.
            desc_part = f"_desc-{desc}" if desc else ""

            if suffix == "eeg":
                path = folder/f"sub-{subject}_task-{task}_run-{run}_eeg.vhdr"
            elif suffix == "gold":
                path = folder/f"sub-{subject}_task-{task}_run-{run}{desc_part}_stc"
            elif suffix == "cov":
                path = folder/f"sub-{subject}_task-{task}_run-{run}_cov.fif"
            elif suffix == "ave":
                path = folder/f"sub-{subject}_task-{task}_run-{run}{desc_part}_ave.fif"
            elif suffix == "stc":
                if desc is None:
                    raise ValueError(f"{suffix} data requires desc entity.")
                path = folder/f"sub-{subject}_task-{task}_run-{run}_desc-{desc}_stc"
        
        elif suffix == "tractogram":
            path = root/f"sub-{subject}"/"dwi"/f"sub-{subject}_model-CSD_desc-DET_tractogram.trk"
        
        elif suffix == "t12diff":
            path = root/f"sub-{subject}"/"xfm"/"final0GenericAffine.mat"

        else:
            raise ValueError(f"Unknown suffix '{suffix}' for `build_simf_path`")

        return path

    def _load_t12diff(self, path):
        """Load subject to diffusion space transform."""
        affine = sp.io.loadmat(str(path))["AffineTransform_double_3_3"].reshape(3,4,order="F")
        return np.concatenate((affine, np.array([[0,0,0,1]])))


class IntraManager(mm.io.DatasetManager):
    """Manages the INTRA datset."""
    def __init__(self, config):
        """INTRA configuration."""
        super().__init__(config)

        # add new root
        self.add_root("spk", self.config["DIR_SPK"])

        # add datatypes
        self.add_datatype("eeg", "spk", self.build_intra_path, self.load_eeg)
        self.add_datatype("ieeg", "spk", self.build_intra_path, self.load_eeg)
        self.add_datatype("eeg-trans", "spk", self.build_intra_path, load_identity_trans)

    def build_intra_path(self, root, subject, suffix, **kwargs):
        """Special INTRA paths."""
        if suffix in ["eeg", "ieeg"]:
            # manage bids entities
            task = kwargs.get("task")
            task = f"_task-{task}" if task else ""

            run = kwargs.get("run")
            run = f"_run-{run}" if run else "_run-*"

            # filename
            if suffix == "eeg":
                eeg_dir = root/f"sub-{subject}"/"eeg"
                path = eeg_dir/f"sub-{subject}{task}{run}_eeg.vhdr"

            else:
                ieeg_dir = root/f"sub-{subject}"/"ieeg"
                path = ieeg_dir/f"sub-{subject}{task}{run}_ieeg.vhdr"

        elif suffix == "eeg-trans":
            path = Path(".")

        else:
            raise ValueError(f"Unknown suffix '{suffix}' for `build_intra_path`.")

        return path

    def load_eeg(self, path):
        """Load EEG for INTRA dataset."""
        files = glob.glob(str(path))
        montage = glob.glob(str(path.parent/"*_electrodes.tsv"))[0]

        dat = []

        for f in files:
            spike = mne.io.read_raw_brainvision(f, verbose="warning")
            dat.append(spike["data"][0])

        dat = np.array(dat)

        # info & montage
        info = spike.info
        montage = mne.channels.read_custom_montage(montage, coord_frame="head")
        montage = mm.analysis.eeg.scale_montage(montage, scale=1e-3)

        # store in epoch object
        epochs = mne.EpochsArray(dat, info, verbose="warning")
        epochs = epochs.set_montage(montage, verbose="warning")

        return epochs


def load_identity_trans(_):
    """
    Load identity transformation matrix.
    """
    return mne.transforms.Transform(fro="head", to="mri")


class SpikeManager(mm.io.DatasetManager):
    """Manages the SIMF dataset."""

    # Subjects whose resection ground truth should come from a manual
    # segmentation (derivatives/mindmap/sub-{subject}/resec/) instead of the
    # automated RAMPS output - e.g. because RAMPS failed or produced a bad
    # mask (see processing_spike.qmd's Quality Assessment section). Keyed by
    # subject -> the `sub-{subject}_{suffix}.nii.gz` filename suffix to use
    # from that folder. Add more subjects here as they're reviewed.
    MANUAL_RESECTION_MASKS = {
        "pat03": "man-recon-seg",
        "pat04": "auto-seg",
        "pat08": "man-recon-seg",
        "pat14": "man-recon-seg",
        "pat24": "man-recon-seg",
        "pat28": "auto-seg",
    }

    def __init__(self, config):
        """SPIKE configuration."""
        super().__init__(config)

        # add new roots
        self.add_root("cart", self.config["DIR_CART"])
        self.add_root("cmp", self.config["DIR_CMP"])
        self.add_root("ramps", self.roots["mm"].parent/"ramps")  # matches resection_masks.py's output_dir

        # subjects come from participants.tsv in the raw root (no sub-* folders exist there)
        tsv = pd.read_csv(self.roots["raw"] / "participants.tsv", sep="\t")
        self.subjects = tsv["participant_id"].str.removeprefix("sub-").tolist()

        # add datatypes
        self.add_datatype("eeg", "cart", self.build_spike_path, self.read_eeg)
        self.add_datatype("T1w", "cmp", self.build_spike_path, mm.anatomy.mri.read_mri)
        self.add_datatype("eeg-trans", "cart", self.build_spike_path, load_identity_trans)
        self.add_datatype("resec-ramp", "ramps", self.build_spike_path, mm.anatomy.mri.read_mri)

    def build_spike_path(self, root, subject, suffix, **kwargs):
        """Build paths for SPIKE derivative files."""
        session = kwargs.get("session")
        task    = kwargs.get("task")
        res     = kwargs.get("res")
        space   = kwargs.get("space")
        desc    = kwargs.get("desc")

        if suffix == "eeg-trans":
            return Path(".")

        if suffix == "resec-ramp":
            if not space or space.lower() not in ("orig", "pre"):
                raise ValueError("resec-ramp requires a 'space' entity ('orig' or 'pre').")

            manual_suffix = self.MANUAL_RESECTION_MASKS.get(subject)
            if manual_suffix is not None:
                # Manual segmentations are drawn directly on the pre-op
                # image, so there is no "orig"-space counterpart the way
                # RAMPS output has - only "pre" is meaningful here.
                if space.lower() != "pre":
                    raise ValueError(
                        f"sub-{subject} uses a manual resection mask "
                        f"(MANUAL_RESECTION_MASKS) - no '{space}'-space variant available."
                    )
                return self.roots["mm"] / f"sub-{subject}" / "resec" / f"sub-{subject}_{manual_suffix}.nii.gz"

            # Non-BIDS-conform RAMPS pipeline output - bypass BIDS entity
            # resolution and build the path directly.
            folder = root / f"sub-{subject}" / "RAMPS_Resection_Mask_Output"
            return folder / f"RAMP_The_resection_mask_in_{space.upper()}.nii.gz"

        if suffix == "eeg":
            if session is None or task is None or desc is None:
                raise ValueError("eeg requires session, task and desc entities.")
            folder = root / f"sub-{subject}" / f"ses-{session}" / "eeg"
            return folder / f"sub-{subject}_ses-{session}_task-{task}_desc-{desc}_eeg.vhdr"

        if suffix == "T1w":
            if session is None:
                raise ValueError("mri requires session entity.")
            res_part   = f"_res-{res}"     if res   else ""
            space_part = f"_space-{space}" if space else ""
            desc_part  = f"_desc-{desc}"   if desc  else ""
            folder = root / f"sub-{subject}" / f"ses-{session}" / "anat"
            return folder / f"sub-{subject}_ses-{session}{res_part}{space_part}{desc_part}_T1w.nii.gz"

        raise ValueError(f"Unknown suffix '{suffix}' for build_spike_path.")


    def read_eeg(self, path):
        """Read Spike EEG."""
        eeg = mne.io.read_raw_brainvision(path, scale=1e-6, misc=[], verbose="warning")

        data = eeg.get_data()
        info = eeg.info

        return mne.EvokedArray(data, info)
