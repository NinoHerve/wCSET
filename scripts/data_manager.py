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

import mne

import mindmap as mm


class IntraManager(mm.io.DatasetManager):
    """Manages the INTRA dataset"""
    def __init__(self, config):
        """Configured configuration"""
        super().__init__(config)

        # add new root
        self.roots["spk"] = self.config["DIR_SPK"]

    def get_subject(self, subject_id):
        """Rewrite missing data for subject reader"""
        # default subject reader (manages mindmap, freesurfer and charm)
        subreader = super().get_subject(subject_id)

        subreader.add_datatype("eeg", "spk", build_intra_path, load_eeg)
        subreader.add_datatype("ieeg", "spk", build_intra_path, load_eeg)
        subreader.add_datatype("eeg-trans", "mm", lambda **_, : Path("."), load_identity_trans)

        return subreader

# ---------
# Builders
# ---------

def build_intra_path(root, subject, datatype, **kwargs):
    """EEG & iEEG path for intra dataset"""
    if datatype in ["eeg", "ieeg"]:
        
        # manage bids entities
        task = kwargs.get("task")
        task = f"_task-{task}" if task else ""

        run = kwargs.get("run")
        run = f"_run-{run}" if run else "_run-*"

        # filename
        if datatype == "eeg":
            eeg_dir = root/f"sub-{subject}"/"eeg"
            return eeg_dir/f"sub-{subject}{task}{run}_eeg.vhdr"
        else:
            ieeg_dir = root/f"sub-{subject}"/"ieeg"
            return ieeg_dir/f"sub-{subject}{task}_*_ieeg.vhdr"

    else:
        raise ValueError(f"Unknown datatype '{datatype}' for build_intra_path.")

# --------
# Loaders
# -------

def load_eeg(path):
    """Load EEGs for intra dataset"""
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
