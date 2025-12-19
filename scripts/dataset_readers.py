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

import glob

import numpy as np
import scipy as sp
import nibabel as nib
import mne
import mne_bids

from mindmap.io.readers import DatasetReader, get_bidspath, read_mri
from mindmap.sensor import build_eeg_template

# --------------------
# SIMF dataset
# ¨¨¨¨¨¨¨¨¨¨¨¨
# Simulation dataset.
# --------------------

class SimfReader(DatasetReader):
    """
    Class to read data from the SIMF dataset.
    """
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, subject, *args, mri_dtype="nib", task=None, run=None, verbose=1):
        """
        Read data for dataset SIMF.

        Parameters
        ----------
        subject: str | int
            Subject name or id.

        *args: list(str)
            Requested data. Data are requested via an abreviated code. See Returns to check 
            available abreviations and associated datatype.

        mri_dtype: str
            MRI dtype (Nibabel or ANTs) for MRI images.

        Returns:
        data: dict
            Dictionnary containing (only) requested data:
            - t1: nib.Nifti1Image | ants.ANTsImage
                Raw T1 weighted MRI.
            - eeg: mne.io.Raw
                Raw eeg simulation.
            - eeg-tmp:mne.Raw
                EEG template (no recordings)
            - t1-fs: nib.Nifti1Image | ants.ANTsImage
                T1-weigthed MRI in freesurfer space.
            - t1-brain: nib.Nifti1Image | ants.ANTsImage
                Brain mask in freesurfer space.
            - white: list[np.ndarray (n,3), np.ndarray (m,3)]
                White surface (boundary between white a gray matter). The surface correponds
                to a mesh defined by its vertex locations (n,3) and triangular faces (m,3).
            - parc: Parcellation
                Parcellation (raw space)
            - ctx: nib.Nifti1Image
                Cortical mask (raw space).
            - src: mne.SourceSpaces
                Source space (freesurfer space).
            - src-raw: mne.SourceSpaces
                Source space before forward solution (freesurfer space).
            - bem: mne.BEM
                Boundary element model for source reconstruction (freesurfer space).
            - bem-sim: mne.BEM
                Boundary element model for simulations (freesurfer space).
            - fwd: mne.Forward
                Forward solution.
            - conn: np.ndarray
                Structural Connectome.
            - ave: mne.Evoked
                Averaged and preprocessed EEG.
            - cset: mne.SourceEstimate
                CSET inverse solution.
            - sloreta: mne.SourceEstimate
                sLORETA inverse solution.
            - wcset: mne.SourceEstimate
                weighted CSET inverse solution.
            - trans: mne.Transformation
                Affine transformation (sensor space -> freesurfer space)
            - tkras2ras: np.ndarray (4,4)
                Affine transformation (freesurfer space -> raw space)
        """
        if isinstance(subject, int):
            subject  = self.config["SUBS"][subject-1]

        name    = self.config["NAME"]
        mm_dir  = self.config["MM_DIR"]     # mindmap directory (derivative)
        cmp_dir = self.config["CMP_DIR"]    # connectome mapper directory (derivative)
        ds_dir  = self.config["DS_DIR"]     # dataset directory (raw)

        data = {
            "name" : name,
            "sub"  : subject,
            "task" : task,
            "run"  : run,
        }

        # dataset specifics (use this section to also overwrite freesurfer and mindmap)
        for key in args:

            # -------------
            # Read Anatomy
            # -------------
            
            # Tractogram
            if key.lower() == "tracto":
                if subject == 10:
                    file = cmp_dir/f"sub-{subject}"/"dwi"/"sub-10_model-CSD_desc-DET_tractogram.trk"
                else:
                    file = cmp_dir/f"sub-{subject}"/"fibtracking"/"sd_stream"/"dwi_denoised_eddy_distc_corrected_1mm_shore_sdstream.trk"
                data[key] = nib.streamlines.load(file)

            # ---------
            # Read EEG
            # ---------

            # raw EEG
            if key.lower() == "eeg":
                # template
                if task is None or run is None:
                    print("No task and/or run provided, loading EEG template ...")
                    file = mm_dir/f"sub-{subject}"/"eeg"/f"sub-{subject}_desc-empty-coreg_ave.fif"
                    
                    # coregistered template
                    if file.exists():
                        print("Coregistered template found, loading ...")
                        data[key] = mne.Evoked(file, verbose="warning")

                    # not coregistered template
                    else:
                        print("No coregistered template found. Building new template ...")
                        eeg = build_eeg_template()
                        data[key] = eeg

                # read raw EEG given task and run
                else:
                    file = mne_bids.BIDSPath(
                        root      = ds_dir,
                        subject   = subject,
                        datatype  = "eeg",
                        task      = task,
                        run       = run,
                    )
                    data[key] = mne_bids.read_raw_bids(file, verbose="warning").load_data(verbose="warning")

            # ---------------------
            # Read ground truth
            # ---------------------

            # full ground truth
            if key.lower() == "gt":
                file = get_bidspath(
                    root     = ds_dir,
                    datatype = "stc",
                    subject  = subject,
                    task     = task,
                    run      = run,
                )
                data[key] = mne.read_source_estimate(file)

            # full ground truth with noise
            if key.lower() == "gtnoise":
                file = get_bidspath(
                    root        = ds_dir,
                    datatype    = "stc",
                    subject     = subject,
                    task        = task,
                    run         = run,
                    description = "noisy",
                )
                data[key] = mne.read_source_estimate(file)

            # ground truth: first event only
            if key.lower() == "gt-1":
                d = self.__call__(subject, "gt", "eeg", "ave", task=task, run=run, verbose=0)

                # extract time interval of first event
                onset = d["eeg"].annotations[0]["onset"]
                tmin  = d["ave"].tmin
                tmax  = d["ave"].tmax

                # crop ground truth
                gt = d["gt"].crop(tmin = onset+tmin, tmax = onset+tmax)
                gt.tmin = d["ave"].tmin

                data[key] = gt

            # ---------------------
            # Read transformations
            # ---------------------

            # EEG coregistration (sensor space in [m] -> freesurfer ras space in [m])
            if key.lower() == "trans":
                data[key] = mne.transforms.Transform("head", "mri")    # identity transformation

            # diffusion ras space in [mm] -> T1-ras space in [mm]
            if key.lower() == "t12diff":
                file      = cmp_dir/f"sub-{subject}"/"xfm"/"final0GenericAffine.mat"
                affine    = sp.io.loadmat(str(file))["AffineTransform_double_3_3"].reshape(3, 4, order="F")
                data[key] = np.concatenate((affine, np.array([[0, 0, 0, 1]])))

        args = list(set(args) - set(data.keys()))   # update list of missing arguments

        # --------------------------------------------------
        # Read FreeSurfer derivatives
        # [t1-fs, t1-brain, white, charm, tkras2ras]
        # --------------------------------------------------

        fs_data = self.read_freesurfer(subject, *args, mri_dtype=mri_dtype)
        data.update(fs_data)
        args = list(set(args) - set(data.keys()))

        # ------------------------------------------------------------------------
        # Read MindMap derivatives
        # [parc, ctx, , src, src-raw, bem, bem-sim, fwd, sc-nfiber, sc-length, \\
        # sc-rb-nfiber, sc-rb-length, eeg, cset, sloreta, wcset, trans]
        # ------------------------------------------------------------------------

        mm_data = self.read_mindmap(subject, *args, task=task, run=run)
        data.update(mm_data)
        args = list(set(args) - set(data.keys()))

        # ------------------------------------------------------------------------
        # Read Standardized data
        # [std-ctx, std-thal, std-mni]
        # ------------------------------------------------------------------------
        std_data = self.read_standards(*args, mri_dtype=mri_dtype)
        data.update(std_data)
        args = list(set(args) - set(data.keys()))

        if verbose == 1:
            self.log(data, args)

        return data




# ---------------------------------
# INTRA dataset
# ¨¨¨¨¨¨¨¨¨¨¨¨¨
# Simultaneous EEG-iEEG recordings
# ----------------------------------


class IntraReader(DatasetReader):
    """
    Class to read data from the SIMF dataset.
    """
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, subject, *args, mri_dtype="nib", task=None, run=None, verbose=1):
        """
        Read data for dataset SIMF.

        Parameters
        ----------
        subject: str | int
            Subject name or id.

        *args: list(str)
            Requested data. Data are requested via an abreviated code. See Returns to check 
            available abreviations and associated datatype.

        mri_dtype: str
            MRI dtype (Nibabel or ANTs) for MRI images.

        Returns:
        data: dict
            Dictionnary containing (only) requested data:
            - t1: nib.Nifti1Image | ants.ANTsImage
                Raw T1 weighted MRI.
            - eeg: mne.io.Raw
                Raw eeg simulation.
            - eeg-tmp:mne.Raw
                EEG template (no recordings)
            - t1-fs: nib.Nifti1Image | ants.ANTsImage
                T1-weigthed MRI in freesurfer space.
            - t1-brain: nib.Nifti1Image | ants.ANTsImage
                Brain mask in freesurfer space.
            - white: list[np.ndarray (n,3), np.ndarray (m,3)]
                White surface (boundary between white a gray matter). The surface correponds
                to a mesh defined by its vertex locations (n,3) and triangular faces (m,3).
            - parc: Parcellation
                Parcellation (raw space)
            - ctx: nib.Nifti1Image
                Cortical mask (raw space).
            - src: mne.SourceSpaces
                Source space (freesurfer space).
            - src-raw: mne.SourceSpaces
                Source space before forward solution (freesurfer space).
            - bem: mne.BEM
                Boundary element model for source reconstruction (freesurfer space).
            - fwd: mne.Forward
                Forward solution.
            - conn: np.ndarray
                Structural Connectome.
            - ave: mne.Evoked
                Averaged and preprocessed EEG.
            - cset: mne.SourceEstimate
                CSET inverse solution.
            - sloreta: mne.SourceEstimate
                sLORETA inverse solution.
            - wcset: mne.SourceEstimate
                weighted CSET inverse solution.
            - trans: mne.Transformation
                Affine transformation (sensor space -> freesurfer space)
            - tkras2ras: np.ndarray (4,4)
                Affine transformation (freesurfer space -> raw space)
        """
        if isinstance(subject, int):
            subject  = self.config["SUBS"][subject-1]

        name    = self.config["NAME"]
        ds_dir  = self.config["DS_DIR"]     # dataset directory (raw)
        spk_dir = self.config["SPK_DIR"]    # spike trials directory (derivative)

        data = {
            "name" : name,
            "sub"  : subject,
            "task" : None,
            "run"  : None,
        }

        # dataset specifics (use this section to also overwrite freesurfer and mindmap)
        for key in args:

            # -------------
            # Read Anatomy
            # -------------
            
            if key.lower() == "t1":
                file = get_bidspath(
                    root     = ds_dir,
                    datatype = "anat",
                    subject  = subject,
                    suffix   = "T1w",
                )
                data[key] = read_mri(file, mri_dtype=mri_dtype)

            # ---------
            # Read EEG
            # ---------

            # raw EEG
            if key.lower() == "eeg":
                # read epochs
                if run is None:
                    files   = glob.glob(str(spk_dir/f"sub-{subject}"/"eeg"/f"sub-{subject}_task-spk_*_eeg.vhdr"))
                    montage = spk_dir/f"sub-{subject}"/"eeg"/f"sub-{subject}_electrodes.tsv"

                    # extract data
                    dat = []

                    for f in files:
                        spike = mne.io.read_raw_brainvision(f, verbose="warning")
                        dat.append(spike["data"][0])
                    dat = np.array(dat)

                    # info & montage
                    info    = spike.info
                    montage = mne.channels.read_custom_montage(montage)

                    # store in Epoch object
                    epochs = mne.EpochsArray(dat, info, verbose="warning")
                    epochs = epochs.set_montage(montage, verbose="warning")

                    data[key] = epochs

                # read single spike
                else:
                    file = get_bidspath(
                        root        = spk_dir,
                        datatype    = "eeg",
                        subject     = subject,
                        task        = task,
                        run         = run,
                    )
                    #montage = file.fpath.parent/f"sub-{subject}_electrodes.tsv"
                    montage = file.parent/f"sub-{subject}_electrodes.tsv"

                    raw     = mne.io.read_raw_brainvision(file)
                    montage = mne.channels.read_custom_montage(montage)
                    raw     = raw.set_montage(montage)

                    data[key] = raw

            # ---------------------
            # Read ground truth
            # ---------------------


            # ---------------------
            # Read transformations
            # ---------------------

            # EEG coregistration (sensor space in [m] -> freesurfer ras space in [m])
            if key.lower() == "trans":
                data[key] = mne.transforms.Transform("head", "mri")    # identity transformation

        args = list(set(args) - set(data.keys()))   # update list of missing arguments

        # --------------------------------------------------
        # Read FreeSurfer derivatives
        # [t1-fs, t1-brain, white, charm, tkras2ras]
        # --------------------------------------------------

        fs_data = self.read_freesurfer(subject, *args, mri_dtype=mri_dtype)
        data.update(fs_data)
        args = list(set(args) - set(data.keys()))

        # ------------------------------------------------------------------------
        # Read MindMap derivatives
        # [parc, ctx, , src, src-raw, bem, bem-sim, fwd, sc-nfiber, sc-length, \\
        # sc-rb-nfiber, sc-rb-length, eeg, cset, sloreta, wcset, trans]
        # ------------------------------------------------------------------------

        mm_data = self.read_mindmap(subject, *args, task=task, run=run)
        data.update(mm_data)
        args = list(set(args) - set(data.keys()))

        # ------------------------------------------------------------------------
        # Read Standardized data
        # [std-ctx, std-thal, std-mni]
        # ------------------------------------------------------------------------
        std_data = self.read_standards(*args, mri_dtype=mri_dtype)
        data.update(std_data)
        args = list(set(args) - set(data.keys()))

        if verbose == 1:
            self.log(data, args)

        return data
