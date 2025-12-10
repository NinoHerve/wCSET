#!/bin/bash
# ---------------------------------------------
# Script: fsrecon.sh
# Author: Nino Herve
# Date: 2025-12-02
# Purpose: Apply freesurfer reconstruction on dataset
# Usage: bash fsrecon.sh file.config.yml [-s:] 
# ---------------------------------------------

# ---------------------
# Check first argument
# ---------------------

# check if first argument is provided
if [ -z "$1" ]; then
    echo "Error: No configuration file provided."
    echo "Usage: $0 <config.yml> [options]"
    exit 1
fi

CONFIG_FILE="$1"
shift

# check if the file has a .yml or .yaml extension
if [[ "$CONFIG_FILE" != *.yml && "$CONFIG_FILE" != *.yaml ]]; then
    echo "Error: Configuration file must have a .yml or .yaml extension."
    exit 1
fi

# check if the file actually exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: File '$CONFIG_FILE' does not exist."
    exit 1
fi

# ------------------
# Parse options
# ------------------

trg_subjects=()
collecting_subs=0
overwrite=0

while (( "$#" )); do
    case "$1" in
        -o)
            overwrite=1
            shift
            continue
            ;;
        -s)
            collecting_subs=1
            shift
            continue
            ;;
        -*) 
            #unknown option
            echo "Invalid option: $1"
            exit 1
            ;;

        *)
            if [[ $collecting_subs -eq 1 ]]; then
                if [[ "$1" =~ ^[0-9]+$ ]]; then
                    trg_subjects+=("sub-$1")
                    shift
                    continue
                else
                    # first non-number stops subject collection
                    collecting_subs=0
                fi
            fi

            # unrecognized positional
            echo "Warning: ignoring unexpected argument: $1"
            shift
            ;;
    esac
done


# ---------------
# Parse yml file
# ---------------

export FREESURFER_HOME=$(yq '.FS_HOME' "$CONFIG_FILE")
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"

export SUBJECTS_DIR=$(yq '.FS_DIR' "$CONFIG_FILE")
export DATASET_DIR=$(yq '.DS_DIR' "$CONFIG_FILE")

# check if dataset dir exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory not found. Check configuration file."
    exit 1
fi

# check if subjects dir exists, if not, ask user if mkdir
if [ ! -d "$SUBJECTS_DIR" ]; then
    echo "Subjects directory does not exist."
    read -p "Make new directory? (y/n): " answer

    if [[ $answer == "y" ]]; then
        mkdir -p "$SUBJECTS_DIR"
    else
        echo "Stopping FreeSurfer reconstruction because subjects directory does not exist."
        exit 1
    fi
fi



# --------------------------------------------
# Get list of subjects and processed subjects
# --------------------------------------------

# list of subject names in dataset
subjects=()
for s in "$DATASET_DIR"/sub-*; do
    [ -d "$s" ] && subjects+=( "$(basename "$s")" )
done

processed=()
for s in "$SUBJECTS_DIR"/sub-*; do
    [ -d "$s" ] && processed+=( "$(basename "$s")" )
done

# define trg_subjects if they were not given as option
if [[ ${#trg_subjects[@]} -eq 0 ]]; then
    trg_subjects=()

    for s in "${subjects[@]}"; do
        skip=0
        for p in "${processed[@]}"; do
            if [[ "$s" == "$p" ]]; then
                skip=1
                break
            fi
        done

        # keep only unprocessed subjects
        if [[ $skip -eq 0 ]]; then
            trg_subjects+=("$s")
        fi
    done
fi


# ----------------------------------------------------------
# Check mismatches
# ----------------------------------------------------------

# check if target subject are in dataset
declare -A subj_map
for s in "${subjects[@]}"; do
    subj_map["$s"]=1
done

missing=()

for t in "${trg_subjects[@]}"; do
    if [[ -z "${subj_map[$t]}" ]]; then
        missing+=("$t")
    fi
done

if [ ${#missing[@]} -gt 0 ]; then
    echo "Error: The following target subjects are missing from the dataset: "
    printf ' %s\n' "${missing[@]}"
    exit 1
fi

# check if any target subject was already processed
if [[ $overwrite -eq 0 ]]; then
    already_processed=()

    declare -A proc_map
    for p in "${processed[@]}"; do
        proc_map["$p"]=1
    done

    for s in "${trg_subjects[@]}"; do
        if [[ -n "${proc_map[$s]}" ]]; then
            already_processed+=("$s")
        fi
    done

    if [ ${#already_processed[@]} -gt 0 ]; then
        echo "Error: The following target subjects are already processed:"
        printf ' %s\n' "${already_processed[@]}"
        exit 1
    fi
fi

# check if any subjects are left to process
if [[ ${#trg_subjects[@]} -eq 0 ]]; then
    echo
    echo "All subjects are already processed"
    exit 0
fi

# --------------------
# Process FreeSurfer
# --------------------

failed_subject=()

for s in "${trg_subjects[@]}"; do
    nifti="$DATASET_DIR/$s/anat/${s}_T1w.nii.gz"

    if [ ! -f "$nifti" ]; then
        echo "Error: Input file for subject $s does not exist: $nifti"
        failed_subjects+=("$s")
        continue
    fi

    echo "Running FreeSurfer recon-all for subject $s ..."

    recon-all -i "$nifti" -s "$s" -all

    if [ $? -ne 0 ]; then
        echo "Warning: recon-all failed for subject $s"
        failed_subjects+=("$s")
    fi
done

# ----------------------------
# Report all failed subjects
# ----------------------------

if [ ${#failed_subjects[@]} -gt 0 ]; then
    echo
    echo "The following subjects failed:"
    printf ' %s\n' "${failed_subjects[@]}"
else
    echo
    echo "All subjects processed successfully."
fi


