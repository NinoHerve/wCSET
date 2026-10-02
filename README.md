# wCSET

Manuscript and analysis code for the validation of wCSET, a connectivity-informed EEG source imaging method, across three studies: simulations (I), a resection cohort (II), and simultaneous EEG/iEEG recordings (III).

Repository: <https://github.com/NinoHerve/wCSET>

## Project structure

```
wCSET/
├── index.qmd                 The manuscript (Quarto article)
├── _quarto.yml               Project config: renders index.qmd only
├── references.bib            Bibliography
├── chicago-author-date-et-al.csl
│
├── processing/               Processing pipelines, one per study (heavy: run manually)
│   ├── processing_simf.qmd     Study I   – simulations (SIMF)
│   ├── processing_spike.qmd    Study II  – resection cohort (SPIKE)
│   ├── processing_intra.qmd    Study III – simultaneous EEG/iEEG (INTRA)
│   └── *.config.yml            Data locations and subject lists for each study
│
├── annex/                    Supplementary material (its own Quarto project)
├── scripts/                  Shared Python code
│   ├── data_manager.py         Dataset readers (paths, subjects) built from *.config.yml
│   ├── figures.py              Shared colors, labels and plot helpers
│   ├── utils.py                Per-subject loops with error handling
│   ├── fsrecon.sh              FreeSurfer recon-all launcher
│   └── view_harmonics.py, review_resection.py   Interactive QC tools
├── results/                  Saved figures and statistics loaded by index.qmd and the annex
├── _freeze/                  Saved execution results of index.qmd (do not edit)
│
├── *.ipynb                   Exploratory notebooks (lambda2 tuning, surrogate harmonics)
└── archive/                  Past material (OHBM 2026 abstract, Radiology Day slides)
```

## Rendering

```bash
quarto render              # manuscript only, reuses the saved results in _freeze/ (fast)
quarto render index.qmd    # manuscript only, re-executes all of its code cells
cd annex && quarto render  # supplementary material
```

The output goes to `_manuscript/`. A plain `quarto render` never runs the processing pipelines or the notebooks.

## Reproducing the processing

The files in `processing/` are long pipelines (source spaces, forward models, connectomes, inverse solutions, statistics). They write to `results/`, which the manuscript then reads. Run them cell by cell in VS Code or Jupyter:

1. Set the data paths (`DIR_*`, `REF_*`, `FS_HOME`) in `processing/*.config.yml`. The datasets are not part of this repository.
2. Run the `setup.py` cell at the top of each file first. It moves the kernel to the project root, so the paths in the following cells (`processing/*.config.yml`, `results/`, `scripts`) resolve from there.

Python dependencies: [`mindmap`](https://github.com/NinoHerve/mindmap), MNE-Python, NumPy, SciPy, Matplotlib. FreeSurfer is needed for surface reconstruction.

## Files not in the repository

These are created locally and ignored by git: `_manuscript/`, `.quarto/`, `index.html`, `index_files/` (Quarto output); `reports/` (per-subject QC reports); `references/` (literature PDFs).
