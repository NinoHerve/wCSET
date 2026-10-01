# BEM comparison — SPIKE cohort, sub-pat10

Companion to the SIMF sub-10 example already in the Annex's "BEM comparison
against MNE-Python's default watershed extraction" section: that example
shows watershed failing MNE's own nesting check outright
(`RuntimeError: Surface inner skull is not completely inside surface outer
skull`). This one shows the complementary case — watershed's raw geometry
*passes* MNE's nesting check for this subject (`mne.make_bem_model` +
`mne.make_bem_solution` both succeed) and is still visibly worse than the
CHARM-based surfaces this manuscript actually uses.

Three separate views, not one combined overlay: a reference panel (CHARM's
tissue segmentation, for anatomical context) plus one panel per method
(CHARM surfaces, watershed surfaces), each screenshotted separately and
compared side by side — rather than layering the segmentation under both
surface panels, which would risk it reading as a strict pixel-level ground
truth it isn't (see caveat below).

## Provenance (no recomputation — copied as-is, source data untouched)

- `watershed_*.surf`: copied from
  `/media/nino/DATA/SPIKE/derivatives/freesurfer-7.4.1/sub-pat10/bem/watershed/`
  — MNE's own `make_watershed_bem()` raw output, never modified.
- `charm_inner_skull.surf` / `charm_outer_skull.surf` / `charm_outer_skin.surf`:
  copied from
  `/media/nino/DATA/SPIKE/derivatives/freesurfer-7.4.1/sub-pat10/bem/{inner_skull,outer_skull,outer_skin}.surf`
  — these filenames currently hold the CHARM-based surfaces, since
  `processing_spike.qmd`'s `segment_head()` overwrites them via
  `mm.anatomy.forward.save_bem_as_freesurfer()`; the untouched watershed
  originals survive only in the `watershed/` subfolder above. These surfaces
  are already transformed into FreeSurfer's tkRAS space (`bem_transform()`
  in `segment_head()`), so they align correctly with `T1.mgz` below.
- `T1.mgz`: copied from the same subject's FreeSurfer
  `mri/T1.mgz` (non-skull-stripped, tkRAS space) — background for the two
  surface commands below.
- `charm_labeling.nii.gz` / `charm_labeling_LUT.txt`: copied from
  `derivatives/charm/sub-pat10/m2m_sub-pat10/segmentation/{labeling.nii.gz,labeling_LUT.txt}`
  — CHARM's own tissue segmentation, with explicit `Bone-Cortical`,
  `Bone-Cancellous`, `Skin`, and `Eyes` labels.
- `charm_T1.nii.gz`: copied from
  `derivatives/charm/sub-pat10/m2m_sub-pat10/T1.nii.gz` — **not** the same
  file as `T1.mgz` above. `labeling.nii.gz` is defined in CHARM's own native
  space (different shape/affine than FreeSurfer's tkRAS-space `T1.mgz`); this
  project applies an explicit `ras2tkras` transform elsewhere to reconcile
  the two (see `bem_transform()`, used when building the `charm_*.surf`
  files above). `charm_T1.nii.gz` and `charm_labeling.nii.gz` share the exact
  same affine (verified directly), so pairing them avoids needing that
  transform here — do **not** load `charm_labeling.nii.gz` against `T1.mgz`,
  they will not align.

**Caveat for the figure caption:** `charm_outer_skull.surf` is fit directly
*from* `charm_labeling.nii.gz`'s bone labels (`build_bem_layers()`), so the
CHARM contour matching this segmentation is expected by construction, not
independent validation. Watershed never sees this segmentation at all (pure
intensity-based flood-fill on the T1), so its contour's agreement or
disagreement with it *is* informative. Caption language should call the
segmentation "CHARM's own tissue segmentation, shown for anatomical
reference" rather than "ground truth."

## Three views (run each interactively, screenshot the same slice/plane)

**1. Reference: CHARM's tissue segmentation (anatomical context only, no surfaces)**
```bash
freeview -v charm_T1.nii.gz \
            charm_labeling.nii.gz:colormap=lut:lut=charm_labeling_LUT.txt:opacity=0.5
```

**2. CHARM surfaces (this manuscript's pipeline):**
```bash
freeview -v T1.mgz \
  -f charm_inner_skull.surf:edgecolor=255,255,255:edgethickness=2 \
     charm_outer_skull.surf:edgecolor=255,255,255:edgethickness=2 \
     charm_outer_skin.surf:edgecolor=230,150,80:edgethickness=2
```

**3. MNE default watershed:**
```bash
freeview -v T1.mgz \
  -f watershed_inner_skull.surf:edgecolor=255,255,255:edgethickness=2 \
     watershed_outer_skull.surf:edgecolor=255,255,255:edgethickness=2 \
     watershed_outer_skin.surf:edgecolor=230,150,80:edgethickness=2 \
     watershed_brain.surf:edgecolor=255,150,180:edgethickness=2
```

Colors: white = skull (inner + outer), beige/orange = scalp, pink = brain
(watershed only — CHARM's `build_bem_layers()` doesn't produce a separate
brain surface, so there's no CHARM counterpart). Pick the same slice/plane
in all three and screenshot for the Annex figure.
