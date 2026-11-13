# Peduncle analysis — organised working version

## Run

1. Extract the entire ZIP to a folder (keep `modules` beside the main script).
2. Open `single_cut_spike_peduncle_transition_roi.py` and edit `REK_FILE` and `OUTPUT_DIR`.
3. In the `xray313` environment, open a terminal in this folder and run:

```powershell
python single_cut_spike_peduncle_transition_roi.py
```

## Structure

- `single_cut_spike_peduncle_transition_roi.py`: **only entry-point**; uses the transition diagnostic Z to anchor the curved 50 mm ROI, calculates existing traits, and saves outputs.
- `modules/barley_analysis_new.py`: REK reading, cleanup, anatomical metrics.
- `modules/peduncle_tracking.py`: peduncle tracking and curved ROI.
- `modules/automatic_seed.py`: seed detection.
- `modules/transition_diagnostic_fixed.py`: transition detection and QC.
- `modules/peduncle_metrics.py`: **only** `per_slice_metrics`, copied from the previous `run_peduncle_analysis.py` without modifying its formulas.
- `archive/cut_spike_detection.py`: earlier method, not used by the current workflow.

## Changes and validation

Only module imports and file organisation were changed. The `per_slice_metrics` function body is unchanged. No thresholds, tracking algorithms, transition rules, ROI calculations, or trait formulas were intentionally modified. The current script still moves non-primary outputs to the `Diagnostic` subfolder.

**Before replacing your working version**, test on a previously processed REK in a new `OUTPUT_DIR`. Compare the transition Z, ROI Z bounds, per-slice CSV, summary CSV, and TIFF. Static syntax checks passed; an end-to-end REK run was not possible here because the Windows source data are not available.

Note: `SHOW_NAPARI=True` opens the viewer; change to `False` if running unattended.
