# Transition-based peduncle analysis — single and batch

Run in the existing `xray313` environment. All image processing, transition selection, centreline ROI and anatomical calculations are delegated to the supplied validated single-REK script; the new batch file handles only database discovery, sample paths, skipping and logging.

## Batch

1. Open `batch_peduncle_transition_roi.py`.
2. Set `LOADING_CYCLES` and `OUTPUT_ROOT`; optionally set `SOURCE_DRIVE_OVERRIDE = "X:"` if the database path uses another drive.
3. First set `DRY_RUN = True` to inspect the database-discovered jobs. **Dry run does not write results.**
4. Then set `DRY_RUN = False` and `MAX_SAMPLES = 2` to validate two samples.
5. Once results match your existing single-REK analysis, set `MAX_SAMPLES = None`.

```powershell
conda activate xray313
cd D:\Projects\0842_OzB_P\Peduncle_Analysis
python batch_peduncle_transition_roi.py
```

Requires `pandas`, `sqlalchemy` and `pymysql` for the database lookup, plus the dependencies of your working single-REK script. Database/network and X: drive must be accessible.

## Loading-cycle selection

`LOADING_CYCLES = "0842-OzB-P"` selects exact `0842-OzB-P` or cycles starting `0842-OzB-P-` (e.g. `0842-OzB-P-01`). `LOADING_CYCLES = "0842-OzB-P-01"` selects that cycle (and nested suffix variants). Database `MeasurementName` must contain the query; this follows the original batch script.

## Results

For each REK:

`OUTPUT_ROOT / <loading_cycle> / <sample_holder> / <barcode>-<cell> /`

The four primary outputs remain in the sample folder: `*_summary.csv`, `*_per_slice.csv`, `*_peduncle.tif`, `*_cut_roi_diagnostic.png`. Other diagnostic outputs go into `Diagnostic/` using the original organiser.

At the root: `batch_all_summary.csv` (one row per processed/skipped sample) and `batch_errors.csv` (only written when errors occur). These are updated after each job. A new run overwrites the batch summary index; sample-level results are preserved when skipped. If no errors occur on a later run, an old `batch_errors.csv` from a prior run is not automatically deleted, so check its timestamps or move it before rerunning.

## Skip-existing logic

A sample is skipped only when the primary summary CSV, per-slice CSV, cut/ROI diagnostic PNG, and (when `SAVE_TIFF=True`) TIFF all exist and are nonempty. The batch summary row for a skipped sample is read from the existing sample summary CSV. Incomplete outputs trigger reprocessing; the single-REK script may reject existing diagnostic files in `Diagnostic/` when it reorganises, so move aside or clear the incomplete sample's output directory before retrying if a collision occurs. Do not delete completed results accidentally.

`OVERWRITE=True` forces reprocessing; it does not automatically clear old diagnostics. Use a new output root for validation or archive old outputs before forcing a rerun.

## Important

- `SHOW_NAPARI` is forced to `False` for unattended batch runs.
- The original single-REK analysis code and modules are copied from the files supplied for this request. No image processing formulas were modified.
- The database and source REK volumes are not available in this environment. Only syntax and lightweight discovery/skip tests were possible; compare two real samples before a large batch.
