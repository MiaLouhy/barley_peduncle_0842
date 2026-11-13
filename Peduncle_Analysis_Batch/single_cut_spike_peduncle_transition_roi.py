"""Experimental one-REK cut-spike peduncle analysis (xray313).

Run from the Peduncle_Analysis project folder with the modules/ directory present. Set REK_FILE, then run:
    python single_cut_spike_peduncle_transition_roi.py

Always inspect the diagnostic PNG/JSON and TIFF before using metrics.
"""
import csv
import json
from pathlib import Path
import shutil 

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules import barley_analysis_new
from modules import peduncle_metrics as saver
from modules.peduncle_tracking import track_peduncle, select_tracked_cut, extract_tracked_structure
from modules.automatic_seed import find_automatic_seed
from modules.transition_diagnostic_fixed import save_transition_diagnostic

# ======================= USER SETTINGS =======================
REK_FILE = Path(r"X:\Result\Ears\xray\0842-OzB-P-01\SH0005\XRAYCT_V_SH0005_260120141824012_ears_1_200mm_084mu_0842-OzB-P-01_260120190311151\XRAYCT_V_P-091179-12_260120141824012_ears_1_200mm_084mu_0842-OzB-P-01-SH0005_260120194159037.rek")
OUTPUT_DIR = Path(r"D:\Projects\0842_OzB_P\12")
PIXEL_SIZE_MM = 0.084
ROI_LENGTH_MM = 50.0
CENTRELINE_SMOOTH_WINDOW = 31  # odd number of slices (~2.6 mm)
CENTRELINE_SMOOTH_POLYORDER = 2
MANUAL_CUT_Z = None  # optional OLD cut anchor; final ROI always uses detected transition
RUN_METRICS = True  # False = first inspect ROI detection only
SAVE_TIFF = True
CLOSING_RADIUS = 1
LUMEN_QUANTILE = None
MIN_LUMEN_AREA_PX = 50
SHOW_NAPARI = False
TRANSITION_REFERENCE_Z = None  # optional visual reference; never affects detection
# Optional: choose a known stem cross-section (original REK Z and Y,X).
TRACK_SEED_Z = None
TRACK_SEED_YX = None  # e.g. (210, 245); useful when automatic seed selects wall
HOLDER_ATTENUATION_THRESHOLD = 4500
TRACK_RADIUS_PX = 40
# =============================================================


def save_csv(path, records):
    if not records:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        fieldnames = list(dict.fromkeys(k for record in records for k in record.keys()))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

def inspect_roi_in_napari(cleaned, z1, z2, diag, final_segmentation):
    import napari

    viewer = napari.Viewer(
        title="Cut-spike peduncle analysis",
        ndisplay=2
    )

    scale = (PIXEL_SIZE_MM,) * 3

    # 1. Full cleaned sample
    viewer.add_image(
        cleaned,
        name="Full cleaned sample",
        colormap="gray",
        scale=scale,
        visible=True,
    )

    # 2. Selected 50 mm ROI
    viewer.add_image(
        cleaned[z1:z2],
        name="Selected peduncle (50 mm)",
        colormap="magenta",
        blending="additive",
        opacity=0.5,
        scale=scale,
        translate=(z1 * PIXEL_SIZE_MM, 0, 0),
        visible=False,
    )

    # 3. Final structural segmentation
    viewer.add_image(
        final_segmentation,
        name="Final peduncle segmentation",
        colormap="cyan",
        blending="additive",
        opacity=0.85,
        scale=scale,
        translate=(z1 * PIXEL_SIZE_MM, 0, 0),
        visible=True,
    )

    # 4 and 5. ROI boundaries
    viewer.add_points(
        np.array([
            [z1, cleaned.shape[1] / 2, cleaned.shape[2] / 2],
            [z2 - 1, cleaned.shape[1] / 2, cleaned.shape[2] / 2],
        ]),
        name="ROI start / cut end",
        size=12,
        face_color=["lime", "orange"],
        scale=scale,
    )

    print("\nNapari layers:")
    print("Gray = full cleaned sample")
    print("Magenta = selected 50 mm ROI")
    print("Cyan = final segmented peduncle")
    print("Green point = ROI start")
    print("Orange point = cut end")

    napari.run()

def isolate_stem_test(cleaned, z1, z2, diag, radius_mm=5.0):
    import numpy as np

    roi = cleaned[z1:z2].copy()
    radius_px = radius_mm / PIXEL_SIZE_MM

    z_profile = np.asarray(diag["z_profile"])
    cx = np.asarray(diag["centroid_x"])
    cy = np.asarray(diag["centroid_y"])

    yy, xx = np.ogrid[:roi.shape[1], :roi.shape[2]]

    for i, z in enumerate(range(z1, z2)):
        x = np.interp(z, z_profile, cx)
        y = np.interp(z, z_profile, cy)

        mask = (xx - x)**2 + (yy - y)**2 <= radius_px**2
        roi[i, ~mask] = 0

    return roi

def organise_output_files(output_dir):
    output_dir = Path(output_dir)
    diagnostic_dir = output_dir / "Diagnostic"

    keep_suffixes = (
        "_cut_roi_diagnostic.png",
        "_peduncle.tif",
        "_per_slice.csv",
        "_summary.csv",
    )

    for file in output_dir.iterdir():
        if not file.is_file():
            continue

        if file.name.endswith(keep_suffixes):
            continue

        diagnostic_dir.mkdir(exist_ok=True)
        destination = diagnostic_dir / file.name

        if destination.exists():
            raise FileExistsError(
                f"Diagnostic file already exists: {destination}"
            )

        shutil.move(str(file), str(destination))

    print(f"Diagnostic files organised in: {diagnostic_dir}")

def main():
    if not REK_FILE.is_file():
        raise FileNotFoundError(f'Edit REK_FILE to point to an existing REK:\n{REK_FILE}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = REK_FILE.stem

    print(f'Loading: {REK_FILE}', flush=True)
    tube = barley_analysis_new.Tube(REK_FILE)
    print('Removing holder tube...', flush=True)
    tube.remove_holder_tube(start_slice=500, stop_slice=2650, method="angular", radial_margin=0, n_angles=360, show_progress=False,)
    print('Thresholding and removing islands...', flush=True)
    tube.threshold_denoise_and_remove_islands(verbose=True)
    cleaned = tube.segmented_data

    # print("\nChecking high-Z attenuation:")

    for z in range(2150, min(2701, cleaned.shape[0]), 50):
        values = cleaned[z]
        foreground = values[values > 0]

        # if foreground.size:
        #     print(
        #         f"Z={z}: "
        #         f"nonzero={foreground.size}, "
        #         f"median={np.median(foreground):.0f}, "
        #         f"P75={np.percentile(foreground, 75):.0f}, "
        #         f"P95={np.percentile(foreground, 95):.0f}, "
        #         f"max={foreground.max()}"
        #     )
        # else:
        #     print(f"Z={z}: no foreground")

    print('Detecting cut end and tracing 50 mm curved centreline...', flush=True)
    auto_seed_diagnostics = None
    seed_z, seed_yx = TRACK_SEED_Z, TRACK_SEED_YX
    if seed_z is None and seed_yx is None:
        print('Finding automatic seed from persistent hollow cross-sections...')
        seed_z, seed_yx, auto_seed_diagnostics = find_automatic_seed(
            cleaned, z_min=500, z_max=2650,
            holder_threshold=HOLDER_ATTENUATION_THRESHOLD)
        print(f'Automatic seed: Z={seed_z}, YX={seed_yx}')
        (OUTPUT_DIR / f'{prefix}_automatic_seed_qc.json').write_text(
            json.dumps(auto_seed_diagnostics, indent=2), encoding='utf-8')
    tracking = track_peduncle(
        cleaned, z_min=500, z_max=2650,
        seed_z=seed_z, seed_xy=seed_yx,
        holder_threshold=HOLDER_ATTENUATION_THRESHOLD,
    )
    print(f"Tracking seed Z={tracking['seed_z']}, YX=({tracking['seed_y']:.1f}, {tracking['seed_x']:.1f}); "
          f"tracked Z=[{tracking['tracked_z_start']}, {tracking['tracked_z_end']}]")
    z1, z2, diag = select_tracked_cut(
        cleaned, tracking, voxel_mm=PIXEL_SIZE_MM, length_mm=ROI_LENGTH_MM,
        manual_cut_z=MANUAL_CUT_Z,
        centreline_smooth_window=CENTRELINE_SMOOTH_WINDOW,
        centreline_smooth_polyorder=CENTRELINE_SMOOTH_POLYORDER,
    )
    # The old cut is retained only as the search anchor for v2 transition detection.
    old_cut_z = int(diag['cut_z'])
    transition = save_transition_diagnostic(
        cleaned, tracking, OUTPUT_DIR / prefix,
        current_cut_z=old_cut_z, voxel_mm=PIXEL_SIZE_MM,
        reference_z=TRANSITION_REFERENCE_Z,
    )
    transition_z = transition['suggested_transition_z']
    if transition_z is None:
        raise RuntimeError(
            f"No usable transition found (confidence={transition['confidence']}). "
            "Review the transition diagnostic; no traits have been calculated."
        )
    transition_z = int(transition_z)
    if not (0 <= transition_z < len(tracking['valid'])) or not tracking['valid'][transition_z]:
        raise RuntimeError(f'Transition Z={transition_z} is not a valid tracked slice; no traits calculated.')

    # Re-select the entire 50 mm ROI from the transition slice. This leaves
    # smoothing, component extraction, and all anatomical formulas unchanged.
    z1, z2, diag = select_tracked_cut(
        cleaned, tracking, voxel_mm=PIXEL_SIZE_MM, length_mm=ROI_LENGTH_MM,
        manual_cut_z=transition_z,
        centreline_smooth_window=CENTRELINE_SMOOTH_WINDOW,
        centreline_smooth_polyorder=CENTRELINE_SMOOTH_POLYORDER,
    )
    diag['cut_detection'] = 'transition_diagnostic_v2'
    diag['previous_cut_z'] = old_cut_z
    diag['transition_confidence'] = transition['confidence']
    diag['transition_method'] = transition['method']
    # print(f"Transition Z={transition_z} (confidence={transition['confidence']}); "
    #       f'previous cut Z={old_cut_z}; final ROI=[{z1},{z2})', flush=True)

    tracked_roi, tracked_mask, tracking_qc = extract_tracked_structure(
        cleaned, tracking, z1, z2, radius_px=TRACK_RADIUS_PX,
        holder_threshold=HOLDER_ATTENUATION_THRESHOLD,
    )
    save_csv(OUTPUT_DIR / f'{prefix}_tracking_qc.csv', tracking_qc)
    diag['tracking_qc_valid_fraction'] = float(np.mean([r['valid'] for r in tracking_qc]))
    diag['tracking_qc_suspicious_slices'] = int(sum(r.get('suspicious', False) for r in tracking_qc))
    if diag['tracking_qc_valid_fraction'] < 0.90:
        print('WARNING: fewer than 90% of ROI slices have a tracked component. Inspect QC and TIFF.')
    if diag['tracking_qc_suspicious_slices']:
        print(f"WARNING: {diag['tracking_qc_suspicious_slices']} suspicious slices. Inspect QC CSV.")
    diag['source_rek'] = str(REK_FILE)
    diag['note'] = ('Prototype: verify cut position and 50 mm ROI in the source volume. '
                    'Cross-sectional metrics use original Z planes, not planes '
                    'perpendicular to the curved stem centreline.')
    (OUTPUT_DIR / f'{prefix}_cut_roi_diagnostic.json').write_text(
        json.dumps(diag, indent=2), encoding='utf-8'
    )

    zz = np.asarray(diag['z_profile'])
    area = np.asarray(diag['cross_section_area_px'])
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    axes[0].plot(zz, area, linewidth=1)
    axes[0].axvline(z1, color='green', linestyle='--', label=f'ROI start z={z1}')
    axes[0].axvline(diag['cut_z'], color='orange', linestyle='--', label=f'Cut candidate z={diag["cut_z"]}')
    axes[0].set(xlabel='Original REK Z slice', ylabel='Tracked component area (pixels)', title='Tracked peduncle region')
    axes[0].legend()
    cand_z = np.asarray(diag['candidate_terminal_z'])
    cand_int = np.asarray([np.nan if v is None else v for v in diag['candidate_terminal_intensity_p75']])
    axes[1].plot(cand_z, cand_int, marker='.', linewidth=1)
    axes[1].axvline(diag['cut_z'], color='orange', linestyle='--')
    axes[1].set(xlabel='Original REK Z slice', ylabel='75th percentile attenuation', title='High-Z cut-end candidates (empty when transition-selected)')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{prefix}_cut_roi_diagnostic.png', dpi=170)
    plt.close(fig)
    print(f'Cut candidate z={diag["cut_z"]}; ROI z=[{z1}, {z2}); '
          f'smoothed centreline length={diag["roi_centreline_length_mm"]:.2f} mm; '
          f'raw length over same ROI={diag["roi_raw_centreline_length_mm"]:.2f} mm', flush=True)
    print('Saved diagnostic JSON and PNG.', flush=True)

    if not RUN_METRICS:
        print('RUN_METRICS=False: stopped after ROI detection.')
        return

    # Keep the full cleaned sample for napari inspection
    full_cleaned = cleaned

    # Disable old intact-spike boundary detection. Retain absolute bounds separately.
    tube.peduncle_bounds = (z1, z2)
    tube.peduncle_data = tracked_roi
    tube.peduncle_structure_data = tracked_roi  # bypass largest-3D-component selection
    tube.segmented_data = tracked_roi

    print('Calculating structural metrics...', flush=True)

    metrics = tube.calculate_cross_section_metrics(
        auto_locate_peduncle=False,
        px_xy_mm=PIXEL_SIZE_MM,
        closing_radius=CLOSING_RADIUS,
        lumen_quantile=LUMEN_QUANTILE,
        min_lumen_area_px=MIN_LUMEN_AREA_PX,
        verbose=True,
    )

    summary = {
        'source_rek': str(REK_FILE),
        'sample_preparation': 'cut_spike',
        'cut_z': diag['cut_z'],
        # 'previous_cut_z': old_cut_z,
        'transition_confidence': transition['confidence'],
        'cut_detection': diag['cut_detection'],
        'peduncle_z_start': z1,
        'peduncle_z_end_exclusive': z2,
        'roi_centreline_length_mm': diag['roi_centreline_length_mm'],
        'pixel_size_mm': PIXEL_SIZE_MM,
        **metrics,
    }
    save_csv(OUTPUT_DIR / f'{prefix}_summary.csv', [summary])
    (OUTPUT_DIR / f'{prefix}_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    rows = saver.per_slice_metrics(tube, PIXEL_SIZE_MM, CLOSING_RADIUS,
                                   LUMEN_QUANTILE, MIN_LUMEN_AREA_PX)
    # Existing per_slice_metrics uses local Z within the transition-based ROI.
    for r in rows:
        r['original_z'] = int(r['z_slice']) + z1 if 'z_slice' in r else None
    save_csv(OUTPUT_DIR / f'{prefix}_per_slice.csv', rows)
    if SAVE_TIFF:
        print('Exporting segmented TIFF...', flush=True)
        tube.export_segmented_tiff(
            OUTPUT_DIR / f'{prefix}_peduncle.tif',
            data=tube.peduncle_structure_data, overwrite=True, compress=False,
        )

    if SHOW_NAPARI:
        print('Opening combined napari viewer...', flush=True)

        inspect_roi_in_napari(
            full_cleaned,
            z1,
            z2,
            diag,
            tube.peduncle_structure_data
        )

    print(f'Done. Inspect outputs in {OUTPUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
    organise_output_files(OUTPUT_DIR)