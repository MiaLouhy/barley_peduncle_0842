
"""Experimental high-Z cut-end localisation and curved-centreline ROI selection.

Requires barley_analysis_new.py. Do not treat an automatic ROI as validated
without checking the saved diagnostic plot and source volume.
"""
import numpy as np
from scipy import ndimage as ndi


def select_cut_peduncle(cleaned, voxel_mm=0.084, length_mm=50.0,
                        manual_cut_z=None, terminal_window_mm=3.0,
                        max_gap_slices=4, min_occupied_pixels=15):
    """Return (z_start, z_end_exclusive, diagnostics).

    The centreline is a smoothed sequence of foreground centroids, so length is
    estimated along the bent stem rather than by Z extent. Assumes stem is
    approximately Z-monotonic; strongly folded stems need a 3D skeleton.
    """
    if cleaned.ndim != 3:
        raise ValueError('Expected (Z,Y,X) volume')

    # Main 3D component excludes disconnected foam/background, when possible.
    mask = cleaned > 0
    labels, n = ndi.label(mask)
    if n == 0:
        raise ValueError('No foreground after cleanup')

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    main_label = int(np.argmax(counts))
    main = labels == main_label
    del labels, mask

    areas = np.count_nonzero(main, axis=(1, 2))
    occupied = areas >= min_occupied_pixels
    z_valid = np.flatnonzero(occupied)
    if z_valid.size == 0:
        raise ValueError('No sufficiently occupied slices')

    terminal_z = int(z_valid[-1])

    # At the cut surface, compact tissue often gives a high intensity percentile.
    # Search only the terminal few mm, not the full stem (nodes may be denser).
    window = max(2, int(round(terminal_window_mm / voxel_mm)))
    lo = max(int(z_valid[0]), terminal_z - window)
    zs = np.arange(lo, terminal_z + 1)
    scores = np.full(zs.size, np.nan)

    for i, z in enumerate(zs):
        values = cleaned[z][main[z]]
        if values.size >= min_occupied_pixels:
            scores[i] = float(np.percentile(values, 75))

    if manual_cut_z is not None:
        cut_z = int(manual_cut_z)
        if cut_z < z_valid[0] or cut_z > terminal_z:
            raise ValueError(f'Manual cut z={cut_z} outside occupied range')
        method = 'manual'
    else:
        if not np.any(np.isfinite(scores)):
            raise ValueError('No usable high-Z intensity measurements')
        cut_z = int(zs[np.nanargmax(scores)])
        method = 'terminal_intensity_peak_candidate'

    # Slice centroids: centreline can curve in XY, but must remain Z-monotonic.
    z0 = int(z_valid[0])
    active_z = np.arange(z0, cut_z + 1)
    cy = np.full(active_z.size, np.nan)
    cx = np.full(active_z.size, np.nan)

    for i, z in enumerate(active_z):
        if areas[z] < min_occupied_pixels:
            continue
        yy, xx = np.nonzero(main[z])
        cy[i] = float(yy.mean())
        cx[i] = float(xx.mean())

    good = np.flatnonzero(np.isfinite(cx))
    if good.size < 3 or good[-1] < len(active_z) - 1 - max_gap_slices:
        raise ValueError('Centreline incomplete near cut end')

    gaps = np.diff(good)
    if gaps.size and gaps.max() > max_gap_slices + 1:
        raise ValueError('Long gap in segmented stem centreline')

    indices = np.arange(len(active_z))
    cy = np.interp(indices, good, cy[good])
    cx = np.interp(indices, good, cx[good])

    # Median suppresses occasional off-axis segmentation, Gaussian smooths trace.
    cy = ndi.gaussian_filter1d(ndi.median_filter(cy, size=9), sigma=5)
    cx = ndi.gaussian_filter1d(ndi.median_filter(cx, size=9), sigma=5)

    steps_mm = voxel_mm * np.sqrt(1 + np.diff(cx)**2 + np.diff(cy)**2)
    backwards_distance = np.r_[0.0, np.cumsum(steps_mm[::-1])]

    if backwards_distance[-1] < length_mm:
        raise ValueError(
            f'Only {backwards_distance[-1]:.1f} mm of centreline below cut; '
            f'need {length_mm:.1f} mm'
        )

    k = int(np.searchsorted(backwards_distance, length_mm))
    start_z = cut_z - k

    if start_z < z0:
        raise ValueError('Peduncle ROI extends below detected plant')

    diag = {
        'cut_z': cut_z,
        'cut_detection': method,
        'terminal_stem_z': terminal_z,
        'cut_to_terminal_distance_mm': round(
            (terminal_z-cut_z)*voxel_mm, 3
        ),
        'roi_z_start': int(start_z),
        'roi_z_end_exclusive': int(cut_z+1),
        'roi_z_extent_mm': round((cut_z-start_z)*voxel_mm, 3),
        'roi_centreline_length_mm': round(
            float(backwards_distance[k]), 3
        ),
        'centreline_length_requested_mm': float(length_mm),
        'candidate_terminal_z': zs.tolist(),
        'candidate_terminal_intensity_p75': [
            None if not np.isfinite(v) else round(float(v), 2)
            for v in scores
        ],
        'z_profile': active_z.tolist(),
        'centroid_x': np.round(cx, 2).tolist(),
        'centroid_y': np.round(cy, 2).tolist(),
        'cross_section_area_px': areas[active_z].tolist(),
    }

    return start_z, cut_z + 1, diag
