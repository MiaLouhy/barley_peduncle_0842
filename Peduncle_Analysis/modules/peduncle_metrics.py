"""Per-slice peduncle measurements extracted unchanged from run_peduncle_analysis.py.

Only the function used by the transition-based workflow is retained.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import closing as sk_closing, disk as sk_disk, remove_small_objects


def per_slice_metrics(tube, px_xy_mm=0.084, closing_radius=1,
                      lumen_quantile=None, min_lumen_area_px=50):
    """Reproduce the uploaded module's per-slice formulae, without changing its summary."""
    source = tube.peduncle_structure_data
    if source is None:
        raise ValueError("Peduncle structure has not been calculated")
    footprint = sk_disk(closing_radius) if closing_radius > 0 else None
    pixel_area = px_xy_mm ** 2
    records = []
    active = np.flatnonzero(np.any(source > 0, axis=(1, 2)))
    for z in active:
        intensity_slice = source[z]
        slice_mask = intensity_slice > 0
        closed_slice = sk_closing(slice_mask, footprint) if footprint is not None else slice_mask
        outer_mask = ndi.binary_fill_holes(closed_slice).astype(bool)
        outer_area_px = int(np.count_nonzero(outer_mask))
        if outer_area_px <= 0:
            continue
        if lumen_quantile is None:
            lumen_mask = (outer_mask & ~closed_slice).astype(bool)
        else:
            outer_values = intensity_slice[outer_mask]
            if not outer_values.size:
                continue
            threshold = float(np.percentile(outer_values, 100 * lumen_quantile))
            candidates = (intensity_slice <= threshold) & outer_mask
            candidates = remove_small_objects(candidates.astype(bool),
                                              min_size=max(1, int(min_lumen_area_px)),
                                              connectivity=2)
            labels, n_labels = ndi.label(candidates)
            if n_labels:
                counts = np.bincount(labels.ravel())
                lumen_mask = labels == int(np.argmax(counts[1:]) + 1)
            else:
                lumen_mask = np.zeros_like(outer_mask, dtype=bool)
        lumen_area_px = int(np.count_nonzero(lumen_mask))
        if lumen_area_px <= 0:
            continue
        wall_mask = outer_mask & ~lumen_mask
        wall_area_px = int(np.count_nonzero(wall_mask))
        if wall_area_px <= 0:
            continue
        d_out = 2 * np.sqrt(outer_area_px / np.pi) * px_xy_mm
        d_in = 2 * np.sqrt(lumen_area_px / np.pi) * px_xy_mm
        yy, xx = np.indices(outer_mask.shape, dtype=float)
        x_coords, y_coords = xx * px_xy_mm, yy * px_xy_mm
        xc = float(np.mean(x_coords[wall_mask]))
        yc = float(np.mean(y_coords[wall_mask]))
        i_x = float(np.sum((y_coords[wall_mask] - yc)**2) * pixel_area)
        i_y = float(np.sum((x_coords[wall_mask] - xc)**2) * pixel_area)
        i_mm4 = i_x + i_y
        records.append({
            "z_slice": int(z),
            "d_out_mm": float(d_out),
            "d_in_mm": float(d_in),
            "wt_mm": float(max((d_out-d_in)/2, 0)),
            "csa_mm2": float(max((outer_area_px-lumen_area_px)*pixel_area, 0)),
            "pr": float(lumen_area_px/outer_area_px),
            "i_mm4": float(i_mm4),
            "z_mm3": float(i_mm4/max(d_out/2, 1e-9)),
            "wall_mean_intensity": float(np.mean(intensity_slice[wall_mask])),
        })
    return records

