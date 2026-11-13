"""Multi-slice automatic seed search for compact hollow peduncle sections.

Conservative experimental candidate ranking; returns a seed and diagnostics.
It does not certify biological identity or repair clipped tissue.
"""
import numpy as np
from scipy import ndimage as ndi
from .peduncle_tracking import _candidates, track_peduncle


def _ring_features(image, candidate, labels):
    sub = labels[candidate['box']] == candidate['label']
    filled = ndi.binary_fill_holes(sub)
    holes = filled & ~sub
    hole_fraction = float(holes.sum() / max(filled.sum(), 1))
    h, w = sub.shape
    aspect = min(h, w) / max(h, w)
    return hole_fraction, aspect


def find_automatic_seed(cleaned, z_min=500, z_max=None,
                        search_start_fraction=0.55, search_end_fraction=0.88,
                        sample_step=35, max_candidates=18,
                        holder_threshold=4500, min_area=35,
                        max_area=2200, max_extent=95,
                        pilot_half_window=90, min_pilot_fraction=0.78,
                        min_hole_fraction=0.035):
    """Return (seed_z, (seed_y, seed_x), diagnostics).

    Rank hollow compact cross-sections, then verify local tracking across Z.
    Fails explicitly if no sufficiently persistent candidate is found.
    """
    z_max = cleaned.shape[0] if z_max is None else min(z_max, cleaned.shape[0])
    span = z_max - z_min
    lo = z_min + int(span * search_start_fraction)
    hi = z_min + int(span * search_end_fraction)
    if hi <= lo:
        raise ValueError('Invalid automatic seed search range')
    kwargs = dict(min_area=min_area, max_area=max_area,
                  max_extent=max_extent, holder_threshold=holder_threshold)
    seeds = []
    for z in range(lo, hi, sample_step):
        labels, candidates = _candidates(cleaned[z], **kwargs)
        for c in candidates:
            hole, aspect = _ring_features(cleaned[z], c, labels)
            if hole < min_hole_fraction or aspect < 0.60 or c['high'] > 0.15:
                continue
            # Strongly favour an annular cross-section, while not insisting
            # on an ideal circle or a particular peduncle diameter.
            ring_score = (3.0 * aspect + 2.0 * min(hole, 0.55)
                          + min(c['compact'], 1.0) - 3.0 * c['high'])
            seeds.append((ring_score, z, c))
    if not seeds:
        raise ValueError('No hollow compact seed found. Flag for manual QC.')
    seeds.sort(key=lambda t: t[0], reverse=True)
    # Avoid spending pilot passes on near-identical seeds.
    shortlisted = []
    for item in seeds:
        _, z, c = item
        if any(abs(z - p[1]) < 45 and
               np.hypot(c['y']-p[2]['y'], c['x']-p[2]['x']) < 15
               for p in shortlisted):
            continue
        shortlisted.append(item)
        if len(shortlisted) >= max_candidates:
            break
    trials = []
    for base_score, z, c in shortlisted:
        low = max(z_min, z-pilot_half_window)
        high = min(z_max, z+pilot_half_window+1)
        try:
            tr = track_peduncle(cleaned, z_min=low, z_max=high,
                                seed_z=z, seed_xy=(c['y'], c['x']),
                                holder_threshold=holder_threshold,
                                min_area=min_area, max_area=max_area,
                                max_extent=max_extent, max_shift_px=5,
                                max_gap_slices=4)
        except ValueError:
            continue
        local_valid = tr['valid'][low:high]
        fraction = float(np.mean(local_valid))
        pts = tr['centers'][low:high][local_valid]
        if len(pts) < 3:
            continue
        jumps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        max_jump = float(np.percentile(jumps, 95))
        score = base_score + 7.0*fraction - 0.3*max_jump
        trials.append(dict(score=score, z=int(z), y=float(c['y']),
                           x=float(c['x']), pilot_valid_fraction=fraction,
                           pilot_p95_shift_px=max_jump))
    if not trials:
        raise ValueError('No automatic seed passed local continuity tracking.')
    trials.sort(key=lambda r:r['score'], reverse=True)
    best = trials[0]
    if best['pilot_valid_fraction'] < min_pilot_fraction:
        raise ValueError('Automatic seed pilot continuity too low: '
                         f"{best['pilot_valid_fraction']:.1%}. Manual QC required.")
    return best['z'], (best['y'], best['x']), dict(
        method='multi_slice_hollow_ring_pilot', selected=best,
        candidates=trials, search_z=[lo,hi],
        warning='Experimental automatic identification; inspect segmentation and QC.')
