"""Experimental shape/attenuation-guided peduncle tracker (Z,Y,X).

No largest-3D-component assumption. Select a compact cross-section in a seed
slice, then track by centroid proximity, radius/area and intensity. Never
silently reconstruct missing plant tissue; report gaps and partial masks.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import savgol_filter


def _candidates(image, min_area=35, max_area=2200, max_extent=95,
                holder_threshold=4500):
    """2D 8-connected candidates, retaining original foreground intensity."""
    binary = image > 0
    labels, n = ndi.label(binary, structure=np.ones((3, 3), bool))
    boxes = ndi.find_objects(labels)
    found = []
    for i, box in enumerate(boxes, 1):
        if box is None:
            continue
        sy, sx = box
        h, w = sy.stop-sy.start, sx.stop-sx.start
        if max(h, w) > max_extent:
            continue
        sub = labels[box] == i
        area = int(sub.sum())
        if not min_area <= area <= max_area:
            continue
        yy, xx = np.nonzero(sub)
        y = float(yy.mean()+sy.start)
        x = float(xx.mean()+sx.start)
        filled = ndi.binary_fill_holes(sub)
        fill_area = int(filled.sum())
        # Fill fraction: annular stem can have lower occupancy than a solid disk.
        aspect = min(h, w) / max(h, w)
        perimeter = sub & ~ndi.binary_erosion(sub)
        p = int(perimeter.sum())
        compact = float(4*np.pi*fill_area/max(p*p, 1))
        values = image[box][sub]
        high = float(np.mean(values >= holder_threshold))
        found.append(dict(label=i, y=y, x=x, area=area, aspect=aspect,
                          compact=min(compact, 1.5), high=high,
                          median=float(np.median(values)), box=box))
    return labels, found


def track_peduncle(cleaned, z_min=500, z_max=None, seed_z=None,
                   seed_xy=None, seed_search_step=20, holder_threshold=4500,
                   max_shift_px=12, min_area=35, max_area=2200,
                   max_extent=95, max_gap_slices=8):
    """Track candidate stem slices. Optional seed_xy=(y,x) for difficult cases.

    Returns arrays of centre positions, validity, area, score; coordinates are
    ORIGINAL REK coordinates. No implicit fallback to holder component.
    """
    if cleaned.ndim != 3:
        raise ValueError('Expected (Z,Y,X)')
    z_max = cleaned.shape[0] if z_max is None else min(int(z_max), cleaned.shape[0])
    if not 0 <= z_min < z_max:
        raise ValueError('Invalid Z range')
    depth = cleaned.shape[0]
    centers = np.full((depth, 2), np.nan)
    areas = np.zeros(depth, dtype=np.int32)
    quality = np.full(depth, np.nan)
    valid = np.zeros(depth, bool)
    kwargs = dict(min_area=min_area, max_area=max_area, max_extent=max_extent,
                  holder_threshold=holder_threshold)

    def seed_score(c):
        return (2.5*c['aspect'] + 1.5*min(c['compact'], 1.0)
                - 3.0*c['high'] - 0.00025*c['area'])

    if seed_z is None:
        # Prefer the high-Z third, where the stem/cut is expected; avoid the
        # last few slices, where end effects and cropping can dominate.
        lo = z_min + (z_max-z_min)//2
        sample_z = range(lo, max(lo+1, z_max-25), seed_search_step)
        seed_choices = []
        for z in sample_z:
            _, cand = _candidates(cleaned[z], **kwargs)
            for c in cand:
                score = seed_score(c)
                if seed_xy is not None:
                    score -= 0.04*np.hypot(c['y']-seed_xy[0], c['x']-seed_xy[1])
                seed_choices.append((score, z, c))
        if not seed_choices:
            raise ValueError('No compact peduncle seed found; set seed_z and seed_xy')
        _, seed_z, seed = max(seed_choices, key=lambda t:t[0])
    else:
        seed_z = int(seed_z)
        _, cand = _candidates(cleaned[seed_z], **kwargs)
        if not cand:
            raise ValueError(f'No eligible cross-section at seed_z={seed_z}')
        if seed_xy is None:
            seed = max(cand, key=seed_score)
        else:
            seed = min(cand, key=lambda c:np.hypot(c['y']-seed_xy[0],c['x']-seed_xy[1]))
            if np.hypot(seed['y']-seed_xy[0],seed['x']-seed_xy[1]) > max_shift_px*2:
                raise ValueError('No component close to requested seed_xy')

    centers[seed_z] = [seed['y'], seed['x']]
    areas[seed_z] = seed['area']
    quality[seed_z] = seed_score(seed)
    valid[seed_z] = True

    for direction in (-1, 1):
        last_center = centers[seed_z].copy()
        last_area = float(seed['area'])
        gap = 0
        for z in range(seed_z+direction, z_min-1 if direction<0 else z_max, direction):
            _, cand = _candidates(cleaned[z], **kwargs)
            chosen = None
            best = -np.inf
            for c in cand:
                distance = float(np.hypot(c['y']-last_center[0], c['x']-last_center[1]))
                if distance > max_shift_px*(1+0.35*gap):
                    continue
                area_change = abs(np.log(max(c['area'],1)/max(last_area,1)))
                score = (seed_score(c) - 0.18*distance - 0.65*area_change)
                if score > best:
                    best, chosen = score, c
            if chosen is None:
                gap += 1
                if gap > max_gap_slices:
                    break
                continue
            centers[z] = [chosen['y'],chosen['x']]
            areas[z] = chosen['area']
            quality[z] = best
            valid[z] = True
            last_center = centers[z].copy()
            last_area = float(chosen['area'])
            gap = 0

    zs = np.flatnonzero(valid)
    if len(zs) < 20:
        raise ValueError(f'Only {len(zs)} tracked slices; inspect seed/thresholds')
    # Interpolate short gaps for the centreline only, not the segmentation.
    interp = np.full_like(centers, np.nan)
    for axis in range(2):
        interp[zs[0]:zs[-1]+1,axis] = np.interp(
            np.arange(zs[0],zs[-1]+1),zs,centers[zs,axis])
    return dict(seed_z=int(seed_z), seed_y=float(seed['y']),
                seed_x=float(seed['x']), valid=valid, centers=centers,
                interpolated_centers=interp, areas=areas, quality=quality,
                tracked_z_start=int(zs[0]), tracked_z_end=int(zs[-1]))


def select_tracked_cut(cleaned, tracking, voxel_mm=0.084, length_mm=50.,
                       manual_cut_z=None, terminal_window_mm=3.,
                       centreline_smooth_window=31, centreline_smooth_polyorder=2):
    """Cut and curved 50mm ROI from tracked stem, never holder-wide centroids."""
    valid = tracking['valid']
    centers = tracking['interpolated_centers']
    terminal = int(tracking['tracked_z_end'])
    if manual_cut_z is not None:
        cut_z = int(manual_cut_z)
        if not valid[cut_z]:
            raise ValueError('Manual cut must lie on a tracked slice')
        method = 'manual_tracked'
    else:
        start = max(tracking['tracked_z_start'],terminal-int(terminal_window_mm/voxel_mm))
        scored = []
        for z in range(start,terminal+1):
            if not valid[z]:
                continue
            # Only the local tracked candidate: avoid high-intensity holder wall.
            cy,cx = tracking['centers'][z]
            y0,y1 = max(0,int(cy)-35),min(cleaned.shape[1],int(cy)+36)
            x0,x1 = max(0,int(cx)-35),min(cleaned.shape[2],int(cx)+36)
            tile=cleaned[z,y0:y1,x0:x1]
            labels,candidates=_candidates(tile,min_area=15,max_area=4000,max_extent=95,
                                           holder_threshold=4500)
            if not candidates:
                continue
            candidate=min(candidates,key=lambda c:np.hypot(c['y']+y0-cy,c['x']+x0-cx))
            values=tile[labels==candidate['label']]
            scored.append((float(np.percentile(values,75)),z))
        if not scored:
            raise ValueError('No tracked cut candidates; set manual_cut_z')
        _,cut_z=max(scored)
        method='tracked_terminal_p75_candidate'
    start_bound=int(tracking['tracked_z_start'])
    if cut_z-start_bound < 2:
        raise ValueError('Not enough tracked stem before cut')
    # Do not bridge long missing sections in ROI selection.
    zs=np.arange(start_bound,cut_z+1)
    if not np.all(np.isfinite(centers[zs])):
        raise ValueError('Incomplete centreline before cut')
    # The original per-slice centroids contain pixel-level jitter. Measuring
    # each raw step accumulates this jitter into a large positive length bias.
    # Smooth the coordinates ONLY for arc-length estimation. Segmentation and
    # cross-sectional metrics still use the original tracked components.
    raw_centers = centers[zs]
    raw_diffs = np.diff(raw_centers, axis=0)
    raw_steps = voxel_mm * np.sqrt(1.0 + np.sum(raw_diffs**2, axis=1))
    raw_backward = np.r_[0.0, np.cumsum(raw_steps[::-1])]
    window = int(centreline_smooth_window)
    if window < 3 or window % 2 != 1:
        raise ValueError('centreline_smooth_window must be an odd integer >= 3')
    poly = int(centreline_smooth_polyorder)
    if poly < 1 or poly >= window:
        raise ValueError('centreline_smooth_polyorder must be >= 1 and < window')
    if len(zs) < window:
        raise ValueError(f'Only {len(zs)} slices; smoothing window is {window}')
    smooth_centers = savgol_filter(raw_centers, window_length=window,
                                   polyorder=poly, axis=0, mode='interp')
    diffs = np.diff(smooth_centers, axis=0)
    step = voxel_mm * np.sqrt(1.0 + np.sum(diffs**2, axis=1))
    backward = np.r_[0.0, np.cumsum(step[::-1])]
    if not np.all(np.isfinite(backward)):
        raise ValueError('Non-finite smoothed centreline length')
    if backward[-1] < length_mm:
        raise ValueError(f'Only {backward[-1]:.2f}mm tracked; need {length_mm}mm')
    k=int(np.searchsorted(backward,length_mm))
    z1=cut_z-k
    diag=dict(cut_z=int(cut_z),cut_detection=method,
              roi_z_start=int(z1),roi_z_end_exclusive=int(cut_z+1),
              roi_centreline_length_mm=float(backward[k]),
              roi_z_extent_mm=float(k*voxel_mm),
              roi_raw_centreline_length_mm=float(raw_backward[k]),
              roi_smoothing_window_slices=window,
              roi_smoothing_polyorder=poly,
              roi_length_method='Savitzky-Golay smoothed XY centroids, 3D arc length',
              roi_start_centreline_overshoot_mm=float(backward[k]-length_mm),
              tracked_seed_z=int(tracking['seed_z']),
              tracked_valid_fraction=float(np.mean(valid[z1:cut_z+1])),
              z_profile=list(map(int,range(z1,cut_z+1))),
              centroid_y=centers[z1:cut_z+1,0].tolist(),
              centroid_x=centers[z1:cut_z+1,1].tolist(),
              smoothed_centroid_y=smooth_centers[-(k+1):,0].tolist(),
              smoothed_centroid_x=smooth_centers[-(k+1):,1].tolist(),
              cross_section_area_px=tracking['areas'][z1:cut_z+1].tolist(),
              candidate_terminal_z=[int(z) for _,z in scored] if manual_cut_z is None else [],
              candidate_terminal_intensity_p75=[float(v) for v,_ in scored] if manual_cut_z is None else [])
    return z1,cut_z+1,diag


def extract_tracked_structure(cleaned, tracking, z1, z2,
                              radius_px=40, min_area=20, max_area=3500,
                              max_extent=95, holder_threshold=4500):
    """Select only a centroid-consistent 2D component on each ROI slice.

    Returns intensity ROI, binary mask, and per-slice QC. Missing slices stay
    empty; no fabricated wall or lumen. Does not recover tube-cropped voxels.
    """
    if z1<0 or z2>cleaned.shape[0] or z1>=z2:
        raise ValueError('Invalid ROI')
    result=np.zeros_like(cleaned[z1:z2])
    qc=[]
    for local,z in enumerate(range(z1,z2)):
        center=tracking['interpolated_centers'][z]
        if not np.all(np.isfinite(center)):
            qc.append(dict(z=z,valid=False,reason='no_centreline'))
            continue
        cy,cx=center
        y0=max(0,int(np.floor(cy-radius_px)))
        y1=min(cleaned.shape[1],int(np.ceil(cy+radius_px))+1)
        x0=max(0,int(np.floor(cx-radius_px)))
        x1=min(cleaned.shape[2],int(np.ceil(cx+radius_px))+1)
        tile=cleaned[z,y0:y1,x0:x1]
        labels,cand=_candidates(tile,min_area=min_area,max_area=max_area,
                                 max_extent=max_extent,holder_threshold=holder_threshold)
        if not cand:
            qc.append(dict(z=z,valid=False,reason='no_compact_component'))
            continue
        chosen=min(cand,key=lambda c:np.hypot(c['y']+y0-cy,c['x']+x0-cx))
        dist=float(np.hypot(chosen['y']+y0-cy,chosen['x']+x0-cx))
        if dist>max(8,radius_px*0.45):
            qc.append(dict(z=z,valid=False,reason='component_far_from_centreline'))
            continue
        keep=labels==chosen['label']
        result[local,y0:y1,x0:x1][keep]=tile[keep]
        # A crop-border touch suggests holder attachment or clipped tissue.
        yy,xx=np.nonzero(keep)
        edge=bool((yy.min()==0) or (xx.min()==0) or
                  (yy.max()==tile.shape[0]-1) or (xx.max()==tile.shape[1]-1))
        qc.append(dict(z=z,valid=True,area=chosen['area'],
                       median_attenuation=chosen['median'],
                       high_attenuation_fraction=chosen['high'],
                       distance_px=dist,roi_edge_touch=edge,
                       suspicious=bool(edge or chosen['high']>0.20)))
    return result,result>0,qc
