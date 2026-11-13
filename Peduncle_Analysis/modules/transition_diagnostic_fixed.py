"""Diagnostic-only peduncle/spike transition analysis (v2).

Drop-in replacement for transition_diagnostic.py. Does not modify seed detection,
tracking, centreline smoothing, cut selection, ROI or trait measurements.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import median_filter
from .peduncle_tracking import _candidates


def _median(a, width=9):
    a = np.asarray(a, float)
    good = np.isfinite(a)
    if good.sum() < 3:
        return np.full_like(a, np.nan)
    v = median_filter(np.interp(np.arange(len(a)), np.flatnonzero(good), a[good]),
                      size=width, mode='nearest')
    v[~good] = np.nan
    return v


def _select_transition(z, area, p75, valid, current_cut_z,
                       baseline_slices=35, confirm_slices=5,
                       area_ratio=1.18, min_area_jump_mm2=0.22):
    """Identify sustained onset of expansion, with attenuation as supporting evidence.

    Does not require attenuation to increase. Returns None when no clear onset.
    Parameters are exploratory, not calibrated to a validated reference set.
    """
    n = len(z)
    candidates = []
    # Search near the terminal region, excluding the beginning of the sampled window.
    # Current cut is only a search anchor, NOT an enforced upper boundary.
    for i in range(baseline_slices + 8, n - confirm_slices):
        if z[i] < current_cut_z - 85 or z[i] > current_cut_z + 25:
            continue
        if not valid[i] or not np.isfinite(area[i]):
            continue
        baseline = area[i-baseline_slices:i-5]
        baseline = baseline[np.isfinite(baseline)]
        ahead = area[i:i+confirm_slices]
        if len(baseline) < baseline_slices * .75 or np.isfinite(ahead).sum() < confirm_slices:
            continue
        b = float(np.median(baseline))
        threshold = max(b * area_ratio, b + min_area_jump_mm2)
        if np.median(ahead) < threshold:
            continue
        # Require expansion to remain present beyond the short confirmation window.
        extended = area[i:i+min(12, n-i)]
        if np.isfinite(extended).sum() < 7 or np.nanmedian(extended) < threshold:
            continue
        # Reject gradual drift: look for local acceleration of area over preceding slices.
        prior = area[max(0, i-7):i]
        if np.isfinite(prior).sum() < 5:
            continue
        local_gain = float(np.nanmedian(ahead) - np.nanmedian(prior))
        if local_gain < max(.12, b * .07):
            continue
        # Attenuation is reported as evidence but NEVER required to increase.
        pb = p75[max(0, i-12):i]
        pa = p75[i:i+confirm_slices]
        intensity_delta = (float(np.nanmedian(pa) - np.nanmedian(pb))
                           if np.isfinite(pb).sum() >= 5 and np.isfinite(pa).sum() >= 3
                           else None)
        candidates.append({'z': int(z[i]), 'baseline_area_mm2': b,
                           'threshold_area_mm2': float(threshold),
                           'confirmed_area_mm2': float(np.nanmedian(ahead)),
                           'local_area_gain_mm2': local_gain,
                           'p75_change': intensity_delta})
    # Prefer a dominant terminal expansion over weak earlier diameter drift.
    strong = [c for c in candidates if c['local_area_gain_mm2'] >= 0.45]
    if strong:
        peak = max(strong, key=lambda c: c['local_area_gain_mm2'])
        # First part of the dominant change, not its maximal derivative.
        neighborhood = [c for c in strong if abs(c['z'] - peak['z']) <= 10 and
                        c['local_area_gain_mm2'] >= 0.35 * peak['local_area_gain_mm2']]
        selected = min(neighborhood, key=lambda c: c['z'])
        return selected['z'], 'provisional_area_expansion', candidates

    # Some spike bases change attenuation strongly without obvious area expansion.
    # This is a lower-confidence fallback; compare against local peduncle baseline.
    intensity_events = []
    for i in range(35, n - confirm_slices):
        if z[i] < current_cut_z - 45 or z[i] > current_cut_z + 25:
            continue
        before = p75[i-35:i-7]
        after = p75[i:i+confirm_slices]
        if np.isfinite(before).sum() < 20 or np.isfinite(after).sum() < confirm_slices:
            continue
        baseline = float(np.nanmedian(before))
        rise = float(np.nanmedian(after) - baseline)
        if rise > max(700., baseline * .22):
            intensity_events.append((int(z[i]), rise))
    if intensity_events:
        peak_z, peak_rise = max(intensity_events, key=lambda x:x[1])
        nearby = [zz for zz, rise in intensity_events
                  if abs(zz - peak_z) <= 14 and rise >= .5 * peak_rise]
        return min(nearby), 'review_attenuation_only', candidates
    return None, 'no_clear_sustained_transition', candidates


def _save_xy_montage(cleaned, tracking, prefix, candidate_z, current_cut_z,
                     reference_z, radius=36):
    centers = tracking['interpolated_centers']
    anchors = [candidate_z - 12, candidate_z - 6, candidate_z - 3,
               candidate_z, candidate_z + 3, candidate_z + 6,
               candidate_z + 12] if candidate_z is not None else [current_cut_z + d for d in (-18,-12,-6,0,6,12)]
    if reference_z is not None:
        anchors.append(int(reference_z))
    # A NumPy array uses `a in centers` to test VALUES, not valid Z indices.
    # Support both array-like centres indexed by original Z and dicts keyed by Z.
    def get_center(slice_z):
        try:
            if isinstance(centers, dict):
                center = centers.get(slice_z)
                if center is None:
                    center = centers.get(str(slice_z))
            else:
                if not (0 <= slice_z < len(centers)):
                    return None
                center = centers[slice_z]
            if center is None:
                return None
            cy, cx = center
            if not np.isfinite([cy, cx]).all():
                return None
            return float(cy), float(cx)
        except (IndexError, KeyError, TypeError, ValueError):
            return None

    anchors = sorted(set(int(a) for a in anchors if a is not None and
                         0 <= int(a) < cleaned.shape[0] and
                         get_center(int(a)) is not None))
    if not anchors:
        print('WARNING: XY montage not saved: no valid tracked centres near transition.', flush=True)
        return None
    # A 2x4 montage has room for at most eight cross-sections.
    anchors = anchors[:8]
    fig, axs = plt.subplots(2, 4, figsize=(12, 6.5))
    for ax, slice_z in zip(axs.flat, anchors):
        cy, cx = get_center(slice_z)
        y, x = int(round(cy)), int(round(cx))
        y0, y1 = max(0,y-radius), min(cleaned.shape[1],y+radius+1)
        x0, x1 = max(0,x-radius), min(cleaned.shape[2],x+radius+1)
        tile = np.asarray(cleaned[slice_z,y0:y1,x0:x1])
        finite = tile[tile > 0]
        vmax = float(np.percentile(finite, 98)) if finite.size else 5000
        ax.imshow(tile, cmap='gray', vmin=0, vmax=max(2500,vmax), origin='upper')
        ax.plot(cx-x0,cy-y0,'r+',ms=9,mew=1.2)
        suffix = ('  suggested' if slice_z == candidate_z else '') + ('  visual' if slice_z == reference_z else '')
        ax.set_title(f'Z={slice_z}{suffix}',fontsize=9)
        ax.axis('off')
    for ax in list(axs.flat)[len(anchors):]:
        ax.axis('off')
    fig.suptitle('Original cleaned XY cross-sections around suggested transition\n'
                 'Red + = tracked centre; display contrast adjusted per slice',fontsize=11)
    fig.tight_layout()
    dest = prefix.parent / (prefix.name+'_transition_xy_montage.png')
    fig.savefig(dest,dpi=160)
    plt.close(fig)
    return str(dest)


def save_transition_diagnostic(cleaned, tracking, output_prefix, current_cut_z,
                               voxel_mm=0.084, window_below_mm=18.0,
                               window_above_mm=1.0, local_radius_px=35,
                               change_window_slices=12, reference_z=None):
    """Write diagnostic CSV/JSON, four-panel plot and XY montage; ROI unchanged."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    lo = max(int(tracking['tracked_z_start']),int(current_cut_z-round(window_below_mm/voxel_mm)))
    hi = min(int(tracking['tracked_z_end']),cleaned.shape[0]-1,
             int(current_cut_z+round(window_above_mm/voxel_mm)))
    rows=[]
    for zval in range(lo,hi+1):
        cy,cx = tracking['interpolated_centers'][zval]
        row=dict(z=int(zval),valid=False,area_px=None,area_mm2=None,
                 filled_area_px=None,hole_area_px=None,hole_fraction=None,
                 median_attenuation=None,p75_attenuation=None,p90_attenuation=None,
                 center_y=None,center_x=None)
        if not np.isfinite([cy,cx]).all():
            rows.append(row);continue
        row['center_y'],row['center_x']=float(cy),float(cx)
        y0=max(0,int(np.floor(cy-local_radius_px)))
        y1=min(cleaned.shape[1],int(np.ceil(cy+local_radius_px))+1)
        x0=max(0,int(np.floor(cx-local_radius_px)))
        x1=min(cleaned.shape[2],int(np.ceil(cx+local_radius_px))+1)
        tile=cleaned[zval,y0:y1,x0:x1]
        labels,candidates=_candidates(tile,min_area=15,max_area=4000,max_extent=95,holder_threshold=4500)
        if not candidates:
            rows.append(row);continue
        chosen=min(candidates,key=lambda c:np.hypot(c['y']+y0-cy,c['x']+x0-cx))
        if np.hypot(chosen['y']+y0-cy,chosen['x']+x0-cx)>12:
            rows.append(row);continue
        mask=labels==chosen['label']
        filled=ndi.binary_fill_holes(mask)
        holes=int(np.count_nonzero(filled & ~mask))
        filled_area=int(np.count_nonzero(filled))
        vals=tile[mask]
        row.update(valid=True,area_px=int(mask.sum()),area_mm2=float(mask.sum()*voxel_mm**2),
                   filled_area_px=filled_area,hole_area_px=holes,
                   hole_fraction=float(holes/max(filled_area,1)),
                   median_attenuation=float(np.median(vals)),
                   p75_attenuation=float(np.percentile(vals,75)),
                   p90_attenuation=float(np.percentile(vals,90)))
        rows.append(row)
    z=np.asarray([r['z'] for r in rows])
    names=('area_mm2','hole_fraction','p75_attenuation')
    profiles={}
    for name in names:
        raw=np.asarray([np.nan if r[name] is None else r[name] for r in rows],float)
        profiles[name]={'raw':raw,'smooth':_median(raw)}
    candidate_z,confidence,candidates=_select_transition(
        z,profiles['area_mm2']['smooth'],profiles['p75_attenuation']['smooth'],
        np.asarray([r['valid'] for r in rows],bool),int(current_cut_z))
    result={'current_cut_z':int(current_cut_z),'suggested_transition_z':candidate_z,
            'suggested_transition_score':None,'reference_z':reference_z,
            'z_range':[int(lo),int(hi)],'valid_fraction':float(np.mean([r['valid'] for r in rows])),
            'method':'sustained area-expansion onset (P75 supporting only; v2)',
            'confidence':confidence,'candidate_expansions':candidates,
            'change_window_slices':int(change_window_slices),
            'note':'Diagnostic only. No changes to ROI, tracking, smoothing or traits. Area is material area; hole fraction is a segmentation proxy. Visually validate on original REK.',
            'rows':rows}
    with (prefix.parent/(prefix.name+'_transition_profile.csv')).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['z'])
        writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True)
    panel=[('area_mm2','Tracked component area (mm²)'),('hole_fraction','Enclosed hole fraction (0–1)'),
           ('p75_attenuation','Tracked tissue P75 attenuation')]
    for ax,(name,ylabel) in zip(axes[:3],panel):
        ax.plot(z,profiles[name]['raw'],color='0.70',lw=.8,label='Raw per-slice')
        ax.plot(z,profiles[name]['smooth'],color='#2268a0',lw=1.8,label='9-slice median (diagnostic only)')
        ax.set_ylabel(ylabel);ax.grid(alpha=.18)
    axes[0].legend(fontsize=8)
    # Show area relative to preceding baseline to make onset evidence interpretable.
    baseline=np.full(len(z),np.nan)
    a=profiles['area_mm2']['smooth']
    for i in range(35,len(z)):
        vals=a[i-35:i-5]
        if np.isfinite(vals).sum()>=23:
            baseline[i]=np.nanmedian(vals)
    axes[3].plot(z,a-baseline,color='#8a5bb5',lw=1.5,label='Area above preceding baseline')
    axes[3].set_ylabel('Area increase (mm²)');axes[3].set_xlabel('Original REK Z slice')
    axes[3].grid(alpha=.18)
    for ax in axes:
        ax.axvline(current_cut_z,color='#d97706',ls='--',lw=1.8,label=f'Existing cut z={current_cut_z}')
        if candidate_z is not None:
            ax.axvline(candidate_z,color='#0f766e',lw=1.8,label=f'Suggested onset z={candidate_z}')
        if reference_z is not None:
            ax.axvline(reference_z,color='#b91c1c',ls=':',lw=1.3,label=f'Visual reference z={reference_z}')
    axes[3].legend(fontsize=8,loc='best')
    fig.suptitle('Peduncle–spike transition diagnostic v2 (ROI unchanged)',fontsize=13)
    fig.tight_layout()
    fig.savefig(prefix.parent/(prefix.name+'_transition_diagnostic.png'),dpi=170)
    plt.close(fig)
    result['xy_montage_path']=_save_xy_montage(cleaned,tracking,prefix,candidate_z,current_cut_z,reference_z)
    if result['xy_montage_path']:
        print(f"Saved XY montage: {result['xy_montage_path']}", flush=True)
    (prefix.parent/(prefix.name+'_transition_diagnostic.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
