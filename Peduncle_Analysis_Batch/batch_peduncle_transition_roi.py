"""Batch runner for the VALIDATED single_cut_spike_peduncle_transition_roi.py.

Only handles database discovery, per-sample paths, skipping and batch logs.
The actual analysis is delegated unchanged to the single-REK script.
Run from the Peduncle_Analysis project folder in xray313.
"""
import csv
import json
import re
import traceback
from datetime import datetime
from pathlib import Path, PureWindowsPath

import pandas as pd
from sqlalchemy import create_engine, text

import single_cut_spike_peduncle_transition_roi as analysis

# ====================== USER SETTINGS ======================
DATABASE_CONNECTION_STRING = "mysql+pymysql://readonlyuser:readonlyuser@129.127.38.50/phenoct"
LOADING_CYCLES = "0842-OzB-P"  # exact LC or prefix; e.g. 0842-OzB-P finds 0842-OzB-P-01
OUTPUT_ROOT = Path(r"D:\Projects\0842_OzB_P\0842_batch")
SOURCE_DRIVE_OVERRIDE = None  # e.g. "X:" when database paths use another drive
SKIP_COMPLETED = True
OVERWRITE = False  # True: reprocess even when results exist
DRY_RUN = False  # True: list jobs only; no image analysis
MAX_SAMPLES = None  # e.g. 3 for a pilot, or None for all
# ===========================================================

# Unlike the older batch script, allow letter-containing barcodes and varied LC names.
REK_PATTERN = re.compile(r"^XRAYCT_V_(?P<barcode>.+)-(?P<cell>\d+)_.*\.rek$", re.I)
LC_PATTERN = re.compile(r"^\d{4}-.+-\d+(?:-\d+)*$", re.I)
SH_PATTERN = re.compile(r"^SH\d+$", re.I)


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def locate_holder(row):
    postpath = PureWindowsPath(str(row['PostPath']))
    postname = str(row['PostName']).replace('XRAYCT_P', 'XRAYCT_V')
    if SOURCE_DRIVE_OVERRIDE:
        postpath = PureWindowsPath(SOURCE_DRIVE_OVERRIDE, *postpath.parts[1:])
    folder = Path(str(postpath)) / postname
    # Match only standalone path components, never fragments inside filenames.
    components = postpath.parts + (postname,)
    sh = next((part for part in reversed(components) if SH_PATTERN.fullmatch(part)), None)
    lc = next((part for part in reversed(components) if LC_PATTERN.fullmatch(part)), None)
    if not sh or not lc:
        raise ValueError(f'Cannot identify loading cycle/sample holder from {folder}')
    return folder, lc, sh


def cycle_matches(lc):
    query = LOADING_CYCLES.strip().casefold()
    value = lc.casefold()
    return value == query or value.startswith(query + '-')


def expected_outputs(folder, prefix):
    required = [folder / f'{prefix}_summary.csv',
                folder / f'{prefix}_per_slice.csv',
                folder / f'{prefix}_cut_roi_diagnostic.png']
    if analysis.SAVE_TIFF:
        required.append(folder / f'{prefix}_peduncle.tif')
    return required


def completed(folder, prefix):
    paths = expected_outputs(folder, prefix)
    return all(p.is_file() and p.stat().st_size > 0 for p in paths)


def read_summary(folder, prefix):
    path = folder / f'{prefix}_summary.csv'
    with path.open(newline='', encoding='utf-8-sig') as f:
        return next(csv.DictReader(f))


def discover_jobs():
    engine = create_engine(DATABASE_CONNECTION_STRING)
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text('SELECT PostPath, PostName, MeasurementName FROM Measurements '
                                  'WHERE MeasurementName LIKE :pattern'), conn,
                             params={'pattern': f'%{LOADING_CYCLES}%'})
    finally:
        engine.dispose()
    print(f'Database rows found: {len(df)}', flush=True)
    holders, failures = {}, []
    for _, row in df.iterrows():
        try:
            folder, lc, sh = locate_holder(row)
            if cycle_matches(lc):
                holders[str(folder).casefold()] = (folder, lc, sh)
        except Exception as exc:
            failures.append({'source_rek': str(row.get('PostPath', '')), 'error': str(exc)})

    jobs = {}
    for holder, lc, sh in sorted(holders.values(), key=lambda v: str(v[0])):
        if not holder.is_dir():
            failures.append({'source_rek': str(holder), 'error': 'Sample holder folder not accessible'})
            print(f'MISSING holder: {holder}', flush=True)
            continue
        for rek in sorted(holder.iterdir()):
            match = REK_PATTERN.fullmatch(rek.name)
            if not rek.is_file() or not match:
                continue
            barcode, cell = match.group('barcode'), match.group('cell')
            # REK stem is used by the validated script for output filenames.
            target = OUTPUT_ROOT / lc / sh / f'{barcode}-{cell}'
            jobs[str(rek).casefold()] = (rek, target, lc, sh, barcode, cell)
    return list(jobs.values()), failures, len(holders)


def process_one(rek, target):
    prefix = rek.stem
    if SKIP_COMPLETED and not OVERWRITE and completed(target, prefix):
        print(f'SKIP existing: {rek.name}', flush=True)
        return read_summary(target, prefix), 'skipped'
    if DRY_RUN:
        return {}, 'dry_run'
    target.mkdir(parents=True, exist_ok=True)
    # Set only the single-REK entry point's configuration; all its calculations
    # and other settings remain unchanged. No viewer in unattended batches.
    analysis.REK_FILE = rek
    analysis.OUTPUT_DIR = target
    analysis.SHOW_NAPARI = False
    analysis.RUN_METRICS = True
    analysis.main()
    # The single script normally organises files after main() in __main__;
    # when imported, we must explicitly do this ourselves.
    analysis.organise_output_files(target)
    if not completed(target, prefix):
        raise RuntimeError('Analysis returned without all required nonempty outputs')
    return read_summary(target, prefix), 'processed'


def main():
    if not LOADING_CYCLES.strip():
        raise ValueError('LOADING_CYCLES cannot be empty')
    if not DRY_RUN:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs, failures, holder_count = discover_jobs()
    if MAX_SAMPLES is not None:
        jobs = jobs[:MAX_SAMPLES]
    print(f'Unique holders: {holder_count}; REK jobs: {len(jobs)}', flush=True)
    summaries = []
    for index, (rek, target, lc, sh, barcode, cell) in enumerate(jobs, 1):
        print(f'\n[{index}/{len(jobs)}] {lc} / {sh} / {barcode}-{cell}\n  {rek}', flush=True)
        try:
            summary, status = process_one(rek, target)
            if status != 'dry_run':
                summaries.append({'loading_cycle': lc, 'sample_holder': sh,
                                  'barcode': barcode, 'cell_id': cell,
                                  'status': status, **summary})
            print(f'  {status.upper()}: {target}', flush=True)
        except Exception as exc:
            print(f'  ERROR: {exc}', flush=True)
            traceback.print_exc()
            failures.append({'source_rek': str(rek), 'error': repr(exc),
                             'time': datetime.now().isoformat()})
        if not DRY_RUN:
            # Incremental logs make interrupted runs recoverable.
            if summaries:
                write_csv(OUTPUT_ROOT / 'batch_all_summary.csv', summaries)
            if failures:
                write_csv(OUTPUT_ROOT / 'batch_errors.csv', failures)
    print(f'\nFinished: {len(summaries)} processed/skipped; {len(failures)} errors', flush=True)
    print(f'Output root: {OUTPUT_ROOT}', flush=True)


if __name__ == '__main__':
    main()
