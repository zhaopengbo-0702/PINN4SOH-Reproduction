"""Stream every mapped published MIT row through the unchanged author's reader.

Separate per-CSV checkpoints preserve the previous 381-cycle signature sample.
Only the seven prespecified non-slope features per curve enter the summary.
Existing helper slope candidates are retained verbatim without selecting one.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd
import psutil

from extract_mit_mirror_features import HASHES, SIZES, ROOT, OUTPUT, author_adapter, curve_candidates
from audit_mit_mirror import sha

FEATURES = {'mean': '{curve} mean', 'std_ddof0': '{curve} std',
    'kurtosis_scipy_biased': '{curve} kurtosis', 'skewness_scipy_biased': '{curve} skewness',
    'entropy_minmax_probability': '{curve} entropy', 'duration_seconds': '{stage} charge time',
    'charge_capacity_difference_Ah': '{stage} Q'}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def clean(value):
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def make_stats():
    return {curve: {key: dict(expected=0, finite=0, sum_abs=0., sum_squared=0., max_abs=0.,
        at_1e_8=0, at_1e_6=0, at_1e_4=0) for key in FEATURES} for curve in ('voltage', 'current')}


def add_stats(stats, record):
    for key, value in stats[record['curve']].items():
        value['expected'] += 1
        delta = record.get('delta_' + key)
        if delta is None or not np.isfinite(delta):
            continue
        absolute = abs(delta)
        value['finite'] += 1
        value['sum_abs'] += absolute
        value['sum_squared'] += delta * delta
        value['max_abs'] = max(value['max_abs'], absolute)
        for threshold, field in [(1e-8, 'at_1e_8'), (1e-6, 'at_1e_6'), (1e-4, 'at_1e_4')]:
            value[field] += absolute <= threshold


def merge_stats(target, source):
    for curve in target:
        for key, data in target[curve].items():
            for field in data:
                if field == 'max_abs':
                    data[field] = max(data[field], source[curve][key][field])
                else:
                    data[field] += source[curve][key][field]


def run(options):
    started = time.monotonic()
    root, output = options.mirror.resolve(), options.output.resolve()
    if output in {root / 'signatures', root / 'signatures_all125'}:
        raise ValueError('Choose a new supplement output; original samples are protected')
    output.mkdir(parents=True, exist_ok=True)
    partitions = output / 'per_csv'
    partitions.mkdir(exist_ok=True)
    mapping_path = root / 'PINN_CSV_to_mirror_cycles.csv'
    with mapping_path.open(encoding='utf-8-sig', newline='') as reader:
        grouped = defaultdict(list)
        for row in csv.DictReader(reader):
            grouped[row['csv']].append(row)
    selected = [(name, sorted(rows, key=lambda row: int(row['csv_row_zero_based'])))
        for name, rows in sorted(grouped.items())
        if not options.files or Path(name).name in options.files]
    if not selected:
        raise ValueError('No CSV files selected')
    adapter, reader_path = author_adapter(ROOT / 'outputs/full_reproduction/feature_audit/preprocessing_source')
    fingerprint = dict(mapping_sha256=sha(mapping_path), reader_sha256=sha(reader_path),
        candidate_helper_sha256=sha(Path(__file__).with_name('audit_xjtu_feature_sample.py')),
        unchanged_adapter_sha256=sha(Path(__file__).with_name('extract_mit_mirror_features.py')),
        streaming_extractor_sha256=sha(Path(__file__)))
    full_source_checks = {}
    stores = {}
    try:
        for date in sorted({row['source_batch_date'] for name, rows in selected for row in rows}):
            path = root / ('within_cycle_' + date + '.h5')
            if path.stat().st_size != SIZES[date] or sha(path) != HASHES[date]:
                raise ValueError('Mirror HDF5 differs from frozen published SHA: ' + date)
            full_source_checks[date] = dict(bytes=SIZES[date], sha256=HASHES[date])
            stores[date] = h5py.File(path, 'r')
        battery = adapter(stores)
        completed = []
        for number, (name, rows) in enumerate(selected, 1):
            part_started = time.monotonic()
            token = Path(name).stem + '_' + hashlib.sha256(name.encode()).hexdigest()[:8]
            directory = partitions / token
            directory.mkdir(exist_ok=True)
            evidence_path = directory / 'PARTITION_VERIFIED.json'
            published_path = ROOT / name
            published_sha = sha(published_path)
            if evidence_path.is_file():
                cached = json.loads(evidence_path.read_text(encoding='utf-8'))
                if (cached.get('fingerprint') == fingerprint and cached.get('published_csv_sha256') == published_sha
                        and cached.get('rows') == len(rows) and all(
                            (directory / path).is_file() and sha(directory / path) == digest
                            for path, digest in cached['artifact_sha256'].items())):
                    completed.append(cached)
                    print('REUSE ' + Path(name).name + ' rows=' + str(len(rows)), flush=True)
                    continue
                raise ValueError('Existing partition differs; retain it and use a new output directory: ' + name)
            published = pd.read_csv(published_path)
            if len(published) != len(rows) or [int(row['csv_row_zero_based']) for row in rows] != list(range(len(rows))):
                raise ValueError('Mapping does not cover each published CSV row exactly once: ' + name)
            stats, statuses = make_stats(), Counter()
            jsonl = directory / 'curve_records.jsonl'
            csv_path = directory / 'fixed14_residuals.csv'
            json_tmp, csv_tmp = jsonl.with_suffix('.jsonl.tmp'), csv_path.with_suffix('.csv.tmp')
            rss_peak = psutil.Process().memory_info().rss
            with json_tmp.open('w', encoding='utf-8', newline='\n') as records, csv_tmp.open('w', encoding='utf-8-sig', newline='') as table:
                writer = None
                for row in rows:
                    truth = published.iloc[int(row['csv_row_zero_based'])]
                    for curve, stage in [('voltage', 'CC'), ('current', 'CV')]:
                        record = {**row, 'curve': curve}
                        try:
                            frame = (battery.get_one_battery_one_cycle_CC_stage(None, row) if curve == 'voltage'
                                else battery.get_one_battery_one_cycle_CV_stage(None, row))
                            values = curve_candidates(frame[curve + (' (V)' if curve == 'voltage' else ' (A)')].to_numpy(),
                                frame['time (min)'].to_numpy(), frame['charge Q (Ah)'].to_numpy())
                            if values is None:
                                record['status'] = 'window_has_fewer_than_four_points'
                            else:
                                record.update(values, status='computed')
                                for feature, column in FEATURES.items():
                                    released = float(truth[column.format(curve=curve, stage=stage)])
                                    record['released_' + feature] = released
                                    record['delta_' + feature] = values[feature] - released
                        except Exception as error:
                            record.update(status='extraction_error', error=type(error).__name__ + ': ' + str(error))
                        record = {key: clean(value) for key, value in record.items()}
                        statuses[record['status']] += 1
                        add_stats(stats, record)
                        records.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
                        flat = {key: record[key] for key in ('csv', 'csv_row_zero_based', 'cell_uid', 'cycle', 'h5_row', 'curve', 'status')}
                        for feature in FEATURES:
                            flat[feature] = record.get(feature)
                            flat['released_' + feature] = record.get('released_' + feature)
                            flat['delta_' + feature] = record.get('delta_' + feature)
                        if writer is None:
                            writer = csv.DictWriter(table, fieldnames=list(flat))
                            writer.writeheader()
                        writer.writerow(flat)
                    if int(row['csv_row_zero_based']) % 128 == 0:
                        rss_peak = max(rss_peak, psutil.Process().memory_info().rss)
            json_tmp.replace(jsonl)
            csv_tmp.replace(csv_path)
            evidence = dict(csv=name, rows=len(rows), curve_records=2 * len(rows), statuses=dict(statuses),
                elapsed_seconds=time.monotonic() - part_started, peak_rss_bytes=rss_peak,
                fingerprint=fingerprint, published_csv_sha256=published_sha, source_checksum=full_source_checks,
                feature_accumulators=stats, artifact_sha256={path.name: sha(path) for path in (jsonl, csv_path)})
            atomic_json(evidence_path, evidence)
            completed.append(evidence)
            print('COMPLETE ' + Path(name).name + ' rows=' + str(len(rows)) + ' seconds=' + str(round(evidence['elapsed_seconds'], 2))
                + ' rss_mib=' + str(round(rss_peak / 1024**2, 1)), flush=True)
            atomic_json(output / 'PROGRESS.json', dict(completed_csvs=len(completed), expected_csvs=len(selected),
                completed_rows=sum(part['rows'] for part in completed), expected_rows=sum(len(rows) for name, rows in selected),
                updated_utc=datetime.now(timezone.utc).isoformat()))
        total_stats, total_statuses = make_stats(), Counter()
        for part in completed:
            merge_stats(total_stats, part['feature_accumulators'])
            total_statuses.update(part['statuses'])
        summary_stats = {}
        for curve in total_stats:
            for feature, data in total_stats[curve].items():
                n = data['finite']
                summary_stats[curve + '/' + feature] = dict(data,
                    mean_absolute_residual=data['sum_abs'] / n if n else None,
                    root_mean_squared_residual=(data['sum_squared'] / n)**.5 if n else None,
                    nonfinite_or_missing=data['expected'] - n)
        summary = dict(format='pinn4soh-mit-all-published-rows-v1',
            scope='All mapped published CSV rows, not every 114738 cycle in the derived mirror',
            selected_csvs=len(selected), expected_published_rows=sum(len(rows) for name, rows in selected),
            completed_csvs=len(completed), completed_published_rows=sum(part['rows'] for part in completed),
            curve_records=sum(part['curve_records'] for part in completed), statuses=dict(total_statuses),
            all_published_rows_covered=len(selected) == 125 and sum(part['rows'] for part in completed) == 85577,
            fixed_non_slope_features=14, residual_summary=summary_stats,
            reader='Unchanged author MITBatteryClass stage selection through unchanged mirror adapter',
            source_checksum=full_source_checks, fingerprint=fingerprint,
            slope_status='Existing six helper candidates retained verbatim; no new slopes or residual-based selection',
            mirror_precision='I/V/Q float32, time float64; source numeric I unchanged; C-rate versus A unit interpretation unresolved',
            selection_uses_feature_error=False, elapsed_seconds=time.monotonic() - started,
            peak_partition_rss_bytes=max(part['peak_rss_bytes'] for part in completed),
            partitions=[dict(csv=part['csv'], rows=part['rows'], artifact_sha256=part['artifact_sha256']) for part in completed])
        atomic_json(output / 'MIT_ALL_ROWS_SUMMARY.json', summary)
        print(json.dumps({key: summary[key] for key in ('selected_csvs', 'completed_published_rows', 'curve_records',
            'statuses', 'all_published_rows_covered', 'elapsed_seconds', 'peak_partition_rss_bytes')}), flush=True)
        return summary
    finally:
        for store in stores.values():
            store.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mirror', type=Path, default=OUTPUT)
    parser.add_argument('--output', type=Path, default=OUTPUT / 'all_rows_supplement_20261110')
    parser.add_argument('--files', nargs='*')
    run(parser.parse_args())
