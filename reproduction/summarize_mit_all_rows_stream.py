"""Combine verified independent MIT CSV partitions without rereading raw HDF5."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np

from extract_mit_all_rows_stream import FEATURES, make_stats, merge_stats, atomic_json
from audit_mit_mirror import ROOT, OUTPUT, sha


def summarize(root, output, require_complete=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    expected, mappings = {}, Counter()
    with (root / 'PINN_CSV_to_mirror_cycles.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            mappings[row['csv']] += 1
    documents, errors = {}, []
    for path in output.rglob('PARTITION_VERIFIED.json'):
        document = json.loads(path.read_text(encoding='utf-8'))
        name = document['csv']
        if name in documents:
            errors.append('Duplicate completed partition: ' + name)
            continue
        if document['rows'] != mappings.get(name):
            errors.append('Partition row count differs from mapping: ' + name)
        for artifact, digest in document['artifact_sha256'].items():
            target = path.parent / artifact
            if not target.is_file() or sha(target) != digest:
                errors.append('Partition artifact SHA mismatch: ' + str(target))
        if sha(ROOT / name) != document['published_csv_sha256']:
            errors.append('Published CSV SHA changed: ' + name)
        document['partition_path'] = path.relative_to(output).as_posix()
        documents[name] = document
    total_stats, statuses = make_stats(), Counter()
    fingerprints = []
    failures, large_residuals = [], []
    for document in documents.values():
        merge_stats(total_stats, document['feature_accumulators'])
        statuses.update(document['statuses'])
        if document['fingerprint'] not in fingerprints:
            fingerprints.append(document['fingerprint'])
        has_large = any(data['max_abs'] > 1e-4 for values in document['feature_accumulators'].values() for data in values.values())
        if set(document['statuses']) - {'computed'} or has_large:
            with (output / Path(document['partition_path']).parent / 'curve_records.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    if row['status'] != 'computed':
                        failures.append(row)
                    elif any(abs(row.get('delta_' + key) or 0) > 1e-4 for key in FEATURES):
                        large_residuals.append({key: value for key, value in row.items()
                            if key in {'csv', 'csv_row_zero_based', 'cell_uid', 'cycle', 'h5_row', 'curve', 'status'}
                            or key in FEATURES or key.startswith('released_') or key.startswith('delta_')})
    if len(fingerprints) > 1:
        errors.append('Partitions used differing reader/helper/mapping/extractor fingerprints')
    missing = sorted(set(mappings) - set(documents))
    residuals = {}
    for curve in total_stats:
        for feature, data in total_stats[curve].items():
            n = data['finite']
            residuals[curve + '/' + feature] = dict(data,
                mean_absolute_residual=data['sum_abs'] / n if n else None,
                root_mean_squared_residual=(data['sum_squared'] / n)**.5 if n else None,
                nonfinite_or_missing=data['expected'] - n)
    result = dict(format='pinn4soh-mit-all-published-rows-combined-v1',
        scope='85577 published feature-table rows across 125 CSVs, not all 114738 mirror cycles',
        expected_csvs=len(mappings), completed_csvs=len(documents),
        expected_published_rows=sum(mappings.values()),
        completed_published_rows=sum(doc['rows'] for doc in documents.values()),
        expected_curve_records=2 * sum(mappings.values()),
        curve_records=sum(doc['curve_records'] for doc in documents.values()), statuses=dict(statuses),
        all_published_rows_covered=not missing and not errors,
        fixed_non_slope_features=14, residual_summary=residuals, fingerprints=fingerprints,
        source_checksums={date: value for doc in documents.values() for date, value in doc['source_checksum'].items()},
        partitions=[{key: doc[key] for key in ('csv', 'rows', 'curve_records', 'statuses', 'elapsed_seconds',
            'peak_rss_bytes', 'artifact_sha256', 'partition_path')} for doc in documents.values()],
        peak_partition_rss_bytes=max((doc['peak_rss_bytes'] for doc in documents.values()), default=0),
        total_partition_compute_seconds=sum(doc['elapsed_seconds'] for doc in documents.values()),
        selection_uses_feature_error=False, mapping_frozen=True, author_reader_unchanged=True,
        current_units='Mirror I native numeric values retained; C-rate/A interpretation unresolved',
        slope_status='Only existing six helper candidates retained verbatim; no new slope proposals or residual-based selection',
        missing_csvs=missing, errors=errors, verified_without_errors=not errors)
    atomic_json(output / 'FAILED_CURVES.json', dict(failed_curve_count=len(failures), records=failures))
    atomic_json(output / 'FIXED14_RESIDUAL_CASES.json', dict(
        threshold='Absolute residual above 1e-4 is a descriptive flag, not a paper-defined equivalence tolerance',
        flagged_curves=len(large_residuals), records=large_residuals))
    atomic_json(output / 'MIT_ALL_ROWS_SUMMARY.json', result)
    with (output / 'fixed14_residual_summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        rows = [dict(feature=key, **value) for key, value in residuals.items()]
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({key: result[key] for key in ('completed_csvs', 'completed_published_rows', 'curve_records',
        'statuses', 'all_published_rows_covered', 'verified_without_errors')}, ensure_ascii=False), flush=True)
    if errors or (require_complete and missing):
        raise RuntimeError('MIT supplement incomplete or invalid: errors=' + str(len(errors)) + ', missing_csvs=' + str(len(missing)))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mirror', type=Path, default=OUTPUT)
    parser.add_argument('--output', type=Path, default=OUTPUT / 'all_rows_supplement_20261110')
    parser.add_argument('--require-complete', action='store_true')
    options = parser.parse_args()
    summarize(options.mirror, options.output, options.require_complete)
