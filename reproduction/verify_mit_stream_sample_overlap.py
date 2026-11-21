"""Check streamed supplements against the preserved 381-cycle signature sample."""
import argparse
import json
from pathlib import Path

from audit_mit_mirror import OUTPUT, sha
from extract_mit_all_rows_stream import FEATURES, atomic_json


def verify(mirror, output, require_complete=False):
    mirror, output = Path(mirror).resolve(), Path(output).resolve()
    original_path = mirror / 'signatures_all125/MIT_mirror_feature_signatures.json'
    previous = json.loads(original_path.read_text(encoding='utf-8'))
    expected_sha = json.loads((mirror / 'signatures_all125/signature_summary.json').read_text(encoding='utf-8'))['signature_json_sha256']
    if sha(original_path) != expected_sha:
        raise RuntimeError('Preserved 381-cycle original sample SHA differs')
    needed = {(row['csv'], row['csv_row_zero_based'], row['curve']): row for row in previous['records']}
    checked, mismatches = set(), []
    for evidence in output.rglob('PARTITION_VERIFIED.json'):
        metadata = json.loads(evidence.read_text(encoding='utf-8'))
        with (evidence.parent / 'curve_records.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                identity = (row['csv'], row['csv_row_zero_based'], row['curve'])
                if identity not in needed:
                    continue
                checked.add(identity)
                for feature in FEATURES:
                    if row.get(feature) != needed[identity].get(feature):
                        mismatches.append(dict(identity=list(identity), feature=feature,
                            original=needed[identity].get(feature), streamed=row.get(feature)))
    missing = sorted(set(needed) - checked)
    result = dict(original_rows=previous['selected_rows'], original_curves=len(needed),
        preserved_original_sha256=expected_sha, original_sample_unchanged=True,
        overlap_curves_verified=len(checked), fixed_non_slope_values_checked=7 * len(checked),
        exact_numeric_agreement=not mismatches, mismatches=mismatches,
        missing_sample_curves=len(missing), full_original_sample_overlap_verified=not missing and not mismatches)
    atomic_json(output / 'PRESERVED_SAMPLE_OVERLAP_VERIFIED.json', result)
    print(json.dumps(result, ensure_ascii=False))
    if mismatches or (require_complete and missing):
        raise RuntimeError('Stream/sample mismatch or incomplete overlap')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mirror', type=Path, default=OUTPUT)
    parser.add_argument('--output', type=Path, default=OUTPUT / 'all_rows_supplement_20261110')
    parser.add_argument('--require-complete', action='store_true')
    options = parser.parse_args()
    verify(options.mirror, options.output, options.require_complete)
