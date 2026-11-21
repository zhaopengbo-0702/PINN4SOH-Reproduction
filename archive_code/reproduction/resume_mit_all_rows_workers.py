"""Resume both disjoint MIT CSV workers from their saved split and partitions."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil

from audit_mit_mirror import OUTPUT
from extract_mit_all_rows_stream import atomic_json, run


def progress(output):
    parts = [json.loads(path.read_text(encoding='utf-8')) for path in output.rglob('PARTITION_VERIFIED.json')]
    counts = Counter()
    for part in parts:
        counts.update(part['statuses'])
    state = dict(updated_utc=datetime.now(timezone.utc).isoformat(), completed_csvs=len(parts),
        expected_csvs=125, completed_rows=sum(part['rows'] for part in parts), expected_rows=85577,
        curve_records=sum(part['curve_records'] for part in parts), expected_curve_records=171154,
        statuses=dict(counts), completed_partitions_are_atomic_and_sha256_manifested=True,
        all_published_rows_covered=len(parts) == 125 and sum(part['rows'] for part in parts) == 85577)
    atomic_json(output / 'PROGRESS.json', state)
    return state


def worker(mirror, output, number):
    split = json.loads((output / 'WORKER_SPLIT.json').read_text(encoding='utf-8'))
    directory = output / ('worker' + str(number))
    directory.mkdir(exist_ok=True)
    with (directory / 'run.log').open('a', encoding='utf-8', buffering=1) as log:
        sys.stdout, sys.stderr = log, log
        print('RESUME ' + datetime.now(timezone.utc).isoformat(), flush=True)
        run(SimpleNamespace(mirror=mirror, output=directory, files=split['groups'][number - 1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mirror', type=Path, default=OUTPUT)
    parser.add_argument('--output', type=Path, default=OUTPUT / 'all_rows_supplement_20261110')
    parser.add_argument('--worker', type=int, choices=(1, 2))
    parser.add_argument('--status', action='store_true')
    options = parser.parse_args()
    mirror, output = options.mirror.resolve(), options.output.resolve()
    if options.worker:
        return worker(mirror, output, options.worker)
    if options.status:
        print(json.dumps(progress(output), ensure_ascii=False, indent=2))
        return
    active = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            arguments = process.info['cmdline'] or []
            if process.pid != os.getpid() and any('extract_mit_all_rows_stream' in arg or 'resume_mit_all_rows_workers' in arg for arg in arguments):
                active.append(process.pid)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    if active:
        raise RuntimeError('MIT worker already running; do not duplicate partitions. Active PIDs: ' + str(active))
    children = []
    for number in (1, 2):
        command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--mirror', str(mirror),
                   '--output', str(output), '--worker', str(number)]
        children.append(subprocess.Popen(command))
    print('Resumed worker PIDs ' + str([child.pid for child in children]), flush=True)
    while any(child.poll() is None for child in children):
        print(json.dumps(progress(output), ensure_ascii=False), flush=True)
        time.sleep(15)
    progress(output)
    if any(child.returncode for child in children):
        raise RuntimeError('One or more MIT workers failed; inspect worker run.log')
    for script in ('summarize_mit_all_rows_stream.py', 'verify_mit_stream_sample_overlap.py'):
        subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).with_name(script)),
            '--mirror', str(mirror), '--output', str(output), '--require-complete'], check=True)


if __name__ == '__main__':
    main()
