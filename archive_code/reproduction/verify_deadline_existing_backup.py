"""Read and verify an existing deadline archive after a summary-return failure.

Never changes the collector output, archive, source data, or mirror. A fresh
external receipt is written only after every member and copy has been checked.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as s:
        for block in iter(lambda: s.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def processes():
    runners = {'search_stage_a.py', 'run_stage_a_cloud.py', 'run_stage_a_confirmation.py',
               'run_transfer_robustness_cloud.py', 'run_transfer_robustness.py'}
    need(Path('/proc').is_dir(), 'Linux /proc required')
    rows = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = p.joinpath('cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            fields = p.joinpath('stat').read_text().rsplit(') ', 1)[1].split()
            if any(Path(a).name in runners for a in args):
                rows.append(dict(pid=int(p.name), ppid=int(fields[1]), state=fields[0], argv=args))
        except (OSError, ValueError, IndexError):
            pass
    need(not [r for r in rows if r['state'] != 'Z'], 'Training runner still alive')
    return rows


def verify(primary, mirror, stop_record, receipt):
    primary, mirror, stop_record, receipt = [Path(v).resolve() for v in (primary, mirror, stop_record, receipt)]
    need(primary != mirror and primary not in mirror.parents and mirror not in primary.parents, 'Copies overlap')
    need(not receipt.exists() and primary not in receipt.parents and mirror not in receipt.parents,
         'Fresh external receipt required')
    before = processes()
    audit, info, inventory, copies = [read(primary / p) for p in
                                    ('DEADLINE_PARTIAL_AUDIT.json', 'ARCHIVE_VERIFIED.json',
                                     'FILE_SHA256_INVENTORY.json', 'SECOND_COPY_VERIFIED.json')]
    need(audit['audit_status'] == 'passed_completed_runs_only', 'Original completed-run audit did not pass')
    need(audit['experiment_status'] == 'stopped_for_deadline_incomplete', 'Partial label missing')
    need(audit['stop_record_sha256'] == sha(stop_record), 'Stop record changed')
    need(info['audit_sha256'] == sha(primary / 'DEADLINE_PARTIAL_AUDIT.json'), 'Audit hash differs')
    need(info['all_member_bytes_sha256_verified'] is True, 'Original member verification missing')
    archive = primary / info['archive']
    need(archive.parent == primary and archive.stat().st_size == info['bytes'] and sha(archive) == info['sha256'],
         'Archive size/hash differs')
    seen = set()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            need(member.isfile() and member.name in inventory and member.name not in seen, 'Unexpected tar member')
            stream, h, n = tar.extractfile(member), hashlib.sha256(), 0
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                n += len(block)
                h.update(block)
            need(n == inventory[member.name]['bytes'] and h.hexdigest() == inventory[member.name]['sha256'],
                 'Tar member bytes/hash differs: ' + member.name)
            seen.add(member.name)
    need(seen == set(inventory) and len(seen) == info['members'], 'Tar member set differs')
    files = sorted(p.name for p in primary.iterdir() if p.is_file())
    need(files == sorted(p.name for p in mirror.iterdir() if p.is_file()), 'Primary/mirror filename sets differ')
    checked_files = []
    for name in files:
        a, b = primary / name, mirror / name
        need(not a.is_symlink() and not b.is_symlink() and a.stat().st_size == b.stat().st_size and sha(a) == sha(b),
             'Primary/mirror bytes/hash differs: ' + name)
        checked_files.append(dict(path=name, bytes=a.stat().st_size, sha256=sha(a)))
    need(copies['all_files_bytes_sha256_equal'] is True and copies['archive_sha256'] == info['sha256'],
         'Original second-copy receipt mismatch')
    for row in copies['files']:
        p = primary / row['path']
        need(p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'Original copy-file receipt differs')
    transfer = read(primary / 'JSON_TRANSFER_MANIFEST.json')
    assembled, n, count = hashlib.sha256(), 0, 0
    for row in sorted(transfer['records'], key=lambda r: r['chunk_index']):
        p = primary / row['wrapper_name']
        need(p.stat().st_size == row['wrapper_bytes'] and sha(p) == row['wrapper_sha256'], 'Wrapper differs')
        wrapper = read(p)
        data = base64.b64decode(wrapper['data_base64'], validate=True)
        need(wrapper['original_zip_sha256'] == info['sha256'] and wrapper['original_zip_bytes'] == info['bytes']
             and wrapper['chunk_index'] == count and wrapper['chunk_offset'] == n
             and len(data) == wrapper['chunk_bytes'] and hashlib.sha256(data).hexdigest() == wrapper['chunk_sha256'],
             'Chunk metadata/content differs')
        assembled.update(data)
        n += len(data)
        count += 1
    need(count == transfer['wrapper_files'] and count > 0 and n == info['bytes'] and assembled.hexdigest() == info['sha256'],
         'Assembled JSON chunks differ from archive')
    after = processes()
    result = dict(format='pinn4soh-existing-deadline-backup-independent-verification-v1',
                  status='passed_existing_backup_without_rearchive', verified_at_utc=datetime.now(timezone.utc).isoformat(),
                  primary=str(primary), mirror=str(mirror), archive_bytes=info['bytes'], archive_sha256=info['sha256'],
                  archive_members_verified=len(seen), wrapper_chunks_verified=count, files_equal=checked_files,
                  audited_complete_training=audit['audited_complete_training'], planned_training=audit['planned_training'],
                  audited_complete_test=audit['audited_complete_test'], planned_test=audit['planned_test'],
                  stop_record_sha256=sha(stop_record), process_snapshot_before=before, process_snapshot_after=after,
                  source_and_existing_outputs_unchanged=True,
                  note='Collector summary-return TypeError occurred after original archive and two-copy receipts were written; this independent pass validates existing bytes. Two directories can share the same NFS.')
    receipt.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return dict(status=result['status'], archive_bytes=info['bytes'], archive_sha256=info['sha256'],
                members=len(seen), training=result['audited_complete_training'], tests=result['audited_complete_test'],
                wrappers=count, equal_files=len(checked_files), receipt=str(receipt), receipt_sha256=sha(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('primary', 'mirror', 'stop-record', 'receipt'):
        parser.add_argument('--' + name, required=True)
    print(json.dumps(verify(**vars(parser.parse_args())), indent=2))
