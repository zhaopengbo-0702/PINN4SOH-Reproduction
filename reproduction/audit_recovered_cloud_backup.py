"""Audit extracted cloud suites against frozen per-task artifact SHA256 lists."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as reader:
        for block in iter(lambda: reader.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def audit(base):
    base = Path(base).resolve()
    if os.name == 'nt' and not str(base).startswith('\\\\?\\'):
        base = Path('\\\\?\\' + str(base))
    workspace = base / 'workspace'
    extraction = json.loads((base / 'LOCAL_EXTRACTION_VERIFIED.json').read_text())
    restore = json.loads((base / 'LOCAL_BACKUP_RESTORE_VERIFIED.json').read_text())
    inventory = json.loads((base / 'EMERGENCY_ARCHIVE_VERIFIED.json').read_text())
    supplement_path = base / 'ROOT_SUPPLEMENT_VERIFIED.json'
    supplement = json.loads(supplement_path.read_text()) if supplement_path.is_file() else None
    if not restore['all_expected_archives_verified'] or not extraction['all_member_sizes_verified']:
        raise RuntimeError('Archive reconstruction or extraction is not verified')
    errors, suites, file_hashes = [], [], {}
    expected_suites = [('pinn4soh_full_paper_stage1', 1880),
                       ('pinn4soh_full_paper_stage2', 240),
                       ('pinn4soh_mit_physical_purity_sensitivity_20261110', 40)]
    archive_names = {row['path'].rstrip('/') for row in inventory['inventory']}
    inventory_errors = []
    for row in inventory['inventory']:
        if not row['is_file']:
            continue
        path = workspace / Path(*Path(row['path']).parts)
        if not path.is_file() or path.stat().st_size != row['bytes']:
            inventory_errors.append(row['path'])
    if inventory_errors:
        errors.append(dict(error='Extracted archive inventory mismatch', paths=inventory_errors))
    for name, expected_count in expected_suites:
        root = workspace / name
        manifest = json.loads((root / 'manifest.json').read_text())
        tasks = manifest['tasks']
        statuses = dict(Counter(task['status'] for task in tasks.values()))
        verified_files, counts, suite_errors, arrays_verified = 0, Counter(), [], 0
        for task_name, task in tasks.items():
            directory = root / 'tasks' / Path(*task_name.split('/'))
            references = task.get('verified_file_sha256', task.get('artifact_sha256', {}))
            for relative, expected_hash in references.items():
                path = directory / Path(*relative.split('/'))
                archive_name = path.relative_to(workspace).as_posix()
                if archive_name not in archive_names:
                    suite_errors.append(dict(task=task_name, file=relative, error='Task-referenced artifact absent from tar inventory'))
                    continue
                if not path.is_file():
                    suite_errors.append(dict(task=task_name, file=relative, error='Task-referenced artifact absent after extraction'))
                    continue
                if path not in file_hashes:
                    file_hashes[path] = sha256(path)
                if file_hashes[path] != expected_hash:
                    suite_errors.append(dict(task=task_name, file=relative, error='Artifact SHA256 differs from suite manifest'))
                    continue
                verified_files += 1
                if relative in ('pred_label.npy', 'true_label.npy', 'model.pth', 'metrics.json', 'splits.json'):
                    counts[relative] += 1
            if task['status'] != 'complete' or task.get('returncode') != 0:
                suite_errors.append(dict(task=task_name, error='Task status/returncode incomplete'))
            try:
                predictions = np.load(directory / 'pred_label.npy', allow_pickle=False)
                measured = np.load(directory / 'true_label.npy', allow_pickle=False)
                if predictions.shape != measured.shape or not predictions.size:
                    raise ValueError('Prediction and label shapes differ or arrays are empty')
                if not np.isfinite(predictions).all() or not np.isfinite(measured).all():
                    raise ValueError('Prediction or label contains nonfinite values')
                arrays_verified += 1
            except Exception as error:
                suite_errors.append(dict(task=task_name, error='Array validation failed: ' + str(error)))
        if len(tasks) != expected_count or statuses != {'complete': expected_count}:
            suite_errors.append(dict(error='Suite task count/status mismatch', expected=expected_count,
                                     actual=len(tasks), statuses=statuses))
        row = dict(name=name, expected_tasks=expected_count, tasks=len(tasks), statuses=statuses,
                   verified_artifact_files=verified_files, required_artifact_counts=dict(counts),
                   prediction_label_pairs_finite_and_shape_matched=arrays_verified,
                   manifest_sha256=sha256(root / 'manifest.json'), errors=suite_errors,
                   verified_without_errors=not suite_errors)
        suites.append(row)
        errors.extend(dict(suite=name, **error) for error in suite_errors)
        print(json.dumps({key: row[key] for key in ('name', 'tasks', 'verified_artifact_files',
              'required_artifact_counts', 'verified_without_errors')}), flush=True)
    files = [path for path in workspace.rglob('*') if path.is_file()]
    counts = Counter(path.name for path in files)
    roots = sorted({path.relative_to(workspace).parts[0] for path in files})
    member_hashes = []
    unsafe_paths = []
    canonical_workspace = workspace.resolve()
    for path in files:
        resolved = path.resolve()
        if canonical_workspace not in resolved.parents or path.is_symlink():
            unsafe_paths.append(str(path))
            continue
        if path not in file_hashes:
            file_hashes[path] = sha256(path)
        member_hashes.append(dict(path=path.relative_to(workspace).as_posix(),
                                 bytes=path.stat().st_size, sha256=file_hashes[path]))
    if unsafe_paths:
        errors.append(dict(error='Unexpected extracted path or symlink', paths=unsafe_paths))
    (base / 'RECOVERED_MEMBER_SHA256.json').write_text(json.dumps(dict(
        format='pinn4soh-recovered-member-sha256-v1', files=len(member_hashes),
        all_paths_within_recovery_workspace=not unsafe_paths, records=member_hashes), indent=2), encoding='utf-8')
    result = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), archive_sha256=inventory['archive_sha256'],
        archive_bytes=inventory['archive_bytes'], archive_members=inventory['members'],
        extracted_files=len(files), extracted_file_bytes=sum(path.stat().st_size for path in files),
        all_expected_archive_members_present=not inventory_errors,
        checkpoint_files=sum(path.suffix in {'.pth', '.pt'} for path in files),
        file_type_counts=dict(notebooks=sum(path.suffix == '.ipynb' for path in files),
            logs=sum(path.suffix == '.log' or path.name in {'logging.txt', 'console.txt'} for path in files),
            log_extension=sum(path.suffix == '.log' for path in files),
            checkpoints=sum(path.suffix in {'.pth', '.pt'} for path in files),
            npy=sum(path.suffix == '.npy' for path in files),
            python=sum(path.suffix == '.py' for path in files),
            json=sum(path.suffix == '.json' for path in files)),
        predicted_arrays=sum(path.name.endswith('pred_label.npy') for path in files),
        measured_arrays=sum(path.name.endswith('true_label.npy') for path in files),
        primary_predicted_arrays=counts['pred_label.npy'], primary_measured_arrays=counts['true_label.npy'],
        roots=roots, suites=suites, unique_files_sha256_verified=len(file_hashes),
        restored_path_safety_verified=not unsafe_paths,
        local_member_sha256_manifest='RECOVERED_MEMBER_SHA256.json',
        root_supplement=supplement,
        scope='All 19 roots selected for emergency cloud snapshot; offline runtimes and raw-data audit files were outside these roots',
        excluded_source_result_artifacts_referenced_by_suite_manifests=[error for error in errors
            if 'absent from tar inventory' in error.get('error', '')],
        errors=errors, verified_without_errors=not errors)
    (base / 'FORMAL_RECOVERY_VERIFIED.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    status = ('# 云端结果本地备份状态\n\n'
        + ('核验通过。' if not errors else '核验发现问题，详情见 FORMAL_RECOVERY_VERIFIED.json。')
        + '\n\n- 原始云端归档：`' + inventory['archive'].split('/')[-1] + '`\n'
        + '- 文件大小：' + format(inventory['archive_bytes'], ',') + ' 字节\n'
        + '- SHA256：`' + inventory['archive_sha256'] + '`\n'
        + '- 8 个下载分片、归档 SHA256、gzip/tar 成员读取与解包大小均已核验。\n'
        + '- 已恢复 ' + format(len(files), ',') + ' 个文件；' + str(result['checkpoint_files'])
        + ' 个模型文件，预测与真实标签数组各 ' + str(counts['pred_label.npy']) + ' 份。\n'
        + '\n| 任务集合 | 完成任务 | SHA256 核验文件 |\n|---|---:|---:|\n'
        + ''.join('| ' + row['name'] + ' | ' + str(row['statuses'].get('complete', 0)) + '/' + str(row['expected_tasks'])
                  + ' | ' + str(row['verified_artifact_files']) + ' |\n' for row in suites)
        + '\n正式任务清单引用的结果文件均已检查是否包含在归档中；缺失数量：'
        + str(len(result['excluded_source_result_artifacts_referenced_by_suite_manifests'])) + '。\n'
        + '\n原目录结构保存在本目录的 `workspace/`；归档和核验 JSON 保留，可再次恢复。云端 NFS 原始目录与归档均保留，本地恢复副本已保存。'
        + '该备份覆盖所选 19 个结果、仓库、工具和流水线根目录；离线运行时与这些目录之外的原始数据审计文件不属于此归档。\n')
    if supplement is not None:
        status += ('\n另已保存根目录补充 ZIP（' + str(supplement['bytes']) + ' 字节、'
            + str(len(supplement['files'])) + ' 个文件），SHA256 和 ZIP CRC 核验通过，'
            + '解包在 `root_supplement/`。补充包包含根目录训练脚本、运行说明、日志、工具包和 notebook 扫描记录。\n'
            + '\n云端 `/workspace` 与 `/root` 的 notebook 扫描结果为空：本次训练采用已保存的 Python 脚本，没有生成 notebook。\n')
    (base / 'BACKUP_STATUS.md').write_text(status, encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('verified_without_errors', 'extracted_files',
          'checkpoint_files', 'predicted_arrays', 'unique_files_sha256_verified')}), flush=True)
    if errors:
        raise RuntimeError('Recovery audit found ' + str(len(errors)) + ' errors')
    return result


if __name__ == '__main__':
    audit(sys.argv[1])
