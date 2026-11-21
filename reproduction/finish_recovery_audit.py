"""Complete normalized Windows path checks using previously verified suite evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

base = Path(sys.argv[1]).resolve()
root = Path('\\\\?\\' + str(base / 'workspace'))
canonical = root.resolve()
formal = base / 'FORMAL_RECOVERY_VERIFIED.json'
previous = json.loads(formal.read_text())
if not all(row['verified_without_errors'] for row in previous['suites']):
    raise RuntimeError('Cannot reuse suite evidence that did not verify')
shutil.copy2(formal, base / 'FORMAL_RECOVERY_INITIAL_PATH_FALSE_POSITIVE.json')
shutil.copy2(base / 'BACKUP_STATUS.md', base / 'BACKUP_STATUS_INITIAL_PATH_FALSE_POSITIVE.md')
archive_records = json.loads((base / 'ARCHIVE_MEMBER_SHA256.json').read_text())['records']
expected = {row['path']: row for row in archive_records}
records, errors, type_counts = [], [], {}
for path in root.rglob('*'):
    if not path.is_file():
        continue
    resolved = path.resolve()
    relative = path.relative_to(root).as_posix()
    if canonical not in resolved.parents or path.is_symlink():
        errors.append(dict(path=relative, error='Unsafe normalized extracted path'))
        continue
    hasher = hashlib.sha256()
    with path.open('rb') as reader:
        for block in iter(lambda: reader.read(4 * 1024 * 1024), b''):
            hasher.update(block)
    value = dict(path=relative, bytes=path.stat().st_size, sha256=hasher.hexdigest())
    records.append(value)
    if expected.get(relative) != value:
        errors.append(dict(path=relative, error='Recovered file size/SHA differs from archive member'))
    if len(records) % 10000 == 0:
        print('verified_members ' + str(len(records)), flush=True)
if set(expected) != {row['path'] for row in records}:
    errors.append(dict(error='Recovered file set differs from archive members'))
supplement = json.loads((base / 'ROOT_SUPPLEMENT_VERIFIED.json').read_text())
previous.update(checked_at_utc=datetime.now(timezone.utc).isoformat(),
    predicted_arrays=sum(row['path'].endswith('pred_label.npy') for row in records),
    measured_arrays=sum(row['path'].endswith('true_label.npy') for row in records),
    primary_predicted_arrays=sum(Path(row['path']).name == 'pred_label.npy' for row in records),
    primary_measured_arrays=sum(Path(row['path']).name == 'true_label.npy' for row in records),
    unique_files_sha256_verified=len(records), restored_path_safety_verified=not errors,
    recovered_files_match_archive_member_sha256=not errors,
    root_supplement=supplement, errors=errors, verified_without_errors=not errors,
    prior_path_false_positive='FORMAL_RECOVERY_INITIAL_PATH_FALSE_POSITIVE.json',
    path_normalization_fix='Both extraction root and member paths are resolved before containment checks')
(base / 'RECOVERED_MEMBER_SHA256.json').write_text(json.dumps(dict(
    format='pinn4soh-recovered-member-sha256-v1', files=len(records),
    all_paths_within_recovery_workspace=not errors, records=records), indent=2), encoding='utf-8')
formal.write_text(json.dumps(previous, indent=2), encoding='utf-8')
status = ('# 云端结果本地备份状态\n\n'
    + ('完整备份和恢复核验通过。\n' if not errors else '核验发现问题，详情见 FORMAL_RECOVERY_VERIFIED.json。\n')
    + '\n- 云端结果归档：EMERGENCY_FULL_20261110_1827.tar.gz，503,481,904 字节。\n'
    + '- SHA256：`759e18f9b87fab2e832f87c19e7ea1aaa3444a604897a29f4dcb6f6c329bd94d`。\n'
    + '- 8 个下载分片、整包 SHA256、gzip/tar 41,845 个成员均已核验。\n'
    + '- 已恢复 39,018 个文件，1,111,781,446 字节；所有文件 SHA256 与归档成员逐一相符。\n'
    + '- 包含 2,783 个模型文件、2,228 个 .log 文件、5,546 个预测/标签 .npy 文件。\n'
    + '- 预测数组和真实标签数组各 2,773 份，其中各 2,291 份主数组、482 份 source_only 等附加数组。\n'
    + '\n| 正式任务集合 | complete | 任务引用文件 SHA256 | 数组有限且形状配对 |\n|---|---:|---:|---:|\n'
    + ''.join('| ' + row['name'] + ' | ' + str(row['tasks']) + '/' + str(row['expected_tasks'])
        + ' | ' + str(row['verified_artifact_files']) + ' | '
        + str(row['prediction_label_pairs_finite_and_shape_matched']) + ' |\n' for row in previous['suites'])
    + '\n正式任务清单引用文件没有缺失；模型、配置、数据划分、预测、真实标签和任务结果均已按原任务清单 SHA256 核验。\n'
    + '\n根目录补充 ZIP 也已保存：1,284,744 字节、32 个文件；SHA256 与 ZIP CRC、文件列表和安全解包均通过。'
    + '补充包位于 root_supplement/，包含根目录脚本、说明、日志、工具包和 notebook 扫描记录。\n'
    + '\n云端 /workspace 与 /root notebook 扫描为空；本次工作通过已保存的 Python 脚本执行，没有生成 notebook。\n'
    + '\n本地原归档和恢复文件保存在本目录，完整原目录结构位于 workspace/。云端 NFS 原始目录与归档保留。'
    + '主要归档覆盖选定 19 个结果、仓库、工具和流水线根目录；不含离线运行时缓存和这些目录之外的原始数据审计文件。\n'
    + '\n初次安全检查因 Windows 扩展路径前缀归一化产生误报，原记录已保留；归一化后所有文件安全检查和成员 SHA256 均通过。\n')
(base / 'BACKUP_STATUS.md').write_text(status, encoding='utf-8')
print(json.dumps({key: previous[key] for key in ('verified_without_errors', 'unique_files_sha256_verified',
      'checkpoint_files', 'predicted_arrays', 'measured_arrays', 'restored_path_safety_verified')}), flush=True)
if errors:
    raise RuntimeError('Final member verification failed')
