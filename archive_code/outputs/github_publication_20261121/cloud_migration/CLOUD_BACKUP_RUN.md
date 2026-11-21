# 云端只读清点与差量归档

此工具只读取原件，创建新的索引与归档目录。没有删除、移动、覆盖、停止训练或修改权限代码。

## 使用前

1. 本地完整 CAS 档案完成并独立验证后，从其 `index.json` 生成小型本地已保存哈希目录：

```powershell
python outputs/github_publication_20261121/cloud_migration/make_known_local_index.py --index outputs/github_publication_20261121/full_archive/index.json --output outputs/github_publication_20261121/cloud_migration/KNOWN_LOCAL_SHA256.json
```

2. 通过获授权的网页终端或文件上传工具，将 `KNOWN_LOCAL_SHA256.json` 放到 `/workspace/KNOWN_LOCAL_SHA256_20261121.json`。小型目录仅包含整文件 SHA 与大小，不含文件内容。云端按内容核验匹配，不依赖 Windows 和 Linux 路径相同。
3. `UPLOAD_CLOUD_BACKUP_ONE_LINE.txt` 是工具本身的 gzip+base64 单行上传命令，含完整 SHA 断言；不打印凭据。工具上传目标为 `/workspace/cloud_inventory_backup_20261121.py`，存在则拒绝覆盖。

## 云端执行

确保没有新的训练写入实验目录后运行：

```bash
python /workspace/cloud_inventory_backup_20261121.py --root /workspace --known-local /workspace/KNOWN_LOCAL_SHA256_20261121.json --output /workspace/CLOUD_MIGRATION_BACKUP_20261121 --max-mib 600 > /tmp/cloud_migration_backup_20261121.log 2>&1
```

输出目录必须不存在。对已有输出不重跑覆盖，需要先检查原状态。运行日志写到源目录外的 `/tmp`，避免清点脚本自身不断增长的日志；结束后可另行复制到结果目录保留。输出目录本身递归排除，其余全部普通文件逐项计算 SHA；符号链接只保留目标字符串，不跟随，特殊文件标为错误。每个文件标记以下一种情况：

- `matched_verified_local_sha256`：内容 SHA 与大小已在上传的本地索引中，避免重复下载。
- `cloud_delta_archived`：云端独有内容按 64 MiB 块去重，写入 ZIP 分包；每包最大 600 MiB，且逐块回读 SHA，再逐文件拼接 SHA 核验。
- `user_discarded_optional_experiments`：名称含已舍弃 Stage A、确认、迁移鲁棒性或 deadline 关键字的相关文件，仍逐文件记录 SHA，不保存到新包中。
- `python_cache`：`__pycache__`、`.pyc`、`.pyo` 仍索引，按缓存排除。
- `runtime_index_only_explicit_policy`：只有执行者明确传入 `--runtime-policy index-only` 时，才对安装目录仅索引、不归档。

**安装环境默认归档。** `legacy37`、`site-packages`、`offline` 等不会静默排除。本地已保存同字节文件时仍优先按 SHA 匹配；否则同实验文件一起打包。如执行者选择 `index-only`，必须在交付中说明这些安装文件未入包，可据源依赖重新建立，而不能声称完整文件副本已经保存。

最终文件：

- `CLOUD_INVENTORY.json`：完整源文件路径、大小、SHA、归档/复用/排除状态、顶层分类统计、分包与块索引。
- `cloud-delta-part-001.zip` 等：云独有内容块，只存不同内容一次。
- `SHA256SUMS.txt`：整包和索引 SHA。
- `PACK_PROGRESS.json`：分包完成进度，不是最终成功证明。

仅 `status=passed_read_only_inventory_delta_backup`、`errors=[]`、全部 `members_verified=true`、`delta_files_reassembled_verified=true` 表示此次索引和差量包核验成功。下载到本地后还必须再次核对整包 SHA，再按索引恢复云独有文件；本地匹配项由原本地档案恢复。输出包不是脱离本地档案即可独立恢复的全副本。

## 不作的动作

工具不删除云目录，不改原文件，不修改原训练或报告。即使归档成功，也没有授权任何后续删除；删除需执行者按照用户当前明确要求另行核对目标与产品恢复方式。
