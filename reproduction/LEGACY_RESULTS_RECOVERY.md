# 旧栈 12 smoke 与 40 control 的完整结果回收

在 12 smoke、40 control 和对照分析写完后，用现代宿主 Python 运行。输出前缀必须全新，名字仅用字母、数字、下划线和连字符；父目录须存在。

```bash
python3 /workspace/collect_legacy_results.py \
  --output-prefix /workspace/legacy_results_20261110
```

默认路径匹配单卡 v3 部署，可用 `--workspace`、`--runtime-prefix`、`--smoke`、`--control`、`--comparison` 覆盖。`--include /workspace/额外证据` 可重复指定，只接受工作区内文件。通常无需覆盖。

归档包含：

- 完整 smoke 目录：12 任务的模型、数组、配置、日志、manifest、launch 和 GPU preflight。
- 完整 40 control 目录，以及配对分析目录的全部 CSV、JSON、说明和分析证据。
- 精确运行时安装证据、分片重组证据、资产版本/SHA 清单；原兼容运行器、patch、smoke/control 计划和单卡工具。
- 所有根目录 `.log`（含 `legacy*_20261110.log`）、legacy 相关根脚本/JSON/说明、`restart_verification_*.json` 和未来 TF32/tf32 sidecar 文件或目录。
- 从现代两 suite 按冻结 `original_task_id` 精确选择的 40 个配对任务完整目录和原 suite manifest，方便离线复核；不会打包其余现代任务。

运行时二进制、离线包资产及 14 个上传大分片继续保存在云卷与已有本地资产中，不进入本次结果回收包。`--without-modern-pairs` 可省掉 40 个现代配对目录；此时独立对照应使用之前已完整校验的本地现代备份。

完成条件只要求 12/40 suite 全部 complete、smoke summary 实证通过、comparison 对全部 40 项有状态记录。`unmatched` 或分析错误也原样保存到证据，不会因结果不利而排除；`comparison_counts` 和 `comparison_analysis_errors` 必须如实审阅，归档成功不等于配对科学结论通过。若需在任务未完成时紧急保存，显式加 `--allow-partial`；`collection_complete` 会为 false。

每个源文件在写入 tar 时计算 SHA256，验证 suite 已存产物 SHA，检查源大小/mtime 未变化；生成完成后重新读 tar，逐成员复核 SHA256。再按每片最多 64 MiB 原始字节生成 base64 JSON，回读解码并验证。不会每处理一个文件重写大清单，也不会修改源结果。

输出位于 `/workspace/legacy_results_20261110/`：

```text
LEGACY_COLLECTION_VERIFIED.json
PINN4SOH_LEGACY_legacy_results_20261110.tar.gz
JSON_TRANSFER_MANIFEST.json
results.legacy.legacy_results_20261110.chunk001.json
results.legacy.legacy_results_20261110.chunk002.json  # 若需要
```

平台下载优先下载两个清单 JSON 和所有 `results.legacy.*.json`；直接 ZIP 下载曾卡住，上轮 base64 JSON 传输已完成整包 SHA 验证。每个 64 MiB 原始分片的 JSON 约 85.4 MiB，可用 `--chunk-mib 32` 降低单文件大小。

本地把本轮文件放在独立下载目录，避免同名历史清单干扰：

```powershell
python reproduction/restore_backup_json.py <本轮下载目录> <新的本地恢复目录> --incremental
```

所有片齐备后可去掉 `--incremental` 严格复核。`LOCAL_BACKUP_RESTORE_VERIFIED.json` 的 `all_expected_archives_verified` 必须为 true；它验证每片 SHA/大小、整包 SHA/大小，以及 gzip/tar 内容可完整读取。后续安全解包应保留目录结构，Windows 长路径需使用扩展路径；再按 `LEGACY_COLLECTION_VERIFIED.json` 的 inventory 逐个核对解包文件 SHA。原下载 JSON、原 tar 和核验清单全部保留。
