# MIT 全发布行补充审计与断点恢复

范围固定为 125 个发布 CSV 的 85,577 行、171,154 条 CC/CV 曲线；不是镜像全部 114,738 cycle。只汇总预先固定的 14 个非斜率特征。作者 reader、原 mirror adapter、固定容量映射和旧 381-cycle 抽样均保持不变；原 helper 的六个斜率候选原样留档，不作新猜测或按残差挑选。

当前本机运行两组互不重叠 CSV worker，每组约 4 万行，worker RSS 各约 280 MiB。精确分工已持久化在 WORKER_SPLIT.json；此前已完成六个 CSV 保存在根 per_csv/，其余在 worker1/per_csv/ 和 worker2/per_csv/。计算不依赖内存中的任务列表。

每个 CSV 完成后，先将 curve_records.jsonl 和 fixed14_residuals.csv 的临时文件原子改名，再原子发布 PARTITION_VERIFIED.json，包含源 CSV SHA、映射/reader/helper/extractor SHA、输出 SHA、行数和残差累加器。掉电或关机只会使正在计算的 CSV 留下 .tmp；重启只重算该 CSV，已完成且 SHA 匹配的分区会显示 REUSE，不覆盖。不要删除完成分区，也不要同时启动重复 worker。

Windows 关机后云端训练仍在平台继续，本地 CPU 补充会停止。开机后在 PowerShell 执行：

```powershell
Set-Location -LiteralPath 'C:\Users\Administrator\Desktop\复现PINN4SOH'
python -X utf8 reproduction/resume_mit_all_rows_workers.py
```

此命令读取固定 WORKER_SPLIT.json，拒绝检测到的重复 worker，恢复两个分区组；父进程每 15 秒更新根 PROGRESS.json。两组完成后自动执行严格汇总与旧抽样交叉验证。原来已启动的两个 worker 也按同一分组写入，正在运行时不要再执行恢复命令。

只查看已持久化进度：

```powershell
python -X utf8 reproduction/resume_mit_all_rows_workers.py --status
```

若两个 worker 已经结束，仅补最终核验：

```powershell
python -X utf8 reproduction/summarize_mit_all_rows_stream.py --require-complete
python -X utf8 reproduction/verify_mit_stream_sample_overlap.py --require-complete
```

完成标志是 MIT_ALL_ROWS_SUMMARY.json 的 all_published_rows_covered=true、completed_published_rows=85577、curve_records=171154，且 errors=[]；再检查提取失败、非有限/缺失数量和每个固定特征的实际残差。完整覆盖不自动意味着零残差、原 MAT 直接复现或所有 16 个特征严格重建。

固定 HDF5 SHA 全部重新核验后才计算。当前第三方镜像 I/V/Q 为 float32、t 为 float64，原生电流数值保留；C-rate/A 物理单位解释仍未解决。统计阈值计数只是残差描述，不能改写为论文给出的精度标准。
