# 单卡正式替代搜索上传与运行

目标是XJTU/2C的显式替代网格64候选×3种子=192次训练，各最多200epoch及显式early-stop。仍不是恢复作者未知搜索史，不覆盖四个数据集全部历史调参，不并入原2120项冻结结果。示例电池级验证划分、u encoder与F的层/宽定义、预测单调loss和完整轨迹归一化限制均见`REPLACEMENT_SEARCH_RUN.md`。

包内原验收搜索入口和辅助文件逐字节保留。包装脚本先核工具包SHA、作者全部Python文件集合及SHA、8个XJTU/2C CSV的SHA；再检查modern依赖、`torch.load(weights_only=...)`支持及恰好一个可见GPU。随后执行1候选×1种子×2epoch的独立GPU烟测（含选定测试），成功才建立192正式计划并串行运行。所有候选完成后冻结validation selection，最后仅3个选定种子checkpoint评估test。单卡、单worker、DataLoader零子进程；正式输出独立目录。

最小环境：Python>=3.8（目标3.10）、PyTorch>=1.13且CUDA可用、NumPy、pandas、scikit-learn及其SciPy依赖。无需新增matplotlib/h5py/PDF或下载数据。当前目标`/usr/bin/python3`、Torch1.13.1+cu116、`PYTHONPATH=/workspace/site-packages`；不使用旧Torch1.7环境，因为原验收入口的`weights_only=True`需要modern接口。前置探针会拒绝无该接口或多卡可见环境，不静默切CPU。

本地打包（不启动云）：

```powershell
python reproduction/package_replacement_search_cloud.py --output outputs/full_reproduction/replacement-search-cloud-20261111-v1.zip
```

通过平台数据管理上传ZIP后，核ZIP SHA与本地manifest一致，再解到新的`/workspace/replacement_search_cloud_20261111_v1`。ZIP根目录即工具目录，解包后含`TOOLS_SHA256.json`。以下为**同一现有单卡平台作业的网页终端**命令；它检查原99扩展pipeline已complete，不申请新平台任务。

```bash
PYTHONPATH=/workspace/site-packages /usr/bin/python3 /workspace/replacement_search_cloud_20261111_v1/run_replacement_search_cloud.py --repo /workspace/PINN4SOH-main --output /workspace/pinn4soh_replacement_grid_20261111 --gpu 0 --prior-pipeline /workspace/legacy_selected_pipeline_20261110_status.json
```

上面只有预检，不训练。确认预检passed后，一行后台执行完整192及选定test、独立审计和归档：

```bash
nohup env PYTHONPATH=/workspace/site-packages CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /usr/bin/python3 /workspace/replacement_search_cloud_20261111_v1/run_replacement_search_cloud.py --repo /workspace/PINN4SOH-main --output /workspace/pinn4soh_replacement_grid_20261111 --gpu 0 --prior-pipeline /workspace/legacy_selected_pipeline_20261110_status.json --execute --evaluate-test > /workspace/replacement_search_pipeline_20261111.log 2>&1 < /dev/null &
```

后台流程不依赖本地浏览器输入。状态：`/workspace/pinn4soh_replacement_grid_20261111_pipeline_status.json`；正式候选状态：输出目录`status.json`；日志如上。包装任务锁和搜索锁防止同一输出重复运行；已有状态但未明确`--resume`会拒绝。失败保持原状态、checkpoint、日志；先定位失败再同一CLI显式加`--resume`，不得另起重复任务。完成worker逐SHA恢复跳过，未完成worker保留旧目录并整候选重跑，不声称中间epoch恢复。

完成后`/workspace/pinn4soh_replacement_grid_20261111_backup/`包含：`SEARCH_COLLECTION_VERIFIED.json`外inventory、`INDEPENDENT_SEARCH_AUDIT.json`、tar.gz、base64 JSON分片和`JSON_TRANSFER_MANIFEST.json`。collector独立从每候选数组重算validation MSE、历史最优epoch、种子均值选择、选定测试指标、数据tensor一致性，核全部artifactSHA，再流式归档，回读每成员SHA，生成逐分片SHA。包中包括运行代码、全部正式checkpoint/日志/数组/worker配置、smoke与收集时pipeline快照。完成时pipeline状态会在collector结束后更新，因此归档内明确只保存`PIPELINE_STATUS_AT_COLLECTION.json`观测快照；最终状态文件需额外下载，不将采集时running冒充final。

归档名`PINN4SOH_LEGACY_SEARCH_...tar.gz`的LEGACY前缀仅兼容现有严格恢复工具的文件名语法，不代表Torch1.7运行。真实runtime在每个result.environment中，collector另有此命名说明。源repo与原CSV不再次复制入包，仅保存完整SHA；它们已在原全量备份中。下载后使用已有`restore_backup_json.py`严格传输manifest恢复，再按inventory安全解压、逐SHA及数组独立复核。
