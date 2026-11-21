# 迁移优化的单卡后续队列

新包装仅接续同一平台作业，不申请另一卡/任务。等待 `/workspace/pinn4soh_stage_a_confirmation_20261111_pipeline_status.json` complete且其60训练+60test+归档verified，之前不导入Torch、不probeGPU、不训练。前序failed会停止保留状态。之后modern probe、两真实GPU短烟测、87选择/控制、独立87矩阵与selection门审计、modern50确认、独立50门审计、modern70test；再legacy probe、两真实烟测、固定modern配置的legacy50确认、独立50门审计、legacy70test，最后独立配对报告与归档。

冻结 `PROTOCOL.json` 和 `run_transfer_robustness.py`不改。云历史源根并非本地extracted目录：wrapper在结果下`verified_source_archive/`建只读用途的目录alias，映射下列三目录到`/workspace/`同名目录：

- `pinn4soh_full_paper_stage1`
- `pinn4soh_legacy_formal_control_20261110`
- `pinn4soh_legacy_selected_extension_20261110`

每个源run严格按协议相对路径访问，40×8证据文件SHA逐个重核，并绑定物理目标CSV、全部历史源CSV、源seed/fingerprint/tensor跨栈相同。`VERIFIED_RELOCATION.json`记录映射；不改任何源run路径记录或协议字节。collector按协议显式枚举证据并resolve实际文件保存真实字节，portable tar不保存alias symlink，也不依赖rglob遍历symlink目录。父pipeline最终状态和日志位于结果外，归档完成后另下载。

预定包解压到 `/workspace/transfer_robustness_20261111_v1`；主结果 `/workspace/pinn4soh_transfer_robustness_20261111`，现代与旧栈分别在其`modern/legacy/`，状态在 `/workspace/pinn4soh_transfer_robustness_20261111_pipeline_status.json`，备份在`/workspace/pinn4soh_transfer_robustness_20261111_backup`。旧栈复用已安装`/workspace/legacy37`，清除继承PYTHONPATH/PYTHONHOME等，明确加载原Python3.7/Torch1.7.1+cu110；modern用现有Python3.10/Torch1.13.1。无依赖自动升级。

未执行参数默认只核只读预检，不probeGPU。真实后台队列命令（仅在包验收、独立review完成后使用）：

```bash
nohup env PYTHONPATH=/workspace/site-packages CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 /usr/bin/python3 /workspace/transfer_robustness_20261111_v1/run_transfer_robustness_cloud.py --repo /workspace/PINN4SOH-main --workspace /workspace --output /workspace/pinn4soh_transfer_robustness_20261111 --prior-pipeline /workspace/pinn4soh_stage_a_confirmation_20261111_pipeline_status.json --stage-a-confirmation-root /workspace/pinn4soh_stage_a_confirmation_20261111 --legacy-prefix /workspace/legacy37 --gpu 0 --execute --wait-prior > /workspace/transfer_robustness_20261111.log 2>&1 < /dev/null &
```

协调器stage锁是先验控制；独立collector在confirmation前重建完整87候选/seed的worker身份、逐数组目标、最优epoch和排序，不能只相信selection SHA。test前重建每lane/futureseed/source-repeat/checkpoint/split/stack的全部50记录，核全部artifact再允许首个test。source-only底层worker原无selection字段，外部`TEST_EVALUATION_GATE.json`绑定global lock、确认marker及完整50result SHA，并核source-only参数无更新；不伪造其内部字段。

最终collector独立重算macro RMSE/MAE/MAPE、十seed均值/样本SD/配对差/固定seed t95区间，保存PNG/PDF及CSV，不选最好seed。区间只表示固定电池/历史源条件下的目标训练随机性。所有87+50+50训练与140test证据、工具、43目标CSV、40源证据真实字节进入tar，逐memberSHA回读，再生成48MiB JSON wrapper，用原严格restore。额外20节有标签valid、历史test已见、全轨迹清理/缩放仍明确保留，不能宣称原论文完全恢复或保证迁移改善。

当前云package属于新增实现，正式部署前必须本地wrapper/collector fixture验收和独立review；本地CPU短烟测不能替代modern/legacy实际GPU兼容检查。
