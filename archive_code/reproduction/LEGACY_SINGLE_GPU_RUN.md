# 单卡旧软件栈：分片恢复、12 smoke 和 40 control

本轮资源固定为一个平台任务、1×RTX 3090；容器内 GPU 0、一个 worker、两条 CPU 线程。先完成 12 个两轮 smoke，再运行 40 个独立正式对照；不启动完整 2120 项旧栈重跑。更新 launcher 会拒绝其他 GPU 列表或 worker 数。

## 最小上传文件

已有平台仓库和 reference 数据仍在时，只需以下 16 个文件：

- `legacy_archive.part001.zip` 至 `legacy_archive.part014.zip`（14 个文件，共 1,338,321,768 字节）。
- `legacy_archive_parts.json`（4,565 字节，SHA256 `9344dbbb49686d10d9a28a625f11ec1669efbe6c8c51c55bc3c84da1f3a316eb`）。
- `legacy-single-gpu-tools-20261110-v3.zip`（小型补充工具包，含重组器、安全源/reference恢复器、单卡 launchers、40 对照计划、分析器、本文档与哈希清单）。不需要旧 `legacy-control-tools-20261110-v1.zip`。

保留 14 个分片为 ZIP 文件，不在同一上传目录自动解出 `legacy_archive.partNNN` 裸分片。所有 ZIP 分片及索引放入 `/workspace/legacy_parts`；工具包解至新的 `/workspace/legacy_single_gpu_v3`。下文路径对应工具包的 `tools/`、`control/` 和 `docs/` 层级。

若新任务没有旧工作卷，另上传 `legacy-source-reference-20261110-v1.zip`（39,235,820 字节，SHA256 `28ded058f73c3c74acda043a94d555e3c9cda34ddee9e84d22588e30caf2f504`）。使用以下恢复器校验整包、CRC、路径、417 文件原指纹、作者 checkpoint 和全部 426 个文件 SHA，再写入全新 `PINN4SOH-main/` 与 `pinn4soh_main_20261109/` 层级。工具包已含恢复器的依赖；stage 路径位于 `/tmp`，必须不存在。

```bash
python3 /workspace/legacy_single_gpu_v3/tools/restore_legacy_source_reference.py \
  --archive /workspace/legacy-source-reference-20261110-v1.zip \
  --stage /tmp/legacy_source_stage_v1 --workspace /workspace
```

这个支持不含现代训练模型与数组。不要 clone 最新主分支替代它，否则固定数据/代码指纹可能改变。已有同名路径时先核对清单，恢复器会停止，不覆盖已有文件。

## 严格重组与安全解包

系统 Python 3.7+ 运行，只用标准库。`--output` 必须不存在，父目录必须已存在，且不能位于分片目录内；原始分片保留不变。

```bash
python3 /workspace/legacy_single_gpu_v3/tools/reassemble_legacy_upload.py \
  --parts /workspace/legacy_parts \
  --output /workspace/legacy_offline_recovered_v1
```

重组器先验证固定索引 SHA、14 个顺序名称、每个外层 ZIP SHA/大小、唯一内层成员、每片 CRC/成员 SHA，再核对整包 1,338,319,780 字节和 SHA256 `e422cc82b6b95062809bf72cea8727beb4cb2a360ab09244dcd00f633948e87f`。完整 ZIP CRC、45 文件和展开总大小再核对通过后安全解至 `payload/`，阻止路径穿越、链接、重复/大小写碰撞、文件/目录冲突和既存路径覆盖；最后再次核对 31 个 runtime 资产 SHA。

成功证据为 `/workspace/legacy_offline_recovered_v1/REASSEMBLY_VERIFIED.json`，其中 status 必须为 `passed`。失败会停止并保留 `REASSEMBLY_FAILED.json` 和局部输出；排查后使用另一个全新输出路径，不把局部输出当可用软件包。原数据、脚本和 Linux 二进制在此步均不执行。

## 安装与探针

隔离前缀 `/workspace/legacy37` 必须全新或空，不能修改当前系统环境。重组得到的 `payload/tools/install_legacy_runtime.py` 与 vendored 依赖固定未改。

```bash
CUDA_VISIBLE_DEVICES=0 python3 /workspace/legacy_offline_recovered_v1/payload/tools/install_legacy_runtime.py \
  --assets /workspace/legacy_offline_recovered_v1/payload/assets \
  --prefix /workspace/legacy37 --require-cuda
```

精确版本：Python 3.7.10、torch 1.7.1+cu110、NumPy 1.20.3、pandas 1.3.5、sklearn 0.24.2、Matplotlib 3.3.4；SciPy 1.7.3 是显式兼容依赖，论文 README 未指定。安装器检查隔离导入路径、真实版本和 CUDA 可用，失败即停。云端仍需通过后续 GPU API 探针，不能以本地静态审计代替。

## 12 项两轮 smoke

以下使用补充工具包中更新的单卡 launcher，而不是 1.338GB 包内的旧副本。补充工具的 installer/probe 与原包保持相同 SHA，单卡政策仅加在 launcher。

```bash
CUDA_VISIBLE_DEVICES=0 python3 /workspace/legacy_single_gpu_v3/tools/launch_legacy_smoke.py \
  --prefix /workspace/legacy37 \
  --support /workspace/legacy_offline_recovered_v1/payload/support \
  --repo /workspace/PINN4SOH-main \
  --output /workspace/pinn4soh_legacy_smoke_20261110 \
  --gpus 0 --workers-per-gpu 1
```

执行前可加 `--dry-run` 检查具体命令；它不会运行任何探针或训练。真实运行先验证精确导入/CUDA11.0、MAPE API、一阶导数损失反传、Conv1d/Dropout、保存重载；任一失败不启动训练。12 项必须全部 complete，`legacy_smoke_summary.json` 的 `smoke_passed: true`，才能继续。

## 40 项正式对照

```bash
CUDA_VISIBLE_DEVICES=0 python3 /workspace/legacy_single_gpu_v3/tools/launch_legacy_control.py \
  --prefix /workspace/legacy37 \
  --support /workspace/legacy_offline_recovered_v1/payload/support \
  --control /workspace/legacy_single_gpu_v3/control \
  --smoke-evidence /workspace/pinn4soh_legacy_smoke_20261110/legacy_smoke_summary.json \
  --repo /workspace/PINN4SOH-main --gpus 0 --workers-per-gpu 1
```

输出固定 `/workspace/pinn4soh_legacy_formal_control_20261110`，因为 24 个迁移任务源依赖指向同目录的 16 个 regular 任务。保留原 seed、split、超参、200 轮上限和早停，只改变隔离软件栈；迁移使用同旧栈 repeat1 复训源权重。可在同输入断点继续时加 `--resume`；不改计划或输出路径后强行续跑。

## 数组独立对照

在现代审计环境运行以下命令，依赖 NumPy，不加载模型 pickle。现代两阶段完整产物必须可读；若不在新任务工作卷，先把新产生的旧栈 suite 完整备份到本地，与已恢复的现代产物在本地比较，不必为分析上传全部现代模型。

```bash
python3 /workspace/legacy_single_gpu_v3/tools/compare_legacy_control.py \
  --plan /workspace/legacy_single_gpu_v3/control/legacy_control_plan.json \
  --legacy-suite /workspace/pinn4soh_legacy_formal_control_20261110 \
  --current-suite /workspace/pinn4soh_full_paper_stage1 \
  --current-suite /workspace/pinn4soh_full_paper_stage2 \
  --output /workspace/pinn4soh_legacy_control_comparison_20261110
```

只进行单种子版本敏感性筛查，报告所有符号变化、预测漂移、优化轮数和方法排序；不发明等效阈值，不把 smoke 当收敛证据，也不把 40 项结果并入原 10 次重复论文统计。操作系统、BLAS、驱动和 GPU 尚不一定等于论文原机器。详细科学范围见 `LEGACY_RUNTIME_RUN.md` 与 control design。
