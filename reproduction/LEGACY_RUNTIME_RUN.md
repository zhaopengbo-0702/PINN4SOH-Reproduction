# 原 README 软件版本的隔离验证

当前平台单卡部署及 14 ZIP 分片恢复请优先使用 `LEGACY_SINGLE_GPU_RUN.md`。以下仍保留原目录层级的科学范围与安装说明；资源统一为一个任务、GPU 0、一个 worker。

本工具只在新的 Linux x86_64 前缀中组装已校验官方包，不调用 Conda 管理器、不修改现有环境。Python 3.7.10、torch 1.7.1、sklearn 0.24.2、NumPy 1.20.3、pandas 1.3.5、Matplotlib 3.3.4 对齐 README；torch 使用官方 `1.7.1+cu110` wheel。SciPy 1.7.3 是明确选择的兼容依赖，论文 README 未指定其版本。操作系统、BLAS、编译器、CUDA 驱动和 GPU 仍须如实记录，不声称论文原机器完全一致。

## 已完成的本地检查

- 31 个官方资产（13 Conda tar、18 wheel）逐个 SHA256、大小、包名、版本、Python 要求及归档路径审计通过。
- wheel 在 Linux/CPython 3.7.10 下必需的 `Requires-Dist` 闭包满足。`dataclasses` 在 Python 3.7 不需要。
- Conda 有 206 条 `info/has_prefix`。使用固定官方 SHA256 的 conda-pack 0.8.1 原始 `prefixes.py` 处理文本及 POSIX 二进制替换，记录每文件前后哈希，二进制长度保持。前缀最长 80 字节，不含空白；所有记录中的二进制占位符均至少 255 字节。
- `_libgcc_mutex` 和 `_openmp_mutex` 是求解器元数据，实际 GCC/OpenMP 库已提供；这不是完整 Conda-manager 环境。Host glibc 必须 >=2.17，由云端探针检查。`ca-certificates` 不在隔离包中，离线训练不依赖 TLS；不在此环境下载数据。
- SciencePlots 未定版本且未提供；实际作者训练入口不导入它，独立重绘不需要它。不能声称原 plotter 的所有风格脚本都已验证。
- 本机为 Windows，未安装可执行 Linux 的 WSL 发行版；本地检查没有运行原 Python 二进制或 GPU smoke。最终以云端实际证据为准。

## 最小上传内容

保持以下相对层级，工具可以整体置于任一外部工具目录；运行器和计划不能预放进 suite 输出目录。

```text
reproduction/install_legacy_runtime.py
reproduction/probe_legacy_runtime.py
reproduction/launch_legacy_smoke.py
reproduction/legacy_vendor/conda_pack/prefixes.py
reproduction/legacy_vendor/conda_pack-0.8.1.dist-info/LICENSE.txt
outputs/full_reproduction/legacy_runtime/*  # 31资产及两个manifest
outputs/full_reproduction/legacy_smoke_support/legacy_compat/*.py  # 4文件
outputs/full_reproduction/legacy_smoke_support/legacy_compat_manifest.json
outputs/full_reproduction/legacy_smoke_support/legacy_smoke_plan.json
outputs/full_reproduction/legacy_smoke_support/compatibility.patch
```

可另带 `audit_legacy_runtime.py`、`legacy_runtime_static_audit.json`、`test_legacy_runtime_tools.py` 与本文档供审阅。静态审计脚本依赖现代宿主环境的 packaging；安装器及云端探针本身仅用标准库与目标包。

## 安装与 GPU 探针

以下把上传根目录写为 `/workspace/legacy_tools_v1`，实际部署替换此根路径即可。仅在独立空目录 `/workspace/legacy37` 安装，不以已存在非空环境作 prefix。

```bash
python3 /workspace/legacy_tools_v1/reproduction/install_legacy_runtime.py \
  --assets /workspace/legacy_tools_v1/outputs/full_reproduction/legacy_runtime \
  --prefix /workspace/legacy37 --require-cuda
```

安装器清除继承的 PYTHONPATH、PYTHONHOME、Conda 和 virtualenv 变量，限制 PATH 与原生库路径，用 `python -I` 验证所有实际导入路径属于 prefix，版本逐项完全匹配。`installation_evidence.json` 记录实际结果及前缀重定位详情。安装探针失败即停止，保留证据供检查；不会自动修改现有环境或继续训练。

## 十二项 GPU smoke

```bash
python3 /workspace/legacy_tools_v1/reproduction/launch_legacy_smoke.py \
  --prefix /workspace/legacy37 \
  --support /workspace/legacy_tools_v1/outputs/full_reproduction/legacy_smoke_support \
  --repo /workspace/PINN4SOH-main \
  --output /workspace/pinn4soh_legacy_smoke_20261110 \
  --gpus 0 --workers-per-gpu 1
```

本轮仅申请一个任务、1×3090，容器内使用 GPU 0、一个 worker、两条 CPU 线程；12 smoke 和 40 control 依次执行。更新 launcher 会拒绝其他 GPU 列表或 worker 数。launcher 用固定 prefix 解释器；卡 0 先检查精确版本、导入位置、CUDA 11.0 wheel、GPU 属性、MAPE、包含一阶导数损失的参数反传、Conv1d/Dropout 反传，以及自行生成的检查点保存/重载。任何失败则不启动 smoke。

12 项来自已冻结 stage1 同 repeat1：四数据集（XJTU1/TJU3/MIT/HUST）×两种 PINN loss 共 8；XJTU1 MLP/CNN 共 2；XJTU1→HUST 两 loss 迁移共 2。保持种子、电池顺序与超参数，明确只运行两轮。使用原 old110 reference/splits 和作者供应检查点验证加载路径。输出全新任务 ID 前缀 `legacy_smoke/`。

`legacy_compat` 只作两种 API 调整：transfer 的两处 `torch.load` 删除 1.7.1 不存在的 `weights_only=True`；coordinator 的 `Path.unlink(missing_ok=True)` 改为捕获 FileNotFoundError。原四个冻结文件未修改。检查点仍受源文件与运行产物 SHA256 约束；旧 torch 使用 pickle，只加载已验证作者/本任务自行生成的检查点。

`legacy_smoke_summary.json` 只有实际 12 个任务全部 complete、coordinator 返回 0 才置 smoke_passed=true。两轮运行只说明训练/测试/载入路径可执行，不说明收敛、论文精度或旧新版稳定性一致。

## 独立完整训练对照

`prepare_legacy_control.py` 已另行冻结 40 项计划，不自动从 smoke 续跑，也不启动全部 2120 项旧栈重跑：四代表组×两 PINN+MLP+CNN=16 regular；12 迁移方向×两 loss=24 fine-tune，每项另有配对 source-only 评估。均 repeat1，同原种子、split、有效参数、200 轮上限与早停。迁移必须依赖本旧栈同 repeat1 的源域训练，不混用现代栈源权重；来源事先固定，不按目标测试误差选源。

额外上传 `launch_legacy_control.py` 与 `outputs/full_reproduction/legacy_control_support/{legacy_control_plan.json,legacy_control_design.json}` 后，只有 12 smoke 实证通过才可运行：

```bash
python3 /workspace/legacy_tools_v1/reproduction/launch_legacy_control.py \
  --prefix /workspace/legacy37 \
  --support /workspace/legacy_tools_v1/outputs/full_reproduction/legacy_smoke_support \
  --control /workspace/legacy_tools_v1/outputs/full_reproduction/legacy_control_support \
  --smoke-evidence /workspace/pinn4soh_legacy_smoke_20261110/legacy_smoke_summary.json \
  --repo /workspace/PINN4SOH-main --gpus 0 --workers-per-gpu 1
```

固定输出为 `/workspace/pinn4soh_legacy_formal_control_20261110`，不要改输出而不重新冻结计划，因为源依赖路径指向该目录。两套软件栈的报告分开生成。逐项保留标准 MAPE/MAE/RMSE/MSE 差值、最佳验证轮数、预测漂移、PINN 相对两基线的排序反转、微调相对 source-only 的符号变化及 G 冻结/F 更新证据。

这一单种子集合用于版本敏感性筛查，不能估计稳定性或作显著性结论。若出现结论反转、明显优化/预测变化，应在受影响配置冻结同种子十次扩展；若问题广泛，再独立审阅完整重跑计划。不会发明论文未给的等效误差阈值，也不会因这一子集无反转而声称全部 2120 项不依赖环境。

## 配对结果分析

`compare_legacy_control.py` 在现代审计环境运行，导入已有冻结 `build_suite_report.py` 独立从保存数组重算每项指标并验证已存值。它不加载 pickle 模型，只校验模型哈希与训练记录。

```bash
python3 /workspace/legacy_control_tools/reproduction/compare_legacy_control.py \
  --plan /workspace/legacy_control_tools/outputs/full_reproduction/legacy_control_support/legacy_control_plan.json \
  --legacy-suite /workspace/pinn4soh_legacy_formal_control_20261110 \
  --current-suite /workspace/pinn4soh_full_paper_stage1 \
  --current-suite /workspace/pinn4soh_full_paper_stage2 \
  --output /workspace/pinn4soh_legacy_control_comparison_20261110
```

若现代全套已结束，可直接读取其两个 suite；无需为了这 40 项重新训练现代栈。已有现代指标汇总不能代替数组/划分/配置核验，因此本工具优先在云端读取原产物。未完成保留 pending；参数、source/data 指纹、种子、划分、测试标签等不一致保留 unmatched 原值和原因，不给可比差值。输出所有模型/指标/迁移方向的正负差值、逐电池差值、优化轮数及预测漂移，不以方向是否有利筛选结果。迁移另核验同 repeat 复训来源、G 不变与 F 更新，源权重本身因运行栈改变而产生的漂移属于整条迁移流程的版本敏感性。

`method_rank_changes.csv` 与 `transfer_effect_changes.csv` 的格数按四指标分别列出，不能当成独立样本数；单 seed 不画均值±标准差或推断显著性。原栈报告不可与主论文图表的十次重复统计混并。
