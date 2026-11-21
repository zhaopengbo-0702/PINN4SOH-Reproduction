# Torch 1.7.1 + CUDA 11.0 在 RTX 3090 的部署诊断

结论：实际固定 wheel 包含 `sm_80`，不含 `sm_86` 的显式架构字符串；这不等于 3090 无法运行。NVIDIA 的 cubin 兼容规则允许同一 major、相同或更高 minor 的 GPU 使用它，因此 sm80→sm86 在规则内。实际 GPU 的库、算子和反传仍以已有 probe 与 12 smoke 为准；`torch.cuda.is_available()` 不能独立证明训练可行。

## 已核对证据

- 固定 wheel `torch-1.7.1+cu110-cp37-cp37m-linux_x86_64.whl` 的 `torch/lib/libtorch_python.so` 包含 `sm_37 sm_50 sm_60 sm_70 sm_75 sm_80`。这个成员 SHA256 为 `2a6c74d1b24289e04f056b9e0a18996c10a03b8cda26f28791bfcbd1816f586c`。
- wheel 的 `torch/cuda/__init__.py::_check_cubins` 以 `sm // 10 == cap_major` 检查兼容，而不是强制 minor 完全相等。`get_arch_list()` 在 1.7.1 已存在，云端可以记录实际返回值。
- [NVIDIA Ampere Compatibility Guide](https://docs.nvidia.com/cuda/ampere-compatibility-guide/index.html) 明确说明 cubin 同 major 向较高 minor 兼容，PTX 向更高 capability 兼容；CUDA 11.0 构建若含 sm80 cubin 或 PTX 可支持 Ampere。
- [CUDA 11.0 Release Notes](https://docs.nvidia.com/cuda/archive/11.0/cuda-toolkit-release-notes/index.html) 说明 CUDA 11.0 加入 compute80/sm80，并列出 Linux 11.0 GA 驱动至少 450.51.05（Update1 为 450.51.06）。3090 本身也需要支持其硬件的驱动；现有现代 CUDA11.6 曾运行只构成间接线索，仍须记录本次实际驱动/分配。
- 原作者供应 XJTU0 checkpoint 的 ZIP pickle 静态检查为 protocol2，仅 OrderedDict、`torch._utils._rebuild_tensor_v2` 和 FloatStorage 这些旧API，降低新序列化格式不兼容风险，但仍由真实加载 smoke 验证。

CUDA11.0 的 sm80 通用运行路径与“可编译 sm86 专用代码”是不同问题；本任务不编译自定义 CUDA 扩展，所以不需要为了 `nvcc -arch=sm_86` 而更换 toolkit。不要将已分配平台驱动替换为离线旧驱动。

## 隔离路径与数据依赖

`launch_legacy_smoke.py`、`launch_legacy_control.py` 由文档中的系统 `python3` 正常启动，顶层从相邻文件导入 `install_legacy_runtime`。若手工改为 `python -I launcher.py`，隔离模式会移除脚本目录，实际本地验证会报 `ModuleNotFoundError: install_legacy_runtime`；最小修复是恢复文档命令，保留脚本不变。

launcher 子进程以 `/workspace/legacy37/bin/python -I` 启动 probe 和 coordinator。两者本身不依赖相邻模块导入，所以有效。coordinator 再以同一个 `sys.executable -u runner.py` 启动 workers，没有传 `-I`；脚本目录因而进入 sys.path，transfer 能导入相邻 run_reproduction，extended 的两种 loss 能导入 run_transfer。worker 随后明确把固定 repo 加到 sys.path 并 chdir(repo)，作者的 Model/utils/dataloader 与相对 data 路径均有定义。不要自行把 `-I` 加到 worker 命令，也不要靠 PYTHONPATH 绕过缺文件。

`isolated_environment` 清除 PYTHONPATH/PYTHONHOME/Conda/virtualenv，限制 PATH 和 LD_LIBRARY_PATH；旧环境不需要当前系统 CUDA toolkit 的 nvcc，官方 wheel 自带运行库，GPU 驱动由容器挂载。安装器的导入检查和实际 probe 会发现动态库混入/缺失。worker 虽无 -I，仍继承已清理的环境与 PYTHONNOUSERSITE=1；记录其完整命令和配置可以复查。

smoke 和 control 都固定依赖 old110 的四个 repeat1 `config.json`/`splits.json`，全体原 417 文件 source/data 指纹，以及 smoke 作者权重 `pretrained model/model_XJTU_0.pth`。control 的 24 个迁移源权重由其自身前16任务产生，不能借现代权重代替。现有平台这些数据仍保留时无需再传 39MB 源包。

## 数值默认值必须额外记录

实际 wheel 的 `torch/include/ATen/Context.h` 中 `allow_tf32_cublas=true`、`allow_tf32_cudnn=true`。3090 支持 TF32，这可能成为旧/现代栈数值差异的组成部分。probe 已覆盖 Linear+sin 的导数损失反传、Conv1d/Dropout反传、检查点保存重载，但当前冻结 probe 未记录 `arch_list`、cuDNN版本、TF32、实际线程、GPU UUID。

根代理要求的双前缀对照可只上传 `compare_runtime_observations.py`，由宿主 Python 执行，每个解释器在独立 `-B -I` 子进程读取导入状态，保留原 TF32 和相关环境变量。`--modern-python` 必须是原现代训练实际解释器；若宿主就是它，可省略。legacy子进程使用与launcher相同的原生库路径，modern子进程保留其原生库环境；两者均清除继承的Python/Conda变量并配置2线程，但不修改正在运行的worker。

```bash
python3 /workspace/compare_runtime_observations.py \
  --legacy-python /workspace/legacy37/bin/python \
  --modern-python /usr/bin/python \
  --modern-site /workspace/site-packages \
  --gpu 0 --output /workspace/two_runtime_observations_20261110_attempt2.json
```

结果记录包版本/导入路径、`arch_list`、cuDNN/TF32、GPU UUID/驱动、实际线程、警告/失败完整stderr和差异。这是新进程默认值的观测，不能证明过去2120个worker运行时的TF32值；原报告没有那些历史字段。

本次 attempt1 的 legacy 通过，而 modern 因 pandas 位于历史 `PYTHONPATH=/workspace/site-packages` 下、被 `-I` 排除而失败。修正版显式提供 `--modern-site`，只给 modern 子进程验证并插入该目录；不执行 `.pth`、不安装或迁移依赖、不注入 legacy，记录目录元数据及每模块来源。原 attempt1 JSON 及新 attempt2 JSON 都保留，collector 的 `two_runtime_observations*.json` 会一起收集。一次只读路径修正不应表述为现代环境本身缺包或训练失败。

另上传独立 sidecar `diagnose_legacy_runtime_observations.py` 记录这些字段，不改冻结 probe 或任何训练默认值。用旧 prefix 的 `-I` 启动这个单文件是安全的，它不导入相邻文件；其进程线程值是旁路观测，worker由原runner设置为2，不能混为同一进程测量。不要在看到 TF32=true 后静默关闭它；如需关闭，应另冻结敏感性实验并保留原结果。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  /workspace/legacy37/bin/python -I /workspace/diagnose_legacy_runtime_observations.py \
  --prefix /workspace/legacy37 \
  --support /workspace/legacy_offline_recovered_v1/payload/support \
  --control /workspace/legacy_single_gpu_v3/control \
  --repo /workspace/PINN4SOH-main \
  --output /workspace/legacy_runtime_observations_20261110.json
```

## 失败时最小修复与完整留证

1. 先保留原状态：安装器 stdout/stderr 和 exit code、`installation_evidence.json`（若已产生）、原prefix路径；probe完整 JSON/警告；`launch.json`、smoke summary、suite manifest/plan/source_inventory、`.coordinator/*.json` 与各 attempt log、失败任务已有 config/splits/console/logging。对这些文件和四 compat runner、launcher、计划计算 SHA；记录平台分配、时间和 nvidia-smi。不要只截图最后一行，也不要删除/改写失败manifest。
2. ZIP/hash/资产失败先重传精确坏片或资产，保持其他文件与计划，使用 fresh输出重组。prefix安装中途失败时另选全新prefix，旧目录保留失败证据；不要混装、pip install -U 或覆盖旧runtime文件来“修好”。
3. import失败先按异常区分路径：launcher相邻模块问题恢复正常系统python命令；worker相邻模块缺失则核对support四文件SHA与父/子命令；作者模块或CSV缺失则核对417指纹与固定reference路径，从已校验备份恢复缺失输入，不clone最新仓库。
4. CUDA unavailable/driver不足：核对仅分配1卡、CUDA_VISIBLE_DEVICES=0、nvidia-smi和驱动容器挂载；不要CPU回退后报告GPU通过。`no kernel image` / `CUBLAS_STATUS_*` / cuDNN错误先保存首个完整trace，可另跑独立探针并加 `CUDA_LAUNCH_BLOCKING=1` 精确定位，输出另存。这是诊断设置，不能在原冻结suite中静默改变运行条件。
5. 若只能更换CUDA wheel或torch版本才能运行，原“README版本验证”保持失败；把新栈作为独立替代实验，另列版本/包hash/工具patch与新suite，不能算原目标通过。不要无证据把nvcc/arch重编译当最小修复。
6. 原输入和环境完全相同的暂态中断才使用原launcher `--resume`；coordinator会保留attempt日志。任何代码、包、计划、硬件条件改变时保存原目录并新建实验/重新冻结，不能修改hash让旧manifest继续。

本地只做静态wheel/序列化检查、官方资料核对与路径dry-run；未执行Linux Python、CUDA kernel或训练，因此以上不是云端已通过的报告。
