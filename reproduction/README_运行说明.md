# PINN4SOH 云端运行器

独立运行器不编辑作者源文件。目标是运行作者当前修正版的四数据集 PINN 主实验：XJTU 六批各十次、MIT 十次、HUST 十次、TJU 三批各十次，共 110 次。CNN/MLP 对照、小样本和迁移学习不包含在这 110 次中。

作者参考提交：`cc3cc5053caf38f04e0665f7f88cb109144d035e`。下载目录是否为 Git checkout 不影响运行；每次输出记录实际源码和数据 SHA-256。

## 云端目录与环境

本次云端作业使用 `node01-3090` 的一张 RTX 3090，镜像为 `public/base:pytorch1131.jupyterlab.tensorboard`。实际环境为 Python 3.10.13、PyTorch 1.13.1+cu116。平台 Web 终端中的目录如下：

- `/workspace/PINN4SOH-main`：作者源码和预处理数据。
- `/workspace/run_reproduction.py`：最初上传的串行运行器。
- `/workspace/parallel_tools/`：后续上传的运行器和并行调度器。
- `/workspace/site-packages`：持久化安装的额外 Python 依赖。
- `/workspace/pinn4soh_main_20261109`：本次 110 次正式实验的统一输出目录。
- `/workspace/pinn4soh_smoke_20261109`：两轮连通性验证的独立输出目录。

`/workspace` 挂载账号持久卷。容器重启后文件仍在，但环境变量需要重新设置。每次新开 Web 终端，先运行：

```bash
export PYTHONPATH="/workspace/site-packages${PYTHONPATH:+:$PYTHONPATH}"
```

保留镜像已有的 CUDA 版 PyTorch，不另外安装 torch。本次云端下载遇到证书链校验问题，采用本机正常 TLS 下载、核对官方哈希后上传的离线 wheel 包。已经安装的依赖无需重复安装；需要恢复时运行：

```bash
python -m zipfile -e /workspace/offline-wheels-cp310-linux.zip /workspace/offline
python -m pip install --no-index --find-links /workspace/offline/wheels -r /workspace/offline/requirements-cloud.txt --target /workspace/site-packages
```

离线包 SHA-256 为 `da5e2e86bb16aff35db9fc20d8db415c49fdb34c7bb3825b7548480fe7a170f0`。安装目录位于持久卷，不依赖容器系统环境中的临时改动。

检查解释器、依赖位置和 GPU：

```bash
python -c "import sys,torch,numpy,pandas; print(sys.version); print('torch:',torch.__version__); print('numpy:',numpy.__version__,numpy.__file__); print('pandas:',pandas.__version__); print('CUDA:',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 连通性验证

在上述环境中，四个数据集各运行首批两轮的命令为：

```bash
python -u /workspace/run_reproduction.py --repo /workspace/PINN4SOH-main --output /workspace/pinn4soh_smoke_20261109 --dataset all --smoke --plots
```

改成 `--dataset XJTU` 可只验证 XJTU 首批。两轮结果只证明代码和数据能够训练，不代表收敛精度。已有验证结果时不要直接重复运行；需要重做可换一个新的输出目录。

## 正式实验

串行运行的命令如下，供首次建立输出目录时使用。已有同一输出目录的训练进程正在运行时，不要再次执行：

```bash
python -u /workspace/run_reproduction.py --repo /workspace/PINN4SOH-main --output /workspace/pinn4soh_main_20261109 --dataset all --repeats 10 --epochs 200 --threads 2 --resume --plots
```

默认最多 200 轮，早停沿用作者。每个训练任务的输出分别保存，作者仓库附带的历史结果不参与汇总。

## 并行续跑

同一张 GPU 上的独立实验可由并行调度器运行，三名 worker 各使用两个 CPU 线程。只由协调进程写统一 manifest 和汇总文件，各 worker 使用独立进程、模型、随机种子和任务目录。并发调度不改变单次实验的模型或训练超参数。

需要恢复工具文件时，将已上传的工具包解压：

```bash
python -m zipfile -e /workspace/parallel-tools-20261110.zip /workspace/parallel_tools
```

先检查进程和只读续跑计划：

```bash
ps -eo pid,ppid,args | grep -E '[r]un_reproduction.py|[r]un_parallel.py'
python -u /workspace/parallel_tools/run_parallel.py --repo /workspace/PINN4SOH-main --output /workspace/pinn4soh_main_20261109 --workers 3 --threads 2 --dry-run
```

只有确认旧串行进程已退出，且没有另一个调度器正在向此输出目录写入时，才启动续跑：

```bash
python -u /workspace/parallel_tools/run_parallel.py --repo /workspace/PINN4SOH-main --output /workspace/pinn4soh_main_20261109 --workers 3 --threads 2 --previous-pid 243 --plots
```

`243` 是本次旧串行进程的 PID；`--previous-pid` 只检查该进程是否退出，不会替用户停止它。重启容器后 PID 可能复用，若保护检查报错，应先核对实际进程，不能直接删除保护参数继续运行。不要让串行运行器与并行调度器同时写同一输出目录；已运行的调度器无需再次启动。需要中断时，在调度器所在终端按 `Ctrl+C`，等待其回收 worker 后再续跑。

并行调度器自动跳过一致且完整的成功任务，重跑失败或中断任务，不需要另加 `--resume`。串行运行器的 `--resume` 使用相同的完成结果检查规则。旧的失败或中断目录重命名为 `.previous.*` 后保留。环境、数据内容或目录顺序发生变化时，应使用新的输出目录。

并发训练的单次耗时包含资源竞争，不能直接与前面的串行耗时比较。实际起止时间和耗时记录在 manifest 中。

只运行某数据集或批次的串行示例，使用独立的新输出目录：

```bash
python -u /workspace/parallel_tools/run_reproduction.py --repo /workspace/PINN4SOH-main --output /workspace/pinn4soh_xjtu_20261109 --dataset XJTU --batches 2C 3C --repeats 10 --resume
```

## 查看和汇总

只读进度查询不会启动训练：

```bash
python /workspace/status_cloud.py --output /workspace/pinn4soh_main_20261109
```

调度器会更新汇总。训练进程全部退出后，也可以手动重新汇总并绘图：

```bash
python /workspace/parallel_tools/run_reproduction.py --output /workspace/pinn4soh_main_20261109 --summarize --plots
```

检查一个已完成任务的最佳权重与保存预测是否一致：

```bash
python /workspace/verify_checkpoint.py --repo /workspace/PINN4SOH-main --run /workspace/pinn4soh_main_20261109/XJTU/2C/Experiment01
```

- `manifest.json`：每次状态、种子、训练耗时、样本数和指标。
- `metrics_per_run.csv`：各次合并测试样本指标。
- `metrics_summary.csv`：每个数据集/批次独立重复的均值及样本标准差 `ddof=1`。
- `RESULTS_说明.md`：中文结果与口径说明。
- `environment.json`、`source_inventory.json`：环境、源码和数据哈希、文件枚举顺序。
- 每次目录内的 `splits.json`：实际训练/验证、测试路径、各电池相邻样本对数量。
- 每次目录内的 `console.txt`、`logging.txt`、`true_label.npy`、`pred_label.npy`、`model.pth`、`metrics_per_battery.csv`：原始日志与结果。

MAPE 用真实 SOH 作分母。作者训练日志和结果分析脚本参数顺序相反，因此日志 MAPE 可能不相同。MAE/RMSE 原始量纲是 SOH 比例，`*_pp` 为百分点；`MAPE_pct` 为相对误差百分数。全样本指标不同于按电池求均值的论文口径。

运行器仅修复保存最佳权重时的引用问题：赋值 `best_model` 时深拷贝字典，确保检查点与最佳验证轮保存的预测一致。不更改损失、优化器、早停、数据划分或训练超参数。HUST 入口的 GPU 编号恢复为单卡容器的 `0`；TJU 的文件枚举行为原样保留并记录。

这是作者 2026 年 9 月 bug 修复后代码的功能复现。`fix.md` 明确指出 alpha、beta 与论文补充材料不同，云镜像也不同于 README 的旧版环境，不能将结果描述为 2024 年论文数值的精确复现。

本文提供运行和恢复方法，不表示 110 次实验已经全部完成。实际完成数量、失败情况和实测结果以云端 manifest、最终执行报告为准。训练和验证结束后，在平台停止云作业释放计算资源；结果仍保存在 `/workspace` 持久卷。
