# StageA单卡云队列

包内新协议：MIT/HUST×PINN/MLP/CNN×9候选×3选择seed=162次，全部六组选择完成后，才用各组已冻结配置/每seed最优checkpoint做18次选定test。Test是历史已观察电池上的探索性比较，不能包装成从未见过的确认性试验。10个future确认seed已预注册，但当前队列不训练它们；后续60次确认训练、旧栈再60次是否兼容和迁移鲁棒性均须另立管道，不能称当前已完成。

工具目标是modern `/usr/bin/python3` Python3.10/Torch1.13.1+cu116，依赖NumPy/pandas/scikit-learn/SciPy。不安装依赖，不启动新平台任务，复用当前单张3090。前置检查原XJTU pipeline状态必须complete，包SHA、作者28Python、全部200个已用CSV SHA、物理UID与6spec/seed注册绑定。新StageA每dataset先1active候选两epochGPU smoke（仅train/valid，不评估test），成功才串行162。确定性GPU探针/实际导数在烟测中检查，失败保留日志且停止。

本地构包：`python reproduction/package_stage_a_cloud.py --output outputs/full_reproduction/stage-a-cloud-20261111-v3.zip`。通过平台上传、核ZIP SHA再解到`/workspace/stage_a_cloud_20261111_v3`，ZIP根即工具目录。v3附runner原验收与collector独立合成/真实烟测验收，不覆盖早期v1/v2。

先预检：

```bash
PYTHONPATH=/workspace/site-packages /usr/bin/python3 /workspace/stage_a_cloud_20261111_v3/run_stage_a_cloud.py --repo /workspace/PINN4SOH-main --output /workspace/pinn4soh_stage_a_20261111 --prior-pipeline /workspace/pinn4soh_replacement_grid_20261111_pipeline_status.json --gpu 0
```

一行后台执行（同一单卡作业；输入脱离终端）：

```bash
nohup env PYTHONPATH=/workspace/site-packages CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 /usr/bin/python3 /workspace/stage_a_cloud_20261111_v3/run_stage_a_cloud.py --repo /workspace/PINN4SOH-main --output /workspace/pinn4soh_stage_a_20261111 --prior-pipeline /workspace/pinn4soh_replacement_grid_20261111_pipeline_status.json --gpu 0 --execute > /workspace/stage_a_pipeline_20261111.log 2>&1 < /dev/null &
```

状态`/workspace/pinn4soh_stage_a_20261111_pipeline_status.json`；各组目录status.json。已有状态不加resume拒绝，同输出文件锁防重复；失败先诊断，必要时原CLI显式`--resume`。各组完成候选逐SHA跳过，不整组重跑。若collector已有不完整输出，不自动覆盖，保留并单独review后恢复归档。

独立collector不导入runner的select；从每个候选数组/physical边界重算macro RMSE、历史最佳epoch，独立排序核三seed均值；严格核3test唯一seed集合、同seed选定checkpoint、来源SHA、全部artifact。先全部SHA，再流式tar回读每成员SHA，48MiB JSON分片带wrapper/chunk/full archive SHA。备份目录`/workspace/pinn4soh_stage_a_20261111_backup`，下载JSON_TRANSFER_MANIFEST、分片、STAGE_A_COLLECTION_VERIFIED外inventory后，用原restore工具严格恢复、安全解包，再本地独立重建原CSV tensor和选择。LEGACY_STAGEA仅为恢复工具文件名语法前缀，实际runtime为modern，每result记录。

包中完整保留新spec、原身份登记、每个checkpoint、日志、loss/梯度/mono违例历史、验证/测试数组，另存200个已用CSV与28作者源码原SHA快照。pipeline状态位于结果根之外，不把归档中仍运行的collector时刻误称最终pipeline状态；最终pipeline JSON和日志另下载。
