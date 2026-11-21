# 固定配置的十个新seed复核

这是Stage A探索性改进实验的后续，不替代论文复现结果。预注册原件位于`outputs/full_reproduction/stage_a9_preregistered_20261111/REGISTRATION.json`，保持字节不变。使用其中固定的十个future seed：2026111111至2026111120；MIT/HUST×PINN/MLP/CNN共60次新训练，随后60次固定test评估。历史test已被观察，不能称全新盲测。

六组原Stage A的全部162选择、18选定test及归档pipeline必须成功。入口独立重验全部原候选数组、冻结selection、原数据/源码SHA及注册，冻结全部六配置后才开始。每组只使用原validation选中的一个配置；不扩grid、不改split、不根据新seed或test改参数，不挑“最好seed”。每个新seed仍只用原train训练、原valid选择epoch，所有60训练成功后才开始固定test。验证数据不并入训练。

训练直接调用原Stage A v3的`search_stage_a.py` worker，文件原字节不改：双端点平均MSE、PDE均值、eval且保留autograd的单调均值项、PINN/MLP相同初始化和训练RNG重置、batch512、warmup5、min30/patience30/max200、完整轨迹minmax与capacity相关3σ均保持原设置。所有60个checkpoint、日志、loss/梯度/违例、数组都保存。物理电池macro RMSE先逐电池等权，再完整报告十seed均值、样本标准差/标准误、t95区间及相同seed的PINN减baseline差值；负差值表示PINN误差较低。区间只反映这些seed的波动，不包含数据集抽样/实验室差异或全轨迹预处理偏差。

全程复用同一作业、单3090、一个worker。`--wait-prior`可后台排队，仅每30秒读前序状态；前序未complete时不训练、不申请新显卡。前序failed/paused等会退出并保留日志。主输出固定为独立目录；完成任务按SHA跳过，已有pipeline必须显式`--resume`且来源/请求完全相同。失败worker目录改名保留后整worker重跑，未提供中途epoch续训。

现代包解压目标：`/workspace/stage_a_confirmation_20261111_v1`，结果：`/workspace/pinn4soh_stage_a_confirmation_20261111`。先在Stage A结束后预检：

```bash
PYTHONPATH=/workspace/site-packages /usr/bin/python3 /workspace/stage_a_confirmation_20261111_v1/run_stage_a_confirmation.py --repo /workspace/PINN4SOH-main --stage-a-root /workspace/pinn4soh_stage_a_20261111 --prior-pipeline /workspace/pinn4soh_stage_a_20261111_pipeline_status.json --output /workspace/pinn4soh_stage_a_confirmation_20261111 --gpu 0
```

后台串行排队（同一个单卡作业内；前序成功才开始）：

```bash
nohup env PYTHONPATH=/workspace/site-packages CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 /usr/bin/python3 /workspace/stage_a_confirmation_20261111_v1/run_stage_a_confirmation.py --repo /workspace/PINN4SOH-main --stage-a-root /workspace/pinn4soh_stage_a_20261111 --prior-pipeline /workspace/pinn4soh_stage_a_20261111_pipeline_status.json --output /workspace/pinn4soh_stage_a_confirmation_20261111 --gpu 0 --execute --wait-prior > /workspace/stage_a_confirmation_20261111.log 2>&1 < /dev/null &
```

状态：`/workspace/pinn4soh_stage_a_confirmation_20261111_pipeline_status.json`。最终collector独立核精确60+60、同seed配置/checkpoint/selection、逐数组macro、history最佳epoch、所有artifact和工具/注册SHA；归档包含200原CSV/28作者源码快照及六个原selection和原plan。流式tar逐成员回读SHA后写48MiB JSON分片，用原严格restore恢复。pipeline最终JSON和主日志位于结果目录外，另下载。

## 旧runtime复核的最小适配设计

现代60使用原`weights_only=True`读取，仅在modern环境执行。本包不运行旧栈60。旧Torch若不接受`weights_only`，最小差异是单独、显式、记录SHA的checkpoint加载适配器：只对本实验自己生成、已验证路径和SHA的state_dict checkpoint移除该关键字，保留map_location及strict load，不改网络、loss、梯度、优化器、epoch、seed、split、排序或任何数值配置。不得全局修改作者源码或悄悄吞掉其他异常。

落地旧栈前先只读记录Python/Torch/CUDA/cuDNN/依赖版本，验证确定性API、TF32开关、导数算子、CNN和两种PINN分支可用；执行小型真实train/valid烟测及save/load bitwise预测检查。若其他API不兼容，逐项形成显式补丁和新验收，不能以“仅加载差异”概括。相同初始seed并不保证跨Torch初始化/RNG算法完全相同，因此应保存初始参数SHA/数据tensor SHA，区分“相同数值超参数”与“跨栈bitwise一致”。只有现代完整冻结后才用同六配置、同十seed和相同数据/worker算法另起独立旧栈目录串行60+60，单独报告每seed差值与环境，不重新调参。本现代管道不把尚未执行的旧栈算作完成。
