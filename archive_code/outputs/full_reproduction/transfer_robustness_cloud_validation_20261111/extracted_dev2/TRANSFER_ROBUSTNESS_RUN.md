# 两个敏感迁移方向的最小探索性改进

新增 `prepare_transfer_robustness.py` 与 `run_transfer_robustness.py`，不修改在线Stage A、原作者代码、冻结2120训练、原十二方向迁移及旧栈对照。当前新工作针对HUST→MIT和XJTU/2C→MIT，不能把此有限优化称作其余十方向的改进确认。

作者已有 `main_adaptation - fine-tuning.py`、AdaPINN、源checkpoint加载、冻结G及各方向调用；本入口补充新的明确选择/确认流程。原 `run_transfer.py` 的模型、来源核验和loss实现已有完整复现证据，直接复用其作者网络和状态hash；不另造物理方程。新的注册、valid-only调参、初始化/冻结策略对照、未来seed及阶段锁是额外研究协议。

源使用已审计的两栈各十次 `paper_prediction_monotonicity` 完整训练：HUST/all和XJTU/2C。每次源run必须完成、声明200epoch上限、best快照深拷贝、日志最佳epoch一致；checkpoint/config/manifest/split/log/tensor identity/source inventory/reload验证SHA均绑定，作者Python及源CSV重hash。两栈同repeat要求源训练seed、源inventory fingerprint和输入tensor身份一致。未来目标seed是新的，但源checkpoint复用历史repeat1..10；不能称十次全新预训练。没有用目标test挑source。

目标固定原先第一节 `data/MIT data/2017-05-12/2017-05-12_battery-1.csv`，其物理UID已核纯。验证与test复用Stage A已冻结的20节/22节物理电池，跨role UID不重合，混合May5/6排除；剩余80节非test不参加训练。**验证集的20节标签用于选参和早停，所以该新协议有额外target验证标签，不能说只用一节带标签target，也不能直接宣称恢复Table4的原一节数据预算。** 所有方法相同这一预算，test仍是历史已看过的探索性评估。

预设有限预算如下，见生成的PROTOCOL.json：

- 每方向九迁移候选：lr∈{0.0001,0.0003,0.001}，各配三策略{全部u可训且alpha0/beta0，全部u可训且alpha0.1/beta_mean0.01，冻结encoder仅调head且alpha0.1/beta_mean0.01}。G始终冻结并eval；冻结encoder也eval，防冻结Dropout仍扰动表示。新mono为mean ReLU，单独eval而保留梯度；data/PDE分支保持明确的训练模式。
- target-only共用一次九候选：相同三lr×weight_decay{0,1e-5,1e-4}，作者Solution_u随机初始化、alpha=beta=0、相同双端点监督。它是同目标数据/监督/选参预算控制；没有source初始化是target-only定义所需。
- 两方向各一原loss控制：保留原预测单调sum、训练态Dropout、alpha0.7/beta50及native常数lr（HUST→MIT0.005、XJTU→MIT0.001）。为对齐该控制使用相同新目标split、宏目标、min30/patience30。它是原loss/lr规则在新split上的matched control，**不是原Table4任务原样重跑**。原始Table4实验已完整保留。
- 全lane最大200epoch、最少30epoch、stale30、batch128、常数lr、不裁剪、逐物理电池RMSE等权选epoch。监督统一两个端点。min30不禁止最佳checkpoint位于早期。checkpoint深拷贝并重载预测逐位相等。
- 选择seed固定2026111101/02/03，分别配历史source repeat1/2/3。只在modern选择：两方向27+27及target-only27共81训练，加两个控制各3共6，总87选择/控制训练。完整矩阵落ALL_SELECTIONS_LOCKED前不允许test；同分按candidate编号/最早epoch。
- 锁定后用固定future target seeds2026111111..2026111120，按repeat1..10配源；两方向新迁移+两个控制+共享target-only，共50训练/栈。旧栈采用modern已选配置，不重新搜优。每栈全部50训练完整SHA核完才允许test，然后两方向各十source-only test。总目标训练87+50+50=187，不含源历史预训练；source-only不训练。所有任务单卡串行、单worker、两CPU线程。

生成新冻结文件（输出必须不存在）：

```powershell
python -X utf8 reproduction/prepare_transfer_robustness.py --archive-root outputs/full_reproduction/legacy_extension_recovered_20261111/restored/extracted --registration outputs/full_reproduction/stage_a9_preregistered_20261111 --repo PINN4SOH --output outputs/full_reproduction/transfer_robustness_preregistered_20261111/PROTOCOL.json
```

默认只dryrun，正式选择另加`--execute`：

```powershell
python reproduction/run_transfer_robustness.py --protocol <PROTOCOL.json> --repo <作者仓库> --archive-root <完整extracted源根> --output <新的modern结果目录> --device 0 --stack modern --phase selection
```

选择完成后，同结果目录用`--phase confirmation --selection-root <modern结果目录> --execute --resume`训练十future种子。独立旧栈目录则`--stack legacy --phase confirmation --selection-root <modern结果目录> --execute`。全部确认训练成功后，各目录同命令加`--phase test --resume`；不能先看test再扩grid。**云部署/现代及旧栈GPU烟测仍未做，不要与在线Stage A并行申请卡。** 当前入口记录实际torch并拒绝stack标签不符。Torch1.7的checkpoint加载/确定性API有明确兼容分支，实际GPU算子支持仍须验证。

`--pause-file`在下个worker前检查；当前worker完整保存后暂停。`--resume`只跳过身份/SHA/数组目标都一致的已完成任务，失败目录保留并重跑整worker，不宣称epoch续训。源码、协议、设备或路径变化需新目录。训练worker只含train/valid路径，正式test的worker只含test路径；主协议可以hash test字节做身份核验，不计算test score。每个result保留初/末u hash、G冻结验证、head冻结验证、环境、每epochloss与固定batch加权梯度诊断、真实预测数组和artifact SHA。

科学限制保持：单个target训练电池及验证划分、全寿命minmax和含capacity的3sigma、历史test已看、历史source重复使用、不同策略Dropout路径不严格逐mask对齐。不承诺调参后一定胜source-only、target-only或原控制。结果应按每seed配对报告相对误差、均值与样本SD，同时展示所有十seed；不只挑最好的seed。

本地功能验证 `python -X utf8 reproduction/test_transfer_robustness.py` 使用三份合成CSV、四lane各两epoch，检查冻结G/encoder、梯度有限性、reload、test路径隔离及完整矩阵选择。它证明执行路径，不是迁移性能证据。正式真实目标有界烟测、云包装、结果独立collector和图表/配对统计仍需独立验收，不能将当前代码存在写成这些都已完成。
