# 代码来源与微信结果技术核对

**截止收尾已完成：2026-11-12 09:00前交付要求已提前落实。** 额外队列不再续跑，22:48:12全部训练/调度/等待进程退出，平台单卡作业已显示“已结束”。[截止版最终交付](../../../明早9点截止_复现最终交付_20261112.md)。

**最终新增结果：59项正式选择训练+2项GPU烟测，选定test0，HUST正式0，确认/迁移训练及test均0。** 新增归档在云端同NFS双目录独立回读通过，**不是异地备份且尚未下载本地**；原主2120/当前默认20及既有本地备份不变。collector最终摘要TypeError保留，并由独立existing verifier回读补证。[最终网页观察转录](../../../outputs/full_reproduction/cloud_live_20261111/deadline_final_closeout_local/LOCAL_UI_OBSERVATION.json)、[最终交付与边界](../../../明早9点截止_复现最终交付_20261112.md)。

**补充澄清：作者当前默认实验并没有缺跑。** 最初110次中已有MIT/HUST各10次，现从完整备份独立核验20/20通过：当前默认pooled标准MAPE为MIT0.689402±0.028627%、HUST0.814535±0.027752%，CSV宏为0.792423±0.050746%、0.799301±0.030026%。MIT单次repeat1 pooled0.635553%。下方表格是主2120的Note3/发表标签项对照，不能替代这批2026默认权重/当前损失结果。此前只突出主2120容易误导版本范围，在此明确纠正。[原数组/配置/源码SHA独立复核](../current_repo_defaults_reaudited_20261111/作者当前默认20次_独立复核.md)。这批结果与微信值接近，但对方缺实际配置、数组和指标口径，不声称严格同协议一致。

本说明只追加证据，不改已冻结2120项结果、登记或作者源码。微信数字来自根代理读到的原文：MIT PINN MAPE 0.656%（论文0.652%）、HUST 0.830%（论文0.778%）；MIT MLP六配置中最好1.646%，SD0.508%，PINN SD0.039%；对方称只改pandas兼容及TJU的pass，未附实际训练脚本、配置、日志或预测数组。因此这些是**尚待原始材料核验的外部报告值**，不能据此认定谁的代码错误。

## 作者本来已经提供什么

| 作者已有文件或资料 | 已有内容及边界 |
|---|---|
| `PINN4SOH/Model/Model.py` | solution、动力学网络、自动求导PDE残差、损失、Adam、调度、训练/验证/测试；不是只有模型壳。 |
| `PINN4SOH/main_XJTU.py`、`main_TJU.py`、`main_MIT.py`、`main_HUST.py`、`dataloader/dataloader.py` | 四数据集常规/小样本入口和已处理特征读取。当前main里的grid_search仅单组alpha/beta循环，不等于补充材料声称但未披露的完整历史搜索。 |
| `PINN4SOH/Model/Compare_Models.py`、`main_comparision.py` | MLP/CNN结构、baseline Trainer与XJTU运行入口。并非没有baseline代码；但直接主入口没有按dataset分派MIT/HUST/TJU。 |
| `PINN4SOH/main_adaptation - fine-tuning.py` | AdaPINN、载入预训练、冻结G和十二方向的相关调用，原生共19配置；不是迁移从零重写。具体batch映射与Table4有差异，须透明区分。 |
| `PINN4SOH/results/`、`results analysis/`、预训练checkpoint | 已有历史数组/分析脚本和11份checkpoint，但未公开这些checkpoint的完整训练seed、split和选择履历，不能与新训练混计。 |

## 我们为何额外写代码

| 类型 | 新文件 | 实际用途与状态 |
|---|---|---|
| 运行与兼容支持 | `reproduction/run_reproduction.py`、`run_extended.py`、`run_full_suite.py`、`launch_legacy_control.py` | 在作者模型/Trainer外增加单任务封装、明确版本/seed/目录、单卡串行、完成恢复、SHA、日志；修复best state_dict未深拷贝造成checkpoint与保存预测不一致。pandas/旧栈适配显式记录；这些支持本身不是新科学方法。 |
| 按论文和已有代码补可运行实验接线 | `reproduction/run_extended.py`、`run_transfer.py`、`make_paper_plan.py` | 将MIT/HUST/TJU的已有loader接入已有baseline Trainer；枚举完整常规/小样本/消融；分开运行发表代码标签项与论文预测项；记录native19与paper_fixed12迁移映射和来源链。主2120已完成，不把后补入口说成作者原已完整提供，也不说公开仓库完全无代码。 |
| 独立验证与保存 | `reproduction/build_suite_report.py`、`compare_legacy_control.py`、`compare_legacy_extension.py`、`collect_suite_evidence_v2.py`、备份/恢复脚本 | 从数组和实际电池边界重算指标，区分MAPE分母、宏/池化、软件栈与seed，保存checkpoint/日志/配置/数组及SHA。是审计与工程工具，不等同新训练。 |
| 缺失原始导出器的透明替代 | `reproduction/export_replacement_features.py`、`validate_replacement_features.py` | 新增16特征导出入口；14项按明确约定核验，两个slope固定端点差/物理秒并标replacement；未恢复作者原两个slope定义，不覆盖作者训练输入。四数据集各三真实周期及五解析/异常检查已有验收。 |
| 未披露历史搜索的透明替代 | `reproduction/search_replacement_grid.py`、`run_replacement_search_cloud.py` | 新预设64候选×3seed=192训练；独立电池验证、完整候选后选择、test隔离。它是新搜索，不是恢复作者未公开的网格/seed/历史成绩；正式进度以根代理云端最新快照为准。 |
| 额外改进研究（停止前历史记录） | `reproduction/search_stage_a.py`、`prepare_stage_a_scientific_protocol.py`、`run_stage_a_cloud.py` | MIT/HUST×三模型×九候选×三选择seed=162的新探索实验。新mean/eval单调、双端点基线、公平初始化、物理电池macro验证，均是明确改变。StageA功能、真实CPU/GPU烟测通过；22:02:39云终端观察57/162选择完成，MIT_PINN complete27/27、MIT_MLP complete27/27，当前MIT_CNN running3/27，尚无18选定test或性能提高结论。60新seed确认waiter18969等待StageA；迁移v2已上传解包25成员、waiter19580等待确认状态，187项目标训练/140test未开始。迁移wrapper/collector工程审计通过；以上是22:02停止前记录。最终59正式+2smoke/0test，确认/迁移0，全部已停止，归档只在云端同NFS双目录核验，未下载。 |

上表所有路径相对工作目录 `C:/Users/Administrator/Desktop/复现PINN4SOH`。原实现缺失或有歧义之处的独立实现已标假设/替代，不能借“写了代码”宣称数值精确复现。

## 最关键的可复核技术差异

**1. 标签单调项没有参数梯度，但PDE项仍然存在。** 发表代码项 `relu(y2-y1).sum()` 只依赖标签，参数梯度恒零。目标仍是双端点监督加`alpha*PDE`；残差通过`u_t/u_x/u`和学到的G反传，不能简称纯MLP。若alpha也为零，才在同初始化、同优化、同监督和随机路径条件下等价监督MLP。当前公开代码另外使用 `relu((u2-u1)*(y1-y2)).sum()`，论文公式版本使用 `relu(u2-u1).sum()`；三者必须明确版本。

**2. 公开baseline主入口只接XJTU。** `main_comparision.py`的`__main__`将model强设为MLP，遍历六个XJTU batch，并无条件调用`load_XJTU_data(args)`。单传`--dataset MIT`只会更改参数/路径标签，不会把数据切到MIT；`--model CNN`也被随后覆盖。因此对方MIT/HUST MLP结果必定还需要额外的调用、交互代码或入口接线。可以复用已有函数而不改作者文件，但应提供实际运行入口。这个事实本身不证明对方用了错误数据。

**3. 拓扑相同不等初始化与训练相同。** 本地只做了构造检查，`Solution_u`与`Compare_Models.MLP`均7781参数，线性层为17→60→60→32→32→1，sin和0.2 Dropout。可是Solution_u额外重置全部线性层Xavier与零bias；baseline MLP仅encoder做Xavier权重初始化，predictor权重/bias和其他bias保留PyTorch默认。baseline仅监督`(x1,y1)`，PINN平均监督两个端点；baseline硬编码stale>10早停、验证对batch等权，PINN按args且验证对样本等权。这些是原公开代码差异；主复现实验保留，新Stage A才对齐初始化/双端点/宏目标。

**4. 默认学习率本身没有已证实冲突。** 公开PINN和baseline默认均Adam、batch512、最多200epoch，预热30epoch，u的lr为0.002→0.01→0.0002；G另有常数lr。不能据微信数字猜对方学习率写错。当前MIT默认alpha/beta为0.5/0.01、HUST为0.6/0.1，而我们冻结Note3任务是MIT1/50、HUST0.5/80；如果对方直接运行最新默认，配置确实不同，但需要其config才能确认实际采用哪组。

**5. 误差统计口径必须匹配。** 作者工具定义`eval_metrix(true,pred)`，若调用为`eval_metrix(pred,true)`则MAPE以预测值作分母；原结果分析还用容量跳变猜边界并丢掉边界样本。我们同时保留该诊断口径和真实CSV边界、measured分母的标准口径；主比较按每CSV等权后跨十seed均值。对方的0.656/1.646是macro还是pooled、MAPE分母/边界/SD的ddof均未由消息确定，不能直接当完全同协议数值。

## 我们已有结果可作什么对照

来自已独立复算的 `outputs/full_reproduction/legacy_extension_recovered_20261111/local_comparison_v2/ten_repeat_summaries.csv`，以下为标准MAPE百分数、十seed均值±样本SD（ddof=1）、发布CSV等权；两栈分开，不与对方未知口径硬合并。

| 数据/模型 | modern | legacy |
|---|---:|---:|
| MIT 标签项PINN | 0.872471±0.040463 | 0.870594±0.036325 |
| MIT MLP | 0.822061±0.093154 | 0.859143±0.073442 |
| MIT CNN | 0.659776±0.039314 | 0.696990±0.068821 |
| HUST 标签项PINN | 0.797987±0.046649 | 0.795020±0.053864 |
| HUST MLP | 0.794585±0.061452 | 0.789361±0.032620 |
| HUST CNN | 0.732670±0.063834 | 0.724719±0.066107 |

这张主2120标签项表没有出现MIT MLP均值1.646%，也不是其所称PINN0.656%的匹配复核；当前默认20次请看文首澄清，不能据此称所有版本都没有接近0.65%的结果。现有证据只能确认存在待解释的实验差异，不能给出对方MLP较差的唯一原因。核对最有价值的材料是：实际entry与模型类、代码commit、config/lr/alpha/beta、split与目录顺序、十seed训练/验证日志、真实与预测数组及指标计算脚本；不需要重写历史结果迎合外部数字。

新Stage A独立审查见 `outputs/full_reproduction/stage_a_validation_20261111/STAGE_A9_INDEPENDENT_PROTOCOL_REVIEW.md`：200CSV、六spec、162预算身份核验通过，现代GPU烟测已通过，正式选择最终保存59项，另2项smoke、0test，确认/迁移0；已按截止指令停止归档，最终转录见文首。作者已有代码、按论文补接口、未知细节替代、额外改进四者继续分开记录。

22:02:39状态的[本地UI观察记录](../cloud_live_20261111/observation_220239/LOCAL_UI_OBSERVATION.json)是网页终端转录，不是下载的云原状态文件；[五文档和冻结SHA核验](动态入口_实时快照_20261111_220239_REVIEW.json)保留版本、额外20节validation标签和真实GPU待执行边界。

[原论文真正剩余八项与新增改进分界](原论文剩余事项_与新增改进分界_20261111.md)解释主矩阵已覆盖、原材料/历史定义/同硬件计时和反例的分别处理；不再把可选旧栈StageA60列为必须补训。

作者当前默认20个checkpoint另完成[CPU复载](../current_repo_defaults_cpu_reload_20261111/README.md)，[独立审查](../current_repo_defaults_cpu_reload_independent_review_20261111/INDEPENDENT_CPU20_REVIEW.json)通过：固定atol1e-6/rtol1e-5为20/20 allclose，prediction bitwise为0/20，最大漂移2.384185791015625e-7 SOH，标签20/20 bitwise、636原输入SHA未变。无训练、未改原输入，原GPU复载覆盖仍仅MIT/HUST各repeat1两次；本次新建X hash不能补造GPU训练时X参考。[101件交付SHA索引](../current_repo_defaults_cpu_reload_20261111/DELIVERY_SHA256_INDEX.json)。

192最终状态原件的[18:22本地下载重试记录](../cloud_live_20261111/FINAL_STATUS_DOWNLOAD_RETRY_1822_LOCAL_RECORD.json)保留网页权限错误和截图；精确网页操作时间未另记，原件仍未下载。云训练不受此次只读下载失败影响。
