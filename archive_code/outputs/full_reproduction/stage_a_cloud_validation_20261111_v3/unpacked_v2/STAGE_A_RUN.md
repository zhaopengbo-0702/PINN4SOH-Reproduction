# Stage A：MIT/HUST的独立改进实验

这是新的调参实验。它保留旧论文复现和XJTU替代搜索，不改写其损失、验证结果或冻结报告。正式计划为两数据集×三方法×9候选×3选择seed，共162次选择训练；每组最优配置由验证电池选择，测试成绩不能决定候选或继续加网格。

PINN九候选：lr_u∈{0.0003,0.001,0.003}，每个lr对应三组(alpha,beta_mean)={(0,0),(0.1,0),(0.1,0.01)}。alpha=0时G参数冻结、不建G优化器、跳过PDE导数计算，lr_F必须0；alpha>0时lr_F=lr_u。三组分别是同网络监督消融、PDE、PDE加小权重单调。MLP和CNN各九候选使用相同三个lr×weight_decay∈{0,1e-5,1e-4}。这是相等候选预算，搜索维度不同，不宣称两个基线也在搜索PINN专属参数。三方法均上限200epoch、warmup5、至少训练30epoch、patience30、batch512、minmax、相同显式物理电池划分、相同3seed。

单调项为`mean(relu(u2_eval-u1_eval))`，临时`solution.eval()`关闭Dropout，**不使用no_grad**，计算图保留。原data/PDE分支仍处于train模式。eval单调分支计算后恢复原模式；模型参数没有被冻结。该项与论文sum及独立Dropout蒙版计算不同，beta_mean0.01不能解释为论文beta0.01。固定诊断batch每epoch保存未加权/加权loss、各项对solution参数的L2梯度范数、实际n及模式；诊断随机数用fork_rng隔离，不改变训练蒙版。每epoch还保存三loss及其加权值、F/G当前lr、实际batch最小/最大点数、clip触发数、梯度有限性、train/eval两支正差分布和分位数。

预设grad_clip_norm=0表示关闭裁剪，不静默修改优化器。lr从0.1×候选lr预热到候选lr，随后余弦降到0.01×候选lr；在每epoch训练开始前更新，F和G使用相同比例。此明确调度不同于原作者只调solution且epoch后step。新worker记录TF32关闭、cuDNN benchmark关闭、deterministic开关；若确定性算子不支持会失败并保留日志，不静默退回。

验证先按每个physical_id汇总其所有CSV段的样本，计算该物理电池RMSE，再对物理电池等权平均，以此选epoch；跨3seed均值选候选。RMSE内部以SOH fraction计算，展示pp乘100不改变排序。同分保留最早epoch/先列候选。每个最优checkpoint深拷贝保存，重载要求验证prediction逐位相等。保存逐物理电池MSE/RMSE/MAE/MAPE和数组，不以pooled MSE代替macro目标。

MLP使用与PINN完全相同的作者`Solution_u`结构和初始化。PINN及MLP在构造完网络后重设训练随机数，DataLoader另用固定generator；功能测试证实alpha=beta=0、同seed同lr时，PINN与MLP的初始solution、输入tensor和两epoch验证prediction完全一致。CNN是作者结构。三方法均用`.5*MSE(u1,y1)+.5*MSE(u2,y2)`成对端点监督，避免原baseline只监督x1的不等暴露。这是公平性改进，不能称作者原baseline结果恢复。

数据仍使用作者离线预处理：每个电池全轨迹所有列3σ剔除（包含capacity）、逐电池完整轨迹归一化、cycle index、相邻配对。验证/测试容量分布影响样本保留集合，完整轨迹影响缩放，因此不属于部署因果评估。新split按物理身份互斥，不能消除这种回顾性预处理限制；更严格train-only scaler应另立协议。

每组spec字段为`label,dataset,model,seeds,splits,fixed,structure,candidates,protocol_evidence,smoke`。每role的paths与physical_ids逐项对应，同一UID可有多CSV但只能落在一个role。runner核路径与UID互斥；身份映射证据还须独立核验，工具不从文件名猜测MIT barcode。正式smoke=false要求九候选、三seed和上述最低schedule；功能烟测smoke=true可减轮数，结果明确不计正式训练。

默认只建计划：

```powershell
python reproduction/search_stage_a.py --spec <StageA-spec.json> --repo PINN4SOH --output <新目录> --device 0
```

正式选择加`--execute`；同一单卡串行跑6组，不并行启动多个脚本。候选worker.json仅含train/valid，test路径不传入。默认不加载test；全部组完成并冻结选择后，才逐组原CLI加`--execute --resume --evaluate-test`评估选择seed。若要10个新确认seed，须另有固定确认计划，不把这三次选择seed测试称十重复确认。现入口尚未包含自动10seed确认或迁移适配。

`--pause-file <路径>`在下个worker前检查哨兵：当前worker完整保存后停止，记录paused_after_completed_worker；去掉哨兵并显式`--resume`继续。完成artifact逐SHA核验后跳过，不改结果；失败/未完成目录保留并重跑整个worker，不声称中间epoch续训。源码、输入、spec或device变化时原plan拒绝恢复，须另开新输出。未加execute不训练。

验证：`test_stage_a.py`核物理电池macro与pooled区分、同UID跨CSV合并、eval保梯度、min30/patience30、warmup/余弦端点、物理ID跨集拒绝、三个真实模型各2候选2epoch、checkpoint逐位reload、候选test隔离、完成resume不改写、alpha=beta0 PINN与MLP精确等价。它的CSV是合成的，只证明执行路径。真实MIT/HUST短烟测和独立spec身份核验另存验收，不能从功能测试宣称性能改善。
