# 本地恢复证据：MIT40、总体MAPE和正式推理计时

以下只读已恢复的原始数组、报告与计时记录，不连接云端、不重新训练。独立调用严格审计器重验440个主常规任务及40个MIT敏感性任务的文件哈希/完整数组/CSV边界指标，均与冻结报告相符。

## MIT40独立敏感性

整文件删除混合May CSV5和CSV6后，101训练/验证CSV与22测试CSV物理身份互斥；保留作者样本split和逐文件全寿命归一化。训练/测试总体都改变，因此差异不能称身份泄漏造成的纯偏差估计。主2120与MIT40在seed、有效参数、数值环境等字段相同；十次重复独立保存，不并入主表。

|实现|主MIT RMSE±SD (pp)|纯净子集 RMSE±SD (pp)|差值 (pp)|
|---|---:|---:|---:|
|PINN/published_label_monotonicity|0.94577±0.03872|0.95213±0.04320|+0.00636|
|MLP/author_current|0.90412±0.08476|0.88472±0.07097|-0.01940|
|PINN/paper_prediction_monotonicity|2.90930±0.01681|2.91405±0.01856|+0.00475|
|CNN/author_current|0.77589±0.05127|0.75451±0.04108|-0.02138|

纯净子集两PINN实现的MAE/MAPE/RMSE均高于MLP和CNN；CNN的均值误差最小。该结果不撤销其他数据集的观察，也不量化混合身份对每种方法的影响。

## 固定standard MAPE的四种总体权重

先在每个seed内合并11组全部82个测试CSV/完整预测点，再跨十个seed取均值和样本SD。四数据集等权时，每个数据集内部各组等权。文件等权仍是发布文件，不是MIT纯物理电池。

|实现|权重|MAPE%±SD|
|---|---|---:|
|paper_prediction_monotonicity|all_82_test_CSV_equal|4.798450±0.012901|
|paper_prediction_monotonicity|all_test_point_pooled|5.530815±0.010909|
|paper_prediction_monotonicity|eleven_group_equal|4.669701±0.010668|
|paper_prediction_monotonicity|four_dataset_equal_within_dataset_group_equal|4.946709±0.007963|
|published_label_monotonicity|all_82_test_CSV_equal|1.087376±0.031465|
|published_label_monotonicity|all_test_point_pooled|0.865541±0.031062|
|published_label_monotonicity|eleven_group_equal|1.029010±0.040935|
|published_label_monotonicity|four_dataset_equal_within_dataset_group_equal|0.988243±0.031131|

未从四种权重中挑选最接近0.87%的结果；原论文总体权重不明，故不能把0.87%视为已精确复现。

## 正式CPU计时

Intel Xeon Gold6230 @2.10GHz、PyTorch1.13.1+cu116；CPU float32 eval/no_grad，batch1000，预热50次，每个独立进程1000次forward取平均，五个独立进程，线程1/2分列。只计F/MLP/CNN前向，不含G/自动微分训练。论文为i5-10400F及Torch1.7.1，硬件软件差异限制直接速度比较。

|模型|线程|参数|每1000样本毫秒±样本SD|
|---|---:|---:|---:|
|F|1|7781|0.3895±0.0711|
|MLP|1|7781|0.3821±0.0604|
|CNN|1|8465|10.5312±0.3339|
|F|2|7781|0.4543±0.0332|
|MLP|2|7781|0.4566±0.0535|
|CNN|2|8465|7.9500±0.1599|

参数数量与论文一致；F/MLP结构相同且计时接近，CNN在本机更慢。计时不支持PINN训练学得更快或BMS部署有效的因果结论。
