# 作者当前默认20次checkpoint的CPU复载

2026-11-11北京时间18:24完成。先MIT/HUST各repeat1作两次烟测，再同一脚本串行复载MIT/HUST各10次。正式20次全部通过固定`atol=1e-6, rtol=1e-5`数值一致性；prediction逐位一致为0/20，不能表述为逐位恢复。最大单元素漂移为`2.384185791015625e-7 SOH`，即约`0.00002384`个SOH百分点。全部Y1标签逐字节一致、原DF与独立CSV重建的四张量逐字节一致。

对照的是已冻结原GPU数组。CPU环境为Python3.9.12 / Torch2.7.0+cpu / NumPy1.26.4 / pandas2.2.2；原GPU为Python3.10.13 / Torch1.13.1+cu116 / NumPy1.24.4 / pandas1.5.3 / 单3090。原始GPU复载证据仍仅MIT/HUST各repeat1两次；这份CPU结果不能扩大该GPU覆盖范围。

输入为`cloud_emergency_recovered/workspace/pinn4soh_main_20261109`的20次原run，通过`current_repo_defaults_reaudited_20261111`的已冻结delivery index与original/recovery证据绑定。417源码/CSV、89条原完成证据、202条恢复证据以及20×关键文件在任何checkpoint restricted load前核SHA。合并636个不同输入文件在执行后再次核SHA，全部未改。没有训练、反传、优化器或写回原run；只用作者Solution_u推理，同时strict加载并检查两网络state字典。

`full_twenty/CPU_RELOAD_SUMMARY.json`是20次汇总；`runs/<dataset>/all/ExperimentNN/`保留每次CPU数组、CPU_RELOAD.json及INPUT_TENSOR_IDENTITY.json。两阶段各有SHA256_INDEX.json，根DELIVERY_SHA256_INDEX.json补充绑定命令、console和两阶段交付，不改旧索引。MIT与HUST所有float64指标的最大绝对变化分别为`1.0283218651596648e-9`、`2.126274377053927e-9`（原始比例单位；这是不同指标中的最大值）。

重建X1/X2/Y1/Y2的dtype、shape、SHA及每CSV边界已保存。**这些X hash是本次新建，旧run没有GPU训练时X tensor hash参照**；不能据此补造原训练时输入逐位一致的历史证据。继承作者整轨迹缩放、包含capacity的3sigma、原测试顺序与batch512；未改变旧数据边界或MIT身份重复。pandas兼容性FutureWarning在full_twenty_console.log保留，四张量独立比较通过，未静默改写作者代码。

脚本的本次调用只支持这个冻结审计对象。当前路径为普通本地文件且输出独立；参数泛化需注意它的resolve后软链检查及mkdir先于results重叠复核两个接口边界，未影响本次绑定运行。失败会保留报告，固定容差不放宽。复现实验中不成立的优势结论也不因CPU复载通过而改变。

命令记录见COMMANDS.txt。两个阶段输入/脚本SHA一致，full阶段检查两次烟测先通过。脚本独立审查由search_runner另存。
