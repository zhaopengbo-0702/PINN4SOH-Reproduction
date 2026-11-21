# 正在运行的XJTU替代搜索只读复核

本复核不修改已上传包及运行源码，不重新启动训练。根执行者确认2026-11-11 12:55:42 GPU smoke通过、正式192项开始；此时间是执行者提供的观测，不由本文件重新确认实时状态。

验证集选epoch逻辑通过源代码复查：每epoch训练后模型eval、验证DataLoader不shuffle、torch.no_grad、仅x1预测；所有样本float64差值的MSE作为目标。严格更低才深拷贝并保存checkpoint，相等保留最早epoch。训练结束重载两网络，按相同顺序、相同batch size重新预测并要求MSE逐数相同。GPU若重载导致数值不一致会失败，不自动放松门槛。最优模型并非最后epoch引用，避免作者原state_dict引用问题。

正式搜索候选worker只接收train/valid路径，未接test路径和test SHA；协调器计划阶段读取全部CSV做SHA，其中test仅字节读取。64个Cartesian候选×3seed完整后，以三seed各自最佳验证MSE均值选候选，完全相同按candidate_id。selection落盘后，仅该候选三seed checkpoint各一次正式test。正式选择没有调用test指标。

需要明确披露：GPU smoke也对正式第4/8测试电池做过一次两epoch选定评估，发生在正式192之前。smoke指标未成为正式选择器输入，网格预先固定，但不能表述为整个会话从未访问测试集或绝对盲测。此前主复现也已观察这些电池，新增协议属于在已有数据上的调参实验。

预处理边界比“test不参与选择”更宽：作者DF先按每电池完整轨迹各列3σ剔除，列包含capacity，因此验证/测试的样本保留集合用到其完整容量分布；随后每电池完整轨迹min-max/z-score并加入cycle index。它是回顾性离线处理，不能称部署时因果归一化或完全无测试标签信息的整个数据管线。本新增协议保留作者数据规则，并未悄悄修正为train-only缩放。

来源核验总体足以检测意外改动：上传预检包SHA及28作者py/8CSV，候选执行前核所有已记source/inputSHA，result绑定worker identity且记录artifactSHA。小缺口：search plan source_sha没有run_reproduction.py，而run_transfer顶层导入该helper；它主要提供工具和定义，不参与已调用loss函数，但作为导入依赖仍应在下载后通过TOOLS_SHA256核其确切字节。包manifest包含该文件。运行代码不补改。

collector已核候选完整路径、配置、源SHA、全部数组MSE与history最佳epoch、原选择、artifactSHA和候选内tensor一致性，再流式tar逐成员回读。补强门槛：其三test检查只有数量和seed属于集合，没有强制三个唯一seed/目录与同seed checkpoint严格一一对应；它的选择重算复用原select函数，也不从CSV独立重建tensor。本地新增`audit_replacement_search_local.py`独立重建64Cartesian×3、全部worker配置/输入hash、精确三seed test目录集合、同seed选定checkpoint lineage、全部CSV清洗/归一化tensor/labels、独立排序选择和测试误差。该审计不导入归档Python、不反序列化checkpoint、不写原结果。

审计命令（待真实归档安全恢复后，不能现在称通过192）：

```powershell
python reproduction/audit_replacement_search_local.py --extracted <恢复根目录> --repo PINN4SOH --collection <SEARCH_COLLECTION_VERIFIED.json> --expected-runs 192 --output <新审计JSON>
```

已用现有真实两候选烟测归档验证同一路径，74成员inventory、31来源、3CSV、13artifactSHA和全部数据重建通过；仅功能验证，不替代云192验收。此本地审计验证checkpoint字节来源，不宣称GPU前向重跑。

科学范围：只有XJTU/2C固定划分、两种结构层/宽、两个α/β值和3优化seed；不恢复作者历史搜索、不覆盖MIT/HUST调参或迁移鲁棒性，也不能将优化seed作为独立电池样本做显著性检验。若新目标转为MIT/HUST及迁移，应先保留当前完整worker产物，再转向新的预注册协议。
