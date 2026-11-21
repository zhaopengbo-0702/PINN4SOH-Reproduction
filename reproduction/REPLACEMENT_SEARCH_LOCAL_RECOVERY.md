# XJTU替代搜索：下载后恢复与独立审计

本说明用于192项XJTU/2C新替代搜索，不是作者未知历史Grid Search。先在全新`outputs/full_reproduction/replacement_search_recovered_20261111/transfer/`中放齐：

- `SEARCH_COLLECTION_VERIFIED.json`
- `JSON_TRANSFER_MANIFEST.json`
- manifest所列的全部`results.search.*.chunkNNN.json`

另下载云端最终`pinn4soh_replacement_grid_20261111_pipeline_status.json`，保存在恢复目录下的`final_pipeline_status.json`，不要将它放入解包后的`extracted/`，否则外inventory精确集合检查会拒绝。归档内`audit/PIPELINE_STATUS_AT_COLLECTION.json`是collector采集时的快照，可能仍为running，不能冒充最终状态。

在项目根运行，恢复输出`restored/`须不存在，其父目录须存在；独立审计输出JSON也须不存在：

```powershell
python -X utf8 reproduction/restore_replacement_search_local.py --transfer outputs/full_reproduction/replacement_search_recovered_20261111/transfer --output outputs/full_reproduction/replacement_search_recovered_20261111/restored --expected-runs 192
python -X utf8 reproduction/audit_replacement_search_local.py --extracted outputs/full_reproduction/replacement_search_recovered_20261111/restored/extracted --repo PINN4SOH --collection outputs/full_reproduction/replacement_search_recovered_20261111/transfer/SEARCH_COLLECTION_VERIFIED.json --expected-runs 192 --output outputs/full_reproduction/replacement_search_recovered_20261111/LOCAL_INDEPENDENT_SEARCH_AUDIT.json
```

第一步复用已核wrapper严格恢复器，核每分片/总归档SHA，要求collector记录192完整候选和全部选定seed测试。逐tar成员流式写到全新目录，拒绝绝对路径、`..`、链接、重复、大小写冲突和Windows危险名，按外inventory写时和落盘后逐SHA/大小复核；不调用不受约束的`extractall`。

第二步只读，依赖本地NumPy/Pandas及已有`PINN4SOH/`，不导入归档Python、不反序列化checkpoint、不重新训练。它核归档外inventory、工具manifest、原作者源码/输入CSV的SHA；独立重建作者清洗/归一化/相邻配对和float32张量，核192个validation数组、每次best epoch、64候选三seed均值选择，再核3次test绑定所选同seed checkpoint和冻结selection，独立算MSE/RMSE/MAE/MAPE。物理前向仍由云训练保存证据支持；本地不把checkpoint字节核验叫重新GPU前向。

主结果/21结论/25图表保持冻结。此搜索保留全电池全轨迹minmax及含capacity的3sigma清洗，历史GPU烟测也已查看同一测试电池；不得宣称部署前瞻处理、从未查看test或完整恢复历史调参。三seed不是三个独立电池测试集。

恢复器验证入口为`outputs/full_reproduction/replacement_search_restore_validation_20261111/`：5普通合成文件恢复、11危险路径/错误scope/已有输出拒绝；另从现成真实2epoch小搜索归档恢复74成员并独立审计2候选/1选定test，13完成artifact、31source、3CSV检查通过。它们验证工具链，不能替代192正式结果审计。
