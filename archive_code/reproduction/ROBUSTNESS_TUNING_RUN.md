# MIT/HUST新调参入口

`search_robustness_tuning.py`是独立新增协议，不修改运行中的XJTU搜索或已冻结复现。每个spec对应一个dataset/model/loss组，使用显式候选列表，支持候选级lr、lr_F、warmup_lr、final_lr、batch_size、normalization_method、alpha、beta。层数与宽度在structure固定，避免一次把结构与优化器维度扩成不可承受全因子搜索。

必需JSON字段：`label,dataset,model,loss_variant,seeds,splits,fixed,structure,candidates,protocol_evidence`。splits与fixed沿用replacement_search_example.json结构；structure包含`u_layers,u_hidden,F_layers_num,F_hidden_dim,alpha,beta`六项；candidates为覆盖上述可调设置的字典列表，未覆盖项沿用fixed/structure。model可PINN/MLP/CNN；PINN显式选`paper_prediction_monotonicity`、`published_label_monotonicity`或`author_current`，baseline用`supervised_pair_mse`。label-only beta的模型参数梯度为零，工具拒绝把beta列入该lane候选。

默认dryrun：

```powershell
python reproduction/search_robustness_tuning.py --spec <预注册spec.json> --repo PINN4SOH --output <全新目录> --device 0
```

执行加`--execute`，恢复加`--resume`。仅单个GPU索引，单个worker，DataLoader零子进程。`--pause-file <路径>`可指定暂停哨兵文件：运行中创建该文件后，当前worker完整训练并保存result/checkpoint，协调器在下个worker前停止并记录`paused_after_completed_worker`。移除哨兵后以原CLI加`--resume`，完成项逐SHA跳过；不会截断当前checkpoint。该开关仅新runner提供，不改变旧XJTU运行源。

为避免重复接触测试，正式搜索建议不加`--evaluate-test`，各候选只train/valid。根协调者等全部方法、全部数据集候选完成且计划冻结后，再各组原CLI加`--execute --resume --evaluate-test`。3seed每组共3次选定checkpoint test；选择文件在任何正式test加载前已落盘，测试误差不参与候选排名。测试阶段不额外重训。计划阶段仅读取test字节做SHA，不计算其指标。

PINN训练复用已验收替代搜索的worker，保留作者MLP/Sin/Dropout、双优化器、residual和选定单调loss。MLP/CNN结构固定作者版本。为公平监督，baseline新训练使用`.5*MSE(f(x1),y1)+.5*MSE(f(x2),y2)`，与PINN数据项暴露相同成对端点；作者原baseline只使用x1，因此这是新公平比较协议，不能冒称原baseline数值恢复。三方法同epoch上限、early-stop、训练/验证电池及sample-weighted validation MSE选epoch/候选。每候选按3seed最佳验证MSE均值选，平局保留候选顺序，epoch平局保留最早。

`protocol_evidence`须解释MIT物理身份、防跨集、HUST预设电池划分和候选预算。入口检查文件路径disjoint，物理电池ID不能仅靠文件名推断，需要单独协议审计。MIT排除已知混合CSV，并从已核身份表验证一个physical ID只能进一个role。HUST保留原20test，非test按预先固定各组1节作valid。以上须由实际生成的spec和证据文件确认，文档不是已完成划分声明。

数据沿用作者DF：NaN/Inf剔除、所有列3σ（包含capacity）、逐电池完整轨迹归一化、cycle index、相邻对。验证/测试样本清理和缩放使用其完整轨迹及容量分布，故属于离线回顾性协议，不能称部署因果评估。不同normalization候选的数据tensorSHA应不同，但同一候选三方法需一致。

`test_robustness_tuning.py`用三节合成CSV对PINN/MLP/CNN各两个候选训练两epoch，验证normalization/batch/lr变更、PINN alpha/beta/lrF变更、完成恢复不改写、test隔离、三方法tensor一致；合成结果仅功能验证。真实MIT/HUST正式数据smoke与全部搜索必须单独核验，不能据功能测试宣称性能改善。
