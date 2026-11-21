# 透明替代超参数搜索

`search_replacement_grid.py`补齐原仓库未发布的搜索入口。补充材料Note 3第14页只说使用Grid Search优化层数、每层神经元、α和β，没有给网格范围、搜索种子、候选成绩或具体选择流程。本工具不声称恢复作者历史搜索，也不把新搜索成绩并入已冻结2120项结果。

`replacement_search_example.json`明确给出64候选×3种子=192个训练。这里只是一个可修改的预设示例：不是论文证据，也不是已完成192项。其网格在执行前完整写入plan，默认不会训练。示例将XJTU第2节作为独立验证电池，保留第4、8节为测试；这是新增电池级验证协议，与作者样本随机切分不同。示例损失是论文预测单调公式；可显式选发表代码标签单调项或作者当前项，各自独立执行。

在项目根目录生成真实数据计划：

```powershell
python reproduction/search_replacement_grid.py --spec reproduction/replacement_search_example.json --repo PINN4SOH --output outputs/replacement_search_plan --device 0
```

经查看plan.json确认搜索规模和全部设置后，执行同一显式网格：

```powershell
python reproduction/search_replacement_grid.py --spec reproduction/replacement_search_example.json --repo PINN4SOH --output outputs/replacement_search_plan --device 0 --execute --resume
```

工具只允许一个GPU索引，按候选、种子串行运行，DataLoader为0子进程。不得同时在平台启动多个本工具任务。CPU功能检查需要`--device cpu --execute --allow-cpu`。默认上限128候选可通过`--max-candidates`显式调整。

候选worker.json仅含train/valid路径，训练只加载这两组。每个候选每个种子按验证集所有样本的MSE选择最优epoch，保存独立深拷贝checkpoint；重载后验证MSE必须逐数一致。各种子的最优验证MSE取均值选候选，完全相等时按candidate_id，单次epoch相等时保留最早epoch。只有完整候选×种子矩阵齐备才生成selection.json。

加`--evaluate-test`时，在selection.json落盘并核SHA后，仅评估选中候选每个种子已选checkpoint，单独保存test数组和指标；测试指标不送回选择器。测试阶段不额外重训。未加该参数时不加载测试数据。协调器在计划阶段会读取测试文件字节计算SHA，这仅做输入审计，不计算测试指标。

结构定义：u使用作者`MLP(17->32)` encoder与固定`Predictor(32->32->1)`；只替换encoder的层数与统一隐藏宽度。F使用作者`MLP(35->1)`并配置层数与统一宽度。MLP的layers_num沿用作者“线性层总数”的含义，包含输出层。Predictor保持固定，不能将u_layers解读为整个u网络层数。激活、Dropout、初始化、Adam双优化器、LR调度、残差及单调loss复用作者和已审计loss helper。作者源码不编辑。

数据：显式train/valid/test电池路径须非空且互不重叠；CSV需16特征与capacity。各电池沿用作者DF的去NaN/Inf、3σ剔除、cycle index重建、逐电池归一化、相邻样本配对。**归一化使用各电池完整轨迹（包括验证和测试电池完整轨迹），不属于部署时因果归一化。** 本工具隔离模型选择与测试，但未将作者预处理改写成train-only缩放。共享种子控制numpy/Python/torch，GPU后端仍可能非确定；不承诺不同框架逐位等价。

恢复：plan、源码SHA、全部输入SHA、设置和GPU选择必须与原计划一致；完成worker逐SHA核验后只跳过不改写。未完成worker目录移至`.incomplete.<time>`保留，重新开始该候选/种子，不声称从中间epoch续训。协调器文件锁阻止同一输出目录双运行。损坏的完成artifact直接拒绝，不静默重训。源码或数据改变须使用新输出目录。

验证入口：`python reproduction/test_replacement_search.py`。它验证测试指标反转不影响选择、缺候选/种子与重叠电池拒绝、8个两epoch合成CSV真实PINN候选worker、仅2个选中test worker、完成结果恢复不改写和SHA篡改拒绝。合成数据只是功能验证，不是科学结果。真实数据小规模两epoch验收及192项完整计划保存在`outputs/full_reproduction/replacement_search_validation_20261111/`；正式192项搜索尚未执行。
