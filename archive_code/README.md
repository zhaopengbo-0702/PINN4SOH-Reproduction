# 全部历史源码、Notebook 与配套文件

这里保留本次代码补充包中的全部源码及同目录的配置、协议和说明，方便直接在 GitHub 点开阅读。原始字节保持不变。

| 范围 | 数量 |
|---|---:|
| 原代码包全部文件路径 | 1042 |
| 其中源码或 Notebook 路径 | 730 |
| 直接按原路径保存的文件 | 819 |
| 其中源码 / 配套文件 | 549 / 270 |
| 递归发布副本采用同字节引用 | 223 |
| 不同源码字节 SHA256 | 245 |
| Notebook | 3 |

[打开完整文件路径索引](源码路径索引.md) · [机器可读 SHA 与映射](CODE_INDEX.json) · [返回成果首页](../README.md)

## 从哪里开始看

- [作者模型](PINN4SOH/Model/Model.py)、[数据加载](PINN4SOH/dataloader/dataloader.py)
- [本项目实验运行器](reproduction/run_reproduction.py)、[主实验计划生成](reproduction/make_paper_plan.py)
- [独立重算主实验](reproduction/review_final2120.py)、[默认参数核验](reproduction/audit_current_defaults_local.py)
- [CPU 模型复载](reproduction/reload_current_defaults_cpu.py)、[替代搜索独立核验](reproduction/audit_replacement_search_local.py)

## Notebook

这三份 Notebook 属于特征处理来源材料，并非本项目重新训练时生成的训练 Notebook；原来源和历史内容保留。

- [CALCE_CS2.ipynb](outputs/full_reproduction/feature_audit/preprocessing_source/CALCE_CS2.ipynb)
- [OxfordBatteryClass.ipynb](outputs/full_reproduction/feature_audit/preprocessing_source/OxfordBatteryClass.ipynb)
- [XJTU数据集分析.ipynb](outputs/full_reproduction/feature_audit/preprocessing_source/XJTU%E6%95%B0%E6%8D%AE%E9%9B%86%E5%88%86%E6%9E%90.ipynb)

## 如何理解历史代码

原路径中保存了不同阶段的作者快照、实验工具、失败诊断、复核和发布工具。同一代码的多份历史副本保持原路径，Git 自身按对象内容去重；发布工作区内部再次复制的仓库文件用索引指向同字节在线文件，避免无限递归。

**历史 Stage A、确认及迁移鲁棒性源码仅作为已写代码保留。未完成的额外调参训练成果已按用户要求撤出交付；保留这些源码不表示重新运行、完成其训练，或支持性能改进结论。**

这些文件未针对当前目录重新接线。历史代码中的绝对路径、环境约定或后续计划是原始上下文，运行前应阅读主仓库 [使用指南](../docs/使用指南.md)，为输入配置实际路径并使用新输出目录。不要仅凭某个脚本存在就认为对应实验已执行。

本目录不包含训练输出、checkpoint、日志或数据；同目录配置和协议仍随原代码保存。较大的完整档案在发布页按清单恢复。云端独有源码另随云差量归档保存，没有再重复复制到此目录。作者源码、第三方 Notebook 及依赖的权利和使用条件归原来源。
