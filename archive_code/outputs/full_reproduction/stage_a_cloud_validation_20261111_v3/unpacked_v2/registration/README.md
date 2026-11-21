# Stage A9 新改进实验预注册

这是2026-11-11新提出的**探索性改进**，与论文复现2120及其所有冻结结论分开。历史测试结果已经被观察，新的固定测试集只能称未用于本轮候选选择的测试电池，不能称从未被看过的确认集。

`REGISTRATION.json`冻结6份正式spec及split/物理身份JSON的SHA、选择与未来确认seed。split选择只使用文件名、身份映射和校验信息，没有读取任何模型分数。生成器`reproduction/prepare_stage_a_scientific_protocol.py`拒绝覆盖，完整重核MIT85577发布行到唯一capacity source映射、逐CSV SHA、UID计数与源审计一致，并验证三集合无物理身份交叉。

| 数据集 | train CSV/物理电池 | validation | test | 排除 |
|---|---:|---:|---:|---|
| MIT | 81/81 | 20/20 | 22/22 | May混合CSV5/6 |
| HUST | 47/47 | 10/10 | 20/20 | 无 |

MIT保留作者numeric ID%5测试集合中的22个纯身份CSV；其余101按原batch分别以电池数字ID升序，位置5、10、15…的UID固定为validation（May7、June7、April6节），其他81train。HUST保留论文20test，每组剩余电池按数字ID最小者1节做validation，其余train。HUST身份是官方原cell key/文件名，不具备独立barcode；MIT身份基于已审计第三方capacity index与原始特征签名，并不是原MAT字节链。

每model独立9候选×3seed，MIT/HUST各3model共162个选择训练。PINN的3lr为0.0003/0.001/0.003，交叉三组(alpha,beta_mean)=(0,0)、(0.1,0)、(0.1,0.01)，G lr在alpha0时0、其余等于solution lr。MLP/CNN各lr3×weight_decay0/1e-5/1e-4。F/G结构固定3/60；所有模型用同样的相邻两端点监督和物理电池等权validation RMSE。PINN单调项mean且eval模式保留自动微分，数据/PDE分支保留train dropout。新协议不能与原论文sum大beta或只x1监督基线混淆。

选择seed固定2026111101、2026111102、2026111103；未来确认seed固定2026111111–2026111120。以每物理电池RMSE等权、再对3seed等权选择；并列按候选ID，不能挑最佳seed。冻结selection后才能评估test。此162项不包括未来10seed×两栈确认，也不包括迁移调参；这些阶段需要对应独立计划。任何结果不佳时仍保存，不在看到test后扩grid以保证必胜。

新runner入口`reproduction/search_stage_a.py`默认dry-run；本目录6spec已通过其`build_plan`schema及文件/身份互斥检查，每份27个正式训练。这里只冻结计划，未在本地启动训练。

```powershell
python -X utf8 reproduction/search_stage_a.py --spec outputs/full_reproduction/stage_a9_preregistered_20261111/mit_pinn_stage_a9.json --repo PINN4SOH --output outputs/full_reproduction/stage_a9_dryrun/mit_pinn
```

单平台任务、单GPU和单worker约束继续适用。候选worker只收到train/valid，test路径和数据不参与训练或选择。checksum核验读取test CSV字节是身份完整性核验，不是选参性能评估。

限制：作者清洗使用包含capacity的全列3sigma删行；每文件全寿命minmax/cycle缩放使用held-out协变量，是离线transductive协议。原test已看，无法靠新seed洗成独立确认。训练/validation完全物理身份隔离消除了原随机sample验证的端点标签复用，但改变了验证协议，改进结果不能回写成论文原数字。单一validation划分还不等于跨组/跨实验室泛化，后续确认应保存每电池/每seed结果和最差情况。
