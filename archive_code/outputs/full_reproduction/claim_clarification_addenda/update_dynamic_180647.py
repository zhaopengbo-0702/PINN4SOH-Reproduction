# -*- coding: utf-8 -*-
"""Update only four dynamic entrances from the recorded 18:06:47 UI observation."""
from pathlib import Path
import re

obs = 'outputs/full_reproduction/cloud_live_20261111/observation_180647/LOCAL_UI_OBSERVATION.json'
shot = 'outputs/full_reproduction/cloud_live_20261111/stage_a27_transfer_queue_waiting_180647.jpg'
review = 'outputs/full_reproduction/claim_clarification_addenda/动态入口_迁移已排队_20261111_180647_REVIEW.json'
docs = [Path('复现结果总入口.md'), Path('本日复现进展_20261111.md'),
        Path('outputs/full_reproduction/执行状态.md'), Path('整篇论文复现覆盖与未复现清单_20261111.md')]
for p in docs:
    s = p.read_text(encoding='utf-8')
    rel = '../../' if p.name == '执行状态.md' else ''
    s = s.replace('17:44:32', '18:06:47').replace('22/162', '27/162')
    s = s.replace('当前`mit_pinn_selection`22/27、整体27/162', 'MIT_PINN已complete27/27，当前`mit_mlp_selection`running0/27、整体27/162')
    s = s.replace('当前mit_pinn_selection22/27、整体27/162', 'MIT_PINN已complete27/27，当前mit_mlp_selection running0/27、整体27/162')
    s = s.replace('`mit_pinn_selection`22/27、整体27/162', 'MIT_PINN complete27/27，MIT_MLP running0/27、整体27/162')
    s = s.replace('两敏感迁移方向代码已通过本地功能与真实CPU烟测验收，187项目标训练尚未云部署', '两敏感迁移方向v2包已上传解包并排队，waiter PID19580等待前序确认，187项目标训练尚未开始')
    s = s.replace('迁移鲁棒性新代码已通过本地功能与真实CPU烟测验收，187项目标训练未云部署', '迁移鲁棒性v2包已上传解包，waiter PID19580等待前序确认，187项目标训练尚未开始')
    s = s.replace('两个敏感迁移方向新代码已通过本地功能与真实CPU烟测验收，187项目标训练未云部署', '两个敏感迁移方向v2包已上传解包，waiter PID19580等待前序确认，187项目标训练尚未开始')
    s = s.replace('新代码本地功能/真实CPU烟测验收通过；87项modern选择/控制、50项modern确认及50项legacy确认共187项目标训练，尚未云部署，不计已训练完成', 'v2包上传解包25成员、PID19580等待前序确认；87项modern选择/控制、50项modern确认及50项legacy确认共187项目标训练、140test尚未开始，本地wrapper/collector工程审计通过')
    s = s.replace('迁移新代码本地验收通过、187项目标训练未云部署', '迁移v2已排队、187项目标训练及140test尚未开始')
    s = s.replace('迁移新代码本地功能与真实CPU烟测验收通过，187项目标训练尚未云部署', '迁移v2包已上传解包并排队，PID19580等待前序确认，187项目标训练/140test尚未开始')
    s = s.replace('后台waiter在18:06:47快照仍写`WAITING_FOR_STAGE_A`，最新可见日志09:44:12Z（上海17:44:12）', '后台waiter在18:06:47快照仍写`WAITING_FOR_STAGE_A`，最新可见日志10:06:43.781364Z（上海18:06:43）')
    s = s.replace('日志`WAITING_FOR_STAGE_A 09:32:11Z`（上海17:32:11）', '最新日志`WAITING_FOR_STAGE_A 10:06:43.781364Z`（上海18:06:43）')
    s = s.replace('日志`WAITING_FOR_STAGE_A 09:32:11Z`', '最新日志`WAITING_FOR_STAGE_A 10:06:43.781364Z`')
    lines = s.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('最新状态证据：'):
            lines[i] = f'最新状态证据：[18:06:47云终端截图]({rel}{shot})（74,318字节，SHA256 `7ee1a31f2752d5c0f5adb45015aedefd5815c90f3a0dac6ae8766594ce390570`）、[本地UI观察记录]({rel}{obs})。观察记录是网页终端转录，**不是下载的云原状态文件**：StageA27/162、MIT_PINN complete27/27，当前MIT_MLP running0/27；单个训练worker PID19586、PPID19585。确认waiter18969和迁移waiter19580均stdin`/dev/null`、children0、`/proc/maps`无libtorch，分别等待StageA及后续确认状态，未并行训练。迁移v2已排队，187目标训练/140test尚未开始；[本次独立REVIEW]({rel}{review})。192最终pipeline仍仅[终端观察complete]({rel}outputs/full_reproduction/claim_clarification_addenda/192搜索最终pipeline_云终端观察_20261111_174432.json)，原final JSON4823字节尚未下载。'
    s = '\n'.join(lines) + '\n'
    operational = f'''
迁移v2包139,771字节、SHA256 `ce4e51c8244fd7adb26d691c0817d5e7d178b8a0a4650a2cc7fc435d9581b081`，安全解包25成员，`TOOLS_SHA256.json` SHA为`4086e950e098f7b8ca0cfc0e920a6c23e9c2f42e3d0c16fe1d105a22103c44c5`；工具在`/workspace/transfer_robustness_20261111_v2`，输出`/workspace/pinn4soh_transfer_robustness_20261111`。本地[wrapper/collector验收]({rel}outputs/full_reproduction/transfer_robustness_cloud_validation_20261111/ACCEPTANCE.json)与[独立代码审查]({rel}outputs/full_reproduction/transfer_robustness_code_review_20261111/CLOUD_WRAPPER_COLLECTOR_FINAL_REVIEW.json)通过。现代/旧栈GPU探测与真实烟测将在前序60训练+60test+归档全部核验完成后执行，当前未执行；20节有标签validation是额外目标域标注预算，train1/valid20/test22不能称全程仅一节带标签target。[运行边界]({rel}reproduction/TRANSFER_ROBUSTNESS_RUN.md)。

**原论文当前没有已证实的主矩阵缺跑。** 真正剩余为MIT原MAT/单位与采集身份记录、原feature exporter和两个slope、卫星program/cycle精确映射、历史grid/seed/损失/backend与发表checkpoint绑定、原聚合/边界/SD约定、相同i5-10400F条件计时，以及已实测不成立的普遍精度/稳定性/迁移优势等结论，需保留证据或反例。新StageA、60个新seed确认和187迁移是额外改进研究。旧栈StageA60仅是可选的改进跨栈核验，当前不新增；不能把它、未报告的机理/阶数消融或BMS部署写成论文必做缺口。
'''
    start = s.index('最新状态证据：')
    end = s.index('\n', start) + 1
    s = s[:end] + operational + s[end:]
    s = re.sub(r'\n{4,}', '\n\n\n', s)
    p.write_text(s, encoding='utf-8')
print('updated', len(docs))
