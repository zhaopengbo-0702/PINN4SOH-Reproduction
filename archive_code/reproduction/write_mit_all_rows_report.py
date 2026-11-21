"""Write the evidence-bounded fixed-feature MIT supplement conclusion."""
from collections import defaultdict
import json
from pathlib import Path

from audit_mit_mirror import OUTPUT, sha

output = OUTPUT / 'all_rows_supplement_20261110'
summary = json.loads((output / 'MIT_ALL_ROWS_SUMMARY.json').read_text(encoding='utf-8'))
overlap = json.loads((output / 'PRESERVED_SAMPLE_OVERLAP_VERIFIED.json').read_text(encoding='utf-8'))
flagged = json.loads((output / 'FIXED14_RESIDUAL_CASES.json').read_text(encoding='utf-8'))
failed = json.loads((output / 'FAILED_CURVES.json').read_text(encoding='utf-8'))
if not summary['all_published_rows_covered'] or not overlap['full_original_sample_overlap_verified']:
    raise RuntimeError('Do not write final report before both strict checks pass')
cases = defaultdict(list)
for row in flagged['records']:
    identity = (Path(row['csv']).name, row['csv_row_zero_based'], row['cycle'])
    cases[identity].append(row)
out = ['# MIT 全发布行固定特征补充审计', '',
    '125 个发布 CSV 的 85,577 行已全部遍历，共 171,154 条 CC/CV 曲线；171,153 条完成计算，1 条 CC 窗口不足四点。'
    '这覆盖发布特征表的全部行，不代表已计算衍生镜像全部 114,738 个 cycle。', '',
    '作者 MITBatteryClass 阶段选择、原 mirror adapter、既定容量来源映射与 helper 均未修改。三个本地 HDF5 的发布 SHA256 全部核验通过，'
    '每个 CSV 分区的原发布文件 SHA256 和输出文件 SHA256 也已复核。提取采用两组互不重叠的 CSV 流式分区，单 worker 观测 RSS 峰值 '
    + f"{summary['peak_partition_rss_bytes']/1024**2:.1f} MiB。", '',
    '旧 381-cycle 样本原文件 SHA256 保持不变；其 762 条曲线的 5,334 个固定非斜率特征值与本次结果逐数相等。', '',
    '本次只汇总预先固定的 14 个非斜率特征。下表保留实际残差，不把完整覆盖写成所有特征零误差重建。'
    '绝对残差 1e-4 仅用于描述异常记录，不是论文给出的等价判据；各特征单位不同，不能用此阈值进行统一科学通过判断。', '',
    '| 曲线/固定特征 | 有限比较数 | 缺失数 | 最大绝对残差 | 平均绝对残差 | >1e-4 数 |',
    '|---|---:|---:|---:|---:|---:|']
for feature, data in summary['residual_summary'].items():
    out.append(f"| {feature} | {data['finite']} | {data['nonfinite_or_missing']} | {data['max_abs']:.10g} | "
        f"{data['mean_absolute_residual']:.10g} | {data['finite']-data['at_1e_4']} |")
out.extend(['', f"有 {flagged['flagged_curves']} 条曲线、{len(cases)} 个发布行至少一个固定特征绝对残差超过 1e-4。"
    '完整原值、发布值和差值在 FIXED14_RESIDUAL_CASES.json；失败记录在 FAILED_CURVES.json。', '',
    '| 发布 CSV | 行号（从 0 开始） | 映射 cycle | CC 时长残差/s | CV 时长残差/s |',
    '|---|---:|---:|---:|---:|'])
for (name, row, cycle), records in sorted(cases.items()):
    durations = {record['curve']: record['delta_duration_seconds'] for record in records}
    out.append('| ' + name + ' | ' + row + ' | ' + cycle + ' | '
        + (f"{durations['voltage']:.10g}" if 'voltage' in durations else '未触发该标记') + ' | '
        + (f"{durations['current']:.10g}" if 'current' in durations else '未触发该标记') + ' |')
out.extend(['', '唯一短窗口为 2017-05-12_battery-45.csv 第 121 行、条码 EL150800460485、映射 cycle 145 的 CC 曲线，'
    '导致七个固定 CC 特征缺失；同一行的 CV 特征也有实质残差。末尾 April 数据新增的差异已完整保留，未为改善匹配重新选择窗口、映射或统计定义。', '',
    '因此，可确认固定阶段/固定统计定义在绝大多数发布行上获得接近数值精度的对应，但存在上述局部例外；'
    '当前证据不足以宣称全部 14 项严格逐行重建，更不能宣称包含两个斜率的全部 16 个特征已重建。', '',
    '来源仍为第三方衍生镜像 DOI 10.5281/zenodo.23984188，而非本次新取得的原始 MAT。镜像 I/V/Q 为 float32、t 为 float64；'
    '保留原生电流数值，C-rate/A 的物理单位解释尚未解决。原 helper 的六个斜率候选只作原样留档，没有增加候选或按残差选择定义。', '',
    '此次审计不改变冻结训练数据、划分、已运行模型及原论文指标。该镜像容量来源映射带有既定匹配假设；'
    '本补充检验固定映射下的曲线特征对应，不将映射本身重新选择为独立验证。', '',
    '机器证据：MIT_ALL_ROWS_SUMMARY.json、fixed14_residual_summary.csv、FAILED_CURVES.json、'
    'FIXED14_RESIDUAL_CASES.json、PRESERVED_SAMPLE_OVERLAP_VERIFIED.json，以及 125 个 PARTITION_VERIFIED.json。'])
(output / 'MIT全发布行补充审计结论.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
artifacts = [path for path in output.rglob('*') if path.is_file() and not path.name.endswith('.tmp')
             and path.name not in {'SUPPLEMENT_ARTIFACT_SHA256.json'}]
evidence = dict(format='pinn4soh-mit-supplement-artifacts-v1', completed_csvs=125,
    published_rows=85577, curve_records=171154, computed_curves=171153, failed_curves=1,
    flagged_curves_at_descriptive_1e_4=flagged['flagged_curves'], flagged_published_rows=len(cases),
    author_reader_unchanged=True, old_sample_preserved_and_overlap_exact=True,
    full14_exact_reconstruction_established=False, full16_reconstruction_established=False,
    artifacts=[dict(path=path.relative_to(output).as_posix(), bytes=path.stat().st_size, sha256=sha(path))
               for path in sorted(artifacts)])
(output / 'SUPPLEMENT_ARTIFACT_SHA256.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
print(json.dumps({key: evidence[key] for key in evidence if key != 'artifacts'}, ensure_ascii=False))
