"""Read-only independent descriptive review of the immutable complete main suite.

This script never changes a training plan, runner, saved result or source report.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'outputs/full_reproduction/final_results_2120/suite_report.json'
EXPECTED_SHA = 'ebdc9f8a7ddeef608ee93a4b4aa6215b8ad06ac4b26e75d744233419f76d9738'
OUT = REPORT.parent / 'independent_review_v2'
LANES = ['published_label_monotonicity', 'paper_prediction_monotonicity']
CONFIGS = [('PINN', LANES[0]), ('PINN', LANES[1]), ('MLP', 'author_current'), ('CNN', 'author_current')]
METRICS = ['MAE', 'MAPE', 'RMSE']
BATCHES = [('XJTU', b) for b in ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']]
BATCHES += [('MIT', 'all'), ('HUST', 'all'), *[('TJU', str(n)) for n in range(3)]]
FIELDS = ['source_fingerprint', 'validation_protocol', 'environment_sha256', 'test_split_sha256', 'repeat_ids', 'seeds']


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def pp(value):
    return f'{value * 100:.4f}'


def main():
    sha = hashlib.sha256(REPORT.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA:
        raise ValueError('Expected the immutable main2120 report')
    report = json.loads(REPORT.read_text(encoding='utf8'))
    if report['metadata_only'] or report['task_status_counts'] != {'included': 2120}:
        raise ValueError('Expected 2120 full-array audited tasks')
    if any(r.get('report_audit_error') for r in report['tasks']):
        raise ValueError('Report contains audit errors')
    primary = [r for r in report['summaries'] if r['aggregation'] == 'battery_macro' and r['convention'] == 'standard']
    if len(primary) != 260 or not all(r['complete_repeat_set'] for r in primary):
        raise ValueError('Expected 260 complete primary summaries')
    lookup = {}
    for r in primary:
        if r['family'] == 'transfer':
            continue
        key = (r['family'], r['dataset'], r['batch'], r['sample_size'], r['model'], r['loss_variant'])
        if key in lookup:
            raise ValueError('Ambiguous nontransfer configuration')
        lookup[key] = r
    cohorts, tables, completeness = {}, defaultdict(list), []
    for family in ['regular', 'small_sample']:
        expected = [(d, b, None) for d, b in BATCHES] if family == 'regular' else [
            (d, b, n) for d, b in BATCHES if d in {'XJTU', 'HUST'} for n in range(1, 5)]
        for d, b, n in expected:
            rows = [lookup[family, d, b, n, m, lane] for m, lane in CONFIGS]
            for field in FIELDS:
                if any(r[field] != rows[0][field] for r in rows):
                    raise ValueError('Unmatched within regular/small cohort: ' + field)
            cohorts[family, d, b, n] = rows
            completeness.append(dict(family=family, dataset=d, batch=b, sample_size=n, arms=4, repeats_per_arm=10,
                                     complete=True, matched_source_environment_test_validation_seeds=True))
            for r in rows:
                tables[family].append({k:r[k] for k in ['family','dataset','batch','paper_group','sample_size','model','loss_variant','comparison_group','n_repeats']}
                    | {m+s:r[m+s] for m in METRICS for s in ['_mean','_sd_ddof0','_sd_ddof1']})
    regular = {k:v for k,v in cohorts.items() if k[0] == 'regular'}
    small = {k:v for k,v in cohorts.items() if k[0] == 'small_sample'}
    assert len(regular) == 11 and len(small) == 28
    saved = {(r['left_comparison_group'], r.get('right_comparison_group'), r.get('metric')):r
        for r in report['comparison_effects'] if r['status'] == 'descriptive_comparison_available'}
    effects, counts, trends = [], [], []
    for key, rows in cohorts.items():
        for left in rows[:2]:
            for right in rows[2:]:
                for metric in METRICS:
                    delta = left[metric+'_mean'] - right[metric+'_mean']
                    if abs(saved[left['comparison_group'], right['comparison_group'], metric]['left_minus_right'] - delta) > 1e-15:
                        raise ValueError('Independent comparator difference disagrees with report')
                    effects.append(dict(family=key[0], dataset=key[1], batch=key[2], sample_size=key[3],
                        loss_variant=left['loss_variant'], comparator=right['model'], metric=metric,
                        left_mean=left[metric+'_mean'], right_mean=right[metric+'_mean'], left_minus_right=delta,
                        direction='lower' if delta < -1e-12 else 'higher' if delta > 1e-12 else 'tie'))
    for family, collection in [('regular',regular), ('small_sample',small)]:
        for lane in LANES:
            for metric in METRICS:
                c = Counter()
                for rows in collection.values():
                    left = next(r for r in rows if r['loss_variant'] == lane)
                    c['lower_both_mean'] += all(left[metric+'_mean'] < r[metric+'_mean'] for r in rows[2:])
                    c['lower_MLP_mean'] += left[metric+'_mean'] < rows[2][metric+'_mean']
                    c['lower_CNN_mean'] += left[metric+'_mean'] < rows[3][metric+'_mean']
                    for ddof in [0,1]:
                        c[f'lower_both_SD_ddof{ddof}'] += all(left[f'{metric}_sd_ddof{ddof}'] < r[f'{metric}_sd_ddof{ddof}'] for r in rows[2:])
                counts.append(dict(family=family,loss_variant=lane,metric=metric,cohorts=len(collection),**c))
    for d,b in [(d,b) for d,b in BATCHES if d in {'XJTU','HUST'}]:
        for m,lane in CONFIGS:
            rows = [lookup['small_sample',d,b,n,m,lane] for n in range(1,5)]
            for metric in METRICS:
                v=[r[metric+'_mean'] for r in rows]
                trends.append(dict(dataset=d,batch=b,model=m,loss_variant=lane,metric=metric,
                    **{f'n{n}':x for n,x in enumerate(v,1)}, n4_minus_n1=v[3]-v[0],n4_lower_n1=v[3]<v[0],
                    every_step_lower=all(y<x for x,y in zip(v[:-1],v[1:])),
                    increases=';'.join(f'n{i+1}->n{i+2}' for i,(x,y) in enumerate(zip(v[:-1],v[1:])) if y>x)))
    one_vs_more=[]
    for d,b in [('XJTU','2C'),('HUST','all')]:
        for lane in LANES:
            left=lookup['small_sample',d,b,1,'PINN',lane]
            for n in [3,4]:
                for m in ['MLP','CNN']:
                    right=lookup['small_sample',d,b,n,m,'author_current']
                    one_vs_more.append(dict(dataset=d,batch=b,loss_variant=lane,comparator=m,baseline_n=n,
                        PINN_n1_RMSE=left['RMSE_mean'],baseline_RMSE=right['RMSE_mean'],
                        left_minus_right=left['RMSE_mean']-right['RMSE_mean'],
                        lower_or_equal=left['RMSE_mean']<=right['RMSE_mean'],equivalence_margin=None))
    transfer=[r for r in report['comparison_effects'] if r['kind']=='fine_tune_vs_source_only']
    assert len(transfer)==48 and all(r['status']=='descriptive_comparison_available' for r in transfer)
    transfer_counts=[]
    for lane in LANES:
        for mode in ['reproduced_checkpoint','supplied_author_checkpoint']:
            rows=[r for r in transfer if r['loss_variant']==lane and r['source_mode']==mode]
            transfer_counts.append(dict(loss_variant=lane,source_mode=mode,directions=len(rows),
                mean_improved=sum(r['left_minus_right']<0 for r in rows),
                individual_paired_repeats_improved=sum(r['paired_repeats_improved'] for r in rows),
                individual_paired_repeats=sum(r['paired_repeats'] for r in rows),
                all_fixed_G_updated_F_verified=all(r['frozen_G_and_updated_F_verified'] for r in rows)))
    cohort_fields=['dataset','paper_group','protocol_id','validation_protocol','source_fingerprint','environment_sha256','test_split_sha256']
    mismatch=[]
    for left in [r for r in primary if r['family']=='transfer' and r['evaluation']=='final']:
        for n in [1,2]:
            candidates=[r for r in primary if r['family']=='small_sample' and r['model']=='PINN' and r['sample_size']==n
                and r['loss_variant']==left['loss_variant'] and all(r[f]==left[f] for f in cohort_fields if f!='environment_sha256')]
            if len(candidates)!=1:
                raise ValueError('Expected unique diagnostic target candidate after excluding environment hash')
            right=candidates[0]
            mismatched=[f for f in cohort_fields+['repeat_ids','seeds'] if left[f]!=right[f]]
            mismatch.append(dict(source=left['source'],target=left['paper_group'],source_mode=left['source_mode'],
                loss_variant=left['loss_variant'],target_only_n=n,left_comparison_group=left['comparison_group'],
                right_comparison_group=right['comparison_group'],mismatched_fields=';'.join(mismatched),
                left_environment_sha256=left['environment_sha256'],right_environment_sha256=right['environment_sha256'],
                strict_comparison='unmatched',training_path_match_proof='not_reviewed_without_payload',
                fine_tune_RMSE=left['RMSE_mean'],target_only_RMSE=right['RMSE_mean'],
                difference_descriptive_only=left['RMSE_mean']-right['RMSE_mean'],
                warning='Different schemas can change environment hash; actual payload comparison required before numerical-equivalence claim'))
    numeric=[r for r in report['published_comparisons'] if r['aggregation']=='battery_macro' and r['convention']=='standard']
    numeric_counts=[]
    for table in sorted(set(r['table'] for r in numeric)):
        for lane in [*LANES,'author_current']:
            rows=[r for r in numeric if r['table']==table and r['loss_variant']==lane]
            if rows:numeric_counts.append(dict(table=table,loss_variant=lane,comparison_count=len(rows),
                printed_precision_matches=sum(bool(r['within_printed_4dp_half_unit']) for r in rows)))
    aggregate=[]
    for lane in LANES:
        rows=[r for r in primary if r['family']=='regular' and r['model']=='PINN' and r['loss_variant']==lane]
        bydataset=defaultdict(list)
        for r in rows:bydataset[r['dataset']].append(r['MAPE_mean'])
        aggregate += [dict(loss_variant=lane,aggregation='eleven_group_equal',MAPE=statistics.mean(r['MAPE_mean'] for r in rows)),
            dict(loss_variant=lane,aggregation='four_dataset_equal_within_dataset_group_equal',
                 MAPE=statistics.mean(statistics.mean(v) for v in bydataset.values()))]
    OUT.mkdir(parents=True,exist_ok=True)
    for name,rows in [('regular_metrics.csv',tables['regular']),('small_sample_metrics.csv',tables['small_sample']),
        ('cohort_completeness.csv',completeness),('matched_effects.csv',effects),('direction_counts.csv',counts),
        ('sample_size_trends.csv',trends),('one_battery_vs_more_baseline.csv',one_vs_more),
        ('transfer_paired_effects.csv',transfer),('transfer_direction_counts.csv',transfer_counts),
        ('target_only_strict_mismatch_diagnostic.csv',mismatch),('published_primary_comparisons.csv',numeric),
        ('published_numeric_match_counts.csv',numeric_counts),('headline_MAPE_partial_aggregations.csv',aggregate)]:
        write_csv(name,rows)
    review=dict(source_report_sha256=sha,source_report_bytes=REPORT.stat().st_size,task_status_counts=report['task_status_counts'],
        independently_checked_summary_effect_differences=len(effects),regular_cohorts=11,small_sample_cohorts=28,
        primary_complete_summaries=260,main_task_breakdown={'regular':440,'small_sample_XJTU_HUST':1120,
            'extra_target_only_MIT_TJU3':80,'transfer':480},direction_counts=counts,transfer_direction_counts=transfer_counts,
        target_only_strict_mismatch_count=len(mismatch),
        target_only_mismatched_field_counts=dict(Counter(r['mismatched_fields'] for r in mismatch)),
        printed_numeric_match_counts=numeric_counts,headline_partial_aggregations=aggregate,
        limitations=['No significance or equivalence claim; no unreported author seeds recovered',
            'battery_macro is released CSV-file macro, not guaranteed physical-cell macro (MIT mixed CSV5)',
            'Original frozen report unchanged; strict target-only comparator is unmatched pending environment payload review',
            'Local independent review recomputes summary differences; saved-array/checkpoint audit occurred on cloud',
            'MIT40 is separate sensitivity and excluded from every main2120 count',
            'Per-CSV and displayed-curve payload is pending; source-report completeness is not full local model backup'])
    (OUT/'independent_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    lines=['# 完整 2120 任务独立结果复核','',
        f'固定报告 SHA-256：`{sha}`，大小 {REPORT.stat().st_size:,} 字节。云端与本地 SHA 已由根代理核对一致。本脚本只读，不修改训练任务、模型、原始数组或冻结报告。','',
        '主套件 2120/2120 纳入，0 报告审计错误：常规 440，XJTU/HUST 小样本 1120，MIT/TJU3 补充 target-only 80，迁移 480。固定主约定为 standard measured-SOH MAPE 和发布 CSV 文件宏平均；260 个主约定汇总均有十个不同 seed/repeat。RMSE×100 的单位为 SOH 百分点（pp）。MIT 混合 CSV5 含两个物理身份，CSV 宏平均不等于物理电池宏平均。','',
        '## 常规及小样本比较','',
        '|范围|PINN实现|指标|低于两基线的组数|低于MLP|低于CNN|样本SD低于两基线|',
        '|---|---|---|---:|---:|---:|---:|']
    for r in counts:lines.append(f"|{r['family']}|{r['loss_variant']}|{r['metric']}|{r['lower_both_mean']}/{r['cohorts']}|{r['lower_MLP_mean']}|{r['lower_CNN_mean']}|{r['lower_both_SD_ddof1']}|")
    lines += ['', '标签单调项只依赖真实标签，对模型参数梯度恒为零；该实现的预测优势不能证明单调物理惩罚的因果作用。预测单调实现的较低 seed SD 可能伴随更高平均误差；稳定与准确需分别判断。所有均值差已独立重算并核对冻结报告的匹配比较。','',
        '|常规组|标签单调PINN RMSE±样本SD (pp)|预测单调PINN|MLP|CNN|',
        '|---|---:|---:|---:|---:|']
    for d,b in BATCHES:
        rows=regular['regular',d,b,None]
        lines.append(f'|{d}/{b}|'+'|'.join(f"{pp(r['RMSE_mean'])}±{pp(r['RMSE_sd_ddof1'])}" for r in rows)+'|')
    lines += ['', '|小样本组|n|标签单调PINN RMSE (pp)|预测单调PINN|MLP|CNN|','|---|---:|---:|---:|---:|---:|']
    for k,rows in small.items():lines.append(f'|{k[1]}/{k[2]}|{k[3]}|'+'|'.join(pp(r['RMSE_mean']) for r in rows)+'|')
    lines += ['', '## 电池数与迁移结果','',
        '样本量趋势保留每一个 n1..4 点及所有相邻回升，见 sample_size_trends.csv。仅首末下降不能写成逐步严格改善。主图 XJTU1/HUST 的 PINN n1 对基线 n3/n4 比较见 one_battery_vs_more_baseline.csv；“接近”没有预设等效界值，不能据胜负计数证明统计等效。','',
        '|PINN实现|源模型|平均RMSE改善方向|配对单次改善|冻结G且更新F|','|---|---|---:|---:|---|']
    for r in transfer_counts:lines.append(f"|{r['loss_variant']}|{r['source_mode']}|{r['mean_improved']}/{r['directions']}|{r['individual_paired_repeats_improved']}/{r['individual_paired_repeats']}|{r['all_fixed_G_updated_F_verified']}|")
    lines += ['', '|PINN实现|源模型|方向|微调RMSE (pp)|对应source-only (pp)|十次改善数|','|---|---|---|---:|---:|---:|']
    for r in sorted(transfer,key=lambda r:(r['loss_variant'],r['source_mode'],r['paper_group'],r['source'])):
        lines.append(f"|{r['loss_variant']}|{r['source_mode']}|{r['source']}→{r['paper_group']}|{pp(r['left_mean'])}|{pp(r['right_mean'])}|{r['paired_repeats_improved']}|")
    lines += ['',
        'target-only n1/n2 的 96 个候选比较被冻结报告严格标为未匹配。独立诊断显示，排除 environment_sha256 后每个候选唯一，dataset/paper_group/protocol/validation/source/test split/repeat/seed 相符；环境哈希仍不同。runner 的环境字段 schema 不同（sklearn/scikit_learn、cuda_runtime/cuda、额外 scipy/threads）可能造成哈希差异，不能把不同哈希直接说成数值版本真的不同，也不能未经实际 config 核对就放宽。目标训练文件路径证明亦待 payload 复核。因此 C13/C14 暂不据此判断迁移优于 target-only。诊断表中的原始均值只作描述。','',
        '## 论文数值与证据边界','',
        '|论文表|实现|对照数|四位小数打印精度匹配数|','|---|---|---:|---:|']
    for r in numeric_counts:lines.append(f"|{r['table']}|{r['loss_variant']}|{r['comparison_count']}|{r['printed_precision_matches']}|")
    lines += ['', '数值匹配严格固定主约定，阈值为绝对差≤0.00005；未从 reversed MAPE 或 pooled 结果中选择最接近论文的一项。零星打印精度相符不等于论文训练协议完全一致。Source Data 的论文值只作参考。','',
        '0.87% 总体 MAPE 的原始权重不明。当前可从主报告求十一组等权与四数据集等权（数据集内各组等权）；逐CSV/全测试样本权重还需 payload，不能混称同一个总体指标。','']
    for r in aggregate:lines.append(f"- {r['loss_variant']} / {r['aggregation']}: MAPE {r['MAPE']*100:.6f}%。")
    lines += ['', '此独立复核不把 GPU 调度吞吐量当作模型学习更快的证据，不推断显著性、等效性、电化学可辨识性或真实 BMS 部署有效性。MIT40 物理身份纯净子集是独立敏感性分析，没有计入主 2120。完整逐CSV及曲线本地证据、云端模型备份和最终图表映射分别保留，不以报告完成代替原始模型归档。','']
    (OUT/'完整2120任务_独立数值复核.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({'output':str(OUT),'main2120':2120,'regular_cohorts':11,'small_cohorts':28,
        'independent_effect_checks':len(effects),'target_only_strict_mismatches':len(mismatch)},ensure_ascii=False))


if __name__ == '__main__':
    main()
