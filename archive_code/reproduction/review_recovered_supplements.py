"""Independent local review of recovered MIT40, full-array aggregates and timing."""
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'outputs/full_reproduction/cloud_emergency_recovered/workspace'
OUT=ROOT/'outputs/full_reproduction/final_results_2120/independent_review_complete'
MAIN=REC/'pinn4soh_full_report_20261110'
MIT=REC/'pinn4soh_mit_physical_purity_report_20261110'


def read(path):return json.loads(path.read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(name,records):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in records for k in r)));w.writeheader();w.writerows(records)
def local_folder(folder):
    if not folder.startswith('/workspace/'):raise ValueError('Unexpected original root')
    p=REC/folder.removeprefix('/workspace/')
    # Deep restored task paths exceed legacy Win32 MAX_PATH even though the
    # extracted files exist. Keep evidence unchanged and use extended syntax.
    if os.name=='nt':p=Path('\\\\?\\'+str(p.resolve()))
    if not p.is_dir():raise ValueError('Recovered folder missing: '+str(p))
    return p


def main():
    OUT.mkdir(exist_ok=True)
    main=read(MAIN/'suite_report.json');mit=read(MIT/'suite_report.json')
    assert sha(MAIN/'suite_report.json')=='ebdc9f8a7ddeef608ee93a4b4aa6215b8ad06ac4b26e75d744233419f76d9738'
    assert sha(MIT/'suite_report.json')=='bdae0e33adb376c0eab333ca6550cdec052b866f314d41fbd91ddeefecb44287'
    assert mit['task_status_counts']=={'included':40} and not mit['metadata_only']
    m=[x for x in main['summaries'] if x['family']=='regular' and x['aggregation']=='battery_macro' and x['convention']=='standard']
    s=[x for x in mit['summaries'] if x['aggregation']=='battery_macro' and x['convention']=='standard']
    comparisons=[]
    for x in s:
        y=next(y for y in m if y['dataset']=='MIT' and y['model']==x['model'] and y['loss_variant']==x['loss_variant'])
        for f in ['seeds','repeat_ids','environment_sha256','parameters_sha256','validation_protocol','source_fingerprint']:
            if x[f]!=y[f]:raise ValueError('MIT40/control mismatched '+f)
        for metric in ['MAE','MAPE','RMSE']:
            comparisons.append(dict(model=x['model'],loss_variant=x['loss_variant'],metric=metric,
                main_mean=y[metric+'_mean'],main_sd_ddof1=y[metric+'_sd_ddof1'],
                pure_mean=x[metric+'_mean'],pure_sd_ddof1=x[metric+'_sd_ddof1'],
                pure_minus_main=x[metric+'_mean']-y[metric+'_mean'],
                same_seeds_environment_parameters=True,
                interpretation='Both training and test populations change; not an isolated estimate of contamination bias'))
    # Recompute both main and sensitivity full-array metrics locally through the existing strict auditor.
    sys.path.insert(0,str(ROOT/'reproduction'))
    import build_suite_report as auditor
    local_checks=[]
    for report_folder,doc,scope in [(MAIN,main,'main2120_regular440'),(MIT,mit,'MIT40')]:
        selected=[r for r in rows(report_folder/'per_repeat.csv') if r['evaluation']=='final'
                  and r['aggregation']=='battery_macro' and r['convention']=='standard'
                  and (scope=='MIT40' or r['family']=='regular')]
        for expected in selected:
            folder=local_folder(expected['folder'])
            # inspect_task does strict hash and independent saved-array/CSV-boundary auditing.
            status,repeats,batteries,proof=auditor.inspect_task(folder,10,False)
            actual=next(r for r in repeats if r['evaluation']=='final' and r['aggregation']=='battery_macro' and r['convention']=='standard')
            for metric in ['MAE','MAPE','RMSE']:
                if abs(actual[metric]-float(expected[metric]))>1e-14:raise ValueError('Local full-array metric differs')
            local_checks.append(dict(scope=scope,task_id=expected['task_id'],local_hash_and_array_audit_passed=True))
    # Headline aggregation uses the recovered full test arrays, not downsampled display curves.
    regular=[r for r in rows(MAIN/'per_repeat.csv') if r['family']=='regular' and r['model']=='PINN'
             and r['evaluation']=='final' and r['aggregation']=='battery_macro' and r['convention']=='standard']
    agg=[]
    for lane in ['published_label_monotonicity','paper_prediction_monotonicity']:
        for repeat in range(1,11):
            selected=[r for r in regular if r['loss_variant']==lane and int(r['repeat'])==repeat]
            assert len(selected)==11
            bydataset=defaultdict(list);file_values=[];pool_error=0.;pool_n=0
            for r in selected:
                folder=local_folder(r['folder'])
                truth=np.load(folder/'true_label.npy',allow_pickle=False).astype(np.float64).reshape(-1)
                pred=np.load(folder/'pred_label.npy',allow_pickle=False).astype(np.float64).reshape(-1)
                point=np.abs((truth-pred)/truth)
                pool_error+=float(point.sum());pool_n+=len(point)
                bydataset[r['dataset']].append(float(r['MAPE']))
                split=next(x for x in read(folder/'splits.json') if x['role']=='test')
                start=0
                for battery in split['batteries']:
                    stop=start+battery['n_pairs'];file_values.append(float(point[start:stop].mean()));start=stop
                assert start==len(point)
            values={'eleven_group_equal':statistics.mean(float(r['MAPE']) for r in selected),
                    'four_dataset_equal_within_dataset_group_equal':statistics.mean(statistics.mean(v) for v in bydataset.values()),
                    'all_82_test_CSV_equal':statistics.mean(file_values),'all_test_point_pooled':pool_error/pool_n}
            assert len(file_values)==82
            for name,value in values.items():agg.append(dict(loss_variant=lane,repeat=repeat,aggregation=name,MAPE=value,
                n_groups=11,n_datasets=4,n_test_CSV=82,n_test_points=pool_n,denominator='measured SOH'))
    agg_summary=[]
    for lane in sorted({r['loss_variant'] for r in agg}):
        for name in sorted({r['aggregation'] for r in agg}):
            v=[r['MAPE'] for r in agg if r['loss_variant']==lane and r['aggregation']==name]
            agg_summary.append(dict(loss_variant=lane,aggregation=name,n_repeats=len(v),
                MAPE_mean=statistics.mean(v),MAPE_sd_ddof1=statistics.stdev(v),
                published_headline_fraction=.0087,mean_minus_published=statistics.mean(v)-.0087))
    efficiency=read(REC/'efficiency_formal_20261110/summary.json')
    c=efficiency['config']
    assert efficiency['status']=='complete' and not efficiency['smoke'] and c['forwards']==1000 and c['batch_size']==1000
    assert c['repeats']==5 and c['warmup_forwards']==50 and c['table_s8_iteration_protocol']
    timing=[]
    for x in efficiency['rows']:
        values=x['raw_seconds_per_1000_samples']
        assert len(values)==5 and abs(statistics.mean(values)-x['mean_seconds_per_1000_samples'])<1e-15
        assert abs(statistics.stdev(values)-x['sample_sd_seconds_per_1000_samples'])<1e-15
        timing.append(dict(model=x['model'],threads=x['threads'],parameters=x['parameters'],n_independent_processes=5,
            mean_seconds_per_1000_samples=x['mean_seconds_per_1000_samples'],
            sample_SD_seconds=x['sample_sd_seconds_per_1000_samples'],
            mean_ms_per_1000_samples=x['mean_seconds_per_1000_samples']*1000,
            paper_seconds=x['paper_reference_seconds_per_1000_samples'],
            measured_to_paper_ratio=x['measured_to_paper_time_ratio']))
    write('MIT40_vs_main_MIT_metrics.csv',comparisons);write('local_recovered_array_checks.csv',local_checks)
    write('headline_MAPE_four_aggregations_per_repeat.csv',agg);write('headline_MAPE_four_aggregations_summary.csv',agg_summary)
    write('formal_CPU_efficiency.csv',timing)
    audit={'main_report_sha256':sha(MAIN/'suite_report.json'),'MIT40_report_sha256':sha(MIT/'suite_report.json'),
        'formal_efficiency_sha256':sha(REC/'efficiency_formal_20261110/summary.json'),
        'local_array_checks':len(local_checks),'scope':'440 regular main +40 separate sensitivity; no training',
        'MIT40_not_merged_into_main2120':True,'timing_independent_process_means_and_SDs_recomputed':True,
        'aggregate_uses_full_saved_arrays':True}
    (OUT/'recovered_supplement_review.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    text=['# 本地恢复证据：MIT40、总体MAPE和正式推理计时','',
        '以下只读已恢复的原始数组、报告与计时记录，不连接云端、不重新训练。独立调用严格审计器重验440个主常规任务及40个MIT敏感性任务的文件哈希/完整数组/CSV边界指标，均与冻结报告相符。','',
        '## MIT40独立敏感性','',
        '整文件删除混合May CSV5和CSV6后，101训练/验证CSV与22测试CSV物理身份互斥；保留作者样本split和逐文件全寿命归一化。训练/测试总体都改变，因此差异不能称身份泄漏造成的纯偏差估计。主2120与MIT40在seed、有效参数、数值环境等字段相同；十次重复独立保存，不并入主表。','',
        '|实现|主MIT RMSE±SD (pp)|纯净子集 RMSE±SD (pp)|差值 (pp)|','|---|---:|---:|---:|']
    for x in comparisons:
        if x['metric']=='RMSE':text.append(f"|{x['model']}/{x['loss_variant']}|{x['main_mean']*100:.5f}±{x['main_sd_ddof1']*100:.5f}|{x['pure_mean']*100:.5f}±{x['pure_sd_ddof1']*100:.5f}|{x['pure_minus_main']*100:+.5f}|")
    text+=['','纯净子集两PINN实现的MAE/MAPE/RMSE均高于MLP和CNN；CNN的均值误差最小。该结果不撤销其他数据集的观察，也不量化混合身份对每种方法的影响。','',
        '## 固定standard MAPE的四种总体权重','',
        '先在每个seed内合并11组全部82个测试CSV/完整预测点，再跨十个seed取均值和样本SD。四数据集等权时，每个数据集内部各组等权。文件等权仍是发布文件，不是MIT纯物理电池。','',
        '|实现|权重|MAPE%±SD|','|---|---|---:|']
    for x in agg_summary:text.append(f"|{x['loss_variant']}|{x['aggregation']}|{x['MAPE_mean']*100:.6f}±{x['MAPE_sd_ddof1']*100:.6f}|")
    text+=['','未从四种权重中挑选最接近0.87%的结果；原论文总体权重不明，故不能把0.87%视为已精确复现。','',
        '## 正式CPU计时','',
        'Intel Xeon Gold6230 @2.10GHz、PyTorch1.13.1+cu116；CPU float32 eval/no_grad，batch1000，预热50次，每个独立进程1000次forward取平均，五个独立进程，线程1/2分列。只计F/MLP/CNN前向，不含G/自动微分训练。论文为i5-10400F及Torch1.7.1，硬件软件差异限制直接速度比较。','',
        '|模型|线程|参数|每1000样本毫秒±样本SD|','|---|---:|---:|---:|']
    for x in timing:text.append(f"|{x['model']}|{x['threads']}|{x['parameters']}|{x['mean_seconds_per_1000_samples']*1000:.4f}±{x['sample_SD_seconds']*1000:.4f}|")
    text+=['','参数数量与论文一致；F/MLP结构相同且计时接近，CNN在本机更慢。计时不支持PINN训练学得更快或BMS部署有效的因果结论。','']
    (OUT/'MIT40_总体指标_正式CPU计时_独立复核.md').write_text('\n'.join(text),encoding='utf8')
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
