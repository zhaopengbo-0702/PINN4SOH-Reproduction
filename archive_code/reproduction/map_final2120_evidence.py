"""Save an explicit, conservative 25 figure/table by 21 claim evidence map."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/full_reproduction'
OUT=BASE/'final_results_2120/independent_review_v2'
PLOT='outputs/full_reproduction/plot_final2120_v3_report_only'
FEATURE='outputs/full_reproduction/feature_audit'


def dump_csv(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def main():
    protocol=json.loads((BASE/'paper_protocol.json').read_text(encoding='utf8'))
    rules=json.loads((ROOT/'reproduction/suite_evidence_rules.json').read_text(encoding='utf8'))
    report=json.loads((BASE/'final_results_2120/suite_report.json').read_text(encoding='utf8'))
    # Each state states exactly what was rendered or audited; a schematic is not a laboratory replication.
    specs={
      'Fig. 1':(['C03','C17','C20'], 'partial_source_and_code_support',
        [f'{PLOT}/schematics/Fig1_computational_framework.mmd',f'{FEATURE}/raw_curve_figures/figure1b_raw_window_example.png'],
        '计算框架示意和预先固定XJTU电池/物理周期的原始充电窗口已生成；示意不验证电化学机制。'),
      'Fig. 2':(['C02'], 'released_data_replot_complete',
        [f'{FEATURE}/processed_figures/figure2_released_capacity.png',f'{FEATURE}/processed_figures/battery_inventory.csv'],
        '55条发布容量轨迹、六批次齐全；横轴为发布行序，不能全部称独立测得的物理周期容量。'),
      'Fig. 3a':(['C03'], 'released_features_replot_complete_raw_partial',
        [f'{FEATURE}/processed_figures/figure3a_released_features.png',f'{FEATURE}/raw_fixed_convention_feature_summary.csv'],
        '八个XJTU1发布CSV的16特征已清洗缩放重绘；原始数据14非斜率特征完成，两个斜率未恢复。'),
      'Fig. 3b':(['C04'], 'rounded_numeric_match_176_of_176',
        [f'{FEATURE}/processed_figures/figure3b_author_cleaned_per_battery_minmax_pooled.png',f'{FEATURE}/processed_figures/pearson_author_cleaned_per_battery_minmax_pooled.csv',f'{FEATURE}/processed_figures/fig3b_paper_rounded_comparison.csv'],
        '作者清洗→逐CSV minmax→按组池化得到176/176个两位小数匹配；这是相关描述，不能识别化学因果。'),
      'Fig. 4a':(['C05'], 'locally_pending_curve_payload',
        [f'{PLOT}/figures/Fig4a_published.png',f'{PLOT}/figures/Fig4a_paperloss.png'],
        '主报告完整但该目录没有per_repeat/预测曲线payload；图明确pending，不能将论文SourceData轨迹当新训练。'),
      'Fig. 4b':(['C05','C06'], 'complete_cloud_training_summary_rendered',
        [f'{PLOT}/figures/Fig4b_published.png',f'{PLOT}/figures/Fig4b_paperloss.png'],
        '11组×4方法损失臂×10重复，MAE/MAPE/RMSE均值与样本SD；两PINN实现分别呈现，共享MLP/CNN。'),
      'Fig. 5':(['C08','C09','C10'], 'complete_cloud_training_summary_rendered',
        [f'{PLOT}/figures/Fig5_published.png',f'{PLOT}/figures/Fig5_paperloss.png'],
        'XJTU1/HUST n1..4完整；显示相邻回升，不把1电池与3/4电池结果声明为等效。'),
      'Fig. 6':(['C12','C18'], 'framework_and_transfer_proof_available',
        [f'{PLOT}/schematics/Fig6_transfer_framework.mmd','outputs/full_reproduction/final_results_2120/independent_review_v2/transfer_paired_effects.csv'],
        '48完整迁移组均有冻结G哈希相同及F更新证明；改善方向依赖损失与源权重，不证明电化学可辨识。'),
      'Table 1':(['C01'], 'released_inventory_count_match_raw_provenance_qualified',
        [f'{FEATURE}/processed_figures/processed_feature_audit.json',f'{FEATURE}/official_raw_inventory.json',f'{FEATURE}/MIT_mirror_audit/MIT报告可用注释.md'],
        '387发布CSV和310705原始行匹配；262原始数据链通过，MIT为派生float32镜像且混合CSV，不能称387独立物理电池已证实。'),
      'Table 2':(['C05','C16'], 'complete_cloud_training_summary_rendered',
        [f'{PLOT}/tables/Table_2_published.png',f'{PLOT}/tables/Table_2_paperloss.png',f'{PLOT}/tables/Table_2_published_difference.png'],
        '固定standard/CSV宏平均；标签PINN22个数值无打印精度匹配，基线44个仅1个；完整差值逐项保留。'),
      'Table 3':(['C08','C09','C10'], 'complete_cloud_training_summary_rendered',
        [f'{PLOT}/tables/Table_3_published.png',f'{PLOT}/tables/Table_3_paperloss.png'],
        '主小样本8组×4臂×10；保留论文HUST n1 CNN MAPE .3614，标签PINN16个主比较仅1个打印精度匹配。'),
      'Table 4':(['C12','C13','C14','C15','C18'], 'transfer_summaries_complete_target_only_strict_match_pending',
        [f'{PLOT}/tables/Table_4_published_reproduced_checkpoint_final.png',f'{PLOT}/tables/Table_4_published_supplied_author_checkpoint_final.png','outputs/full_reproduction/final_results_2120/independent_review_v2/target_only_strict_mismatch_diagnostic.csv'],
        '12方向×2损失×2源模式×10微调及配对source-only完整。n1/n2数值可描述，但96严格比较因环境schema哈希不同尚未匹配。'),
      'Fig. S1':(['C01','C02'], 'external_laboratory_evidence_only',
        ['outputs/full_reproduction/paper_protocol.json'],
        '原实验装置只核文献来源；云端代码不能复现实验室硬件或重新采集原始电池。'),
      'Fig. S2':(['C02','C03'], 'raw_protocol_illustration_partial',
        [f'{FEATURE}/raw_curve_figures/RAW_CURVE_FIGURES.json','outputs/full_reproduction/paper_protocol.json'],
        '原始物理曲线与六批协议有来源核对；未另绘与论文S2逐点相同的实验图。'),
      'Fig. S3':(['C02','C03'], 'raw_protocol_illustration_partial',
        [f'{FEATURE}/raw_curve_figures/RAW_CURVE_FIGURES.json','outputs/full_reproduction/paper_protocol.json'],
        '同上，原始曲线说明协议，不声明独立复现电池循环实验。'),
      'Fig. S4':(['C02','C03'], 'source_grounded_raw_curves_rendered_with_interpolation_caveat',
        [f'{FEATURE}/raw_curve_figures/figureS4_raw_six_batch_curves.png',f'{FEATURE}/raw_curve_figures/figureS4_capacity_source.csv',f'{FEATURE}/raw_curve_figures/CAPTIONS.md'],
        '六批首电池原始充放电和容量来源重绘；批3Note1/原S4图注不一致、无周期容量测试插值问题明确保留。'),
      'Fig. S5':(['C08','C09','C11'], 'complete_cloud_training_summary_rendered',
        [f'{PLOT}/figures/FigS5_published.png',f'{PLOT}/figures/FigS5_paperloss.png'],
        '六XJTU批×四样本量×四臂×十重复=960任务，batch1与Fig5重叠只算一次；作者三模型口径为720。'),
      'Table S1':(['C02'], 'published_protocol_source_only_not_new_timing_experiment',
        ['outputs/full_reproduction/paper_protocol.json'],
        'satellite放电时间协议已转录来源；未声明逐记录原始时间重新恢复整个S1。'),
      'Table S2':(['C05','C11','C19'], 'published_ids_and_cloud_audit_available_local_payload_pending',
        ['outputs/full_reproduction/paper_protocol.json','outputs/validation_overlap_43_20261110/summary.json'],
        '11组论文测试ID/数量已转录，实际tensor成员审计43组完成；逐CSV最终payload本地复核仍待到达。'),
      'Table S3':(['C05','C06','C16'], 'locally_pending_per_csv_payload',
        [f'{PLOT}/tables/Table_S3_published_1.png'],
        '论文逐CSV参考表存在；当前report-only没有per_battery_summaries，不能用组均值填逐文件值。'),
      'Table S4':(['C05','C06','C16'], 'locally_pending_per_csv_payload',
        [f'{PLOT}/tables/Table_S4_published_1.png'],
        'TJU逐文件十次均值待最终payload，不把论文参考值当实际复现。'),
      'Table S5':(['C05','C06','C16'], 'locally_pending_per_csv_payload',
        [f'{PLOT}/tables/Table_S5_published_1.png'],
        'HUST逐文件十次均值待最终payload。'),
      'Table S6':(['C05','C06','C16','C19'], 'locally_pending_per_csv_payload_mit_identity_qualified',
        [f'{PLOT}/tables/Table_S6_published_1.png',f'{FEATURE}/MIT_mirror_audit/MIT报告可用注释.md'],
        'MIT逐文件均值待payload；May CSV5混合物理身份，发布文件指标不得改称独立物理电池指标。'),
      'Table S7':(['C08','C09','C11'], 'complete_cloud_training_summary_rendered',
        [f'{PLOT}/tables/Table_S7_published_1.png',f'{PLOT}/tables/Table_S7_paperloss_1.png'],
        '六批n1..4完整，批1与主表重叠；标签PINN48个比较中2个四位小数匹配，数值偏差保留。'),
      'Table S8':(['C07'], 'parameter_audit_available_formal_timing_pending_locally',
        ['outputs/efficiency_smoke_local_20261110/summary.json'],
        '本地smoke核7781/7781/8465参数，但只有5次forward不是正式1000次；正式云端计时文件待本地导出，不能冒称论文速度重现。'),
    }
    assert len(specs)==25 and set(specs)=={x['id'] for x in rules['figure_table_inventory']}
    figure_rows=[]
    for item in rules['figure_table_inventory']:
        claims,state,paths,note=specs[item['id']]
        # Pending per-CSV page labels may vary; resolve the actual saved filename only.
        actual=[]
        for path in paths:
            p=ROOT/path
            if p.exists():actual.append(path)
            elif p.parent.exists() and p.name.startswith('Table_S'):
                stem=p.name.rsplit('_',1)[0]
                found=sorted(p.parent.glob(stem+'*.png'))
                actual.extend(x.relative_to(ROOT).as_posix() for x in found[:1])
            else:actual.append(path+' [not_yet_locally_present]')
        figure_rows.append(dict(figure_table=item['id'],scope=item['scope'],claim_ids=';'.join(claims),status=state,
            artifacts=';'.join(actual),interpretation=note))
    verdicts={
      'C01':('count_match_with_identity_and_provenance_limits','387发布CSV、310705原始行确实相符；清洗后303952行/303565相邻对。MIT125文件不是已证明的125纯净物理电池，原始MAT未取得。'),
      'C02':('descriptive_support','55发布轨迹映射六组并重绘；原始曲线说明容量测试/插值边界，六批均为独立测量完整容量的说法不成立。'),
      'C03':('partial_support_full_feature_reconstruction_unresolved','262原始电池225128发布行的14个非斜率有限特征按固定约定全匹配；MIT381镜像周期抽查支持14特征精度，两斜率和MIT原MAT/电流单位仍未解决。'),
      'C04':('rounded_descriptive_replication_not_causal','Fig3b176/176打印两位小数相符；化学体系比协议更重要仅为有混杂的描述，不构成因果识别。'),
      'C05':('implementation_dependent_mixed_results','标签单调PINN每个指标9/11低于两基线，合27/33；MIT/HUST均未优于任一基线。预测单调为0/33，不能合并成同一PINN结论。'),
      'C06':('descriptive_seed_SD_support_qualified','标签单调31/33组指标样本SD低于两基线，预测单调33/33；ddof0方向相同。较小SD不说明平均误差较小或统计显著。'),
      'C07':('parameter_counts_supported_formal_timing_not_locally_verified','F与MLP均7781，CNN8465。已见本地计时只是smoke、非1000次正式基准；云端正式文件待导出，硬件/软件也异于论文。'),
      'C08':('broad_best_in_all_contradicted_results_mixed','标签单调MAPE/RMSE各25/28同时优于两基线，RWn1/HUSTn1/HUSTn4输MLP但全28均胜CNN；预测单调各5/28胜两基线。'),
      'C09':('general_endpoint_trend_support_not_strict_monotonicity','标签单调RMSE7/7 n4低于n1、5/7严格逐步下降；预测单调4/7首末下降、1/7逐步下降；MLP/CNN14/14首末下降、8/14逐步下降。'),
      'C10':('direction_specific_no_equivalence_claim','XJTU1/HUST主图合8个n1对n3/n4基线RMSE比较，标签单调4/8胜（XJTU全胜、HUST全输），预测单调2/8胜；无预设等效界值。'),
      'C11':('execution_complete_exact_local_test_id_payload_pending','六批小样本三模型口径720运行已齐，另预测单调240运行；与主图batch1不重复计。完整split与逐CSVpayload待本地复核。'),
      'C12':('source_mode_and_loss_dependent_support','复现源权重两损失均12/12方向平均改善，单次改善标签111/120、预测120/120。作者源权重标签11/12（111/120）、预测8/12（85/120）；48组均冻结G并更新F。'),
      'C13':('strict_comparison_unmatched_pending_environment_payload','48个n1候选在环境哈希上不符，已定位runner环境schema差异的可能来源。不得未核实际版本/线程/目标训练路径而宣布优于target-only。'),
      'C14':('strict_comparison_unmatched_pending_environment_payload','所有n2比较仍严格未匹配；XJTU1/TJU3相关24方向损失源模式组合只能列原始数值，不能宣布因果或公平优越。'),
      'C15':('exploratory_only','完整迁移矩阵和化学标签可用于描述；MIT/HUST同属LFP但目标难度、源误差与预训练源混杂，不能推断化学相同保证迁移更好。'),
      'C16':('headline_not_reproduced_under_available_explicit_weights','标签单调十一组等权MAPE1.029010%、四数据集等权0.988243%；预测单调4.669701%/4.946709%。均非0.87%；文件等权/样本池化待payload。'),
      'C17':('implementation_discrepancy_verified_no_loss_ablation_claim','原作者当前损失relu((u2-u1)*(y1-y2))、发布标签损失relu(y2-y1)、预测单调relu(u2-u1)不同；标签项参数梯度恒零。只比较实现不能证明物理项贡献。'),
      'C18':('reuse_usefulness_partial_not_electrochemical_identifiability','48组G不变/F更新证据均通过；source-only改善依源模式/损失，支持部分可复用预测结构，不识别真实电化学机制。'),
      'C19':('universal_no_leakage_claim_not_supported','43组样本split保持tensor且验证x1标签68.50%–84.47%作为训练x2被PINN直接使用；CSV级独立测试不等于MIT物理身份独立（1256/14170测试点身份见过训练）。另每整电池全寿命minmax含测试未来极值。未量化各项性能偏差。'),
      'C20':('first_order_source_verified_superiority_unavailable','源码确用u_t/u_x一阶导，未做导数阶数消融；准确率/复杂度折中更优不能由代码存在推得。'),
      'C21':('broad_causal_speed_and_BMS_claim_unavailable','精度优势依实现/任务而变，标签单调项零梯度；无匹配收敛时间实验或BMS部署证据。调度效率不能当模型学得更快。'),
    }
    claim_rows=[]
    for c in protocol['claims']:
        state,note=verdicts[c['id']]
        figs=[f['figure_table'] for f in figure_rows if c['id'] in f['claim_ids'].split(';')]
        claim_rows.append(dict(claim=c['id'],published_statement=c['statement'],status=state,
            evidence_figures_tables=';'.join(figs),independent_conclusion=note,
            decision_rule=next(x['decision_rule'] for x in rules['claim_rules'] if x['id']==c['id'])))
    matrix=[{'figure_table':f['figure_table'],**{c['id']:'relevant_evidence' if c['id'] in f['claim_ids'].split(';') else '' for c in protocol['claims']}} for f in figure_rows]
    long=[dict(figure_table=f['figure_table'],claim=c['id'],relevant=c['id'] in f['claim_ids'].split(';'),
        artifact_status=f['status'],claim_status=verdicts[c['id']][0]) for f in figure_rows for c in protocol['claims']]
    assert len(long)==525
    chemistry=[]
    for lane in ['published_label_monotonicity','paper_prediction_monotonicity']:
        for mode in ['reproduced_checkpoint','supplied_author_checkpoint']:
            for same in [True,False]:
                rows=[x for x in report['comparison_effects'] if x['kind']=='fine_tune_vs_source_only' and x['loss_variant']==lane
                      and x['source_mode']==mode and ({x['source'],x['paper_group']}=={'MIT','HUST'})==same]
                chemistry.append(dict(loss_variant=lane,source_mode=mode,class_label='MIT_HUST_both_LFP' if same else 'other_chemistry_pairs',
                    directions=len(rows),mean_RMSE_improved=sum(x['left_minus_right']<0 for x in rows),
                    direction_equal_mean_relative_RMSE_reduction=statistics.mean(1-x['left_mean']/x['right_mean'] for x in rows),
                    interpretation='Exploratory, unadjusted for target difficulty and source-only error; no causal inference'))
    dump_csv('figure_table_evidence_map_25.csv',figure_rows);dump_csv('claim_decisions_21.csv',claim_rows)
    dump_csv('figure25_claim21_matrix.csv',matrix);dump_csv('figure25_claim21_long_525.csv',long)
    dump_csv('transfer_chemistry_exploratory.csv',chemistry)
    lines=['# 25 项图表 × 21 项结论证据映射','',
      '本映射针对主 2120 任务固定报告，并结合已落地特征/物理身份/验证成员审计。训练完成、图已渲染、论文数值打印匹配和科学结论支持是四种不同状态。MIT40 不计入此主表。论文 Source Data 仅作参考，图中的实际结果来自独立训练报告；原始实验装置和未提供导出器的部分不冒称已重做。','',
      '当前图表目录为 v3 report-only：Table4选择显式sample_size=1修复v2把完整迁移格误列pending的问题；原v1/v2、训练数据和suite_report保持原样。逐CSV与曲线payload尚未进入此目录，Fig4a/S3–S6因此保留pending。原报告C13/C14严格环境匹配缺口同样保留。','',
      '|图表|关联结论|已落地证据与边界|','|---|---|---|']
    for r in figure_rows:lines.append(f"|{r['figure_table']}|{r['claim_ids']}|{r['interpretation']}|")
    lines += ['', '|结论|裁定|独立结论与限制|','|---|---|---|']
    for r in claim_rows:lines.append(f"|{r['claim']}|{r['status']}|{r['independent_conclusion']}|")
    lines += ['', '所有525个图表-结论组合见 figure25_claim21_long_525.csv；宽矩阵见 figure25_claim21_matrix.csv。有关联表示提供该结论的某部分证据，不表示支持该结论；对应状态与限制必须一起阅读。逐项实际文件路径见 figure_table_evidence_map_25.csv。','']
    (OUT/'25图表_21结论_逐项证据映射.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({'figure_table_count':len(figure_rows),'claim_count':len(claim_rows),'matrix_pairs':len(long),
        'output':str(OUT),'missing_artifact_paths':[p for r in figure_rows for p in r['artifacts'].split(';') if '[not_yet_locally_present]' in p]},ensure_ascii=False))


if __name__=='__main__':main()
