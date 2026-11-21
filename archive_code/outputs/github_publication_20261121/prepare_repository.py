"""Prepare the Chinese, browsable publication without altering frozen evidence."""
from pathlib import Path
import os, shutil, hashlib, json

BASE = Path('\\\\?\\' + os.getcwd())
PUB = BASE / 'outputs/github_publication_20261121'
DEST = PUB / 'repository'
DEST.mkdir(parents=True, exist_ok=True)
copied = []

def copy_file(src, dst=None):
    source = BASE / src
    target = DEST / (dst or src)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    copied.append({'path': str(target.relative_to(DEST)).replace('\\', '/'),
                   'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bytes':target.stat().st_size})

def tree(src, limit=None, suffixes=None):
    source = BASE / src
    for p in sorted(source.rglob('*')):
        if not p.is_file() or any(x in p.parts for x in ['__pycache__','.git']): continue
        if limit is not None and p.stat().st_size > limit: continue
        if suffixes and p.suffix.lower() not in suffixes: continue
        copy_file(str(p.relative_to(BASE)))

F = 'outputs/full_reproduction/'
for name in ['final_results_2120','plot_final2120_complete_v4','regular_per_CSV_curves_repeat1_v4',
             'current_repo_defaults_reaudited_20261111','current_repo_defaults_cpu_reload_20261111',
             'current_repo_defaults_cpu_reload_independent_review_20261111',
             'environment_schema_supplement_final','satellite_protocol_supplement_20261110_final',
             'replacement_feature_validation_20261111','plans','source_data']:
    tree(F+name, limit=12*1024*1024, suffixes={'.py','.md','.json','.csv','.png','.pdf','.txt','.npy'})
for name in ['legacy_extension_recovered_20261111','replacement_search_recovered_20261111']:
    p=BASE/F/name
    for f in p.iterdir():
        if f.is_file() and f.suffix in {'.md','.json','.csv'}: copy_file(str(f.relative_to(BASE)))
for name in ['direction_stability_v2','local_comparison_v2']:
    tree(F+'legacy_extension_recovered_20261111/'+name,limit=12*1024*1024)
for name in ['FEATURE_AUDIT.md','RAW_FEATURE_INTEGRITY.json','RAW_FEATURE_SUMMARY_STATUS.json',
             'raw_fixed_convention_feature_summary.csv','raw_fixed_convention_residuals_per_cell.csv',
             'raw_cycle_alignment_summary.csv','raw_candidate_residuals_summary.csv',
             'slope_all_frozen_candidates_summary.csv','processed_figures','raw_curve_figures']:
    s=F+'feature_audit/'+name
    if (BASE/s).is_dir(): tree(s, limit=3*1024*1024)
    else: copy_file(s)
s=F+'feature_audit/MIT_mirror_audit/all_rows_supplement_20261110'
for p in (BASE/s).iterdir():
    if p.is_file() and p.suffix in {'.md','.csv','.json'}: copy_file(str(p.relative_to(BASE)))
for s in ['C03_峰度历史措辞澄清_20261110.md']:
    copy_file(F+'claim_clarification_addenda/'+s)
for s in ['paper_protocol.json','paper_protocol.md','legacy_extension_selection_20261110.json']:
    if (BASE/F/s).exists():copy_file(F+s)

# Keep the exact historical helper version used by the completed main matrix.
hist=F+'cloud_emergency_recovered/workspace/full_tools_v1'
tree(hist,limit=4*1024*1024,suffixes={'.py','.json','.md','.csv','.txt'})
for p in (BASE/'reproduction').iterdir():
    if p.is_file() and p.suffix in {'.py','.ps1','.md','.json','.csv','.txt'}:
        if any(x in p.name.lower() for x in ['stage_a','robustness','deadline']):continue
        copy_file(str(p.relative_to(BASE)))
tree('reproduction/legacy_vendor',suffixes={'.py','.txt','.md'})

# Author sources are attributed snapshots; data and trained weights are release assets.
up=BASE/F/'cloud_emergency_recovered/workspace/PINN4SOH-main'
for p in up.rglob('*'):
    if not p.is_file() or '__pycache__' in p.parts:continue
    rel=p.relative_to(up)
    if rel.parts[0] in {'data','results','pretrained model'}:continue
    if p.suffix not in {'.py','.md','.txt','.csv'}:continue
    copy_file(str(p.relative_to(BASE)),str(Path('PINN4SOH')/rel))

(DEST/'.gitignore').write_text('''__pycache__/
*.py[cod]
.venv/
runs/
downloads/
release_assets/
*.part
*.tmp
.DS_Store
''',encoding='utf-8')
(DEST/'docs/.nojekyll').write_text('',encoding='utf-8')
(DEST/'PUBLICATION_FILES.json').write_text(json.dumps({'format':'pinn4soh-browsable-publication-v1','original_bytes_preserved':True,'files':copied},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'files':len(copied),'bytes':sum(x['bytes'] for x in copied)},ensure_ascii=True))
