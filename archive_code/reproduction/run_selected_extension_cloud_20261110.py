#!/usr/bin/env python3
"""Run the frozen selected legacy extension and collect its evidence on cloud."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    workspace = Path('/workspace')
    tools = workspace / 'legacy_extension_tools_v2' / 'tools'
    overlay = workspace / 'legacy_independent_overlay_20261110'
    assert json.loads((overlay / 'identity_lineage_overlay.json').read_text())['assessment'] == 'passed_all40'
    for name, smoke in [('legacy_cpu_smoke', True), ('legacy_cpu_formal', False)]:
        result = json.loads((overlay / name / 'summary.json').read_text())
        assert result['status'] == 'complete' and result['smoke'] is smoke
    support = workspace / 'legacy_selected_support_20261110'
    design = json.loads((support / 'extension_design.json').read_text())
    assert design['new_task_count'] == 99 and design['analysis_task_count'] == 110
    current = ['/workspace/pinn4soh_full_paper_stage1', '/workspace/pinn4soh_full_paper_stage2']
    common = ['--extension', str(support), '--legacy-control-suite', '/workspace/pinn4soh_legacy_formal_control_20261110']
    suite = ['--legacy-extension-suite', '/workspace/pinn4soh_legacy_selected_extension_20261110']
    comparison = '/workspace/legacy_selected_comparison_20261110'
    screening = '/workspace/pinn4soh_legacy_control_comparison_20261110/summary.json'
    steps = [
        ('training', [sys.executable, str(tools / 'launch_legacy_extension.py'), '--prefix', '/workspace/legacy37',
                      '--support', '/workspace/legacy_offline_recovered_v1/payload/support', *common,
                      '--repo', '/workspace/PINN4SOH-main', '--gpus', '0', '--workers-per-gpu', '1']),
        ('comparison', [sys.executable, str(tools / 'compare_legacy_extension.py'), *common, *suite,
                        '--current-suites', *current, '--output', comparison]),
        ('extension_collection', [sys.executable, str(tools / 'collect_legacy_extension_results.py'),
                                   '--workspace', '/workspace', *common, *suite, '--current-suites', *current,
                                   '--comparison', comparison, '--screening-report', screening,
                                   '--output-prefix', '/workspace/legacy_selected_backup_20261110']),
        ('control_overlay_collection', [sys.executable, '/workspace/collect_legacy_results.py',
            '--output-prefix', '/workspace/legacy_results_with_overlay_20261110',
            '--include', str(overlay), '--include', '/workspace/audit_legacy_control_overlay.py',
            '--include', '/workspace/reconstruct_control_data_identity.py',
            '--include', '/workspace/benchmark_efficiency_legacy.py',
            '--include', '/workspace/compare_legacy_control.py', '--include', '/workspace/build_suite_report.py',
            '--include', '/workspace/legacy_cpu_pipeline_20261110.log',
            '--include', '/workspace/legacy_cpu_pipeline_20261110_attempt2.log',
            '--include', str(Path(__file__).resolve())]),
    ]
    status_path = workspace / 'legacy_selected_pipeline_20261110_status.json'
    assert not status_path.exists(), 'Use a separately reviewed resume after an interrupted attempt'
    state = {'status': 'running', 'steps': [], 'pid': os.getpid()}
    environment = dict(os.environ, PYTHONPATH='/workspace/site-packages', CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1')
    def save():
        state['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(status_path)
    try:
        for name, command in steps:
            record = {'name': name, 'status': 'running', 'command': command}
            state['steps'].append(record)
            save()
            print('START', name, flush=True)
            record['returncode'] = subprocess.call(command, cwd=workspace, env=environment)
            record['status'] = 'complete' if record['returncode'] == 0 else 'failed'
            save()
            if record['returncode']:
                raise RuntimeError(name + ' failed: ' + str(record['returncode']))
            print('COMPLETE', name, flush=True)
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
