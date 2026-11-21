"""Check recovery artifacts, strict completion, and per-task checksum enforcement."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import collect_legacy_results as collector
import restore_backup_json as restore


def document(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(root):
    smoke = root / 'pinn4soh_legacy_smoke_20261110'
    control = root / 'pinn4soh_legacy_formal_control_20261110'
    for suite, count in [(smoke / 'suite', 12), (control, 40)]:
        tasks = {}
        for number in range(count):
            name = 'task%02d' % number
            path = suite / 'tasks' / name / 'metrics.json'
            document(path, dict(rmse=0.01))
            tasks[name] = dict(status='complete', verified_file_sha256={
                'metrics.json': hashlib.sha256(path.read_bytes()).hexdigest()})
        document(suite / 'manifest.json', dict(tasks=tasks, status='complete'))
    document(smoke / 'legacy_smoke_summary.json', dict(smoke_passed=True, counts={'complete': 12}, returncode=0))
    document(root / 'pinn4soh_legacy_control_comparison_20261110/summary.json',
             dict(task_counts={'matched': 40}, errors=[], assessment='complete_single_seed_version_screen'))
    document(root / 'legacy37/installation_evidence.json', dict(versions_passed=True))
    recovered = root / 'legacy_offline_recovered_v1'
    for relative in ['REASSEMBLY_VERIFIED.json', 'payload/support/legacy_compat_manifest.json',
                     'payload/tools/install.py', 'payload/assets/legacy_asset_manifest.json',
                     'payload/assets/legacy_torch_manifest.json']:
        document(recovered / relative, {})
    plan = {'tasks': [{'original_task_id': 'modern/task%02d' % number} for number in range(40)]}
    document(root / 'legacy_single_gpu_v3/control/legacy_control_plan.json', plan)
    for name, begin, end in [('pinn4soh_full_paper_stage1', 0, 16), ('pinn4soh_full_paper_stage2', 16, 40)]:
        tasks = {}
        for number in range(begin, end):
            task_id = 'modern/task%02d' % number
            path = root / name / 'tasks' / task_id / 'metrics.json'
            document(path, {'rmse': 0.02})
            tasks[task_id] = dict(status='complete', verified_file_sha256={
                'metrics.json': hashlib.sha256(path.read_bytes()).hexdigest()})
        document(root / name / 'manifest.json', dict(tasks=tasks, status='complete'))
    # The package must not carry runtime or uploaded assets.
    document(root / 'legacy37/lib/site-packages/large_runtime.json', {'not': 'selected'})
    document(recovered / 'payload/assets/large_package.whl', {'not': 'selected'})
    document(root / 'restart_verification_20261110.json', {'passed': True})
    document(root / 'legacy_tf32_sidecar_20261110/settings.json', {'enabled': True})
    document(root / 'legacy_smoke_20261110.log', {'status': 'complete'})
    document(root / 'two_runtime_observations_20261110.json', {'modern': {'passed': False}, 'legacy': {'passed': True}})
    document(root / 'two_runtime_observations_20261110_attempt2.json', {'modern': {'passed': True}, 'legacy': {'passed': True}})


class LegacyCollectionTests(unittest.TestCase):
    def test_complete_archive_restores_byte_exact_and_excludes_runtime(self):
        with tempfile.TemporaryDirectory(prefix='pinn_legacy_recovery_') as temp:
            root = Path(temp)
            fixture(root)
            output = root / 'recovery'
            state = collector.collect(root, output)
            self.assertTrue(state['collection_complete'])
            self.assertEqual(state['modern_task_pairs'], 40)
            self.assertEqual(state['artifact_sha256_checks'], 92)
            names = {row['path'] for row in state['inventory']}
            self.assertNotIn('legacy37/lib/site-packages/large_runtime.json', names)
            self.assertNotIn('legacy_offline_recovered_v1/payload/assets/large_package.whl', names)
            self.assertIn('restart_verification_20261110.json', names)
            self.assertIn('legacy_tf32_sidecar_20261110/settings.json', names)
            self.assertIn('legacy_smoke_20261110.log', names)
            self.assertIn('two_runtime_observations_20261110.json', names)
            self.assertIn('two_runtime_observations_20261110_attempt2.json', names)
            restored = restore.restore(output, root / 'restored')
            archive = Path(state['archive'])
            self.assertEqual((root / 'restored' / archive.name).read_bytes(), archive.read_bytes())
            self.assertTrue(restored[0]['sha256_verified'])

    def test_incomplete_smoke_refuses_default_but_can_save_partial(self):
        with tempfile.TemporaryDirectory(prefix='pinn_legacy_partial_') as temp:
            root = Path(temp)
            fixture(root)
            path = root / 'pinn4soh_legacy_smoke_20261110/legacy_smoke_summary.json'
            document(path, dict(smoke_passed=False, counts={'complete': 11}, returncode=1))
            with self.assertRaises(RuntimeError):
                collector.collect(root, root / 'strict')
            self.assertFalse((root / 'strict').exists())
            state = collector.collect(root, root / 'partial', allow_partial=True)
            self.assertFalse(state['collection_complete'])
            self.assertTrue(state['verified_without_errors'])

    def test_modified_task_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='pinn_legacy_mismatch_') as temp:
            root = Path(temp)
            fixture(root)
            document(root / 'pinn4soh_legacy_formal_control_20261110/tasks/task00/metrics.json', {'rmse': 99})
            with self.assertRaisesRegex(RuntimeError, 'artifact SHA mismatch'):
                collector.collect(root, root / 'mismatch')
            self.assertFalse((root / 'mismatch/LEGACY_COLLECTION_VERIFIED.json').exists())


if __name__ == '__main__':
    with contextlib.redirect_stdout(io.StringIO()):
        unittest.main()
