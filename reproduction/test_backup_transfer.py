"""Exercise partial-download recovery, source integrity, and emergency tar restore."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TransferTests(unittest.TestCase):
    def test_incomplete_archive_does_not_block_other_archive(self):
        with tempfile.TemporaryDirectory(prefix='pinn_partial_transfer_') as temp:
            root = Path(temp)
            for number in (1, 2):
                with zipfile.ZipFile(root / f'results.{number:03d}.zip', 'w') as archive:
                    archive.writestr('metrics.json', '{"rmse":0.0123}')
            (root / 'VERIFIED_BACKUP.json').write_text('{"verified_without_errors":true}')
            wrapper = load('wrap_backup_json')
            wrapper.MAX_CHUNK = 73
            wrapper.wrap_backup(root)
            incomplete = sorted(root.glob('results.001.zip.chunk*.json'))[-1]
            renamed = incomplete.with_suffix('.json.crdownload')
            incomplete.rename(renamed)
            restore = load('restore_backup_json')
            output = root / 'restored'
            results = restore.restore(root, output, incremental=True)
            self.assertEqual([row['original_zip_name'] for row in results], ['results.002.zip'])
            index = json.loads((output / 'LOCAL_BACKUP_RESTORE_INDEX.json').read_text())
            self.assertEqual(index['verified_archives'], 1)
            self.assertEqual(index['pending_archives'][0]['original_zip_name'], 'results.001.zip')
            self.assertEqual(index['pending_files'][0]['status'], 'downloading')
            self.assertFalse(index['all_expected_archives_verified'])
            renamed.rename(incomplete)
            results = restore.restore(root, output)
            self.assertEqual(len(results), 2)
            self.assertTrue(results[1]['reused_verified'])
            index = json.loads((output / 'LOCAL_BACKUP_RESTORE_INDEX.json').read_text())
            self.assertTrue(index['all_expected_archives_verified'])
            for number in (1, 2):
                name = f'results.{number:03d}.zip'
                self.assertEqual((output / name).read_bytes(), (root / name).read_bytes())

    def test_emergency_archive_restores_and_checks_tar(self):
        with tempfile.TemporaryDirectory(prefix='pinn_emergency_transfer_') as temp:
            root = Path(temp)
            source = root / 'metrics.json'
            source.write_text('{"rmse":0.011}')
            target = root / 'EMERGENCY_FULL_20261110_1827.tar.gz'
            with tarfile.open(target, 'w:gz') as archive:
                archive.add(source, arcname='suite/metrics.json')
            finalizer = load('finalize_emergency_archive')
            finalizer.CHUNK = 73
            finalizer.finalize(target)
            transfer = root / (target.name + '_transfer')
            restore = load('restore_backup_json')
            results = restore.restore(transfer, root / 'restored')
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]['members'], 1)
            self.assertEqual((root / 'restored' / target.name).read_bytes(), target.read_bytes())

    def test_corrupt_wrapper_preserves_previously_restored_archive(self):
        with tempfile.TemporaryDirectory(prefix='pinn_corrupt_transfer_') as temp:
            root = Path(temp)
            with zipfile.ZipFile(root / 'results.001.zip', 'w') as archive:
                archive.writestr('evidence.txt', 'evidence')
            (root / 'VERIFIED_BACKUP.json').write_text('{"verified_without_errors":true}')
            load('wrap_backup_json').wrap_backup(root)
            restore = load('restore_backup_json')
            output = root / 'restored'
            restore.restore(root, output)
            before = (output / 'results.001.zip').read_bytes()
            path = root / 'results.001.zip.json'
            data = path.read_bytes()
            marker = data.find(b'"data_base64":"') + len(b'"data_base64":"')
            path.write_bytes(data[:marker] + b'AAAA' + data[marker + 4:])
            restore.restore(root, output, incremental=True)
            index = json.loads((output / 'LOCAL_BACKUP_RESTORE_INDEX.json').read_text())
            self.assertEqual(index['verified_archives'], 0)
            self.assertTrue(index['errors'])
            self.assertEqual((output / 'results.001.zip').read_bytes(), before)


if __name__ == '__main__':
    with contextlib.redirect_stdout(io.StringIO()):
        unittest.main()
