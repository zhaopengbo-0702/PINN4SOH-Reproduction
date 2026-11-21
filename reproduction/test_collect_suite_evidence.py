import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import collect_suite_evidence as evidence


class TextEvidenceTest(unittest.TestCase):
    def fixture(self, root):
        source = root / "source"
        source.mkdir()
        records = []
        for name, data in [("table.csv", b"\xef\xbb\xbfmodel,RMSE\r\nPINN,0.01\r\n"),
                           ("nodes.part001.json", '{"value":"\u5b9e\u9a8c","rows":[1,2,3]}\n'.encode())]:
            (source / name).write_bytes(data)
            records.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        (source / "suite_evidence_index.json").write_text(json.dumps(dict(files=records)), encoding="utf-8")
        return source

    def test_exact_bytes_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.fixture(root)
            index = evidence.pack(source, root / "download", .0006)
            self.assertGreaterEqual(len(index["parts"]), 2)
            result = evidence.restore(root / "download", root / "restored")
            self.assertTrue(result["all_original_bytes_and_sha256_verified"])
            for path in source.iterdir():
                self.assertEqual(path.read_bytes(), (root / "restored" / path.name).read_bytes())

    def test_tamper_and_existing_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.fixture(root)
            index = evidence.pack(source, root / "download", .01)
            restored = root / "restored"
            restored.mkdir()
            (restored / "table.csv").write_text("other evidence", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                evidence.restore(root / "download", restored)
            part = root / "download" / index["parts"][0]["path"]
            part.write_bytes(part.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "Invalid downloaded"):
                evidence.restore(root / "download", root / "new")

    def test_escape_and_executable_rejected(self):
        for name in ["../model.json", "model.pth", "runner.py", "/root/a.json"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                evidence.safe_name(name)


if __name__ == "__main__":
    unittest.main()
