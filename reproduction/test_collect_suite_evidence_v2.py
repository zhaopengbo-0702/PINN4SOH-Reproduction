"""Verify oversized UTF-8 file roundtrip, caps, and tamper rejection."""

import json
from pathlib import Path
import tempfile
import unittest

from collect_suite_evidence_v2 import encoded, pack, restore, sha256


class CollectorTest(unittest.TestCase):
    def fixture(self, root):
        source = root / "source"
        source.mkdir()
        files = {
            "per_battery.csv": b"\xef\xbb\xbf" + ('battery,comment,value\r\n电池😀,"quote\\slash\t\"",1.23\r\n' * 1200).encode("utf-8"),
            "empty.csv": b"",
            "notes.md": "内容与换行\n🌏\n".encode("utf-8"),
        }
        records = []
        for name, data in files.items():
            (source / name).write_bytes(data)
            records.append(dict(path=name, bytes=len(data), sha256=sha256(data)))
        (source / "suite_evidence_index.json").write_bytes(encoded(dict(files=records)))
        return source, files

    def test_roundtrip_and_part_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, original = self.fixture(root)
            result = pack(source, root / "parts", .004)
            self.assertGreater(len(result["parts"]), 5)
            self.assertLessEqual(result["maximum_actual_part_bytes"], int(.004 * 1024 * 1024))
            self.assertTrue(restore(root / "parts", root / "restored")["all_original_bytes_and_sha256_verified"])
            for name, data in original.items():
                self.assertEqual(data, (root / "restored" / name).read_bytes())
            self.assertEqual((source / "suite_evidence_index.json").read_bytes(),
                             (root / "restored" / "suite_evidence_index.json").read_bytes())

    def test_tampered_part_and_conflicting_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = self.fixture(root)
            result = pack(source, root / "parts", .004)
            restore(root / "parts", root / "restored")
            (root / "restored" / "empty.csv").write_text("different")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                restore(root / "parts", root / "restored")
            part = root / "parts" / result["parts"][0]["path"]
            part.write_bytes(part.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "part hash"):
                restore(root / "parts", root / "other")

    def test_missing_chunk_even_after_part_hash_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = self.fixture(root)
            pack(source, root / "parts", .004)
            index_path = root / "parts" / "suite_text_evidence_v2_index.json"
            index = json.loads(index_path.read_text())
            part_path = root / "parts" / index["parts"][0]["path"]
            document = json.loads(part_path.read_text(encoding="utf-8"))
            document["chunks"].pop(0)
            data = encoded(document)
            part_path.write_bytes(data)
            index["parts"][0].update(bytes=len(data), sha256=sha256(data), chunks=len(document["chunks"]))
            index_path.write_bytes(encoded(index))
            with self.assertRaisesRegex(ValueError, "Missing|hash mismatch"):
                restore(root / "parts", root / "restored")


if __name__ == "__main__":
    unittest.main()
