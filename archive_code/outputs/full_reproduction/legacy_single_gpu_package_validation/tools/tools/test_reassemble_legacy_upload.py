"""Small adversarial fixtures for reconstruction; never execute legacy binaries."""

import hashlib
import io
import json
from pathlib import Path
import stat
import struct
import tempfile
import unittest
import warnings
import zipfile

import reassemble_legacy_upload as rebuild


def zip_bytes(items):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_STORED) as archive:
        for name, content in items:
            if isinstance(name, str):
                item = zipfile.ZipInfo()
                item.create_system = 3
                # Keep deliberately hostile backslashes even on Windows, whose
                # ZipInfo constructor would otherwise turn them into slashes.
                item.filename = name
            else:
                item = name
            archive.writestr(item, content)
    return stream.getvalue()


class ReassemblyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pinn-reassembly-")
        self.root = Path(self.temp.name)
        self.parts = self.root / "parts"
        self.parts.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def make_fixture(self, payload=None):
        if payload is None:
            payload = zip_bytes([("tools/example.py", b"print('fixture only')\n"),
                                 ("assets/example.bin", bytes(range(256)) * 8)])
        chunk = (len(payload) + 13) // 14
        rows = []
        for number in range(1, 15):
            member = "legacy_archive.part%03d" % number
            block = payload[(number - 1) * chunk:number * chunk]
            path = self.parts / (member + ".zip")
            with zipfile.ZipFile(path, "x", zipfile.ZIP_STORED) as archive:
                archive.writestr(member, block)
            rows.append(dict(archive=path.name, member=member, bytes=len(block),
                             sha256=hashlib.sha256(block).hexdigest(), archive_sha256=rebuild.sha256(path)))
        index = dict(format="pinn4soh-offline-archive-parts-v1", original="fixture.zip",
                     bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(), parts=rows)
        self.index_path = self.parts / rebuild.INDEX_NAME
        self.index_path.write_text(json.dumps(index), encoding="utf-8")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = archive.infolist()
        self.pins = dict(index_sha256=rebuild.sha256(self.index_path), original="fixture.zip",
                         bytes=len(payload), sha256=index["sha256"], part_count=14, part_bytes=chunk,
                         payload_entries=len(infos), payload_bytes=sum(item.file_size for item in infos))
        return payload, index

    def repin_index(self, index):
        self.index_path.write_text(json.dumps(index), encoding="utf-8")
        self.pins["index_sha256"] = rebuild.sha256(self.index_path)

    def test_fourteen_parts_roundtrip_and_no_overwrite(self):
        payload, _ = self.make_fixture()
        output = self.root / "fresh"
        result = rebuild.rebuild(self.parts, output, self.pins, verify_assets=False)
        self.assertEqual(result["status"], "passed")
        self.assertEqual((output / "fixture.zip").read_bytes(), payload)
        self.assertEqual((output / "payload/tools/example.py").read_text(), "print('fixture only')\n")
        self.assertEqual(len(result["part_results"]), 14)
        with self.assertRaisesRegex(ValueError, "fresh nonexistent"):
            rebuild.rebuild(self.parts, output, self.pins, verify_assets=False)
        self.assertEqual((output / "fixture.zip").read_bytes(), payload)

    def test_changed_index_fails_before_output_creation(self):
        self.make_fixture()
        self.index_path.write_bytes(self.index_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "index SHA256"):
            rebuild.rebuild(self.parts, self.root / "fresh", self.pins, verify_assets=False)
        self.assertFalse((self.root / "fresh").exists())

    def test_wrong_part_order_rejected_even_with_rehashed_fixture_index(self):
        _, index = self.make_fixture()
        index["parts"][0], index["parts"][1] = index["parts"][1], index["parts"][0]
        self.repin_index(index)
        with self.assertRaisesRegex(ValueError, "identity/order"):
            rebuild.rebuild(self.parts, self.root / "fresh", self.pins, verify_assets=False)

    def test_missing_or_extra_part_rejected(self):
        self.make_fixture()
        (self.parts / "legacy_archive.part015.zip").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "Missing, extra"):
            rebuild.rebuild(self.parts, self.root / "fresh", self.pins, verify_assets=False)

    def test_changed_outer_zip_sha_rejected(self):
        self.make_fixture()
        path = self.parts / "legacy_archive.part001.zip"
        raw = bytearray(path.read_bytes())
        raw[64] ^= 1
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "archive SHA256"):
            rebuild.rebuild(self.parts, self.root / "fresh", self.pins, verify_assets=False)

    def test_member_sha_is_checked_independently(self):
        _, index = self.make_fixture()
        index["parts"][0]["sha256"] = "0" * 64
        self.repin_index(index)
        with self.assertRaisesRegex(ValueError, "member SHA256"):
            rebuild.rebuild(self.parts, self.root / "fresh", self.pins, verify_assets=False)
        self.assertTrue((self.root / "fresh/REASSEMBLY_FAILED.json").is_file())
        self.assertFalse((self.root / "fresh/REASSEMBLY_VERIFIED.json").exists())

    def test_total_sha_is_checked_independently(self):
        _, index = self.make_fixture()
        index["sha256"] = self.pins["sha256"] = "1" * 64
        self.repin_index(index)
        with self.assertRaisesRegex(ValueError, "stream identity"):
            rebuild.rebuild(self.parts, self.root / "fresh", self.pins, verify_assets=False)

    def test_inner_zip_crc_damage_is_rejected(self):
        payload = bytearray(zip_bytes([("tools/a", bytes(range(256)) * 4)]))
        name_len, extra_len = struct.unpack_from("<HH", payload, 26)
        payload[30 + name_len + extra_len + 10] ^= 1
        self.make_fixture(bytes(payload))
        with self.assertRaises((ValueError, zipfile.BadZipFile)):
            rebuild.rebuild(self.parts, self.root / "fresh", self.pins, verify_assets=False)
        self.assertFalse((self.root / "fresh/payload").exists())

    def assert_unsafe_payload(self, payload):
        path = self.root / "unsafe.zip"
        path.write_bytes(payload)
        with self.assertRaises(ValueError):
            rebuild.safe_extract_archive(path, self.root / "extracted", 100, 100000)
        self.assertFalse((self.root / "extracted").exists())

    def test_path_traversal_and_windows_paths_rejected(self):
        for name in ["../outside", "/absolute", "C:/absolute", "a\\b", "a/../b", "a//b", "a/./b", "NUL.txt", "a."]:
            with self.subTest(name=name):
                self.assert_unsafe_payload(zip_bytes([(name, b"bad")]))

    def test_symlink_member_rejected(self):
        item = zipfile.ZipInfo("link")
        item.create_system = 3
        item.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.assert_unsafe_payload(zip_bytes([(item, b"../outside")]))

    def test_duplicate_and_case_collision_rejected(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assert_unsafe_payload(zip_bytes([("tools/a", b"a"), ("tools/a", b"b")]))
        self.assert_unsafe_payload(zip_bytes([("tools/a", b"a"), ("TOOLS/A", b"b")]))

    def test_file_directory_prefix_collision_rejected(self):
        self.assert_unsafe_payload(zip_bytes([("tools", b"a"), ("tools/a", b"b")]))

    def test_extraction_size_budget_rejected_before_writing(self):
        path = self.root / "large.zip"
        path.write_bytes(zip_bytes([("large", b"a" * 100)]))
        with self.assertRaisesRegex(ValueError, "size exceeds"):
            rebuild.safe_extract_archive(path, self.root / "extracted", 10, 99)
        self.assertFalse((self.root / "extracted").exists())

    def test_duplicate_json_key_rejected(self):
        path = self.root / "duplicate.json"
        path.write_text('{"x": 1, "x": 2}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            rebuild.read_json(path)


if __name__ == "__main__":
    unittest.main()
