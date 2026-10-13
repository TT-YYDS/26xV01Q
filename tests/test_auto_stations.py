import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pointcloud_app.desktop_runtime import STATIONS, save_station_mapping, suggest_station_mapping


class AutoStationsTests(unittest.TestCase):
    def setUp(self):
        self.storage = TemporaryDirectory()
        self.root = Path(self.storage.name)
        self.addCleanup(self.storage.cleanup)

    def files(self, names):
        result = [self.root / (name + ".ply") for name in names]
        for i, path in enumerate(result):
            path.touch()
            os.utime(path, ns=(1_700_000_000_000_000_000 + i * 1_000_000_000,) * 2)
        return result

    def test_timestamp_sequence_is_original_acquisition_not_display_order(self):
        files = self.files(["z", "c", "q", "a", "h", "x"])
        mapping, source = suggest_station_mapping(list(reversed(files)))
        self.assertEqual(source, "file_mtime_legacy")
        self.assertEqual([mapping[p.name] for p in files],
                         ["minus50", "minus28", "minus10", "reference", "plus37", "plus47"])
        self.assertFalse((self.root / "station_order.json").exists())

    def test_same_timestamp_is_rejected(self):
        files = self.files(["a", "b", "c", "d", "e"])
        os.utime(files[1], ns=(files[0].stat().st_mtime_ns,) * 2)
        with self.assertRaisesRegex(ValueError, "时间相同"):
            suggest_station_mapping(files)

    def test_direct_labels(self):
        files = self.files(list(STATIONS[:5]))
        mapping, source = suggest_station_mapping(files)
        self.assertEqual(source, "filename_station_label")
        self.assertEqual(mapping, {p.name: p.stem for p in files})

    def test_numbered_files_use_sequence(self):
        files = self.files(["scan_3", "scan_1", "scan_2", "scan_5", "scan_4"])
        mapping, source = suggest_station_mapping(files)
        self.assertEqual(source, "filename_view_number")
        self.assertEqual(mapping["scan_1.ply"], "minus50")
        self.assertEqual(mapping["scan_4.ply"], "reference")

    def test_duplicate_numbers_rejected(self):
        files = self.files(["scan_1_a", "scan_1_b", "scan_2", "scan_3", "scan_4"])
        with self.assertRaises(ValueError):
            suggest_station_mapping(files)

    def test_partial_labels_rejected(self):
        files = self.files(["reference", "b", "c", "d", "e"])
        with self.assertRaises(ValueError):
            suggest_station_mapping(files)

    def test_manifest_priority_and_provenance(self):
        files = self.files(["a", "b", "c", "d", "e"])
        mapping = dict(zip([p.name for p in files], STATIONS[:5]))
        manifest = save_station_mapping(files, mapping, suggestion_source="filename_view_number")
        self.assertEqual(suggest_station_mapping(files), (mapping, "station_order.json"))
        self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["suggestion_source"],
                         "filename_view_number")
        manifest.write_text("broken", encoding="utf-8")
        with self.assertRaises(ValueError):
            suggest_station_mapping(files)


if __name__ == "__main__":
    unittest.main()
