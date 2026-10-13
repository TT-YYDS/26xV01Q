import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

# Public test fixture only; never used as an application default.
with patch.dict(os.environ, {"POINTCLOUD_WEB_SECRET_KEY": "unit-test-only-not-a-real-secret!"}):
    import media_web_app


class ExternalSectionHandoffTests(unittest.TestCase):
    def test_external_preview_is_loaded_without_pointcloud_upload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            preview_root = Path(temporary_directory)
            token = "a" * 32
            package = preview_root / token
            section = package / "media_processing" / "vertical_section"
            section.mkdir(parents=True)
            (section / "vertical_section_points.ply").write_text("ply", encoding="ascii")
            (package / "pointcloud_side_preview.png").write_bytes(b"png")
            (package / "manifest.json").write_text(
                json.dumps(
                    {
                        "token": token,
                        "files": [],
                        "analysis_pointcloud": "media_processing/vertical_section/vertical_section_points.ply",
                        "mode": media_web_app.EXTERNAL_SECTION_MODE,
                        "section_points": 1234,
                        "source_file_count": 6,
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(media_web_app, "POINTCLOUD_PREVIEW_ROOT", preview_root):
                response = media_web_app.app.test_client().get(f"/?section_token={token}")

            self.assertEqual(response.status_code, 200)
            page = response.get_data(as_text=True)
            self.assertIn("最终侧面截面", page)
            self.assertIn('"section_points": 1234', page)
            self.assertIn('"source_file_count": 6', page)


if __name__ == "__main__":
    unittest.main()
