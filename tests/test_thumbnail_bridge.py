from pathlib import Path
import unittest

from covermorph.thumbnail_bridge import (
    ThumbnailBridgeError,
    ThumbnailBridgeRequest,
    ThumbnailBridgeResponse,
    read_request_json,
    standard_output_paths,
    write_response_json,
)


class ThumbnailBridgeContractTests(unittest.TestCase):
    def test_request_accepts_unicode_project_path(self):
        req = ThumbnailBridgeRequest.from_dict({
            "protocol_version": 1,
            "request_id": "abc",
            "action": "generate",
            "project_dir": r"D:\일본 칠리랩\男_001",
            "title": "目が合っただけなのに",
        })
        self.assertEqual(req.action, "generate")
        self.assertIn("일본", req.project_dir)

    def test_edit_requires_instruction(self):
        with self.assertRaises(ThumbnailBridgeError):
            ThumbnailBridgeRequest.from_dict({
                "protocol_version": 1,
                "action": "edit",
                "project_dir": r"D:\p",
            })

    def test_response_round_trip_is_unicode_json(self):
        text = write_response_json(ThumbnailBridgeResponse(
            request_id="1", action="status", project_dir=r"D:\한글", message="정상"
        ))
        self.assertIn("정상", text)
        self.assertIn('"ok":true', text)

    def test_standard_paths(self):
        paths = standard_output_paths(Path("p"))
        self.assertEqual(paths["canvas_clean"].name, "canvas_clean.png")
        self.assertEqual(paths["project_manifest"].name, "project_manifest.json")

    def test_read_rejects_bad_json(self):
        with self.assertRaises(ThumbnailBridgeError):
            read_request_json("{bad")


if __name__ == "__main__":
    unittest.main()


class ThumbnailBridgeRuntimeTests(unittest.TestCase):
    def test_status_handler_reports_structured_capabilities(self):
        from covermorph.thumbnail_bridge_runtime import handle_request
        import tempfile
        with tempfile.TemporaryDirectory(prefix="브리지 상태 ") as tmp:
            req = ThumbnailBridgeRequest.from_dict({
                "protocol_version": 1,
                "request_id": "status-1",
                "action": "status",
                "project_dir": tmp,
            })
            response = handle_request(req)
            self.assertTrue(response.ok)
            self.assertIn("status", response.outputs)
            payload = response.outputs["status"]
            self.assertTrue(payload["capabilities"]["headless"])
            self.assertTrue(payload["project_dir_writable"])

    def test_generate_is_not_falsely_reported_as_real_ai(self):
        from unittest.mock import patch
        from covermorph import thumbnail_bridge_runtime
        from covermorph.thumbnail_bridge_runtime import handle_request
        import tempfile
        missing = {"status": "model_missing", "reference_status": "model_missing", "model": {"failure_reason": "missing"}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                thumbnail_bridge_runtime, "_environment", lambda request: (missing, Path(tmp), "none")):
            req = ThumbnailBridgeRequest.from_dict({
                "protocol_version": 1,
                "request_id": "gen-1",
                "action": "generate",
                "project_dir": tmp,
                "prompt": "textless scene",
            })
            response = handle_request(req)
            self.assertFalse(response.ok)
            self.assertEqual(response.error_code, "MODEL_NOT_READY")
            self.assertFalse((Path(tmp) / "canvas_clean.png").exists())
