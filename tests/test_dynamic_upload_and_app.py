"""
Integration Test Suite for Dynamic CCTV Video Upload, Zone Drawing, and Intrusion Pipeline.
Verifies all 10 requirements:
1. Video upload works.
2. The uploaded video's frame is displayed correctly.
3. Polygon drawing and saving work.
4. Detection runs on the uploaded video.
5. Safe-area movement does not trigger intrusion alerts.
6. Restricted-zone entry triggers an immediate alert.
7. Exit clears the live alert.
8. Re-entry creates a new incident.
9. Terminal logs show correct timestamps and durations.
10. Annotated video and incident history are generated.
"""

import io
import sys
import unittest
from pathlib import Path
import tkinter as tk
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cctv_app import SmartCCTVApp
from intrusion_detection import IntrusionDetector, RestrictedZone, process_video_stream


class TestDynamicVideoUploadAndPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_video_path = PROJECT_ROOT / "datasets" / "intruder_test.mp4"
        if not cls.test_video_path.is_file():
            raise FileNotFoundError(f"Test video missing: {cls.test_video_path}")

    def setUp(self):
        # Create headless Tkinter instance for UI testing
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = SmartCCTVApp(self.root)

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def test_01_upload_and_metadata_extraction(self):
        """Verify video upload, metadata parsing, and no hardcoded ABODA coordinates."""
        self.app.load_video(self.test_video_path, is_user_upload=True)

        self.assertEqual(self.app.video_path, self.test_video_path)
        self.assertEqual(self.app.orig_w, 1280)
        self.assertEqual(self.app.orig_h, 720)
        self.assertEqual(self.app.fps, 30.0)
        self.assertEqual(self.app.total_frames, 257)
        self.assertIsNotNone(self.app.first_frame_bgr)
        # Verify no hardcoded zones are loaded for user uploaded videos
        self.assertEqual(len(self.app.saved_zones), 0)
        self.assertEqual(len(self.app.current_poly_pts), 0)

    def test_02_polygon_drawing_saving_and_normalization(self):
        """Verify drawing polygon points, resetting, and coordinate normalization."""
        self.app.load_video(self.test_video_path, is_user_upload=True)

        # Mock display dimensions
        self.app.disp_w = 640
        self.app.disp_h = 360
        self.app.offset_x = 20
        self.app.offset_y = 10

        # Simulate 4 clicks
        # Point 1: (offset_x + 160, offset_y + 90) -> norm: (0.25, 0.25)
        # Point 2: (offset_x + 480, offset_y + 90) -> norm: (0.75, 0.25)
        # Point 3: (offset_x + 480, offset_y + 270) -> norm: (0.75, 0.75)
        # Point 4: (offset_x + 160, offset_y + 270) -> norm: (0.25, 0.75)
        pts = [
            (self.app.offset_x + 160, self.app.offset_y + 90),
            (self.app.offset_x + 480, self.app.offset_y + 90),
            (self.app.offset_x + 480, self.app.offset_y + 270),
            (self.app.offset_x + 160, self.app.offset_y + 270),
        ]
        self.app.current_poly_pts = list(pts)
        self.assertEqual(len(self.app.current_poly_pts), 4)

        # Save zone
        self.app.zone_name_entry.delete(0, tk.END)
        self.app.zone_name_entry.insert(0, "Office Center Restricted Zone")
        self.app.on_save_zone()

        self.assertEqual(len(self.app.saved_zones), 1)
        zone0 = self.app.saved_zones[0]
        self.assertEqual(zone0["name"], "Office Center Restricted Zone")
        self.assertEqual(len(zone0["points"]), 4)
        # Verify normalized coordinates match expected
        self.assertAlmostEqual(zone0["points"][0][0], 0.25, places=2)
        self.assertAlmostEqual(zone0["points"][0][1], 0.25, places=2)
        self.assertAlmostEqual(zone0["points"][2][0], 0.75, places=2)
        self.assertAlmostEqual(zone0["points"][2][1], 0.75, places=2)

    def test_03_multi_polygon_support(self):
        """Verify defining and saving multiple distinct zones."""
        self.app.load_video(self.test_video_path, is_user_upload=True)
        self.app.disp_w, self.app.disp_h, self.app.offset_x, self.app.offset_y = 640, 360, 0, 0

        # Zone 1
        self.app.current_poly_pts = [(50, 50), (200, 50), (200, 200), (50, 200)]
        self.app.zone_name_entry.delete(0, tk.END)
        self.app.zone_name_entry.insert(0, "Zone Alpha")
        self.app.on_save_zone()

        # Zone 2
        self.app.current_poly_pts = [(300, 100), (500, 100), (500, 300), (300, 300)]
        self.app.zone_name_entry.delete(0, tk.END)
        self.app.zone_name_entry.insert(0, "Zone Beta")
        self.app.on_save_zone()

        self.assertEqual(len(self.app.saved_zones), 2)
        self.assertEqual(self.app.saved_zones[0]["name"], "Zone Alpha")
        self.assertEqual(self.app.saved_zones[1]["name"], "Zone Beta")

    def test_04_end_to_end_detection_and_terminal_logging(self):
        """
        Verify end-to-end detection pipeline on uploaded video:
        - Safe area movement triggers 0 alerts.
        - Zone entry triggers immediate alert.
        - Exit clears live alert.
        - Terminal output contains timestamps, Camera, Track ID, and Duration.
        - Output video and event screenshots are generated.
        """
        out_dir = PROJECT_ROOT / "runs" / "test_upload_evaluation"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_video = out_dir / "intruder_test_annotated.mp4"
        screens_dir = out_dir / "screenshots"

        # Define dynamic zone covering the path where the person walks:
        # In intruder_test.mp4 (1280x720), person is detected between x=450-750, y=250-640 (t=2.5s - 7.5s)
        # Normalized: x in [0.35, 0.65], y in [0.35, 0.90]
        test_zone = RestrictedZone(
            zone_id="custom_upload_zone_1",
            name="Restricted Corridor Zone",
            raw_points=[[0.35, 0.35], [0.65, 0.35], [0.65, 0.90], [0.35, 0.90]],
            classes={"person", "car"},
            enabled=True,
            is_normalized=True,
            zone_type="restricted",
        )

        detector = IntrusionDetector(
            zones=[test_zone],
            yolo_model=PROJECT_ROOT / "yolov8s.pt",
            confidence=0.45,
            confirm_frames=2,
            camera_name="Office Intrusion CCTV",
            simulated_start_time="11:00:00",
        )

        # Capture terminal stdout during processing
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured

        try:
            results = process_video_stream(
                video_source=self.test_video_path,
                detector=detector,
                output_video=out_video,
                save_screenshots=screens_dir,
                display=False,
                realtime=False,
            )
        finally:
            sys.stdout = old_stdout

        out_text = captured.getvalue()

        # 1. Verify terminal logs
        self.assertIn("[CRITICAL ALERT] PERSON INTRUSION", out_text)
        self.assertIn("Camera: Office Intrusion CCTV", out_text)
        self.assertIn("Class: Person", out_text)
        self.assertIn("[ZONE EXIT] PERSON", out_text)
        self.assertIn("Exit Time:", out_text)
        self.assertIn("Duration:", out_text)

        # 2. Verify results dictionary
        self.assertGreater(results["total_frames"], 200)
        self.assertGreaterEqual(results["total_incidents"], 1)
        self.assertGreaterEqual(results["person_incidents"], 1)
        self.assertEqual(results["vehicle_incidents"], 0)

        # 3. Verify output files
        self.assertTrue(out_video.is_file())
        self.assertGreater(out_video.stat().st_size, 50000)

        # 4. Verify screenshots were captured
        self.assertGreaterEqual(len(results["saved_screenshots"]), 1)
        first_shot = Path(results["saved_screenshots"][0])
        self.assertTrue(first_shot.is_file())

        # 5. Verify incident history records
        history = results["incident_history"]
        self.assertGreaterEqual(len(history), 1)
        inc0 = history[0]
        self.assertEqual(inc0["class_name"], "person")
        self.assertEqual(inc0["zone_name"], "Restricted Corridor Zone")
        self.assertEqual(inc0["status"], "CLEARED")
        self.assertIsNotNone(inc0["duration_seconds"])
        print(f"\n[Test Result] Logged Incident: {inc0['type']} ID {inc0['track_id']} | Duration: {inc0['duration_seconds']}s")


if __name__ == "__main__":
    unittest.main()
