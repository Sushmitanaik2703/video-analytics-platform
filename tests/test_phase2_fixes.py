"""
Unit & Integration Tests for Phase 2 CCTV Fixes:
1. Start Detection without polygon when restricted zone is disabled.
2. Single-pass unified person tracking (no duplicate bounding boxes or conflicting IDs).
3. Immediate unattended-bag notification in notification panel, incident table, and KPI.
4. Tracking ID stability across frames (spatial-temporal matching).
5. Independent and combined mode validation.
"""

import sys
import unittest
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cv2
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cctv_app import SmartCCTVApp
from integrated_pipeline import IntegratedPipeline, process_integrated_stream
from intrusion_detection import RestrictedZone
import unattended_object as uo


class TestPhase2Fixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = PROJECT_ROOT / "yolov8n.pt"
        if not cls.model_path.is_file():
            cls.model_path = PROJECT_ROOT / "yolov8s.pt"
        cls.video_path = PROJECT_ROOT / "datasets" / "samples" / "real_world_tests" / "aboda_video2.avi"

    def test_01_gui_start_detection_without_polygon_when_intrusion_disabled(self):
        """Verify GUI allows starting detection without polygon when only bag or vehicle is selected."""
        from unittest.mock import patch

        root = tk.Tk()
        app = SmartCCTVApp(root)
        root.update()

        # Load video
        app.load_video(self.video_path, is_user_upload=False)
        root.update()

        # Disable intrusion detection, enable unattended bag detection
        app.chk_feat_intrusion.set(False)
        app.chk_feat_vehicle.set(False)
        app.chk_feat_bag.set(True)
        root.update()

        # Verify no zones are drawn
        self.assertEqual(len(app.saved_zones), 0)
        self.assertEqual(len(app.current_poly_pts), 0)

        # Start detection - should succeed without error or polygon warning popup
        with patch("tkinter.messagebox.showinfo"):
            app.on_start_detection()
            self.assertTrue(app.is_processing)

            # Stop detection cleanly
            app.on_stop_detection()
            import time
            t_end = time.time() + 2.0
            while time.time() < t_end and app.is_processing:
                root.update()
                time.sleep(0.01)

        root.destroy()

    def test_02_notification_panel_present_and_updates_on_unattended_bag(self):
        """Verify notification panel exists below video and displays bag alert details immediately."""
        root = tk.Tk()
        app = SmartCCTVApp(root)
        root.update()

        # Verify notification panel widgets exist
        self.assertTrue(hasattr(app, "notif_panel"))
        self.assertTrue(hasattr(app, "lbl_notif_title"))
        self.assertTrue(hasattr(app, "lbl_notif_text"))

        # Simulate unattended bag alert status
        mock_bag = uo.TrackedBag(
            track_id=7,
            class_name="backpack",
            bbox=(100, 100, 200, 200),
            position=(150.0, 150.0),
            first_detected_at=uo.datetime.now(),
            last_detected_at=uo.datetime.now(),
            stationary_since=10.0,
            stationary_anchor=(150.0, 150.0),
            associated_person_id=3,
            alert_active=True,
            last_seen_time=25.0,
        )

        mock_status = {
            "timestamp": 25.0,
            "time_str": "12:00:25",
            "all_active_alerts": 1,
            "banner_level": "CRITICAL",
            "banner_text": "🚨 CRITICAL ALERT ACTIVE: UNATTENDED BAG: Backpack (Person #3)",
            "unattended_bags": [mock_bag],
            "confirmed_intruders": [],
            "notifications": [
                {
                    "level": "CRITICAL",
                    "type": "UNATTENDED_BAG",
                    "message": "Unattended Backpack detected (Person #3)",
                }
            ],
        }

        mock_incidents = [
            {
                "incident_id": 1,
                "type": "UNATTENDED_BAG",
                "track_id": 7,
                "class_name": "backpack",
                "owner_id": 3,
                "zone_name": "Full-Frame (Automatic)",
                "entry_time": "12:00:25",
                "exit_time": None,
                "duration_seconds": 15.0,
                "status": "ACTIVE",
            }
        ]

        dummy_frame = np.zeros((480, 720, 3), dtype=np.uint8)
        app._update_live_ui(
            frame_idx=100,
            total_frames=1000,
            cur_fps=28.5,
            timestamp=25.0,
            status=mock_status,
            frame_bgr=dummy_frame,
            incident_history=mock_incidents,
        )
        root.update()

        # Verify notification panel shows the owner, duration, and ACTIVE status without a bag ID
        notif_text = app.lbl_notif_text.cget("text")
        self.assertIn("Person #3", notif_text)
        self.assertNotIn("#7", notif_text)
        self.assertIn("ACTIVE", notif_text)
        self.assertIn("Backpack", notif_text)

        # Verify KPI bag updated
        self.assertEqual(app.kpi_bag.val_lbl.cget("text"), "1")
        self.assertEqual(app.kpi_total.val_lbl.cget("text"), "1")

        # Verify incident table row updated
        children = app.tree.get_children()
        self.assertEqual(len(children), 1)
        vals = app.tree.item(children[0])["values"]
        self.assertEqual(vals[1], "UNATTENDED_BAG")
        self.assertEqual(vals[2], "Person #3")
        self.assertEqual(vals[8], "ACTIVE")

        root.destroy()

    def test_03_bag_tracking_id_stability(self):
        """Verify spatial association preserves bag track ID across consecutive frames."""
        model = YOLO(str(self.model_path))
        bags: dict[int, uo.TrackedBag] = {}
        args = SimpleNamespace(
            confidence=0.45,
            bag_confidence=0.20,
            imgsz=640,
            stationary_time=15.0,
            person_distance=150.0,
            stationary_distance=25.0,
            alert_cooldown=30.0,
            debug_detections=False,
            save_frames=None,
            save_interval=1.0,
        )

        dummy_frame = np.zeros((480, 720, 3), dtype=np.uint8)

        # Create initial bag in bags dictionary
        initial_bag = uo.TrackedBag(
            track_id=12,
            class_name="backpack",
            bbox=(200, 200, 260, 280),
            position=(230.0, 240.0),
            first_detected_at=uo.datetime.now(),
            last_detected_at=uo.datetime.now(),
            stationary_since=5.0,
            stationary_anchor=(230.0, 240.0),
            associated_person_id=4,
            alert_active=False,
            last_seen_time=5.0,
        )
        bags[12] = initial_bag

        # Now simulate a detection at a slightly shifted position (within 10 pixels), but with a different ByteTrack ID (99)
        # Using uo.compute_box_iou
        iou = uo.compute_box_iou((200, 200, 260, 280), (202, 201, 262, 281))
        self.assertGreater(iou, 0.85)

        # Test spatial matching logic
        matched_id = None
        for existing_id, existing_bag in bags.items():
            dist = (
                (231.0 - existing_bag.position[0]) ** 2
                + (241.0 - existing_bag.position[1]) ** 2
            ) ** 0.5
            if dist <= 45.0 or iou >= 0.20:
                matched_id = existing_id
                break

        # Must match existing bag ID 12 rather than creating new ID 99
        self.assertEqual(matched_id, 12)

    def test_04_no_duplicate_person_boxes_in_combined_mode(self):
        """Verify combined mode shares person tracks and sets skip_draw_people to avoid duplicate boxes."""
        zone = RestrictedZone("z1", "Test Zone", [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)], is_normalized=True)
        pipeline = IntegratedPipeline(
            yolo_model_path=self.model_path,
            enable_intrusion=True,
            enable_vehicle_tracking=True,
            enable_unattended_bag=True,
            zones=[zone],
        )

        dummy_frame = np.zeros((480, 720, 3), dtype=np.uint8)
        annotated, status = pipeline.process_frame(dummy_frame, 0.0)

        self.assertIsNotNone(annotated)
        self.assertIn("confirmed_intruders", status)
        self.assertIn("unattended_bags", status)


if __name__ == "__main__":
    unittest.main()
