from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unattended_object as uo
from cctv_app import SmartCCTVApp
from integrated_pipeline import IntegratedPipeline
from intrusion_detection import IntrusionDetector, RestrictedZone


class FakeBox:
    def __init__(self, bbox: tuple[int, int, int, int], class_id: int):
        self.xyxy = torch.tensor([bbox], dtype=torch.float32)
        self.cls = torch.tensor([class_id])
        self.conf = torch.tensor([0.9])


class FakeBoxes(list):
    def __init__(self, boxes: list[FakeBox], track_ids: list[int]):
        super().__init__(boxes)
        self.id = torch.tensor(track_ids)


class FakeModel:
    names = {0: "person", 24: "backpack", 26: "handbag", 28: "suitcase"}

    def __init__(self):
        self.detections = FakeBoxes([], [])

    def track(self, *args, **kwargs):
        return [SimpleNamespace(boxes=self.detections, names=self.names)]


class TestAlertLifecycleFixes(unittest.TestCase):
    def test_completed_video_keeps_unresolved_incident_active_in_ui(self):
        app = SimpleNamespace(
            btn_start=Mock(),
            btn_stop=Mock(),
            status_badge=Mock(),
            status_text=Mock(),
            btn_play_video=Mock(),
            btn_open_folder=Mock(),
            btn_export_log=Mock(),
            _populate_screenshots_gallery=Mock(),
            kpi_total=SimpleNamespace(val_lbl=Mock()),
            kpi_person=SimpleNamespace(val_lbl=Mock()),
            kpi_vehicle=SimpleNamespace(val_lbl=Mock()),
            kpi_bag=SimpleNamespace(val_lbl=Mock()),
            kpi_active=SimpleNamespace(val_lbl=Mock()),
            kpi_fps=SimpleNamespace(val_lbl=Mock()),
            lbl_notif_text=Mock(),
            lbl_notif_title=Mock(),
            notif_panel=Mock(),
            _sync_incident_table=Mock(),
            success_color="#98c379",
        )
        results = {
            "total_frames": 20,
            "elapsed_seconds": 1.0,
            "total_incidents": 1,
            "person_incidents": 0,
            "vehicle_incidents": 1,
            "unattended_bag_incidents": 0,
            "average_fps": 20.0,
            "incident_history": [{"status": "ACTIVE"}],
        }

        with patch("cctv_app.messagebox.showinfo"):
            SmartCCTVApp._on_pipeline_completed(app, results, Path("annotated.mp4"))

        app.kpi_active.val_lbl.config.assert_called_once_with(text="1")
        self.assertIn(
            "remain active",
            app.lbl_notif_text.config.call_args.kwargs["text"],
        )

    def test_bag_alert_latches_until_bag_moves_and_alerts_without_owner_track(self):
        args = SimpleNamespace(
            confidence=0.5,
            bag_confidence=0.25,
            imgsz=640,
            stationary_time=1.0,
            person_distance=30.0,
            stationary_distance=20.0,
            alert_cooldown=30.0,
            debug_detections=False,
        )
        bags: dict[int, uo.TrackedBag] = {}
        bag_box = (90, 90, 110, 110)

        uo.process_frame(
            FakeModel(),
            np.zeros((240, 320, 3), dtype=np.uint8),
            bags,
            args,
            0.0,
            pre_detected_boxes=FakeBoxes([FakeBox(bag_box, 24)], [7]),
        )
        alert_frame = uo.process_frame(
            FakeModel(),
            np.zeros((240, 320, 3), dtype=np.uint8),
            bags,
            args,
            1.1,
            pre_detected_boxes=FakeBoxes([FakeBox(bag_box, 24)], [7]),
        )
        self.assertTrue(bags[7].alert_active)
        self.assertEqual(alert_frame[90, 90].tolist(), [0, 0, 255])

        nearby_person = uo.TrackedPerson(3, (100.0, 100.0), (90, 90, 110, 110))
        uo.process_frame(
            FakeModel(),
            np.zeros((240, 320, 3), dtype=np.uint8),
            bags,
            args,
            1.2,
            external_people={3: nearby_person},
            pre_detected_boxes=FakeBoxes([FakeBox(bag_box, 24)], [7]),
        )
        self.assertTrue(bags[7].alert_active)

        moved_box = (150, 90, 170, 110)
        for frame_num in range(3, 7):
            uo.process_frame(
                FakeModel(),
                np.zeros((240, 320, 3), dtype=np.uint8),
                bags,
                args,
                1.0 + frame_num / 10.0,
                external_people={
                    3: uo.TrackedPerson(3, (160.0, 100.0), moved_box)
                },
                pre_detected_boxes=FakeBoxes([FakeBox(moved_box, 24)], [7]),
            )
            self.assertTrue(bags[7].alert_active)

        uo.process_frame(
            FakeModel(),
            np.zeros((240, 320, 3), dtype=np.uint8),
            bags,
            args,
            1.7,
            external_people={
                3: uo.TrackedPerson(3, (160.0, 100.0), moved_box)
            },
            pre_detected_boxes=FakeBoxes([FakeBox(moved_box, 24)], [7]),
        )
        self.assertFalse(bags[7].alert_active)

    def test_active_bag_remains_red_and_unresolved_when_detection_is_lost(self):
        args = SimpleNamespace(
            confidence=0.5,
            bag_confidence=0.25,
            imgsz=640,
            stationary_time=1.0,
            person_distance=30.0,
            stationary_distance=20.0,
            alert_cooldown=30.0,
            debug_detections=False,
        )
        bag = uo.TrackedBag(
            track_id=7,
            class_name="backpack",
            bbox=(90, 90, 110, 110),
            position=(100.0, 100.0),
            first_detected_at=uo.datetime.now(),
            last_detected_at=uo.datetime.now(),
            stationary_since=0.0,
            stationary_anchor=(100.0, 100.0),
            alert_active=True,
            alert_anchor=(100.0, 100.0),
            last_seen_time=0.0,
            class_history=["backpack"],
        )
        model = FakeModel()
        pipeline = SimpleNamespace(
            enable_intrusion=False,
            enable_vehicle_tracking=False,
            enable_unattended_bag=True,
            yolo_bag_model=model,
            unattended_bags={7: bag},
            bag_cfg=args,
            _logged_bag_alert_ids={7},
            incident_history=[{
                "type": "UNATTENDED_BAG",
                "status": "ACTIVE",
                "track_id": 7,
                "entry_timestamp": 0.0,
                "entry_time": "00:00:00",
                "class_name": "backpack",
            }],
            event_log=[],
            camera_name="Test Camera",
        )

        frame, status = IntegratedPipeline.process_frame(
            pipeline,
            np.zeros((240, 320, 3), dtype=np.uint8),
            5.0,
        )

        self.assertEqual(status["all_active_alerts"], 1)
        self.assertTrue(bag.alert_active)
        self.assertEqual(pipeline.incident_history[0]["status"], "ACTIVE")
        self.assertFalse(
            any(notification["type"] == "BAG_RESOLVED" for notification in status["notifications"])
        )
        self.assertEqual(frame[90, 90].tolist(), [0, 0, 255])

        model.detections = FakeBoxes([FakeBox((150, 90, 170, 110), 24)], [7])
        for timestamp in (5.1, 5.2, 5.3, 5.4):
            _, status = IntegratedPipeline.process_frame(
                pipeline,
                np.zeros((240, 320, 3), dtype=np.uint8),
                timestamp,
            )
            self.assertEqual(pipeline.incident_history[0]["status"], "ACTIVE")
            self.assertEqual(status["all_active_alerts"], 1)

        _, status = IntegratedPipeline.process_frame(
            pipeline,
            np.zeros((240, 320, 3), dtype=np.uint8),
            5.5,
        )
        self.assertEqual(pipeline.incident_history[0]["status"], "RESOLVED")
        self.assertEqual(status["all_active_alerts"], 0)
        self.assertTrue(
            any(notification["type"] == "BAG_RESOLVED" for notification in status["notifications"])
        )

    def test_vehicle_alert_is_not_cleared_on_missed_detections(self):
        zone = RestrictedZone(
            zone_id="restricted",
            name="Restricted Zone",
            raw_points=[[100, 100], [300, 100], [300, 300], [100, 300]],
            classes={"car"},
            is_normalized=False,
            zone_type="restricted",
        )
        detector = IntrusionDetector(
            zones=[zone],
            confirm_frames=1,
            max_lost_frames=2,
        )
        inside = [{"track_id": 42, "class_name": "car", "bbox": (180, 180, 220, 220)}]
        outside = [{"track_id": 42, "class_name": "car", "bbox": (20, 20, 60, 60)}]

        status = detector.process_tracks(inside, 400, 400, 0.0)
        self.assertTrue(status["active_intrusion"])
        for frame_num in range(1, 6):
            status = detector.process_tracks([], 400, 400, frame_num * 0.1)
        self.assertTrue(status["active_intrusion"])
        self.assertEqual(detector.incident_history[0]["status"], "ACTIVE")

        detector.process_tracks(outside, 400, 400, 0.6)
        self.assertEqual(detector.incident_history[0]["status"], "CLEARED")

    def test_parking_violation_stays_active_while_vehicle_moves_inside_zone(self):
        zone = RestrictedZone(
            zone_id="no-parking",
            name="No Parking",
            raw_points=[[100, 100], [300, 100], [300, 300], [100, 300]],
            classes={"car"},
            is_normalized=False,
            zone_type="no_parking",
            parking_time_threshold=0.1,
            motion_threshold=20.0,
        )
        detector = IntrusionDetector(zones=[zone], confirm_frames=1)
        detector.process_tracks(
            [{"track_id": 9, "class_name": "car", "bbox": (180, 180, 220, 220)}],
            400,
            400,
            0.0,
        )
        status = detector.process_tracks(
            [{"track_id": 9, "class_name": "car", "bbox": (180, 180, 220, 220)}],
            400,
            400,
            0.2,
        )
        self.assertTrue(status["active_intrusion"])

        status = detector.process_tracks(
            [{"track_id": 9, "class_name": "car", "bbox": (220, 180, 260, 220)}],
            400,
            400,
            0.3,
        )
        self.assertTrue(status["active_intrusion"])
        self.assertTrue(status["confirmed_intruders"][0].parking_violation)
        self.assertEqual(detector.incident_history[0]["status"], "ACTIVE")

        detector.process_tracks(
            [{"track_id": 9, "class_name": "car", "bbox": (20, 20, 60, 60)}],
            400,
            400,
            0.4,
        )
        self.assertNotIn((9, "no-parking"), detector.tracked_intruders)
        self.assertEqual(detector.incident_history[0]["status"], "CLEARED")


if __name__ == "__main__":
    unittest.main()
