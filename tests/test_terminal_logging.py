"""
Unit tests for terminal logging of person and vehicle restricted-zone intrusion alerts and zone exits.
"""

import io
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path("D:/Miniproject/video-analytics-platform")
sys.path.insert(0, str(PROJECT_ROOT))

from intrusion_detection import IntrusionDetector, RestrictedZone


class TestTerminalLogging(unittest.TestCase):

    def setUp(self):
        self.zone = RestrictedZone(
            zone_id="restricted_zone_test",
            name="Hazardous Equipment Zone",
            raw_points=[[100, 100], [400, 100], [400, 400], [100, 400]],
            classes={"person", "car", "truck"},
            enabled=True,
            is_normalized=False,
            zone_type="restricted",
        )
        self.detector = IntrusionDetector(
            zones=[self.zone],
            confirm_frames=1,
            max_lost_frames=3,
            camera_name="Factory CCTV",
            simulated_start_time="10:15:00",
        )

    def test_person_entry_exit_and_no_duplicates(self):
        """Test person entry alert, exit logging, single dispatch (no duplicates), and duration."""
        captured_output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            # 1. Frame at t=20s (10:15:20) - Person 7 enters
            tracks_entry = [{
                "track_id": 7,
                "class_name": "person",
                "bbox": (150, 150, 250, 350), # feet at (200, 350) inside zone
                "conf": 0.90,
            }]
            self.detector.process_tracks(tracks_entry, 720, 480, 20.0)

            out_1 = captured_output.getvalue()
            self.assertIn("[CRITICAL ALERT] PERSON INTRUSION", out_1)
            self.assertIn("Camera: Factory CCTV", out_1)
            self.assertIn("Track ID: 7", out_1)
            self.assertIn("Class: Person", out_1)
            self.assertIn("Entry Time: 10:15:20", out_1)

            # 2. Frame at t=25s (10:15:25) - Person 7 still inside (MUST NOT DUPLICATE ALERT)
            captured_output.truncate(0)
            captured_output.seek(0)
            self.detector.process_tracks(tracks_entry, 720, 480, 25.0)
            out_2 = captured_output.getvalue()
            self.assertEqual(out_2.strip(), "", "Duplicate alert printed on subsequent frame!")

            # 3. Frame at t=35s (10:15:35) - Person 7 moves outside zone
            tracks_exit = [{
                "track_id": 7,
                "class_name": "person",
                "bbox": (500, 150, 600, 350), # feet at (550, 350) outside zone
                "conf": 0.90,
            }]
            captured_output.truncate(0)
            captured_output.seek(0)
            self.detector.process_tracks(tracks_exit, 720, 480, 35.0)
            out_3 = captured_output.getvalue()

            self.assertIn("[ZONE EXIT] PERSON", out_3)
            self.assertIn("Camera: Factory CCTV", out_3)
            self.assertIn("Track ID: 7", out_3)
            self.assertIn("Class: Person", out_3)
            self.assertIn("Entry Time: 10:15:20", out_3)
            self.assertIn("Exit Time: 10:15:35", out_3)
            self.assertIn("Duration: 15 seconds", out_3)

            # 4. Verify incident history persists even after exit
            history = self.detector.get_incident_history()
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["status"], "CLEARED")
            self.assertEqual(history[0]["track_id"], 7)
            self.assertEqual(history[0]["duration_seconds"], 15)

        finally:
            sys.stdout = old_stdout

    def test_vehicle_reentry_creates_new_event(self):
        """Test vehicle intrusion alert and re-entry triggering a fresh event."""
        captured_output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            # 1. Entry at t=10s (10:15:10) - Car 12 enters
            tracks_entry = [{
                "track_id": 12,
                "class_name": "car",
                "bbox": (150, 150, 300, 300),
                "conf": 0.88,
            }]
            self.detector.process_tracks(tracks_entry, 720, 480, 10.0)
            out_1 = captured_output.getvalue()

            self.assertIn("[CRITICAL ALERT] VEHICLE INTRUSION", out_1)
            self.assertIn("Camera: Factory CCTV", out_1)
            self.assertIn("Track ID: 12", out_1)
            self.assertIn("Class: Car", out_1)
            self.assertIn("Entry Time: 10:15:10", out_1)

            # 2. Exit at t=22s (10:15:22)
            tracks_outside = [{
                "track_id": 12,
                "class_name": "car",
                "bbox": (500, 150, 650, 300),
                "conf": 0.88,
            }]
            captured_output.truncate(0)
            captured_output.seek(0)
            self.detector.process_tracks(tracks_outside, 720, 480, 22.0)
            out_2 = captured_output.getvalue()

            self.assertIn("[ZONE EXIT] VEHICLE", out_2)
            self.assertIn("Exit Time: 10:15:22", out_2)
            self.assertIn("Duration: 12 seconds", out_2)

            # 3. Re-entry at t=40s (10:15:40) - MUST PRINT A BRAND NEW EVENT
            captured_output.truncate(0)
            captured_output.seek(0)
            self.detector.process_tracks(tracks_entry, 720, 480, 40.0)
            out_3 = captured_output.getvalue()

            self.assertIn("[CRITICAL ALERT] VEHICLE INTRUSION", out_3)
            self.assertIn("Entry Time: 10:15:40", out_3)

            # 4. Verify incident history holds both events permanently
            history = self.detector.get_incident_history()
            self.assertEqual(len(history), 2)
            self.assertEqual(history[0]["status"], "CLEARED")
            self.assertEqual(history[0]["entry_time"], "10:15:10")
            self.assertEqual(history[1]["status"], "ACTIVE")
            self.assertEqual(history[1]["entry_time"], "10:15:40")

        finally:
            sys.stdout = old_stdout


if __name__ == "__main__":
    unittest.main()
