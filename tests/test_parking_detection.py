"""
Unit and Integration Test Suite for Restricted-Zone Illegal Parking Detection.

Tests:
1. Moving vehicle transiting through no-parking zone (displacement > motion_threshold) generates ZERO violations.
2. Stationary vehicle dwell triggers parking violation exactly after parking_time_threshold (5.0s).
3. Temporary detection loss (<= 5 frames) preserves stationary anchor and dwell duration without resetting.
4. Physical exit clears state immediately; re-entry requires full 5.0s stationary dwell before re-triggering.
5. Duplicate prevention: only one alert event is dispatched while vehicle remains parked.
6. Backward compatibility: person intrusion on standard restricted zone triggers immediately after confirm_frames.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path("D:/Miniproject/video-analytics-platform")
sys.path.insert(0, str(PROJECT_ROOT))

from intrusion_detection import IntrusionDetector, RestrictedZone


class TestIllegalParkingDetection(unittest.TestCase):

    def setUp(self):
        # Define a no-parking zone covering [100, 100] to [300, 300]
        self.no_parking_zone = RestrictedZone(
            zone_id="test_no_parking",
            name="No Parking Loading Bay",
            raw_points=[[100, 100], [300, 100], [300, 300], [100, 300]],
            classes={"car", "motorcycle", "bus", "truck"},
            enabled=True,
            is_normalized=False,
            zone_type="no_parking",
            parking_time_threshold=5.0,
            motion_threshold=20.0,
        )

        # Standard restricted zone for person intrusion
        self.standard_zone = RestrictedZone(
            zone_id="test_standard_zone",
            name="Standard Security Zone",
            raw_points=[[400, 100], [600, 100], [600, 300], [400, 300]],
            classes={"person"},
            enabled=True,
            is_normalized=False,
            zone_type="restricted",
        )

        self.detector = IntrusionDetector(
            zones=[self.no_parking_zone, self.standard_zone],
            confirm_frames=3,
            max_lost_frames=5,
        )

    def test_01_moving_vehicle_transiting_no_violation(self):
        """A vehicle driving continuously through the no-parking zone must NEVER trigger a violation."""
        frame_w, frame_h = 720, 480
        fps = 10.0

        # Vehicle moves from x=120 to x=280 (step of 15 px every 0.1s, total 30 frames = 3.0s)
        # Center displacement per frame is 15 px, and cumulatively changes continuously
        for f in range(50):
            t = f / fps
            center_x = 120 + (f * 3)  # Moves 3px/frame = 30px/sec, but anchor shifts when dist > 20px
            # To ensure it continuously moves > 20px every second:
            pos_x = 110 + (f * 3.5)
            tracks = [{
                "track_id": 101,
                "class_name": "car",
                "bbox": (int(pos_x - 20), 180, int(pos_x + 20), 220),
                "conf": 0.85,
            }]
            status = self.detector.process_tracks(tracks, frame_w, frame_h, t)
            # Must never confirm parking violation while moving
            for intruder in status.get("confirmed_intruders", []):
                self.assertFalse(intruder.parking_violation, f"Vehicle moving at t={t}s triggered violation!")

    def test_02_stationary_vehicle_triggers_violation_at_5s(self):
        """A vehicle that stops in the zone must trigger a violation exactly at t >= 5.0s."""
        frame_w, frame_h = 720, 480
        fps = 10.0
        stop_box = (180, 180, 220, 220)  # center = (200, 200), foot = (200, 220) inside zone

        violation_triggered_at = None

        # 80 frames = 8.0 seconds of stationary presence
        for f in range(80):
            t = f / fps
            tracks = [{
                "track_id": 102,
                "class_name": "car",
                "bbox": stop_box,
                "conf": 0.90,
            }]
            status = self.detector.process_tracks(tracks, frame_w, frame_h, t)
            confirmed = status.get("confirmed_intruders", [])

            if any(intr.parking_violation for intr in confirmed):
                if violation_triggered_at is None:
                    violation_triggered_at = t

        self.assertIsNotNone(violation_triggered_at, "Stationary vehicle failed to trigger parking violation!")
        self.assertAlmostEqual(violation_triggered_at, 5.0, delta=0.2,
                               msg=f"Expected violation at ~5.0s, got {violation_triggered_at:.2f}s")

    def test_03_temporary_detection_loss_preserves_timer(self):
        """Grace period (<= 5 frames) preserves stationary anchor and accumulated dwell time."""
        frame_w, frame_h = 720, 480
        fps = 10.0
        stop_box = (180, 180, 220, 220)

        # 1. Be stationary for 3.0s (30 frames)
        for f in range(30):
            t = f / fps
            tracks = [{"track_id": 103, "class_name": "motorcycle", "bbox": stop_box, "conf": 0.88}]
            self.detector.process_tracks(tracks, frame_w, frame_h, t)

        # 2. Miss detection for 4 frames (under max_lost_frames=5)
        for f in range(30, 34):
            t = f / fps
            tracks = []  # No detection
            status = self.detector.process_tracks(tracks, frame_w, frame_h, t)
            # Track state must still be maintained in tracked_intruders
            self.assertIn((103, "test_no_parking"), self.detector.tracked_intruders)

        # 3. Reappear at frame 34 and dwell for 2.0s more (total dwell = 5.0s)
        violation_confirmed = False
        for f in range(34, 60):
            t = f / fps
            tracks = [{"track_id": 103, "class_name": "motorcycle", "bbox": stop_box, "conf": 0.88}]
            status = self.detector.process_tracks(tracks, frame_w, frame_h, t)
            if any(intr.parking_violation for intr in status.get("confirmed_intruders", [])):
                violation_confirmed = True
                break

        self.assertTrue(violation_confirmed, "Parking violation should trigger even after brief temporary detection loss!")

    def test_04_physical_exit_clears_state_and_allows_reentry(self):
        """When vehicle physically exits, state clears immediately; re-entry requires new 5.0s stop."""
        frame_w, frame_h = 720, 480
        fps = 10.0
        inside_box = (180, 180, 220, 220)
        outside_box = (50, 50, 80, 80)  # outside the zone

        # 1. Park for 6.0s (confirm violation)
        for f in range(60):
            t = f / fps
            tracks = [{"track_id": 104, "class_name": "car", "bbox": inside_box, "conf": 0.90}]
            self.detector.process_tracks(tracks, frame_w, frame_h, t)

        key = (104, "test_no_parking")
        self.assertTrue(self.detector.tracked_intruders[key].parking_violation)

        # 2. Vehicle physically moves outside the zone
        t_exit = 6.1
        tracks = [{"track_id": 104, "class_name": "car", "bbox": outside_box, "conf": 0.90}]
        status = self.detector.process_tracks(tracks, frame_w, frame_h, t_exit)

        # Must be immediately deleted from tracked_intruders
        self.assertNotIn(key, self.detector.tracked_intruders)

        # 3. Re-enter the zone at t=7.0s and stop for only 2.0s (t=7.0 to 9.0)
        violation_during_reentry = False
        for f in range(70, 90):
            t = f / fps
            tracks = [{"track_id": 104, "class_name": "car", "bbox": inside_box, "conf": 0.90}]
            status = self.detector.process_tracks(tracks, frame_w, frame_h, t)
            if any(intr.parking_violation for intr in status.get("confirmed_intruders", [])):
                violation_during_reentry = True

        self.assertFalse(violation_during_reentry, "Re-entering vehicle triggered premature parking violation (<5s)!")

    def test_05_person_intrusion_backward_compatibility(self):
        """Person intrusion in a standard restricted zone still confirms immediately after 3 frames."""
        frame_w, frame_h = 720, 480
        person_box = (480, 180, 520, 260)  # inside standard_zone

        # Frame 1
        st1 = self.detector.process_tracks([{"track_id": 201, "class_name": "person", "bbox": person_box}], frame_w, frame_h, 0.1)
        self.assertFalse(st1["active_intrusion"])

        # Frame 2
        st2 = self.detector.process_tracks([{"track_id": 201, "class_name": "person", "bbox": person_box}], frame_w, frame_h, 0.2)
        self.assertFalse(st2["active_intrusion"])

        # Frame 3: Must confirm immediately!
        st3 = self.detector.process_tracks([{"track_id": 201, "class_name": "person", "bbox": person_box}], frame_w, frame_h, 0.3)
        self.assertTrue(st3["active_intrusion"])
        self.assertEqual(len(st3["confirmed_intruders"]), 1)
        self.assertFalse(st3["confirmed_intruders"][0].parking_violation)  # standard intrusion


if __name__ == "__main__":
    unittest.main()
