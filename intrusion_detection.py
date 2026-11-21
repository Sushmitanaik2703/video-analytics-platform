"""
Restricted-Zone Intrusion Detection Module (Member 2).

Detects unauthorized access of tracked people (and vehicles) into predefined
geometrical security zones (convex or concave polygons).

Key Capabilities:
1. Multi-zone polygon configuration from JSON (normalized 0.0-1.0 or pixel coordinates).
2. Ground-plane intrusion testing using the bottom-center feet contact point:
   feet_point = ((x1 + x2) // 2, y2).
3. Temporal persistence smoothing: requires >= 3 consecutive processed frames inside
   the zone before confirming intrusion.
4. Single-pass YOLO inference + ByteTrack tracking with zero redundant model runs.
5. High-visibility security rendering (semi-transparent zone fill, red alert banner,
   intruder bounding box highlighting).
6. Non-blocking audio alarm dispatch with alert cooldown.
7. Real-time video playback synchronization for live CCTV demonstration.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import winsound
except ImportError:
    winsound = None

import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class RestrictedZone:
    """Represents a single restricted zone boundary."""

    zone_id: str
    name: str
    raw_points: list[list[float]]
    classes: set[str] = field(default_factory=lambda: {"person"})
    enabled: bool = True
    is_normalized: bool = True
    zone_type: str = "restricted"             # "restricted" (immediate) or "no_parking" (stationary dwell)
    parking_time_threshold: float = 5.0      # Configurable stationary seconds before parking violation
    motion_threshold: float = 20.0           # Max pixel displacement to consider vehicle stationary

    @classmethod
    def from_dict(cls, data: dict) -> RestrictedZone:
        zone_id = str(data.get("id", f"zone_{id(data)}"))
        name = str(data.get("name", "Restricted Area"))
        points = data.get("points", [])
        classes = set(data.get("classes", ["person"]))
        enabled = bool(data.get("enabled", True))
        zone_type = str(data.get("zone_type", "restricted")).lower()
        parking_time_threshold = float(data.get("parking_time_threshold", 5.0))
        motion_threshold = float(data.get("motion_threshold", 20.0))

        # Check if coordinates are normalized (all between 0.0 and 1.0)
        is_normalized = True
        for pt in points:
            if pt[0] > 1.0 or pt[1] > 1.0:
                is_normalized = False
                break

        return cls(
            zone_id=zone_id,
            name=name,
            raw_points=points,
            classes=classes,
            enabled=enabled,
            is_normalized=is_normalized,
            zone_type=zone_type,
            parking_time_threshold=parking_time_threshold,
            motion_threshold=motion_threshold,
        )

    def get_pixel_polygon(self, frame_w: int, frame_h: int) -> np.ndarray:
        """Convert zone points to an integer pixel coordinate contour array."""
        pixel_pts = []
        for pt in self.raw_points:
            if self.is_normalized:
                px = int(round(pt[0] * frame_w))
                py = int(round(pt[1] * frame_h))
            else:
                px = int(round(pt[0]))
                py = int(round(pt[1]))
            # Clamp to frame boundary
            px = max(0, min(frame_w - 1, px))
            py = max(0, min(frame_h - 1, py))
            pixel_pts.append([px, py])
        return np.array(pixel_pts, dtype=np.int32).reshape((-1, 1, 2))

    def contains_point(self, point: tuple[int, int], frame_w: int, frame_h: int) -> bool:
        """
        Test if a 2D point lies inside the polygon.
        Uses cv2.pointPolygonTest: returns >= 0 if inside or on the boundary.
        """
        contour = self.get_pixel_polygon(frame_w, frame_h)
        if len(contour) < 3:
            return False
        dist = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), measureDist=False)
        return dist >= 0


@dataclass
class IntruderTrack:
    """State tracking for an entity inside a restricted zone."""

    track_id: int
    class_name: str
    zone_id: str
    zone_name: str
    consecutive_frames: int = 0
    first_detected_time: float = 0.0
    last_detected_time: float = 0.0
    confirmed: bool = False
    feet_point: tuple[int, int] = (0, 0)
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    lost_frames: int = 0
    # Parking detection attributes:
    stationary_anchor: tuple[float, float] = (0.0, 0.0)
    stationary_since: float = 0.0
    is_stationary: bool = False
    parking_violation: bool = False
    alert_fired: bool = False
    dwell_duration: float = 0.0
    # Schedule & Alert attributes:
    alert_level: str = "NORMAL"         # "NORMAL", "WARNING" (daytime), "CRITICAL" (night or parking)
    alert_type: str = "NONE"            # "DAYTIME_WARNING", "NIGHT_INTRUSION", "PARKING_VIOLATION", "VEHICLE_INTRUSION"
    time_mode_at_entry: str = "daytime" # "daytime" or "night"
    current_time_mode: str = "daytime"  # dynamically updated
    notification_sent: bool = False
    alert_escalated: bool = False       # True if switched from warning to critical due to schedule transition
    first_detected_time_str: str = ""   # Human-readable entry time HH:MM:SS
    entry_logged: bool = False          # Prevent duplicate terminal alert prints
    exit_logged: bool = False           # Prevent duplicate terminal exit prints


def load_zones_from_file(config_path: str | Path) -> list[RestrictedZone]:
    """Load restricted zone definitions from a JSON configuration file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Zone config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_zones = data.get("zones", [])
    zones = [RestrictedZone.from_dict(z) for z in raw_zones if z.get("enabled", True)]
    return zones


def play_audio_alert(frequency: int = 1800, duration_ms: int = 350) -> None:
    """Dispatch non-blocking audio beep on Windows in a background thread."""
    if winsound is not None:

        def _beep():
            try:
                winsound.Beep(frequency, duration_ms)
            except Exception:
                pass

        threading.Thread(target=_beep, daemon=True).start()


class IntrusionDetector:
    """
    Modular Restricted-Zone Intrusion Detection Engine.
    
    Can operate in two modes:
    1. Standalone: Accepts raw frames and runs YOLO + ByteTrack internally.
    2. Integrated: Accepts pre-computed bounding boxes/tracks from an upstream YOLO runner.
    """

    def __init__(
        self,
        zones: list[RestrictedZone] | str | Path | None = None,
        yolo_model: YOLO | None = None,
        confidence: float = 0.50,
        confirm_frames: int = 3,
        alert_cooldown: float = 5.0,
        imgsz: int = 640,
        max_lost_frames: int = 5,
        schedule_mode: str = "auto",
        simulated_start_time: str | None = None,
        camera_name: str | None = None,
    ):
        if isinstance(zones, (str, Path)):
            self.zones = load_zones_from_file(zones)
        elif isinstance(zones, list):
            self.zones = zones
        elif zones is None:
            default_cfg = Path(__file__).resolve().parent / "config" / "default_zones.json"
            self.zones = load_zones_from_file(default_cfg) if default_cfg.is_file() else []
        else:
            self.zones = []

        if isinstance(yolo_model, (str, Path)):
            self.model = YOLO(str(yolo_model))
        else:
            self.model = yolo_model
        self.confidence = confidence
        self.confirm_frames = confirm_frames
        self.alert_cooldown = alert_cooldown
        self.imgsz = imgsz
        self.max_lost_frames = max_lost_frames
        self.schedule_mode = schedule_mode
        self.simulated_start_time = simulated_start_time
        self.camera_name = camera_name or "CCTV Camera"

        # Tracking state: key = (track_id, zone_id)
        self.tracked_intruders: dict[tuple[int, str], IntruderTrack] = {}
        self.last_audio_alert_time: float = 0.0
        self.event_log: list[dict] = []
        self.incident_history: list[dict] = []
        self._base_wall_time: datetime.datetime | None = None

    def get_current_time_mode(self, video_timestamp: float) -> tuple[str, str]:
        """
        Determine whether current time is 'daytime' (08:00 - 18:00) or 'night' (18:00 - 08:00).
        Returns (mode, formatted_time_str) where mode is 'daytime' or 'night'.
        """
        if self.simulated_start_time:
            parts = [int(p) for p in self.simulated_start_time.split(":")]
            h = parts[0]
            m = parts[1] if len(parts) > 1 else 0
            s = parts[2] if len(parts) > 2 else 0
            base_dt = datetime.datetime.combine(datetime.date.today(), datetime.time(h, m, s))
            curr_dt = base_dt + datetime.timedelta(seconds=video_timestamp)
            curr_t = curr_dt.time()
            time_str = curr_dt.strftime("%H:%M:%S")
        elif self.schedule_mode == "daytime":
            base_dt = datetime.datetime.combine(datetime.date.today(), datetime.time(10, 0, 0))
            curr_dt = base_dt + datetime.timedelta(seconds=video_timestamp)
            curr_t = curr_dt.time()
            time_str = curr_dt.strftime("%H:%M:%S")
        elif self.schedule_mode == "night":
            base_dt = datetime.datetime.combine(datetime.date.today(), datetime.time(22, 0, 0))
            curr_dt = base_dt + datetime.timedelta(seconds=video_timestamp)
            curr_t = curr_dt.time()
            time_str = curr_dt.strftime("%H:%M:%S")
        else:
            if self._base_wall_time is None:
                self._base_wall_time = datetime.datetime.now()
            curr_dt = self._base_wall_time + datetime.timedelta(seconds=video_timestamp)
            curr_t = curr_dt.time()
            time_str = curr_dt.strftime("%H:%M:%S")

        t_val = curr_t.hour * 60 + curr_t.minute + curr_t.second / 60.0
        # 08:00 is 480 mins, 18:00 is 1080 mins
        if 480.0 <= t_val < 1080.0:
            return "daytime", time_str
        else:
            return "night", time_str

    def process_tracks(
        self,
        tracks: list[dict],
        frame_w: int,
        frame_h: int,
        timestamp: float,
    ) -> dict:
        """
        Evaluate intrusion rules on pre-tracked objects.
        Each item in `tracks`:
          {'track_id': int, 'class_name': str, 'bbox': (x1, y1, x2, y2), 'conf': float}
        """
        time_mode, current_time_str = self.get_current_time_mode(timestamp)
        active_track_zone_keys = set()
        confirmed_intruders: list[IntruderTrack] = []
        any_confirmed_intrusion = False
        new_notifications: list[dict] = []

        # Identify all track IDs present in the current frame
        present_track_ids = {trk["track_id"] for trk in tracks}

        VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}

        for trk in tracks:
            track_id = trk["track_id"]
            class_name = trk.get("class_name", "person")
            x1, y1, x2, y2 = trk["bbox"]

            # Ground-plane contact point: bottom-center of bounding box
            feet_point = (int(round((x1 + x2) / 2.0)), int(round(y2)))
            center_int_pt = (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0)))

            for zone in self.zones:
                if not zone.enabled:
                    continue

                # Filter by allowed/monitored classes for this zone
                if zone.classes and class_name not in zone.classes:
                    continue

                key = (track_id, zone.zone_id)
                is_vehicle = class_name in VEHICLE_CLASSES
                if is_vehicle:
                    is_inside = zone.contains_point(feet_point, frame_w, frame_h) or zone.contains_point(center_int_pt, frame_w, frame_h)
                else:
                    is_inside = zone.contains_point(feet_point, frame_w, frame_h)

                if is_inside:
                    active_track_zone_keys.add(key)
                    center_pt = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    is_vehicle_no_parking = (zone.zone_type == "no_parking" and class_name in VEHICLE_CLASSES)

                    if key not in self.tracked_intruders:
                        # First frame entering the zone
                        self.tracked_intruders[key] = IntruderTrack(
                            track_id=track_id,
                            class_name=class_name,
                            zone_id=zone.zone_id,
                            zone_name=zone.name,
                            consecutive_frames=1,
                            first_detected_time=timestamp,
                            first_detected_time_str=current_time_str,
                            last_detected_time=timestamp,
                            confirmed=(self.confirm_frames <= 1 or (is_vehicle and not is_vehicle_no_parking)),
                            feet_point=feet_point,
                            bbox=(x1, y1, x2, y2),
                            lost_frames=0,
                            stationary_anchor=center_pt if is_vehicle_no_parking else (0.0, 0.0),
                            stationary_since=timestamp if is_vehicle_no_parking else 0.0,
                            is_stationary=True if is_vehicle_no_parking else False,
                            parking_violation=False,
                            alert_fired=False,
                            dwell_duration=0.0,
                            alert_level="NORMAL",
                            alert_type="NONE",
                            time_mode_at_entry=time_mode,
                            current_time_mode=time_mode,
                            notification_sent=False,
                            alert_escalated=False,
                            entry_logged=False,
                            exit_logged=False,
                        )
                    else:
                        intruder = self.tracked_intruders[key]
                        intruder.lost_frames = 0
                        intruder.consecutive_frames += 1
                        intruder.last_detected_time = timestamp
                        intruder.feet_point = feet_point
                        intruder.bbox = (x1, y1, x2, y2)
                        intruder.current_time_mode = time_mode

                        if is_vehicle_no_parking:
                            # Evaluate center-point displacement from stationary anchor
                            ax, ay = intruder.stationary_anchor
                            cx, cy = center_pt
                            dist = ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5

                            if dist > zone.motion_threshold:
                                # Vehicle moved beyond threshold: reset anchor and stationary timer
                                intruder.stationary_anchor = center_pt
                                intruder.stationary_since = timestamp
                                intruder.is_stationary = False
                                intruder.parking_violation = False
                                intruder.dwell_duration = 0.0
                            else:
                                # Vehicle remains stationary within threshold
                                intruder.is_stationary = True
                                dwell = max(0.0, timestamp - intruder.stationary_since)
                                intruder.dwell_duration = dwell
                                if dwell >= zone.parking_time_threshold:
                                    intruder.parking_violation = True
                                    intruder.confirmed = True
                                    intruder.alert_level = "CRITICAL"
                                    intruder.alert_type = "PARKING_VIOLATION"
                                    if not intruder.alert_fired:
                                        any_confirmed_intrusion = True
                                        intruder.alert_fired = True
                                        alert_msg = (
                                            f"\n[CRITICAL ALERT] VEHICLE INTRUSION (ILLEGAL PARKING)\n"
                                            f"Camera: {self.camera_name}\n"
                                            f"Track ID: {track_id}\n"
                                            f"Class: {class_name.capitalize()}\n"
                                            f"Entry Time: {intruder.first_detected_time_str or current_time_str}\n"
                                            f"Violation Time: {current_time_str}\n"
                                            f"Dwell: {dwell:.1f}s\n"
                                        )
                                        print(alert_msg, flush=True)
                                        notif = {
                                            "level": "CRITICAL",
                                            "type": "PARKING_VIOLATION",
                                            "track_id": track_id,
                                            "zone_name": zone.name,
                                            "timestamp": timestamp,
                                            "time_str": current_time_str,
                                            "message": f"[ILLEGAL PARKING ALERT] Vehicle ID {track_id} stationary in '{zone.name}' for {dwell:.1f}s at {current_time_str} (t={timestamp:.2f}s).",
                                        }
                                        new_notifications.append(notif)
                                        self.event_log.append(notif)
                                        print(f"  {notif['message']}", flush=True)
                        else:
                            if intruder.consecutive_frames >= self.confirm_frames:
                                intruder.confirmed = True

                    intruder = self.tracked_intruders[key]

                    # For confirmed person intrusion or general vehicle intrusion:
                    if intruder.confirmed:
                        if class_name == "person":
                            if intruder.alert_level == "NORMAL":
                                intruder.alert_level = "CRITICAL"
                                intruder.alert_type = "PERSON_INTRUSION"
                            if not intruder.entry_logged:
                                alert_msg = (
                                    f"\n[CRITICAL ALERT] PERSON INTRUSION\n"
                                    f"Camera: {self.camera_name}\n"
                                    f"Track ID: {track_id}\n"
                                    f"Class: Person\n"
                                    f"Entry Time: {current_time_str}\n"
                                )
                                print(alert_msg, flush=True)
                                intruder.entry_logged = True
                                notif = {
                                    "level": "CRITICAL",
                                    "type": "PERSON_INTRUSION",
                                    "track_id": track_id,
                                    "class_name": "person",
                                    "zone_name": zone.name,
                                    "timestamp": timestamp,
                                    "time_str": current_time_str,
                                    "message": f"[RESTRICTED ZONE PERSON INTRUSION] Critical Red Alert! Person ID {track_id} entered '{zone.name}' at {current_time_str} (t={timestamp:.2f}s).",
                                }
                                new_notifications.append(notif)
                                self.event_log.append(notif)
                                self.incident_history.append({
                                    "incident_id": len(self.incident_history) + 1,
                                    "event": "INTRUSION_ENTRY",
                                    "type": "PERSON_INTRUSION",
                                    "camera": self.camera_name,
                                    "track_id": track_id,
                                    "class_name": "person",
                                    "zone_name": zone.name,
                                    "entry_time": current_time_str,
                                    "entry_timestamp": timestamp,
                                    "exit_time": None,
                                    "exit_timestamp": None,
                                    "duration_seconds": None,
                                    "status": "ACTIVE",
                                })
                                intruder.notification_sent = True
                        elif class_name in VEHICLE_CLASSES and zone.zone_type != "no_parking":
                            if intruder.alert_level == "NORMAL":
                                intruder.alert_level = "CRITICAL"
                                intruder.alert_type = "VEHICLE_INTRUSION"
                            if not intruder.entry_logged:
                                alert_msg = (
                                    f"\n[CRITICAL ALERT] VEHICLE INTRUSION\n"
                                    f"Camera: {self.camera_name}\n"
                                    f"Track ID: {track_id}\n"
                                    f"Class: {class_name.capitalize()}\n"
                                    f"Entry Time: {current_time_str}\n"
                                )
                                print(alert_msg, flush=True)
                                intruder.entry_logged = True
                                notif = {
                                    "level": "CRITICAL",
                                    "type": "VEHICLE_INTRUSION",
                                    "track_id": track_id,
                                    "class_name": class_name,
                                    "zone_name": zone.name,
                                    "timestamp": timestamp,
                                    "time_str": current_time_str,
                                    "message": f"[RESTRICTED ZONE VEHICLE INTRUSION] Critical Red Alert! Vehicle ID {track_id} entered '{zone.name}' at {current_time_str} (t={timestamp:.2f}s).",
                                }
                                new_notifications.append(notif)
                                self.event_log.append(notif)
                                self.incident_history.append({
                                    "incident_id": len(self.incident_history) + 1,
                                    "event": "INTRUSION_ENTRY",
                                    "type": "VEHICLE_INTRUSION",
                                    "camera": self.camera_name,
                                    "track_id": track_id,
                                    "class_name": class_name,
                                    "zone_name": zone.name,
                                    "entry_time": current_time_str,
                                    "entry_timestamp": timestamp,
                                    "exit_time": None,
                                    "exit_timestamp": None,
                                    "duration_seconds": None,
                                    "status": "ACTIVE",
                                })
                                intruder.notification_sent = True

                        confirmed_intruders.append(intruder)
                        if not is_vehicle_no_parking or intruder.parking_violation:
                            any_confirmed_intrusion = True

        # Helper for logging confirmed zone exit
        def _record_zone_exit(intruder_obj: IntruderTrack, exit_t: float, exit_t_str: str):
            if intruder_obj.confirmed and not intruder_obj.exit_logged:
                intruder_obj.exit_logged = True
                cat = "PERSON" if intruder_obj.class_name == "person" else "VEHICLE"
                dur = max(1, int(round(exit_t - intruder_obj.first_detected_time)))
                ent_str = intruder_obj.first_detected_time_str or exit_t_str
                exit_msg = (
                    f"\n[ZONE EXIT] {cat}\n"
                    f"Camera: {self.camera_name}\n"
                    f"Track ID: {intruder_obj.track_id}\n"
                    f"Class: {intruder_obj.class_name.capitalize()}\n"
                    f"Entry Time: {ent_str}\n"
                    f"Exit Time: {exit_t_str}\n"
                    f"Duration: {dur} seconds\n"
                )
                print(exit_msg, flush=True)

                exit_record = {
                    "level": "INFO",
                    "type": "ZONE_EXIT",
                    "track_id": intruder_obj.track_id,
                    "class_name": intruder_obj.class_name,
                    "zone_name": intruder_obj.zone_name,
                    "timestamp": exit_t,
                    "time_str": exit_t_str,
                    "entry_time": ent_str,
                    "exit_time": exit_t_str,
                    "duration_seconds": dur,
                    "message": f"[ZONE EXIT] {intruder_obj.class_name.capitalize()} ID {intruder_obj.track_id} exited '{intruder_obj.zone_name}' at {exit_t_str} (t={exit_t:.2f}s). Alert cleared.",
                }
                new_notifications.append(exit_record)
                self.event_log.append(exit_record)

                # Keep and update incident history record permanently
                for inc in reversed(self.incident_history):
                    if inc.get("track_id") == intruder_obj.track_id and inc.get("status") == "ACTIVE":
                        inc["status"] = "CLEARED"
                        inc["exit_time"] = exit_t_str
                        inc["exit_timestamp"] = exit_t
                        inc["duration_seconds"] = dur
                        break

        # Handle tracks not detected inside their respective zone in this frame
        for key in list(self.tracked_intruders):
            if key in active_track_zone_keys:
                continue

            track_id, zone_id = key
            intruder = self.tracked_intruders[key]

            if track_id in present_track_ids:
                # Confirmed physical exit:
                # The entity was actively detected in this frame, but its contact point is outside this zone.
                # Clear active intrusion state immediately so re-entry requires full confirmation.
                _record_zone_exit(intruder, timestamp, current_time_str)
                del self.tracked_intruders[key]
            else:
                # Temporary detection loss:
                # The entity was completely missing from detections (e.g. occlusion, blur, temporary miss).
                intruder.lost_frames += 1
                if intruder.lost_frames > self.max_lost_frames:
                    # Grace period expired: entity disappeared from camera view
                    _record_zone_exit(intruder, intruder.last_detected_time, current_time_str)
                    del self.tracked_intruders[key]
                else:
                    # Entity is within grace period: preserve existing intrusion state
                    if intruder.confirmed:
                        confirmed_intruders.append(intruder)
                        z_obj = next((z for z in self.zones if z.zone_id == intruder.zone_id), None)
                        if z_obj is None or z_obj.zone_type != "no_parking" or intruder.parking_violation:
                            any_confirmed_intrusion = True

        # Audio alert trigger
        if any_confirmed_intrusion:
            if (timestamp - self.last_audio_alert_time) >= self.alert_cooldown:
                self.last_audio_alert_time = timestamp
                has_critical = any(intr.alert_level == "CRITICAL" for intr in confirmed_intruders)
                if has_critical:
                    play_audio_alert(frequency=2000, duration_ms=450)
                else:
                    play_audio_alert(frequency=1200, duration_ms=200)

        return {
            "timestamp": timestamp,
            "time_mode": time_mode,
            "time_str": current_time_str,
            "active_intrusion": any_confirmed_intrusion,
            "confirmed_intruders": confirmed_intruders,
            "all_zone_tracks": list(self.tracked_intruders.values()),
            "notifications": new_notifications,
        }

    def get_incident_history(self) -> list[dict]:
        """Return the complete permanent audit log of all recorded intrusion incidents."""
        return list(self.incident_history)

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
    ) -> tuple[np.ndarray, dict]:
        """
        Standalone frame processing with integrated YOLOv8 + ByteTrack.
        """
        if self.model is None:
            raise RuntimeError("YOLO model not initialized for standalone process_frame()")

        frame_h, frame_w = frame.shape[:2]

        # COCO class mapping for security monitoring
        COCO_CLASS_MAP = {
            0: "person",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }
        CLASS_NAME_TO_COCO = {name: cid for cid, name in COCO_CLASS_MAP.items()}

        monitored_classes = {0}  # Person always monitored
        for z in self.zones:
            if z.enabled and z.classes:
                for c in z.classes:
                    if c in CLASS_NAME_TO_COCO:
                        monitored_classes.add(CLASS_NAME_TO_COCO[c])

        # Single YOLO inference + ByteTrack run
        results = self.model.track(
            frame,
            conf=self.confidence,
            classes=sorted(list(monitored_classes)),
            imgsz=self.imgsz,
            device=0 if torch.cuda.is_available() else "cpu",
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes
        track_ids = (
            boxes.id.int().cpu().tolist()
            if boxes is not None and boxes.id is not None
            else []
        )

        tracks = []
        if boxes is not None:
            for idx, box in enumerate(boxes):
                if idx >= len(track_ids):
                    continue
                t_id = int(track_ids[idx])
                conf = float(box.conf[0].item())
                if conf < self.confidence:
                    continue
                cls_id = int(box.cls[0].item()) if box.cls is not None else 0
                class_name = COCO_CLASS_MAP.get(cls_id, "person")
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                tracks.append({
                    "track_id": t_id,
                    "class_name": class_name,
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                })

        status = self.process_tracks(tracks, frame_w, frame_h, timestamp)
        status["all_frame_tracks"] = tracks
        annotated_frame = self.annotate_frame(frame.copy(), status)
        return annotated_frame, status

    def annotate_frame(
        self,
        frame: np.ndarray,
        status: dict,
    ) -> np.ndarray:
        """
        Render zones, intruder highlights, and security alert banners.
        """
        frame_h, frame_w = frame.shape[:2]
        confirmed_intruders: list[IntruderTrack] = status.get("confirmed_intruders", [])
        time_mode = status.get("time_mode", "daytime")
        current_time_str = status.get("time_str", "12:00:00")
        all_frame_tracks = status.get("all_frame_tracks", [])
        intruder_map = {intr.track_id: intr for intr in confirmed_intruders}

        # 1. Draw Zones (Rule 3: Keep the restricted-zone polygon yellow)
        for zone in self.zones:
            if not zone.enabled:
                continue
            contour = zone.get_pixel_polygon(frame_w, frame_h)
            # Restricted zone polygon is always yellow (0, 255, 255)
            color_border = (0, 255, 255)
            status_tag = "RESTRICTED AREA"

            # Draw clean yellow box boundary outline (no color fill)
            cv2.polylines(frame, [contour], isClosed=True, color=color_border, thickness=2)

            # Zone label badge
            top_pt = contour[np.argmin(contour[:, 0, 1])][0]
            label_text = f"ZONE: {zone.name} [{status_tag}]"
            cv2.putText(
                frame,
                label_text,
                (max(10, top_pt[0]), max(25, top_pt[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color_border,
                2,
                cv2.LINE_AA,
            )

        # 2. Render Tracked Entities
        # Rule 1: Detect and track all people with green bounding boxes and IDs
        # Rule 2: Detect and track all vehicles with blue bounding boxes and IDs
        # Rule 4 & 5: When person OR vehicle enters, change bounding box to RED and keep active
        # Rule 6: Clear alert when object exits (box returns to green/blue)
        for trk in all_frame_tracks:
            track_id = trk["track_id"]
            class_name = trk.get("class_name", "person")
            x1, y1, x2, y2 = trk["bbox"]
            is_veh = class_name in {"car", "motorcycle", "bus", "truck"}

            if track_id in intruder_map:
                intruder = intruder_map[track_id]
                duration = max(0.0, status["timestamp"] - intruder.first_detected_time)
                box_color = (0, 0, 255)  # Bright Red
                tag = f"CRITICAL INTRUDER | {class_name.upper()} #{track_id} ({duration:.1f}s)"

                # Ground contact marker in red
                feet_x = (x1 + x2) // 2
                feet_y = y2
                cv2.circle(frame, (feet_x, feet_y), 5, (0, 0, 255), -1)
            elif is_veh:
                box_color = (255, 0, 0)  # Pure Blue (BGR)
                tag = f"{class_name.capitalize()} #{track_id}"
            else:
                box_color = (0, 255, 0)  # Pure Green (BGR)
                tag = f"Person #{track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), (0, 0, 0), -1)
            cv2.putText(
                frame,
                tag,
                (x1 + 2, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                box_color,
                1,
                cv2.LINE_AA,
            )

        # 3. Top Header Alert Banner
        # Rule 4 & 5: When person OR vehicle enters, show an immediate red alert banner
        # Rule 6: Clear the alert when the object exits
        if status.get("active_intrusion") and confirmed_intruders:
            banner_h = 42
            banner_color = (0, 0, 220)  # Red banner
            cv2.rectangle(frame, (0, 0), (frame_w, banner_h), banner_color, -1)
            first_intr = confirmed_intruders[0]
            alert_text = (
                f"CRITICAL ALERT: RESTRICTED ZONE INTRUSION | ZONE: {first_intr.zone_name.upper()} | "
                f"{first_intr.class_name.upper()} ID: {first_intr.track_id} | [{current_time_str}]"
            )
            cv2.putText(
                frame,
                alert_text,
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # 4. Schedule & Clock Telemetry Badge (Top Right)
        sched_label = f"SCHEDULE: {time_mode.upper()} (08:00-18:00) | {current_time_str}"
        (sw, sh), _ = cv2.getTextSize(sched_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        bx = frame_w - sw - 20
        by = 52 if status.get("active_intrusion") else 10
        cv2.rectangle(frame, (bx - 6, by), (frame_w - 10, by + sh + 10), (30, 30, 30), -1)
        sched_color = (0, 215, 255) if time_mode == "daytime" else (0, 100, 255)
        cv2.putText(
            frame,
            sched_label,
            (bx, by + sh + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            sched_color,
            1,
            cv2.LINE_AA,
        )

        return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restricted-Zone Intrusion Detection System (Member 2)"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Path to input video file")
    source.add_argument("--webcam", action="store_true", help="Use default webcam (index 0)")

    parser.add_argument(
        "--zones",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "default_zones.json",
        help="Path to zone configuration JSON (default: config/default_zones.json)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to YOLO model checkpoint (default: yolov8s.pt or yolov8n.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.50,
        help="Minimum person confidence threshold (default: 0.50)",
    )
    parser.add_argument(
        "--confirm-frames",
        type=int,
        default=3,
        help="Consecutive frames required inside zone to confirm intrusion (default: 3)",
    )
    parser.add_argument(
        "--max-lost-frames",
        type=int,
        default=5,
        help="Maximum temporary lost frames before deleting intruder state (default: 5)",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Optional path to output annotated video file",
    )
    parser.add_argument(
        "--save-screenshots",
        type=Path,
        default=None,
        help="Directory to save intrusion event screenshots",
    )
    parser.add_argument(
        "--schedule-mode",
        choices=["auto", "daytime", "night"],
        default="auto",
        help="Operating schedule mode (default: auto, using current system clock)",
    )
    parser.add_argument(
        "--simulated-time",
        type=str,
        default=None,
        help="Simulated start time HH:MM or HH:MM:SS (advances with video playback)",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default=None,
        help="Camera / stream display name for alerts (default: auto-derived from video filename)",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display live visual window (default: True, use --no-display for headless execution)",
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize playback to original video speed (default: True, use --no-realtime for uncapped)",
    )
    parser.add_argument(
        "--interactive-zone",
        action="store_true",
        default=False,
        help="Interactively draw restricted-zone polygon(s) on the video's first frame",
    )
    return parser.parse_args()


def interactive_draw_zones(
    video_source: int | str | Path,
    classes: set[str] | list[str] | None = None,
    save_path: str | Path | None = None,
) -> list[RestrictedZone]:
    """
    Open the first frame of a video in an interactive OpenCV window.
    Allows user to click points on the frame to define one or multiple restricted zones.
    Keybindings:
      - Left-Click: Add polygon vertex
      - Right-Click: Undo last vertex
      - Enter / Space: Save current polygon into zones list
      - 'r': Reset current polygon vertices
      - 'c': Clear all saved zones
      - 's' or 'q': Finish and return configured zones
    """
    cap = cv2.VideoCapture(video_source if isinstance(video_source, int) else str(video_source))
    if not cap.isOpened():
        print(f"Error: Could not open {video_source} for interactive zone drawing.", file=sys.stderr)
        return []

    ret, frame0 = cap.read()
    cap.release()
    if not ret or frame0 is None:
        print(f"Error: Could not read first frame from {video_source}.", file=sys.stderr)
        return []

    h, w = frame0.shape[:2]
    win_name = "Smart CCTV - Draw Restricted Zones (Click Points | Enter/Space: Save Zone | S/Q: Done)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    saved_zones_raw: list[list[list[float]]] = []
    current_pts: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        nonlocal current_pts
        if event == cv2.EVENT_LBUTTONDOWN:
            current_pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if current_pts:
                current_pts.pop()

    cv2.setMouseCallback(win_name, on_mouse)

    target_classes = set(classes) if classes else {"person", "car", "motorcycle", "bus", "truck"}

    while True:
        canvas = frame0.copy()

        # Draw already saved zones in yellow
        for idx, sz in enumerate(saved_zones_raw):
            pts_arr = (np.array(sz) * [w, h]).astype(np.int32)
            cv2.polylines(canvas, [pts_arr], isClosed=True, color=(0, 255, 255), thickness=2)
            top_pt = pts_arr[np.argmin(pts_arr[:, 1])]
            cv2.putText(
                canvas,
                f"Zone {idx + 1} [RESTRICTED]",
                (max(10, top_pt[0]), max(25, top_pt[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        # Draw current points and lines
        for pt in current_pts:
            cv2.circle(canvas, pt, 5, (0, 0, 255), -1)
            cv2.circle(canvas, pt, 7, (0, 255, 255), 1)

        if len(current_pts) > 1:
            for i in range(len(current_pts) - 1):
                cv2.line(canvas, current_pts[i], current_pts[i + 1], (0, 255, 255), 2)

        # Instruction HUD
        info = (
            f"Saved Zones: {len(saved_zones_raw)} | Current Points: {len(current_pts)} | "
            "L-Click: Add | R-Click: Undo | Enter: Save Zone | R: Reset | S/Q: Finish"
        )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 35), (20, 20, 20), -1)
        cv2.putText(canvas, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(win_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):  # Enter or Space: Save current zone
            if len(current_pts) >= 3:
                norm_pts = [[round(x / w, 4), round(y / h, 4)] for (x, y) in current_pts]
                saved_zones_raw.append(norm_pts)
                print(f"[Zone] Saved Zone {len(saved_zones_raw)} with {len(current_pts)} vertices.")
                current_pts = []
            else:
                print("Notice: A polygon zone requires at least 3 points.")
        elif key == ord("r"):  # Reset current polygon
            current_pts = []
            print("[Zone] Reset current polygon points.")
        elif key == ord("c"):  # Clear all zones
            saved_zones_raw = []
            current_pts = []
            print("[Zone] Cleared all zones.")
        elif key in (ord("s"), ord("q"), 27):  # Finish
            if len(current_pts) >= 3:
                norm_pts = [[round(x / w, 4), round(y / h, 4)] for (x, y) in current_pts]
                saved_zones_raw.append(norm_pts)
                print(f"[Zone] Saved Zone {len(saved_zones_raw)} with {len(current_pts)} vertices.")
            break

    cv2.destroyWindow(win_name)

    zones_list: list[RestrictedZone] = []
    for idx, sz in enumerate(saved_zones_raw):
        z_id = f"custom_zone_{idx + 1}"
        z_name = f"Restricted Zone {idx + 1}"
        rz = RestrictedZone(
            zone_id=z_id,
            name=z_name,
            raw_points=sz,
            classes=target_classes,
            enabled=True,
            is_normalized=True,
            zone_type="restricted",
        )
        zones_list.append(rz)

    if save_path is not None and zones_list:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "zones": [
                {
                    "id": z.zone_id,
                    "name": z.name,
                    "points": z.raw_points,
                    "classes": sorted(list(z.classes)),
                    "enabled": z.enabled,
                    "zone_type": z.zone_type,
                }
                for z in zones_list
            ]
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[Config] Saved {len(zones_list)} zones to {p}")

    return zones_list


def process_video_stream(
    video_source: int | str | Path,
    detector: IntrusionDetector,
    output_video: Path | str | None = None,
    save_screenshots: Path | str | None = None,
    display: bool = False,
    realtime: bool = False,
    progress_callback: object | None = None,
    stop_event: object | None = None,
) -> dict:
    """
    Execute intrusion detection and tracking pipeline over a video stream.
    Streams playback time, logs alerts to terminal, saves annotated output and event screenshots,
    and returns comprehensive audit summary.
    """
    source_desc = f"Webcam {video_source}" if isinstance(video_source, int) else str(video_source)
    capture = cv2.VideoCapture(video_source if isinstance(video_source, int) else str(video_source))
    if not capture.isOpened():
        print(f"Error: Could not open {source_desc}", file=sys.stderr)
        return {}

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if not isinstance(video_source, int) else 0

    print(f"[Video] {source_desc}: {width}x{height} @ {fps:.1f} FPS ({total_frames} frames)")

    writer = None
    if output_video is not None:
        out_path = Path(output_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    screens_dir = Path(save_screenshots) if save_screenshots is not None else None
    if screens_dir is not None:
        screens_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    start_wall_time = time.time()
    saved_intruder_ids = set()
    saved_screenshot_paths: list[str] = []

    try:
        while True:
            if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
                print("Processing halted by user request.", flush=True)
                break

            ret, frame = capture.read()
            if not ret:
                break

            msec = capture.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = msec / 1000.0 if msec > 0 else frame_idx / fps

            annotated_frame, status = detector.process_frame(frame, timestamp)

            # Screenshot on confirmed intrusion events or warnings
            if screens_dir is not None:
                for notif in status.get("notifications", []):
                    shot_key = f"{notif['type']}_{notif['track_id']}"
                    if shot_key not in saved_intruder_ids:
                        saved_intruder_ids.add(shot_key)
                        prefix = notif["type"].lower()
                        shot_path = screens_dir / f"{prefix}_track_{notif['track_id']}_t{notif['timestamp']:.2f}s.jpg"
                        cv2.imwrite(str(shot_path), annotated_frame)
                        saved_screenshot_paths.append(str(shot_path))
                        print(f"  [Event] Saved Event Screenshot: {shot_path.name}")

            # Terminal reflection when video plays (time and alert occurred, like unattended bag)
            for notif in status.get("notifications", []):
                print(f"  >>> [ALERT OCCURRED] Frame {frame_idx:4d} (t={timestamp:.2f}s) | Clock: {status.get('time_str')} | {notif['message']} <<<", flush=True)

            total_sec = total_frames / fps if fps > 0 else 0
            m_curr, s_curr = int(timestamp // 60), int(timestamp % 60)
            m_tot, s_tot = int(total_sec // 60), int(total_sec % 60)
            time_disp = f"{m_curr:02d}:{s_curr:02d} / {m_tot:02d}:{s_tot:02d} (t={timestamp:5.2f}s)"

            confirmed = status.get("confirmed_intruders", [])
            sample_interval = max(1, int(round(fps)))
            if confirmed:
                if frame_idx % sample_interval == 0:
                    intr_strs = [f"{intr.class_name.capitalize()} ID {intr.track_id} in '{intr.zone_name}'" for intr in confirmed]
                    print(f"  [PLAYING {time_disp}] Frame {frame_idx:4d} | [ALERT ACTIVE] {', '.join(intr_strs)}", flush=True)
            else:
                if frame_idx % (sample_interval * 3) == 0 or frame_idx == 1:
                    print(f"  [PLAYING {time_disp}] Frame {frame_idx:4d} | Status: Normal (No Intrusions)", flush=True)

            if writer is not None:
                writer.write(annotated_frame)

            if progress_callback is not None:
                elapsed_now = time.time() - start_wall_time
                fps_now = frame_idx / elapsed_now if elapsed_now > 0 else 0.0
                try:
                    progress_callback({
                        "frame_idx": frame_idx,
                        "total_frames": total_frames,
                        "timestamp": timestamp,
                        "fps": fps_now,
                        "status": status,
                        "annotated_frame": annotated_frame,
                    })
                except Exception:
                    pass

            if display:
                cv2.imshow("Smart CCTV - Restricted-Zone Intrusion Detection", annotated_frame)

            if realtime and not isinstance(video_source, int):
                frame_interval = 1.0 / fps
                target_next_time = start_wall_time + (frame_idx + 1) * frame_interval
                time_remaining = target_next_time - time.time()

                if time_remaining > 0:
                    wait_ms = max(1, int(time_remaining * 1000))
                    key = cv2.waitKey(wait_ms) & 0xFF if display else ord(" ")
                else:
                    key = cv2.waitKey(1) & 0xFF if display else ord(" ")
                    lag = time.time() - target_next_time
                    if lag > frame_interval:
                        skips = min(int(lag / frame_interval), 2)
                        for _ in range(skips):
                            if not capture.grab():
                                break
                            frame_idx += 1
            else:
                key = cv2.waitKey(1) & 0xFF if display else ord(" ")

            if key == ord("q"):
                break

            frame_idx += 1
            if frame_idx % 100 == 0:
                elapsed = time.time() - start_wall_time
                proc_fps = frame_idx / elapsed if elapsed > 0 else 0
                print(f"  Processed {frame_idx}/{total_frames} frames ({proc_fps:.1f} FPS)...")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    total_time = time.time() - start_wall_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0.0
    print(f"[Done] Processed {frame_idx} frames in {total_time:.2f}s ({avg_fps:.1f} FPS).")
    if output_video is not None:
        print(f"[Saved] Output video: {output_video}")

    history = detector.get_incident_history()
    VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
    person_incidents = sum(1 for inc in history if inc.get("class_name") == "person")
    vehicle_incidents = sum(1 for inc in history if inc.get("class_name") in VEHICLE_CLASSES)

    return {
        "total_frames": frame_idx,
        "elapsed_seconds": total_time,
        "fps": avg_fps,
        "incident_history": history,
        "total_incidents": len(history),
        "person_incidents": person_incidents,
        "vehicle_incidents": vehicle_incidents,
        "saved_screenshots": saved_screenshot_paths,
        "output_video": str(output_video) if output_video else None,
    }


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    # Resolve model checkpoint
    if args.model is not None:
        model_path = args.model.resolve()
    else:
        candidate_s = project_root / "yolov8s.pt"
        candidate_n = project_root / "yolov8n.pt"
        model_path = candidate_s if candidate_s.is_file() else candidate_n

    if not model_path.is_file():
        print(f"Error: YOLO model not found: {model_path}", file=sys.stderr)
        return 1

    if args.video is not None and not args.video.is_file():
        print(f"Error: Video file not found: {args.video}", file=sys.stderr)
        return 1

    print(f"[Init] Loading YOLO model: {model_path}...")
    yolo_model = YOLO(str(model_path))

    # Determine friendly camera name
    if args.camera_name:
        cam_name = args.camera_name
    elif args.webcam:
        cam_name = "Webcam 0"
    elif args.video:
        stem = args.video.stem.lower()
        if "factory" in stem:
            cam_name = "Factory CCTV"
        elif "logistics" in stem:
            cam_name = "Logistics CCTV"
        elif "substation" in stem:
            cam_name = "Substation CCTV"
        elif "aboda" in stem:
            cam_name = "Perimeter CCTV"
        else:
            cam_name = f"{args.video.stem.replace('_', ' ').title()} CCTV"
    else:
        cam_name = "CCTV Camera"

    if args.interactive_zone:
        if args.video is None:
            print("Error: --interactive-zone requires a video file passed via --video", file=sys.stderr)
            return 1
        print(f"[Interactive] Opening {args.video.name} for interactive zone drawing...")
        zones = interactive_draw_zones(args.video)
        if not zones:
            print("No zones defined. Exiting.", file=sys.stderr)
            return 1
    else:
        print(f"[Init] Loading Restricted Zones: {args.zones}...")
        zones = args.zones

    detector = IntrusionDetector(
        zones=zones,
        yolo_model=yolo_model,
        confidence=args.confidence,
        confirm_frames=args.confirm_frames,
        max_lost_frames=args.max_lost_frames,
        schedule_mode=args.schedule_mode,
        simulated_start_time=args.simulated_time,
        camera_name=cam_name,
    )

    print(f"[Config] Schedule Mode: {args.schedule_mode} | Simulated Time: {args.simulated_time or 'Current Clock'}")

    res = process_video_stream(
        video_source=0 if args.webcam else args.video,
        detector=detector,
        output_video=args.output_video,
        save_screenshots=args.save_screenshots,
        display=args.display,
        realtime=args.realtime,
    )

    return 0 if res else 1


if __name__ == "__main__":
    sys.exit(main())
