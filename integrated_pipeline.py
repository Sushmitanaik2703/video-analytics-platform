"""
Integrated AI Analytics Pipeline for Smart CCTV Monitoring System

Orchestrates:
1. Restricted-Zone Intrusion Detection (intrusion_detection.py)
2. Vehicle Detection & Tracking (vehicle_tracking.py)
3. Unattended-Bag Detection (unattended_object.py)

Each feature can be toggled independently or combined seamlessly.
Preserves all original detection logic, thresholds, and model weights without modification.
"""

from __future__ import annotations

import datetime
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Import unmodified original modules
from intrusion_detection import IntrusionDetector, RestrictedZone
from vehicle_tracking import VehicleTracker
import unattended_object as uo


class IntegratedPipeline:
    """
    Unified multi-feature AI video analytics engine.
    Wraps existing IntrusionDetector, VehicleTracker, and Unattended Object logic.
    """

    def __init__(
        self,
        yolo_model_path: Path | str,
        enable_intrusion: bool = True,
        enable_vehicle_tracking: bool = False,
        enable_unattended_bag: bool = False,
        zones: list[RestrictedZone] | None = None,
        confidence: float = 0.45,
        bag_confidence: float = 0.25,
        confirm_frames: int = 2,
        stationary_time: float = 15.0,
        person_distance: float = 150.0,
        stationary_distance: float = 25.0,
        alert_cooldown: float = 30.0,
        sched_mode: str = "auto",
        camera_name: str = "CCTV Camera",
    ):
        self.camera_name = camera_name
        self.enable_intrusion = enable_intrusion
        self.enable_vehicle_tracking = enable_vehicle_tracking
        self.enable_unattended_bag = enable_unattended_bag
        self.zones = zones or []
        self.confidence = confidence

        self.incident_history: list[dict] = []
        self.event_log: list[dict] = []
        self._logged_bag_alert_ids: set[int] = set()

        # 1. Initialize Intrusion Detector if enabled
        self.intrusion_detector: IntrusionDetector | None = None
        if self.enable_intrusion:
            model_intrusion = YOLO(str(yolo_model_path))
            self.intrusion_detector = IntrusionDetector(
                zones=self.zones,
                yolo_model=model_intrusion,
                confidence=confidence,
                confirm_frames=confirm_frames,
                max_lost_frames=5,
                schedule_mode=sched_mode,
                camera_name=camera_name,
            )

        # 2. Initialize Standalone Vehicle Tracker if enabled and intrusion is off
        # (When intrusion is active, IntrusionDetector already tracks vehicles in real-time)
        self.vehicle_tracker: VehicleTracker | None = None
        if self.enable_vehicle_tracking and not self.enable_intrusion:
            model_vehicle = YOLO(str(yolo_model_path))
            self.vehicle_tracker = VehicleTracker(
                yolo_model=model_vehicle,
                confidence=confidence,
                imgsz=640,
                max_lost_frames=5,
            )

        # 3. Initialize Unattended Bag State if enabled
        self.unattended_bags: dict[int, uo.TrackedBag] = {}
        self.yolo_bag_model: YOLO | None = None
        self.bag_cfg = SimpleNamespace(
            confidence=confidence,
            bag_confidence=bag_confidence,
            imgsz=640,
            stationary_time=stationary_time,
            person_distance=person_distance,
            stationary_distance=stationary_distance,
            alert_cooldown=alert_cooldown,
            debug_detections=False,
            save_frames=None,
            save_interval=1.0,
        )
        if self.enable_unattended_bag:
            self.yolo_bag_model = YOLO(str(yolo_model_path))

    def process_frame(self, frame: np.ndarray, timestamp: float) -> tuple[np.ndarray, dict]:
        """
        Process a single video frame through all active feature modules.
        Returns the annotated frame and consolidated status dictionary.
        """
        annotated = frame.copy()
        current_time_str = datetime.datetime.now().strftime("%H:%M:%S")

        status: dict = {
            "timestamp": timestamp,
            "time_str": current_time_str,
            "notifications": [],
            "confirmed_intruders": [],
            "unattended_bags": [],
            "active_vehicles": [],
            "all_active_alerts": 0,
            "banner_text": None,
            "banner_level": "NORMAL",
        }

        tracked_people: dict[int, uo.TrackedPerson] = {}

        # ------------------- 1. RESTRICTED-ZONE INTRUSION -------------------
        if self.enable_intrusion and self.intrusion_detector is not None:
            annotated, int_status = self.intrusion_detector.process_frame(annotated, timestamp)
            status["notifications"].extend(int_status.get("notifications", []))
            status["confirmed_intruders"] = int_status.get("confirmed_intruders", [])

            # Extract tracked persons from IntrusionDetector for single-identity person tracking
            for trk in int_status.get("all_frame_tracks", []):
                if trk.get("class_name") == "person":
                    t_id = trk["track_id"]
                    x1, y1, x2, y2 = trk["bbox"]
                    pos = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    tracked_people[t_id] = uo.TrackedPerson(
                        track_id=t_id,
                        position=pos,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                    )

            # Ingest intrusion incident history into master incident list
            for inc in self.intrusion_detector.get_incident_history():
                if not any(x.get("incident_id") == inc.get("incident_id") and x.get("type") == inc.get("type") for x in self.incident_history):
                    self.incident_history.append(inc)
                else:
                    # Update status, exit time, duration for cleared incidents
                    for x in self.incident_history:
                        if x.get("incident_id") == inc.get("incident_id") and x.get("type") == inc.get("type"):
                            x["status"] = inc.get("status")
                            x["exit_time"] = inc.get("exit_time")
                            x["duration_seconds"] = inc.get("duration_seconds")

        # ------------------- 2. STANDALONE VEHICLE TRACKING -------------------
        elif self.enable_vehicle_tracking and self.vehicle_tracker is not None:
            annotated, veh_data = self.vehicle_tracker.process_frame(annotated, timestamp)
            status["active_vehicles"] = list(self.vehicle_tracker.active_tracks.values())

        # ------------------- 3. UNATTENDED BAG DETECTION -------------------
        if self.enable_unattended_bag and self.yolo_bag_model is not None:
            # When intrusion detection is active, share its tracked persons and skip drawing duplicate person boxes
            annotated = uo.process_frame(
                model=self.yolo_bag_model,
                frame=annotated,
                bags=self.unattended_bags,
                args=self.bag_cfg,
                timestamp=timestamp,
                external_people=tracked_people if self.enable_intrusion else None,
                skip_draw_people=self.enable_intrusion,
            )

            # Check for active unattended bag alerts
            active_unattended = [b for b in self.unattended_bags.values() if b.alert_active]
            status["unattended_bags"] = active_unattended

            # Track new unattended bag alert events
            for bag in active_unattended:
                if bag.track_id not in self._logged_bag_alert_ids:
                    self._logged_bag_alert_ids.add(bag.track_id)
                    stationary_duration = max(0.0, timestamp - bag.stationary_since)
                    owner_text = f"Person ID {bag.associated_person_id}" if bag.associated_person_id is not None else "Owner Unknown / Away"

                    print(f"\n[CRITICAL ALERT] UNATTENDED BAG DETECTED", flush=True)
                    print(f"Camera: {self.camera_name}", flush=True)
                    print(f"Track ID: {bag.track_id}", flush=True)
                    print(f"Class: {bag.class_name.capitalize()}", flush=True)
                    print(f"Owner: {owner_text}", flush=True)
                    print(f"Stationary Dwell: {stationary_duration:.1f}s", flush=True)
                    print(f"Alert Time: {current_time_str}\n", flush=True)

                    notif = {
                        "level": "CRITICAL",
                        "type": "UNATTENDED_BAG",
                        "track_id": bag.track_id,
                        "class_name": bag.class_name,
                        "zone_name": "Full-Frame (Automatic)",
                        "timestamp": timestamp,
                        "time_str": current_time_str,
                        "message": f"[UNATTENDED BAG ALERT] Critical! {bag.class_name.capitalize()} #{bag.track_id} stationary for {stationary_duration:.1f}s (owner: {owner_text}) at {current_time_str}.",
                    }
                    status["notifications"].append(notif)
                    self.event_log.append(notif)

                    self.incident_history.append({
                        "incident_id": len(self.incident_history) + 1,
                        "event": "UNATTENDED_BAG",
                        "type": "UNATTENDED_BAG",
                        "camera": self.camera_name,
                        "track_id": bag.track_id,
                        "class_name": bag.class_name,
                        "zone_name": "Full-Frame (Automatic)",
                        "entry_time": current_time_str,
                        "entry_timestamp": timestamp,
                        "exit_time": None,
                        "exit_timestamp": None,
                        "duration_seconds": round(stationary_duration, 1),
                        "status": "ACTIVE",
                        "owner_id": bag.associated_person_id,
                    })

            # Check if any previously alerted bag was retrieved / cleared
            for inc in reversed(self.incident_history):
                if inc.get("type") == "UNATTENDED_BAG" and inc.get("status") == "ACTIVE":
                    bag_id = inc.get("track_id")
                    cur_bag = self.unattended_bags.get(bag_id)
                    # A lost detection alone is not proof that the bag was picked up.
                    if cur_bag is not None and not cur_bag.alert_active:
                        dur = max(1, int(round(timestamp - inc["entry_timestamp"])))
                        inc["status"] = "RESOLVED"
                        inc["exit_time"] = current_time_str
                        inc["exit_timestamp"] = timestamp
                        inc["duration_seconds"] = dur
                        if bag_id in self._logged_bag_alert_ids:
                            self._logged_bag_alert_ids.remove(bag_id)

                        print(f"\n[ALERT RESOLVED] UNATTENDED BAG CLEARED", flush=True)
                        print(f"Camera: {self.camera_name}", flush=True)
                        print(f"Track ID: {bag_id}", flush=True)
                        print(f"Resolution Time: {current_time_str}", flush=True)
                        print(f"Total Duration: {dur}s\n", flush=True)

                        res_notif = {
                            "level": "INFO",
                            "type": "BAG_RESOLVED",
                            "track_id": bag_id,
                            "class_name": inc.get("class_name"),
                            "zone_name": "Full-Frame",
                            "timestamp": timestamp,
                            "time_str": current_time_str,
                            "entry_time": inc.get("entry_time"),
                            "exit_time": current_time_str,
                            "duration_seconds": dur,
                            "message": f"[BAG RESOLVED] Unattended bag #{bag_id} resolved at {current_time_str} ({dur}s).",
                        }
                        status["notifications"].append(res_notif)
                        self.event_log.append(res_notif)

        # ------------------- 4. COMPOSITE BANNER & ALERTS -------------------
        alert_parts = []
        if status["confirmed_intruders"]:
            int_summaries = [f"{i.class_name.capitalize()} #{i.track_id} in '{i.zone_name}'" for i in status["confirmed_intruders"]]
            alert_parts.append(f"INTRUSION: {', '.join(int_summaries)}")

        if status.get("unattended_bags"):
            bag_summaries = [f"{b.class_name.capitalize()} #{b.track_id}" for b in status["unattended_bags"]]
            alert_parts.append(f"UNATTENDED BAG: {', '.join(bag_summaries)}")

        status["all_active_alerts"] = len(status["confirmed_intruders"]) + len(status.get("unattended_bags", []))

        if alert_parts:
            status["banner_level"] = "CRITICAL"
            status["banner_text"] = "🚨 CRITICAL ALERT ACTIVE: " + " | ".join(alert_parts)
            # Draw prominent top alert banner on annotated frame
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 44), (0, 0, 220), -1)
            cv2.putText(
                annotated,
                status["banner_text"][:85],
                (14, 30),
                cv2.FONT_HERSHEY_DUPLEX,
                0.72,
                (255, 255, 255),
                2,
            )
        else:
            status["banner_level"] = "NORMAL"
            status["banner_text"] = "● ALL RESTRICTED ZONES & AREAS SECURE"

        return annotated, status

    def get_incident_history(self) -> list[dict]:
        """Return full permanent incident audit log."""
        return list(self.incident_history)

    def get_summary(self) -> dict:
        """Return consolidated summary counts."""
        tot = len(self.incident_history)
        p_inc = sum(1 for x in self.incident_history if x.get("class_name") == "person" and x.get("type") != "UNATTENDED_BAG")
        v_inc = sum(1 for x in self.incident_history if x.get("class_name") in {"car", "motorcycle", "bus", "truck"})
        bag_inc = sum(1 for x in self.incident_history if x.get("type") == "UNATTENDED_BAG")
        return {
            "total_incidents": tot,
            "person_incidents": p_inc,
            "vehicle_incidents": v_inc,
            "unattended_bag_incidents": bag_inc,
            "incident_history": list(self.incident_history),
        }


def process_integrated_stream(
    video_source: int | str | Path,
    pipeline: IntegratedPipeline,
    output_video: Path | str | None = None,
    save_screenshots: Path | str | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    stop_event: threading.Event | None = None,
) -> dict:
    """
    Execute integrated multi-feature detection pipeline over a video stream.
    Streams playback time, dispatches UI callbacks, and saves annotated video/screenshots.
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

    print(f"[Integrated Video] {source_desc}: {width}x{height} @ {fps:.1f} FPS ({total_frames} frames)")
    print(f"[Integrated Video] Intrusion: {pipeline.enable_intrusion} | Vehicles: {pipeline.enable_vehicle_tracking} | Unattended Bags: {pipeline.enable_unattended_bag}")

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
    saved_alert_keys = set()
    saved_screenshot_paths: list[str] = []
    last_terminal_time = 0.0

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

            annotated_frame, status = pipeline.process_frame(frame, timestamp)

            # Terminal playback + alert logging
            now_wall = time.time()
            mins = int(timestamp) // 60
            secs = int(timestamp) % 60
            time_disp = f"{mins:02d}:{secs:02d}"
            if status["all_active_alerts"] > 0 and (now_wall - last_terminal_time >= 1.0):
                last_terminal_time = now_wall
                confirmed = status.get("confirmed_intruders", [])
                bags = status.get("unattended_bags", [])
                if confirmed:
                    intr_strs = [f"{i.class_name.capitalize()} #{i.track_id}→'{i.zone_name}'" for i in confirmed]
                    print(f"[VIDEO {time_disp}] ⚠ INTRUSION ALERT: {', '.join(intr_strs)}", flush=True)
                if bags:
                    bag_strs = [f"{b.class_name.capitalize()} #{b.track_id}" for b in bags]
                    print(f"[VIDEO {time_disp}] ⚠ UNATTENDED BAG: {', '.join(bag_strs)}", flush=True)
            elif now_wall - last_terminal_time >= 10.0:
                last_terminal_time = now_wall
                inc_count = len(pipeline.incident_history)
                print(f"[VIDEO {time_disp}] Frame {frame_idx}/{total_frames} | Incidents so far: {inc_count} | Normal", flush=True)

            # Capture screenshots on new alert transitions
            if screens_dir is not None and status["all_active_alerts"] > 0:
                for intruder in status.get("confirmed_intruders", []):
                    key = f"intruder_{intruder.track_id}_{intruder.zone_name}"
                    if key not in saved_alert_keys:
                        saved_alert_keys.add(key)
                        fname = f"intrusion_track{intruder.track_id}_t{timestamp:.2f}s.jpg"
                        out_p = screens_dir / fname
                        cv2.imwrite(str(out_p), annotated_frame)
                        saved_screenshot_paths.append(str(out_p))

                for bag in status.get("unattended_bags", []):
                    key = f"bag_{bag.track_id}"
                    if key not in saved_alert_keys:
                        saved_alert_keys.add(key)
                        fname = f"unattended_bag_track{bag.track_id}_t{timestamp:.2f}s.jpg"
                        out_p = screens_dir / fname
                        cv2.imwrite(str(out_p), annotated_frame)
                        saved_screenshot_paths.append(str(out_p))

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
                        "incident_history": pipeline.get_incident_history(),
                    })
                except Exception:
                    pass

            frame_idx += 1

    finally:
        capture.release()
        if writer is not None:
            writer.release()

    total_time = time.time() - start_wall_time
    summary = pipeline.get_summary()
    summary.update({
        "total_frames": frame_idx,
        "elapsed_seconds": total_time,
        "average_fps": frame_idx / total_time if total_time > 0 else 0.0,
        "saved_screenshots": saved_screenshot_paths,
        "output_video": str(output_video) if output_video else None,
    })

    print(f"\n[Integrated Video] Complete: Processed {frame_idx} frames in {total_time:.2f}s ({summary['average_fps']:.1f} FPS).")
    print(f"[Integrated Video] Total Incidents: {summary['total_incidents']} (Person: {summary['person_incidents']}, Vehicle: {summary['vehicle_incidents']}, Bags: {summary['unattended_bag_incidents']})\n")
    return summary
