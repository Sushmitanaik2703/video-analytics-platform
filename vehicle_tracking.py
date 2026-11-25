"""
Vehicle Detection and Tracking Module (Member 1 / Unified CCTV Architecture).

Detects and persistently tracks vehicles (cars, motorcycles, buses, trucks)
in CCTV surveillance footage using YOLOv8s and ByteTrack.

Key Features:
1. Four targeted COCO vehicle classes:
   - Car: Class ID 2
   - Motorcycle: Class ID 3
   - Bus: Class ID 5
   - Truck: Class ID 7
2. Dual-mode architecture:
   - Standalone: Performs internal YOLOv8s + ByteTrack inference on raw video frames.
   - Upstream Integration: Ingests pre-tracked bounding boxes to avoid duplicate
     YOLO inference when combined with person/luggage pipelines.
3. State tracking per vehicle:
   - Persistent Track ID
   - Vehicle Class Name & ID
   - Bounding Box (x1, y1, x2, y2)
   - Geometric Center Point ((x1 + x2) // 2, (y1 + y2) // 2)
   - Confidence Score
   - First-seen & Last-seen Timestamps
   - Motion Trajectory History (recent center points trail)
   - Detection Observation Counter
4. Robustness safeguards:
   - Configurable lost-frame grace period (max_lost_frames=5) for occlusion recovery.
   - Defensive bounding box sanitization (handles malformed, inverted, or out-of-bounds boxes).
   - Track ID collision and reuse safety.
5. High-visibility security rendering:
   - Distinct color palettes per vehicle category.
   - Motion trajectory polyline trails with visual heading.
   - Top-left CCTV telemetry dashboard (class breakdown, unique count, FPS).
6. 1.0x real-time playback synchronization for live CCTV simulation.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Standard COCO vehicle class mappings
VEHICLE_CLASS_MAP: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Reverse mapping: class name -> class ID
VEHICLE_NAME_TO_ID: dict[str, int] = {name: cid for cid, name in VEHICLE_CLASS_MAP.items()}

# High-contrast color palette (BGR) for CCTV overlay
VEHICLE_PALETTE: dict[str, tuple[int, int, int]] = {
    "car": (0, 220, 100),         # Emerald Green
    "motorcycle": (255, 200, 0),   # Amber Cyan
    "bus": (0, 140, 255),         # Safety Orange
    "truck": (180, 0, 220),       # Deep Magenta
}
DEFAULT_VEHICLE_COLOR: tuple[int, int, int] = (0, 255, 255)  # Yellow fallback


@dataclass
class VehicleTrack:
    """State tracking for a single vehicle entity across video frames."""

    track_id: int
    class_id: int
    class_name: str
    bbox: tuple[int, int, int, int]
    center_point: tuple[int, int]
    conf: float
    first_seen_time: float
    last_seen_time: float
    trajectory: list[tuple[int, int]] = field(default_factory=list)
    lost_frames: int = 0
    total_observations: int = 1


class VehicleTracker:
    """
    Modular Vehicle Detection and Tracking Engine.

    Supports both standalone video processing and upstream track ingestion.
    """

    def __init__(
        self,
        yolo_model: YOLO | None = None,
        confidence: float = 0.40,
        imgsz: int = 640,
        max_lost_frames: int = 5,
        max_trajectory_points: int = 30,
        device: str | int | None = None,
    ):
        self.model = yolo_model
        self.confidence = confidence
        self.imgsz = imgsz
        self.max_lost_frames = max_lost_frames
        self.max_trajectory_points = max_trajectory_points

        # Resolve compute device
        if device is not None:
            self.device = device
        else:
            self.device = 0 if torch.cuda.is_available() else "cpu"

        # Active tracking records: key = track_id
        self.active_tracks: dict[int, VehicleTrack] = {}

        # Historical telemetry counters
        self.unique_track_ids: set[int] = set()
        self.class_unique_counts: dict[str, set[int]] = {c: set() for c in VEHICLE_CLASS_MAP.values()}
        self.total_observations: int = 0

    def reset(self) -> None:
        """Clear all active tracks and reset historical telemetry."""
        self.active_tracks.clear()
        self.unique_track_ids.clear()
        for s in self.class_unique_counts.values():
            s.clear()
        self.total_observations = 0

    def process_tracks(
        self,
        tracks: list[dict],
        frame_w: int,
        frame_h: int,
        timestamp: float,
    ) -> dict:
        """
        Ingest and maintain vehicle tracks from an external detector/tracker.

        Each item in `tracks`:
          {
            'track_id': int,
            'class_id': int (or 'class_name': str),
            'bbox': (x1, y1, x2, y2),
            'conf': float
          }
        """
        active_ids_in_frame: set[int] = set()

        for trk in tracks:
            # Defensive validation: track_id must be valid positive integer
            raw_id = trk.get("track_id")
            if raw_id is None or int(raw_id) <= 0:
                continue
            track_id = int(raw_id)

            # Resolve class name and ID
            raw_cls = trk.get("class_name")
            raw_cid = trk.get("class_id")

            if raw_cls is not None and str(raw_cls).lower() in VEHICLE_NAME_TO_ID:
                class_name = str(raw_cls).lower()
                class_id = VEHICLE_NAME_TO_ID[class_name]
            elif raw_cid is not None and int(raw_cid) in VEHICLE_CLASS_MAP:
                class_id = int(raw_cid)
                class_name = VEHICLE_CLASS_MAP[class_id]
            else:
                # Not a vehicle target class
                continue

            conf = float(trk.get("conf", 1.0))
            if conf < self.confidence:
                continue

            # Defensive bounding box validation & clamping
            raw_bbox = trk.get("bbox", (0, 0, 0, 0))
            if len(raw_bbox) != 4:
                continue

            try:
                x1, y1, x2, y2 = map(int, map(round, raw_bbox))
            except (ValueError, TypeError):
                continue

            # Check for inverted or zero-area coordinates
            if x2 <= x1 or y2 <= y1:
                continue

            # Clamp coordinates to frame dimensions
            x1 = max(0, min(frame_w - 1, x1))
            y1 = max(0, min(frame_h - 1, y1))
            x2 = max(0, min(frame_w, x2))
            y2 = max(0, min(frame_h, y2))

            if (x2 - x1) <= 2 or (y2 - y1) <= 2:
                continue

            # Geometric center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            center_point = (center_x, center_y)

            active_ids_in_frame.add(track_id)
            self.total_observations += 1
            self.unique_track_ids.add(track_id)
            self.class_unique_counts[class_name].add(track_id)

            if track_id in self.active_tracks:
                # Update existing track
                vt = self.active_tracks[track_id]
                vt.bbox = (x1, y1, x2, y2)
                vt.center_point = center_point
                vt.conf = conf
                vt.last_seen_time = timestamp
                vt.lost_frames = 0
                vt.total_observations += 1
                # Update class if higher confidence observation
                if conf >= vt.conf:
                    vt.class_id = class_id
                    vt.class_name = class_name
                # Append to trajectory
                vt.trajectory.append(center_point)
                if len(vt.trajectory) > self.max_trajectory_points:
                    vt.trajectory = vt.trajectory[-self.max_trajectory_points:]
            else:
                # New track creation
                vt = VehicleTrack(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    bbox=(x1, y1, x2, y2),
                    center_point=center_point,
                    conf=conf,
                    first_seen_time=timestamp,
                    last_seen_time=timestamp,
                    trajectory=[center_point],
                    lost_frames=0,
                    total_observations=1,
                )
                self.active_tracks[track_id] = vt

        # Handle missing / stale tracks with grace period
        stale_ids = []
        for track_id, vt in self.active_tracks.items():
            if track_id not in active_ids_in_frame:
                vt.lost_frames += 1
                if vt.lost_frames > self.max_lost_frames:
                    stale_ids.append(track_id)

        for sid in stale_ids:
            del self.active_tracks[sid]

        # Calculate current telemetry breakdown
        active_counts_by_class: dict[str, int] = {c: 0 for c in VEHICLE_CLASS_MAP.values()}
        for vt in self.active_tracks.values():
            if vt.lost_frames == 0:  # Only count currently visible in this frame
                active_counts_by_class[vt.class_name] = active_counts_by_class.get(vt.class_name, 0) + 1

        return {
            "timestamp": timestamp,
            "active_vehicles": list(self.active_tracks.values()),
            "active_counts_by_class": active_counts_by_class,
            "total_active_visible": sum(active_counts_by_class.values()),
            "total_tracked": len(self.active_tracks),
            "total_unique_vehicles": len(self.unique_track_ids),
            "class_unique_counts": {c: len(ids) for c, ids in self.class_unique_counts.items()},
            "total_observations": self.total_observations,
        }

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        fps_metric: float = 0.0,
    ) -> tuple[np.ndarray, dict]:
        """
        Standalone frame processing with internal YOLOv8s + ByteTrack inference.
        """
        if self.model is None:
            raise RuntimeError("YOLO model is not initialized for standalone process_frame()")

        frame_h, frame_w = frame.shape[:2]

        # Target class IDs: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
        target_classes = list(VEHICLE_CLASS_MAP.keys())

        # Single YOLOv8 inference + ByteTrack run
        results = self.model.track(
            frame,
            conf=self.confidence,
            classes=target_classes,
            imgsz=self.imgsz,
            device=self.device,
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

        tracks: list[dict] = []
        if boxes is not None:
            for idx, box in enumerate(boxes):
                if idx >= len(track_ids):
                    continue
                t_id = int(track_ids[idx])
                conf = float(box.conf[0].item())
                if conf < self.confidence:
                    continue
                cid = int(box.cls[0].item())
                if cid not in VEHICLE_CLASS_MAP:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                tracks.append({
                    "track_id": t_id,
                    "class_id": cid,
                    "class_name": VEHICLE_CLASS_MAP[cid],
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                })

        status = self.process_tracks(tracks, frame_w, frame_h, timestamp)
        annotated_frame = self.annotate_frame(frame.copy(), status, fps=fps_metric)
        return annotated_frame, status

    def annotate_frame(
        self,
        frame: np.ndarray,
        status: dict,
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Render vehicle bounding boxes, labels, trajectory trails, and telemetry HUD.
        """
        frame_h, frame_w = frame.shape[:2]
        active_vehicles: list[VehicleTrack] = status.get("active_vehicles", [])

        # 1. Draw Trajectory Trails (Rendered first so bounding boxes stay on top)
        for vt in active_vehicles:
            color = VEHICLE_PALETTE.get(vt.class_name, DEFAULT_VEHICLE_COLOR)
            pts = vt.trajectory
            if len(pts) >= 2:
                for i in range(1, len(pts)):
                    # Fade trail thickness towards older points
                    thickness = max(1, int(round(3 * (i / len(pts)))))
                    cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

        # 2. Draw Vehicle Bounding Boxes and Header Badges
        for vt in active_vehicles:
            if vt.lost_frames > 0:
                # If currently missing in grace period, render with dashed-style or muted color
                color = (128, 128, 128)
            else:
                color = VEHICLE_PALETTE.get(vt.class_name, DEFAULT_VEHICLE_COLOR)

            x1, y1, x2, y2 = vt.bbox

            # Main bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Center point marker
            cx, cy = vt.center_point
            cv2.circle(frame, (cx, cy), 4, color, -1)
            cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1)

            # Badge tag: [VEHICLE_CLASS] ID: {track_id} ({conf:.2f})
            duration = max(0.0, status["timestamp"] - vt.first_seen_time)
            badge_text = f"{vt.class_name.upper()} #{vt.track_id} ({vt.conf:.2f})"
            (tw, th), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)

            badge_y1 = max(0, y1 - th - 8)
            badge_y2 = y1
            badge_x2 = min(frame_w, x1 + tw + 8)

            # Badge background
            cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
            # Badge text (dark text on bright background)
            text_color = (0, 0, 0) if vt.class_name in {"car", "motorcycle"} else (255, 255, 255)
            cv2.putText(
                frame,
                badge_text,
                (x1 + 4, max(th + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                text_color,
                1,
                cv2.LINE_AA,
            )

        # 3. Top Telemetry HUD Banner
        hud_h = 42
        hud_overlay = frame.copy()
        cv2.rectangle(hud_overlay, (0, 0), (frame_w, hud_h), (25, 25, 25), -1)
        cv2.addWeighted(hud_overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, hud_h), (frame_w, hud_h), (60, 60, 60), 1)

        counts = status.get("active_counts_by_class", {})
        fps_text = f"FPS: {fps:.1f}" if fps > 0 else "FPS: --"
        total_unique = status.get("total_unique_vehicles", 0)

        hud_title = "SMART CCTV | VEHICLE MONITORING"
        cv2.putText(frame, hud_title, (14, 26), cv2.FONT_HERSHEY_DUPLEX, 0.60, (0, 220, 255), 1, cv2.LINE_AA)

        # Class counts breakdown with color-coded pills
        offset_x = 340
        for cname in ["car", "motorcycle", "bus", "truck"]:
            c_color = VEHICLE_PALETTE[cname]
            c_count = counts.get(cname, 0)
            c_text = f"{cname.upper()}: {c_count}"
            (cw, ch), _ = cv2.getTextSize(c_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            if offset_x + cw + 20 < frame_w - 180:
                cv2.circle(frame, (offset_x, 22), 4, c_color, -1)
                cv2.putText(frame, c_text, (offset_x + 8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 230), 1, cv2.LINE_AA)
                offset_x += cw + 22

        # Telemetry metrics on right
        stats_text = f"UNIQUE: {total_unique} | {fps_text}"
        (sw, _), _ = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(frame, stats_text, (frame_w - sw - 14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 200), 1, cv2.LINE_AA)

        return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart CCTV Vehicle Detection and Tracking (COCO Classes: Car, Motorcycle, Bus, Truck)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Path to input surveillance video file")
    source.add_argument("--webcam", action="store_true", help="Use live default webcam (device 0)")

    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to YOLO model checkpoint (default: yolov8s.pt in project root)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.40,
        help="Minimum vehicle detection confidence threshold (default: 0.40)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        choices=[640, 1280],
        help="YOLO inference image size (default: 640)",
    )
    parser.add_argument(
        "--max-lost-frames",
        type=int,
        default=5,
        help="Grace period frames before purging stale vehicle tracks (default: 5)",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Optional path to save annotated output video (.mp4)",
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize playback to original video speed (default: True, use --no-realtime for max speed)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process before exiting",
    )
    return parser.parse_args()


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
        print(f"Error: YOLO model checkpoint not found: {model_path}", file=sys.stderr)
        return 1

    source_desc = "Webcam 0" if args.webcam else str(args.video)
    capture = cv2.VideoCapture(0 if args.webcam else str(args.video))
    if not capture.isOpened():
        print(f"Error: Could not open video source: {source_desc}", file=sys.stderr)
        return 1

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if not args.webcam else 0

    print("=" * 65)
    print("SMART CCTV - VEHICLE DETECTION AND TRACKING")
    print(f"Source: {source_desc} ({width}x{height} @ {fps:.1f} FPS, {total_frames} frames)")
    print(f"Model: {model_path.name} | Conf: {args.confidence} | imgsz: {args.imgsz}")
    print(f"Classes: Car (2), Motorcycle (3), Bus (5), Truck (7)")
    print("=" * 65)

    print(f"[Init] Loading YOLO model: {model_path.name}...")
    yolo_model = YOLO(str(model_path))

    tracker = VehicleTracker(
        yolo_model=yolo_model,
        confidence=args.confidence,
        imgsz=args.imgsz,
        max_lost_frames=args.max_lost_frames,
    )

    writer = None
    if args.output_video is not None:
        args.output_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output_video), fourcc, fps, (width, height))
        print(f"[Output] Saving annotated video to: {args.output_video}")

    frame_idx = 0
    start_wall_time = time.time()
    last_fps_calc_time = time.time()
    fps_calc_counter = 0
    measured_fps = fps

    try:
        while True:
            if args.max_frames is not None and frame_idx >= args.max_frames:
                print(f"[Limit] Reached max requested frames: {args.max_frames}")
                break

            ret, frame = capture.read()
            if not ret:
                break

            msec = capture.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = msec / 1000.0 if msec > 0 else frame_idx / fps

            # Running FPS estimation
            fps_calc_counter += 1
            if (time.time() - last_fps_calc_time) >= 0.5:
                measured_fps = fps_calc_counter / (time.time() - last_fps_calc_time)
                last_fps_calc_time = time.time()
                fps_calc_counter = 0

            annotated_frame, status = tracker.process_frame(frame, timestamp, fps_metric=measured_fps)

            if writer is not None:
                writer.write(annotated_frame)

            cv2.imshow("Smart CCTV - Vehicle Detection & Tracking", annotated_frame)

            # 1.0x Real-time playback synchronization
            if args.realtime and not args.webcam:
                frame_interval = 1.0 / fps
                target_next_time = start_wall_time + (frame_idx + 1) * frame_interval
                time_remaining = target_next_time - time.time()

                if time_remaining > 0:
                    wait_ms = max(1, int(time_remaining * 1000))
                    key = cv2.waitKey(wait_ms) & 0xFF
                else:
                    key = cv2.waitKey(1) & 0xFF
                    lag = time.time() - target_next_time
                    if lag > frame_interval:
                        skips = min(int(lag / frame_interval), 2)
                        for _ in range(skips):
                            if not capture.grab():
                                break
                            frame_idx += 1
            else:
                key = cv2.waitKey(1) & 0xFF

            if key in {ord("q"), 27}:
                print("[User] Exiting on 'q' press.")
                break

            frame_idx += 1

    finally:
        total_elapsed = time.time() - start_wall_time
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    avg_fps = frame_idx / total_elapsed if total_elapsed > 0 else 0.0
    print("\n" + "=" * 65)
    print("PROCESSING SUMMARY")
    print(f"Total Frames Processed: {frame_idx}")
    print(f"Elapsed Time: {total_elapsed:.2f}s | Average Speed: {avg_fps:.1f} FPS")
    print(f"Total Unique Vehicles Tracked: {len(tracker.unique_track_ids)}")
    print(f"Total Detection Observations: {tracker.total_observations}")
    for cname, ids in tracker.class_unique_counts.items():
        print(f"  - {cname.capitalize()}: {len(ids)} unique vehicles")
    print("=" * 65)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
