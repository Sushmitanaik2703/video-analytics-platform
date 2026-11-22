import argparse
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
try:
    import winsound
except ImportError:
    winsound = None

import cv2
import torch
from ultralytics import YOLO


PERSON_CLASS_ID = 0
BAG_CLASS_IDS = {24, 26, 28}
TRACKED_CLASS_IDS = [PERSON_CLASS_ID, *sorted(BAG_CLASS_IDS)]
TRACK_EXPIRY_SECONDS = 2.0
PICKUP_CONFIRMATION_FRAMES = 5
MAX_CLASS_HISTORY = 30


def get_model_class_ids(model: YOLO) -> tuple[int, set[int], list[int]]:
    names = model.names
    person_id = 0
    bag_ids = set()
    for cid, cname in names.items():
        if cname == "person":
            person_id = int(cid)
        elif cname in {"backpack", "handbag", "suitcase"}:
            bag_ids.add(int(cid))
    if not bag_ids:
        # Fallback to standard COCO IDs
        person_id = PERSON_CLASS_ID
        bag_ids = BAG_CLASS_IDS
    tracked_ids = [person_id, *sorted(bag_ids)]
    return person_id, bag_ids, tracked_ids


@dataclass
class TrackedPerson:
    track_id: int
    position: tuple[float, float]
    bbox: tuple[int, int, int, int]


@dataclass
class TrackedBag:
    track_id: int
    class_name: str
    bbox: tuple[int, int, int, int]
    position: tuple[float, float]
    first_detected_at: datetime
    last_detected_at: datetime
    stationary_since: float
    stationary_anchor: tuple[float, float]
    associated_person_id: int | None = None
    last_alert_at: float | None = None
    alert_active: bool = False
    alert_anchor: tuple[float, float] | None = None
    pickup_motion_frames: int = 0
    last_seen_time: float = 0.0
    class_history: list[str] = field(default_factory=list)
    anchor_id: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect possibly unattended bags in a video or webcam stream."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Path to a video file")
    source.add_argument("--webcam", action="store_true", help="Use the default webcam")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to YOLO model checkpoint (default: yolov8n.pt if available, else yolov8s.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.50,
        help="Minimum person confidence from 0 to 1 (default: 0.50)",
    )
    parser.add_argument(
        "--bag-confidence",
        type=float,
        default=0.25,
        help="Minimum bag confidence from 0 to 1 (default: 0.25)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        choices=[640, 1280],
        help="YOLO inference image size: 640 (default) or 1280 (high-resolution)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=3,
        help="Run detection on every Nth frame for faster video processing (default: 3)",
    )
    parser.add_argument(
        "--stationary-time",
        type=float,
        default=15.0,
        help="Seconds a bag must remain stationary (default: 15)",
    )
    parser.add_argument(
        "--person-distance",
        type=float,
        default=150.0,
        help="Nearby-person/moved-away distance in pixels (default: 150)",
    )
    parser.add_argument(
        "--stationary-distance",
        type=float,
        default=25.0,
        help="Maximum bag-center movement in pixels to count as stationary (default: 25)",
    )
    parser.add_argument(
        "--alert-cooldown",
        type=float,
        default=30.0,
        help="Seconds before the same bag can alert again (default: 30)",
    )
    parser.add_argument(
        "--debug-detections",
        action="store_true",
        help="Print and draw every detected class and confidence, not just people and bags",
    )
    parser.add_argument(
        "--save-frames",
        type=Path,
        help="Directory for periodically saving annotated frames",
    )
    parser.add_argument(
        "--save-interval",
        type=float,
        default=1.0,
        help="Seconds between saved frames (default: 1)",
    )
    args = parser.parse_args()

    for option in ("confidence", "bag_confidence"):
        if not 0 <= getattr(args, option) <= 1:
            parser.error(f"--{option.replace('_', '-')} must be between 0 and 1")
    if args.imgsz not in (640, 1280):
        parser.error(f"--imgsz must be either 640 or 1280, got {args.imgsz}")
    if args.frame_stride < 1:
        parser.error("--frame-stride must be at least 1")
    for option in (
        "stationary_time",
        "person_distance",
        "stationary_distance",
        "alert_cooldown",
    ):
        if getattr(args, option) < 0:
            parser.error(f"--{option.replace('_', '-')} must not be negative")
    if args.save_interval <= 0:
        parser.error("--save-interval must be greater than 0")

    return args


def compute_box_iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box1_area = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    box2_area = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    return float(inter_area / float(box1_area + box2_area - inter_area))


def box_details(
    box: object,
) -> tuple[int, tuple[int, int, int, int], tuple[float, float]]:
    coordinates = box.xyxy[0].cpu().tolist()
    bbox = tuple(int(value) for value in coordinates)
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    return int(box.cls[0].item()), bbox, center


def frame_timestamp(
    capture: cv2.VideoCapture,
    is_webcam: bool,
    previous_timestamp: float | None,
    fps: float,
) -> float:
    if is_webcam:
        return time.monotonic()

    timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if previous_timestamp is None:
        return max(0.0, timestamp)
    if timestamp <= previous_timestamp:
        timestamp = previous_timestamp + (1.0 / fps if fps > 0 else 1.0 / 25.0)
    return timestamp


def update_bag(
    bag: TrackedBag,
    bbox: tuple[int, int, int, int],
    position: tuple[float, float],
    class_name: str,
    timestamp: float,
    person_distance: float,
    stationary_distance: float,
    people: dict[int, TrackedPerson],
) -> None:
    bag.bbox = bbox
    bag.position = position
    bag.last_detected_at = datetime.now()
    bag.last_seen_time = timestamp

    # Update rolling class vote accumulator
    bag.class_history.append(class_name)
    if len(bag.class_history) > MAX_CLASS_HISTORY:
        bag.class_history.pop(0)
    bag.class_name = Counter(bag.class_history).most_common(1)[0][0]

    # Evaluate physical displacement from stationary anchor
    movement = (
        (position[0] - bag.stationary_anchor[0]) ** 2
        + (position[1] - bag.stationary_anchor[1]) ** 2
    ) ** 0.5
    if bag.alert_active:
        alert_anchor = bag.alert_anchor or bag.stationary_anchor
        alert_movement = (
            (position[0] - alert_anchor[0]) ** 2
            + (position[1] - alert_anchor[1]) ** 2
        ) ** 0.5
        pickup_threshold = max(stationary_distance * 2.0, 50.0)
        if alert_movement > pickup_threshold:
            bag.pickup_motion_frames += 1
        else:
            bag.pickup_motion_frames = 0

        if bag.pickup_motion_frames >= PICKUP_CONFIRMATION_FRAMES:
            bag.alert_active = False
            bag.alert_anchor = None
            bag.pickup_motion_frames = 0
            bag.stationary_anchor = position
            bag.stationary_since = timestamp
    elif movement > stationary_distance:
        bag.stationary_anchor = position
        bag.stationary_since = timestamp

    # Owner association: frozen once set; only associates if owner is currently unknown
    if bag.associated_person_id is None:
        nearby_people = [
            (
                (
                    (person.position[0] - position[0]) ** 2
                    + (person.position[1] - position[1]) ** 2
                )
                ** 0.5,
                person.track_id,
            )
            for person in people.values()
        ]
        nearby_people = [item for item in nearby_people if item[0] <= person_distance]
        if nearby_people:
            bag.associated_person_id = min(nearby_people)[1]


def play_alert_sound(frequency: int = 1500, duration_ms: int = 400) -> None:
    """Non-blocking audio alert on Windows via winsound in a daemon thread."""
    if winsound is not None:
        def _beep():
            try:
                winsound.Beep(frequency, duration_ms)
            except Exception:
                pass
        threading.Thread(target=_beep, daemon=True).start()


def draw_box(
    frame: object,
    bbox: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def process_frame(
    model: YOLO,
    frame: object,
    bags: dict[int, TrackedBag],
    args: argparse.Namespace,
    timestamp: float,
    external_people: dict[int, TrackedPerson] | None = None,
    skip_draw_people: bool = False,
    pre_detected_boxes: object | None = None,
) -> object:
    person_cls_id, bag_cls_ids, tracked_cls_ids = get_model_class_ids(model)
    inference_confidence = min(args.confidence, args.bag_confidence)

    if pre_detected_boxes is not None:
        boxes = pre_detected_boxes
        result_names = model.names
    else:
        # If external persons are already provided, infer only bag classes for speed
        infer_classes = sorted(list(bag_cls_ids)) if external_people is not None else tracked_cls_ids
        results = model.track(
            frame,
            conf=inference_confidence,
            classes=None if args.debug_detections else infer_classes,
            imgsz=args.imgsz,
            device=0 if torch.cuda.is_available() else "cpu",
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        result_names = result.names

    people: dict[int, TrackedPerson] = dict(external_people) if external_people is not None else {}
    detected_bags: list[
        tuple[int | None, str, tuple[int, int, int, int], tuple[float, float]]
    ] = []

    if args.debug_detections:
        print(f"t={timestamp:.2f}s result.boxes is None: {boxes is None}")
        print(
            f"t={timestamp:.2f}s result.boxes.id is None: "
            f"{boxes is None or boxes.id is None}"
        )
    track_ids = (
        boxes.id.int().cpu().tolist()
        if boxes is not None and boxes.id is not None
        else []
    )
    if boxes is not None:
        for index, box in enumerate(boxes):
            class_id, bbox, position = box_details(box)
            confidence = float(box.conf[0].item())
            class_name = str(result_names[class_id])
            track_id = int(track_ids[index]) if index < len(track_ids) else None

            if args.debug_detections:
                id_label = f" ID {track_id}" if track_id is not None else " ID unavailable"
                print(
                    f"t={timestamp:.2f}s {class_name} "
                    f"(class={class_id}) confidence={confidence:.3f}{id_label}"
                )
                draw_box(
                    frame,
                    bbox,
                    f"{class_name} {confidence:.2f}{id_label}",
                    (255, 255, 0),
                )

            # Person detection (only populate if external_people was not supplied)
            if class_id == person_cls_id and confidence >= args.confidence:
                if external_people is None:
                    if track_id is None:
                        if not args.debug_detections and not skip_draw_people:
                            draw_box(frame, bbox, "person | ID unavailable", (255, 180, 0))
                        continue
                    people[track_id] = TrackedPerson(track_id, position, bbox)

            # Bag detection (backpack, handbag, suitcase)
            elif class_id in bag_cls_ids and confidence >= args.bag_confidence:
                # Filter out tiny noise slivers (e.g., width < 14 or height < 14)
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                if bw < 14 or bh < 14:
                    continue
                detected_bags.append((track_id, class_name, bbox, position))

    # --- Robust Spatial-Temporal Bag Association (Anchor-Based Stability) ---
    # Maintains stable bag track IDs across consecutive frames even if ByteTrack drops or switches IDs
    matched_bag_keys: set[int] = set()
    next_bag_virtual_id = max(bags.keys(), default=0) + 1

    for raw_track_id, class_name, bbox, position in detected_bags:
        matched_id: int | None = None
        best_dist = float("inf")

        # Priority 1: Match existing bag with identical ByteTrack track_id within spatial boundary
        if raw_track_id is not None and raw_track_id in bags and raw_track_id not in matched_bag_keys:
            dist = (
                (position[0] - bags[raw_track_id].position[0]) ** 2
                + (position[1] - bags[raw_track_id].position[1]) ** 2
            ) ** 0.5
            if bags[raw_track_id].alert_active or dist <= max(args.stationary_distance * 2.0, 50.0):
                matched_id = raw_track_id
                best_dist = dist

        # Priority 2: Match by spatial proximity or IoU to an existing tracked bag
        if matched_id is None:
            for existing_id, existing_bag in bags.items():
                if existing_id in matched_bag_keys:
                    continue
                iou = compute_box_iou(bbox, existing_bag.bbox)
                dist = (
                    (position[0] - existing_bag.position[0]) ** 2
                    + (position[1] - existing_bag.position[1]) ** 2
                ) ** 0.5
                spatial_limit = (
                    max(args.stationary_distance * 2.0, 50.0)
                    if existing_bag.alert_active
                    else max(args.stationary_distance * 1.5, 45.0)
                )
                if (dist <= spatial_limit or iou >= 0.20) and dist < best_dist:
                    best_dist = dist
                    matched_id = existing_id

        # Update matched existing bag (preserves stable track ID and stationary_since)
        if matched_id is not None:
            matched_bag_keys.add(matched_id)
            bag = bags[matched_id]
            update_bag(
                bag,
                bbox,
                position,
                class_name,
                timestamp,
                args.person_distance,
                args.stationary_distance,
                people,
            )
        else:
            # New bag: assign consistent ID
            assigned_id = raw_track_id if (raw_track_id is not None and raw_track_id not in bags) else next_bag_virtual_id
            if assigned_id == next_bag_virtual_id:
                next_bag_virtual_id += 1
            bag = TrackedBag(
                track_id=assigned_id,
                class_name=class_name,
                bbox=bbox,
                position=position,
                first_detected_at=datetime.now(),
                last_detected_at=datetime.now(),
                stationary_since=timestamp,
                stationary_anchor=position,
                last_seen_time=timestamp,
                class_history=[class_name],
                anchor_id=assigned_id,
            )
            bags[assigned_id] = bag
            matched_bag_keys.add(assigned_id)
            update_bag(
                bag,
                bbox,
                position,
                class_name,
                timestamp,
                args.person_distance,
                args.stationary_distance,
                people,
            )

    # Remove bag tracks that have not been detected for more than TRACK_EXPIRY_SECONDS (2.0s)
    for track_id in list(bags):
        if (
            timestamp - bags[track_id].last_seen_time > TRACK_EXPIRY_SECONDS
            and not bags[track_id].alert_active
        ):
            del bags[track_id]

    # Only draw people if skip_draw_people is False (prevents duplicate boxes)
    if not args.debug_detections and not skip_draw_people:
        for person in people.values():
            draw_box(
                frame,
                person.bbox,
                f"person ID {person.track_id}",
                (255, 180, 0),
            )

    warning_active = False
    for bag in bags.values():
        stationary_duration = max(0.0, timestamp - bag.stationary_since)
        person_moved_away = all(
            (
                (person.position[0] - bag.position[0]) ** 2
                + (person.position[1] - bag.position[1]) ** 2
            ) ** 0.5
            > args.person_distance
            for person in people.values()
        )
        conditions_met = (
            stationary_duration >= args.stationary_time and person_moved_away
        )
        if not bag.alert_active and conditions_met and (
            bag.last_alert_at is None
            or timestamp - bag.last_alert_at >= args.alert_cooldown
        ):
            bag.alert_active = True
            bag.last_alert_at = timestamp
            bag.alert_anchor = bag.position
            bag.pickup_motion_frames = 0
            play_alert_sound()

        warning_active = warning_active or bag.alert_active
        person_text = (
            f"person {bag.associated_person_id}"
            if bag.associated_person_id is not None
            else "person unknown"
        )
        if bag.alert_active:
            label = (
                f"UNATTENDED BAG | ID: {bag.track_id} | "
                f"{stationary_duration:.1f}s | "
                f"pos ({int(bag.position[0])}, {int(bag.position[1])}) | {person_text}"
            )
        else:
            label = (
                f"{bag.class_name} ID {bag.track_id} | "
                f"{stationary_duration:.1f}s | "
                f"pos ({int(bag.position[0])}, {int(bag.position[1])}) | {person_text}"
            )
        draw_box(
            frame,
            bag.bbox,
            label,
            (0, 0, 255) if bag.alert_active else (0, 220, 0),
        )
        if bag.alert_active:
            cv2.putText(
                frame,
                "UNATTENDED OBJECT",
                (bag.bbox[0], min(frame.shape[0] - 10, bag.bbox[3] + 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )

    if warning_active:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (0, 0, 220), -1)
        cv2.putText(
            frame,
            "ALERT: UNATTENDED BAG DETECTED",
            (20, 32),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
    return frame


def main() -> int:
    args = parse_args()
    if args.model is not None:
        model_path = args.model.resolve()
    else:
        candidate_n = Path(__file__).resolve().parent / "yolov8n.pt"
        candidate_s = Path(__file__).resolve().parent / "yolov8s.pt"
        model_path = candidate_n if candidate_n.is_file() else candidate_s

    if not model_path.is_file():
        print(f"Error: YOLO model not found: {model_path}", file=sys.stderr)
        return 1
    if args.video is not None and not args.video.is_file():
        print(f"Error: Video file not found: {args.video}", file=sys.stderr)
        return 1

    try:
        model = YOLO(str(model_path))
    except Exception as error:
        print(f"Error: Could not load YOLO model: {error}", file=sys.stderr)
        return 1

    capture = cv2.VideoCapture(0 if args.webcam else str(args.video))
    if not capture.isOpened():
        source_name = "webcam (camera index 0)" if args.webcam else str(args.video)
        print(f"Error: Could not open {source_name}.", file=sys.stderr)
        capture.release()
        return 1

    bags: dict[int, TrackedBag] = {}
    fps = capture.get(cv2.CAP_PROP_FPS)
    previous_timestamp: float | None = None
    next_save_time: float | None = None
    exit_code = 0
    if args.save_frames is not None:
        try:
            args.save_frames.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            print(
                f"Error: Could not create frame output directory "
                f"{args.save_frames}: {error}",
                file=sys.stderr,
            )
            capture.release()
            return 1
    try:
        frame_number = 0
        while True:
            success, frame = capture.read()
            if not success:
                if args.webcam:
                    print(
                        "Error: Could not read a frame from the webcam.",
                        file=sys.stderr,
                    )
                    exit_code = 1
                break

            current_frame_number = frame_number
            frame_number += 1
            if current_frame_number % args.frame_stride != 0:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            timestamp = frame_timestamp(
                capture, args.webcam, previous_timestamp, fps
            )
            previous_timestamp = timestamp
            try:
                annotated_frame = process_frame(model, frame, bags, args, timestamp)
            except Exception as error:
                print(f"Error: Could not process video frame: {error}", file=sys.stderr)
                exit_code = 1
                break
            if args.save_frames is not None and (
                next_save_time is None or timestamp >= next_save_time
            ):
                output_path = args.save_frames / f"frame_{current_frame_number:06d}.jpg"
                if not cv2.imwrite(str(output_path), annotated_frame):
                    print(
                        f"Error: Could not save annotated frame: {output_path}",
                        file=sys.stderr,
                    )
                    exit_code = 1
                    break
                next_save_time = timestamp + args.save_interval
            cv2.imshow("Smart CCTV - Unattended Object Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
