import torch
import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


PERSON_CLASS_ID = 0
BAG_CLASS_IDS = {24, 26, 28}
TRACKED_CLASS_IDS = [PERSON_CLASS_ID, *sorted(BAG_CLASS_IDS)]
TRACK_EXPIRY_SECONDS = 2.0


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
    last_seen_time: float = 0.0


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
        help="Path to YOLO model checkpoint (default: yolov8s.pt if available, else yolov8n.pt)",
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
        timestamp = previous_timestamp + (1.0 / fps if fps > 0 else 1.0 / 30.0)
    return timestamp


def update_bag(
    bag: TrackedBag,
    bbox: tuple[int, int, int, int],
    position: tuple[float, float],
    timestamp: float,
    person_distance: float,
    stationary_distance: float,
    people: dict[int, TrackedPerson],
) -> None:
    bag.bbox = bbox
    bag.position = position
    bag.last_detected_at = datetime.now()
    bag.last_seen_time = timestamp

    movement = (
        (position[0] - bag.stationary_anchor[0]) ** 2
        + (position[1] - bag.stationary_anchor[1]) ** 2
    ) ** 0.5
    if movement > stationary_distance:
        bag.stationary_anchor = position
        bag.stationary_since = timestamp
        bag.alert_active = False

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
) -> object:
    person_cls_id, bag_cls_ids, tracked_cls_ids = get_model_class_ids(model)
    inference_confidence = min(args.confidence, args.bag_confidence)
    results = model.track(
        frame,
        conf=inference_confidence,
        classes=None if args.debug_detections else tracked_cls_ids,
        imgsz=args.imgsz,
        device=0 if torch.cuda.is_available() else "cpu",
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )
    result = results[0]
    people: dict[int, TrackedPerson] = {}
    detected_bags: list[
        tuple[int, str, tuple[int, int, int, int], tuple[float, float]]
    ] = []

    boxes = result.boxes
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
            class_name = str(result.names[class_id])
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

            if class_id == person_cls_id and confidence >= args.confidence:
                if track_id is None:
                    if not args.debug_detections:
                        draw_box(frame, bbox, "person | ID unavailable", (255, 180, 0))
                    continue
                people[track_id] = TrackedPerson(track_id, position, bbox)
            elif class_id in bag_cls_ids and confidence >= args.bag_confidence:
                if track_id is None:
                    if not args.debug_detections:
                        draw_box(
                            frame,
                            bbox,
                            f"{class_name} | ID unavailable",
                            (0, 220, 0),
                        )
                    continue
                detected_bags.append((track_id, class_name, bbox, position))

    for track_id, class_name, bbox, position in detected_bags:
        bag = bags.get(track_id)
        if bag is None:
            bag = TrackedBag(
                track_id=track_id,
                class_name=class_name,
                bbox=bbox,
                position=position,
                first_detected_at=datetime.now(),
                last_detected_at=datetime.now(),
                stationary_since=timestamp,
                stationary_anchor=position,
                last_seen_time=timestamp,
            )
            bags[track_id] = bag
        update_bag(
            bag,
            bbox,
            position,
            timestamp,
            args.person_distance,
            args.stationary_distance,
            people,
        )

    for track_id in list(bags):
        if timestamp - bags[track_id].last_seen_time > TRACK_EXPIRY_SECONDS:
            del bags[track_id]

    if not args.debug_detections:
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
        associated_person = people.get(bag.associated_person_id)
        if associated_person is not None:
            dist_to_owner = (
                (associated_person.position[0] - bag.position[0]) ** 2
                + (associated_person.position[1] - bag.position[1]) ** 2
            ) ** 0.5
            person_moved_away = dist_to_owner > args.person_distance
        elif bag.associated_person_id is not None:
            # Owner was previously identified but has now walked out of view/off-screen
            # Verify no other person is currently standing directly beside the bag
            min_dist_to_anyone = min(
                [
                    (
                        (p.position[0] - bag.position[0]) ** 2
                        + (p.position[1] - bag.position[1]) ** 2
                    ) ** 0.5
                    for p in people.values()
                ],
                default=float("inf"),
            )
            person_moved_away = min_dist_to_anyone > args.person_distance
        else:
            # Unclaimed bag standing alone with no associated person
            min_dist_to_anyone = min(
                [
                    (
                        (p.position[0] - bag.position[0]) ** 2
                        + (p.position[1] - bag.position[1]) ** 2
                    ) ** 0.5
                    for p in people.values()
                ],
                default=float("inf"),
            )
            person_moved_away = min_dist_to_anyone > args.person_distance
        conditions_met = (
            stationary_duration >= args.stationary_time and person_moved_away
        )
        if not conditions_met:
            bag.alert_active = False
        elif not bag.alert_active and (
            bag.last_alert_at is None
            or timestamp - bag.last_alert_at >= args.alert_cooldown
        ):
            bag.alert_active = True
            bag.last_alert_at = timestamp

        warning_active = warning_active or bag.alert_active
        person_text = (
            f"person {bag.associated_person_id}"
            if bag.associated_person_id is not None
            else "person unknown"
        )
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
                "POSSIBLE UNATTENDED OBJECT",
                (bag.bbox[0], min(frame.shape[0] - 10, bag.bbox[3] + 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )

    if warning_active:
        cv2.putText(
            frame,
            "POSSIBLE UNATTENDED OBJECT",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
    return frame


def main() -> int:
    args = parse_args()
    if args.model is not None:
        model_path = args.model.resolve()
    else:
        candidate_s = Path(__file__).resolve().parent / "yolov8s.pt"
        candidate_n = Path(__file__).resolve().parent / "yolov8n.pt"
        model_path = candidate_s if candidate_s.is_file() else candidate_n

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
                output_path = args.save_frames / f"frame_{frame_number:06d}.jpg"
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
            frame_number += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
