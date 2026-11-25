"""
Experimental CLI runner for Hybrid Owner Re-identification and Return Detection (Phase 1).

Executes unattended object detection with OSNet Re-ID owner return and retrieval verification.
Outputs annotated videos and key scenario frames.
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

from .owner_reid_pipeline import HybridOwnerReIDPipeline
from .reid_extractor import OSNetExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid Owner Re-identification & Return Detection (Phase 1)"
    )
    parser.add_argument("--video", type=Path, required=True, help="Path to input video file")
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Path to output annotated video file",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to YOLO model checkpoint (default: yolov8s.pt, or best.pt for Video 5)",
    )
    parser.add_argument(
        "--reid-weights",
        type=Path,
        default=None,
        help="Path to OSNet weights checkpoint",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.50,
        help="Minimum person confidence (default: 0.50)",
    )
    parser.add_argument(
        "--bag-confidence",
        type=float,
        default=0.25,
        help="Minimum bag confidence (default: 0.25)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        choices=[640, 1280],
        help="YOLO inference image size: 640 (default) or 1280",
    )
    parser.add_argument(
        "--stationary-time",
        type=float,
        default=15.0,
        help="Stationary threshold in seconds (default: 15.0)",
    )
    parser.add_argument(
        "--person-distance",
        type=float,
        default=150.0,
        help="Owner separation distance in pixels (default: 150.0)",
    )
    parser.add_argument(
        "--stationary-distance",
        type=float,
        default=25.0,
        help="Bag-center movement tolerance in pixels (default: 25.0)",
    )
    parser.add_argument(
        "--alert-cooldown",
        type=float,
        default=30.0,
        help="Alert cooldown in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--reid-threshold",
        type=float,
        default=0.65,
        help="OSNet cosine similarity threshold for owner match (default: 0.65)",
    )
    parser.add_argument(
        "--confirm-frames",
        type=int,
        default=5,
        help="Consecutive frames required to confirm owner return (default: 5)",
    )
    parser.add_argument(
        "--save-screenshots",
        type=Path,
        default=None,
        help="Directory to save key event screenshots",
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize playback to original video speed (default: True, use --no-realtime for uncapped)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    if args.model is not None:
        model_path = args.model.resolve()
    else:
        video_name = str(args.video).lower() if args.video is not None else ""
        if "video5" in video_name:
            model_path = (
                project_root
                / "runs"
                / "unattended_yolov8n_targeted_v2"
                / "weights"
                / "best.pt"
            )
        else:
            model_path = project_root / "yolov8s.pt"

    if not model_path.is_file():
        alt_path = (
            project_root / "runs" / "unattended_yolov8n_targeted_v2" / "weights" / "best.pt"
            if model_path.name == "yolov8s.pt"
            else project_root / "yolov8s.pt"
        )
        if alt_path.is_file():
            model_path = alt_path
        else:
            print(f"Error: YOLO model not found: {model_path}", file=sys.stderr)
            return 1

    if not args.video.is_file():
        print(f"Error: Video file not found: {args.video}", file=sys.stderr)
        return 1

    print(f"[Init] Loading YOLO model from {model_path}...")
    yolo_model = YOLO(str(model_path))

    reid_weights = args.reid_weights
    if reid_weights is None:
        reid_weights = (
            Path(__file__).resolve().parent / "weights" / "osnet_x0_25_msmt17.pth"
        )
    print(f"[Init] Loading OSNet Re-ID extractor from {reid_weights}...")
    extractor = OSNetExtractor(weights_path=reid_weights)

    pipeline = HybridOwnerReIDPipeline(
        yolo_model=yolo_model,
        reid_extractor=extractor,
        confidence=args.confidence,
        bag_confidence=args.bag_confidence,
        stationary_time=args.stationary_time,
        person_distance=args.person_distance,
        stationary_distance=args.stationary_distance,
        alert_cooldown=args.alert_cooldown,
        imgsz=args.imgsz,
        reid_threshold=args.reid_threshold,
        confirm_frames=args.confirm_frames,
    )

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        print(f"Error: Could not open video {args.video}", file=sys.stderr)
        return 1

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Video] {args.video.name}: {total_frames} frames, {width}x{height} @ {fps:.1f} FPS")

    writer = None
    if args.output_video is not None:
        args.output_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output_video), fourcc, fps, (width, height))

    if args.save_screenshots is not None:
        args.save_screenshots.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    previous_timestamp = None
    saved_events = set()

    start_wall_time = time.time()
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            msec = capture.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = msec / 1000.0 if msec > 0 else frame_idx / fps
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                timestamp = previous_timestamp + (1.0 / fps)
            previous_timestamp = timestamp

            annotated_frame, status = pipeline.process_frame(frame, timestamp)

            # Check for screenshot trigger events
            if args.save_screenshots is not None:
                if status["active_alert"] and "unattended_alert" not in saved_events:
                    saved_events.add("unattended_alert")
                    cv2.imwrite(
                        str(args.save_screenshots / f"{args.video.stem}_unattended_alert.jpg"),
                        annotated_frame,
                    )
                    print(f"  [Event] Unattended Alert saved at t={timestamp:.2f}s")

                has_returned_owner = any(
                    b["owner_returned"] and not b["bag_retrieved"]
                    for b in status["bags"].values()
                )
                if has_returned_owner and "owner_returned" not in saved_events:
                    saved_events.add("owner_returned")
                    cv2.imwrite(
                        str(args.save_screenshots / f"{args.video.stem}_owner_returned.jpg"),
                        annotated_frame,
                    )
                    print(f"  [Event] Owner Returned saved at t={timestamp:.2f}s")

                if status["active_retrieval"] and "bag_retrieved" not in saved_events:
                    saved_events.add("bag_retrieved")
                    cv2.imwrite(
                        str(args.save_screenshots / f"{args.video.stem}_bag_retrieved.jpg"),
                        annotated_frame,
                    )
                    print(f"  [Event] Bag Retrieved saved at t={timestamp:.2f}s")

            if writer is not None:
                writer.write(annotated_frame)

            # Display annotated video at original video playback speed
            cv2.imshow("Smart CCTV - Owner Re-Identification", annotated_frame)

            if args.realtime:
                frame_interval = 1.0 / fps
                target_next_time = start_wall_time + (frame_idx + 1) * frame_interval
                time_remaining = target_next_time - time.time()

                if time_remaining > 0:
                    # Processing was faster than frame duration: wait remaining time to match original video speed
                    wait_ms = max(1, int(time_remaining * 1000))
                    key = cv2.waitKey(wait_ms) & 0xFF
                else:
                    # Processing took longer: minimal UI wait and catch up with real-time clock
                    key = cv2.waitKey(1) & 0xFF
                    lag = time.time() - target_next_time
                    if lag > frame_interval:
                        # Skip intermediate frames to prevent slow-motion playback lag
                        skip_count = min(int(lag / frame_interval), 2)
                        for _ in range(skip_count):
                            if not capture.grab():
                                break
                            frame_idx += 1
            else:
                key = cv2.waitKey(1) & 0xFF

            # Press Q to exit
            if key == ord('q'):
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
        cv2.destroyAllWindows()

    total_time = time.time() - start_wall_time
    print(f"[Done] Finished {frame_idx} frames in {total_time:.2f}s ({frame_idx / total_time:.1f} FPS).")
    if args.output_video is not None:
        print(f"[Saved] Output video: {args.output_video}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
