import argparse
import json
import math
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect objects in a video or webcam stream."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Path to a video file")
    source.add_argument("--webcam", action="store_true", help="Use the default webcam")

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum detection confidence (default: 0.50)",
    )

    return parser.parse_args()



def main() -> int:
    args = parse_args()
    model_path = Path(__file__).resolve().parent / "yolov8s.pt"

    if not model_path.is_file():
        print(f"Error: YOLO model not found: {model_path}", file=sys.stderr)
        return 1

    try:
        model = YOLO(str(model_path))
    except Exception as error:
        print(f"Error: Could not load YOLO model: {error}", file=sys.stderr)
        return 1

    if args.webcam:
        capture = cv2.VideoCapture(0)
    else:
        capture = cv2.VideoCapture(str(args.video))

    output_dir = Path(__file__).resolve().parent / "outputs"
    video_output_path = output_dir / "annotated_video.mp4"
    detections_output_path = output_dir / "detections.jsonl"
    writer = None
    detections_file = None
    frame_number = 0

    try:
        if not capture.isOpened():
            if args.webcam:
                print("Error: Could not open the webcam (camera index 0).", file=sys.stderr)
            else:
                print(f"Error: Could not open video file: {args.video}", file=sys.stderr)
            return 1

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            print(f"Error: Could not create output folder {output_dir}: {error}", file=sys.stderr)
            return 1

        while True:
            success, frame = capture.read()
            if not success:
                if args.webcam:
                    print("Error: Could not read a frame from the webcam.", file=sys.stderr)
                break

            frame_number += 1
            if writer is None:
                height, width = frame.shape[:2]
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                if not math.isfinite(fps) or fps <= 0:
                    fps = 20.0 if args.webcam else 30.0

                try:
                    writer = cv2.VideoWriter(
                        str(video_output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                except Exception as error:
                    print(f"Error: Could not create output video {video_output_path}: {error}", file=sys.stderr)
                    return 1

                if not writer.isOpened():
                    print(f"Error: Could not open output video for writing: {video_output_path}", file=sys.stderr)
                    return 1

                try:
                    detections_file = detections_output_path.open("w", encoding="utf-8")
                except OSError as error:
                    print(f"Error: Could not open detections file {detections_output_path}: {error}", file=sys.stderr)
                    return 1

            results = model(
    frame,
    conf=args.confidence,
    classes=[0, 24, 26, 28],
    device="cpu",
    verbose=False
)
            result = results[0]
            annotated_frame = result.plot()
            writer.write(annotated_frame)

            detections = []
            for box in result.boxes:
                class_id = int(box.cls.item())
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": result.names[class_id],
                        "confidence": float(box.conf.item()),
                        "bbox_xyxy": [float(value) for value in box.xyxy[0].tolist()],
                    }
                )

            frame_record = {"frame_number": frame_number, "detections": detections}
            detections_file.write(json.dumps(frame_record) + "\n")
            cv2.imshow("Smart CCTV Monitoring System - Object Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if detections_file is not None:
            detections_file.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())