import argparse
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
        default=0.50,
        help="Minimum detection confidence (default: 0.50)",
    )

    return parser.parse_args()



def main() -> int:
    args = parse_args()
    model_path = Path(__file__).resolve().parent / "yolov8n.pt"

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

    if not capture.isOpened():
        if args.webcam:
            print("Error: Could not open the webcam (camera index 0).", file=sys.stderr)
        else:
            print(f"Error: Could not open video file: {args.video}", file=sys.stderr)
        capture.release()
        return 1

    try:
        while True:
            success, frame = capture.read()
            if not success:
                if args.webcam:
                    print("Error: Could not read a frame from the webcam.", file=sys.stderr)
                break

            results = model(
    frame,
    conf=args.confidence,
    device="cpu",
    verbose=False
)
            annotated_frame = results[0].plot()
            cv2.imshow("Smart CCTV Monitoring System - Object Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())