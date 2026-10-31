
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


# COCO class IDs:
# 0 = person
# 24 = backpack
# 26 = handbag
# 28 = suitcase
DETECTION_CLASSES = [0, 24, 26, 28]

BAG_CLASSES = {
    "backpack",
    "handbag",
    "suitcase",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect people and bags in a video or webcam stream."
    )

    source = parser.add_mutually_exclusive_group(required=True)

    source.add_argument(
        "--video",
        type=Path,
        help="Path to a video file",
    )

    source.add_argument(
        "--webcam",
        action="store_true",
        help="Use the default webcam",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum detection confidence (default: 0.35)",
    )

    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0.0 and 1.0")

    return args


def main() -> int:
    args = parse_args()

    project_dir = Path(__file__).resolve().parent
    model_path = project_dir / "yolov8s.pt"

    # Check whether the model exists
    if not model_path.is_file():
        print(
            f"Error: YOLO model not found: {model_path}",
            file=sys.stderr,
        )
        return 1

    # Load YOLO model
    try:
        model = YOLO(str(model_path))
    except Exception as error:
        print(
            f"Error: Could not load YOLO model: {error}",
            file=sys.stderr,
        )
        return 1

    # Initialize video source
    if args.webcam:
        capture = cv2.VideoCapture(0)
    else:
        capture = cv2.VideoCapture(str(args.video))

    # Output paths
    output_dir = project_dir / "outputs"

    video_output_path = output_dir / "annotated_video.mp4"
    detections_output_path = output_dir / "detections.jsonl"

    writer = None
    detections_file = None

    frame_number = 0
    fps = 20.0 if args.webcam else 30.0

    try:
        # Check video/webcam connection
        if not capture.isOpened():
            if args.webcam:
                print(
                    "Error: Could not open webcam (camera index 0).",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Error: Could not open video file: {args.video}",
                    file=sys.stderr,
                )

            return 1

        # Create output directory
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            print(
                f"Error: Could not create output directory: {error}",
                file=sys.stderr,
            )
            return 1

        print("Smart CCTV Object Detection Started")
        print("Detecting: Person, Backpack, Handbag, Suitcase")
        print("Press 'q' to stop detection.")

        while True:
            success, frame = capture.read()

            if not success:
                if args.webcam:
                    print(
                        "Webcam stream ended or could not read frame.",
                        file=sys.stderr,
                    )
                break

            frame_number += 1

            # Initialize output writer and JSONL file
            # using the first frame's dimensions and source FPS.
            if writer is None:
                height, width = frame.shape[:2]

                source_fps = float(
                    capture.get(cv2.CAP_PROP_FPS)
                )

                if math.isfinite(source_fps) and source_fps > 0:
                    fps = source_fps

                try:
                    writer = cv2.VideoWriter(
                        str(video_output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                except Exception as error:
                    print(
                        f"Error: Could not create output video: {error}",
                        file=sys.stderr,
                    )
                    return 1

                if not writer.isOpened():
                    print(
                        f"Error: Could not open output video: "
                        f"{video_output_path}",
                        file=sys.stderr,
                    )
                    return 1

                try:
                    detections_file = detections_output_path.open(
                        "w",
                        encoding="utf-8",
                    )
                except OSError as error:
                    print(
                        f"Error: Could not open detections file: {error}",
                        file=sys.stderr,
                    )
                    return 1

            # Run YOLO detection
            results = model(
                frame,
                conf=args.confidence,
                classes=DETECTION_CLASSES,
                device="cpu",
                verbose=False,
            )

            result = results[0]

            # Generate annotated frame
            annotated_frame = result.plot()

            # Save annotated frame to output video
            writer.write(annotated_frame)

            # Process detections
            detections = []

            for box in result.boxes:
                class_id = int(box.cls.item())

                # Original YOLO class name
                source_class_name = result.names[class_id]

                # Standardize bag-related classes
                if source_class_name in BAG_CLASSES:
                    class_name = "bag"
                else:
                    class_name = source_class_name

                # Bounding box coordinates
                bbox = [
                    round(float(value), 2)
                    for value in box.xyxy[0].tolist()
                ]

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "source_class_name": source_class_name,
                        "confidence": round(
                            float(box.conf.item()), 4
                        ),
                        "bbox_xyxy": bbox,
                    }
                )

            # Approximate timestamp based on source FPS
            timestamp_sec = (frame_number - 1) / fps

            # Create one JSON record for every processed frame,
            # including frames where no objects are detected.
            frame_record = {
                "frame_number": frame_number,
                "timestamp_sec": round(timestamp_sec, 3),
                "detections": detections,
            }

            detections_file.write(
                json.dumps(frame_record) + "\n"
            )

            # Display annotated video
            cv2.imshow(
                "Smart CCTV Monitoring System - Object Detection",
                annotated_frame,
            )

            # Stop when Q is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nDetection stopped by user.")

    except Exception as error:
        print(
            f"Error during detection: {error}",
            file=sys.stderr,
        )
        return 1

    finally:
        # Release all resources
        capture.release()

        if writer is not None:
            writer.release()

        if detections_file is not None:
            detections_file.close()

        cv2.destroyAllWindows()

    print("\nDetection completed.")
    print(f"Annotated video: {video_output_path}")
    print(f"Detection data: {detections_output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
