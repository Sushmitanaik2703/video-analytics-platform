
import cv2
import joblib
import pandas as pd
import os
import sys
from pathlib import Path
from ultralytics import YOLO

# ---------------- PATHS ----------------

# Main fight-detection folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Random Forest model
MODEL_PATH = PROJECT_DIR / "models" / "fight_detection_final.pkl"

# YOLOv8s weights are stored in the repository root
YOLO_MODEL_PATH = PROJECT_DIR.parent / "yolov8s.pt"

# Import existing feature extractor
sys.path.append(str(Path(__file__).resolve().parent))

from extract_features import extract_video_features


# ---------------- MAIN DEMO ----------------

def run_demo(video_path):

    if not os.path.exists(video_path):
        print("Video not found!")
        return

    # Check model files
    if not MODEL_PATH.exists():
        print("Random Forest model not found:", MODEL_PATH)
        return

    if not YOLO_MODEL_PATH.exists():
        print("YOLOv8s model not found:", YOLO_MODEL_PATH)
        return

    # Load Random Forest model
    print("Loading Random Forest model...")
    model = joblib.load(MODEL_PATH)

    # Load YOLOv8s model
    print("Loading YOLOv8s model...")
    yolo_model = YOLO(str(YOLO_MODEL_PATH))

    # ---------------- FEATURE EXTRACTION ----------------

    print("\nAnalyzing video for possible physical altercation...")

    features = extract_video_features(video_path)

    if features is None:
        print("Feature extraction failed.")
        return

    X = pd.DataFrame([features])
    X = X[model.feature_names_in_]

    # Predict using Random Forest
    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]
    fight_probability = probabilities[1] * 100

    print("\n---------- DETECTION RESULT ----------")

    print("Fight Probability:", round(fight_probability, 2), "%")

    if prediction == 1:
        print("Result: POSSIBLE PHYSICAL ALTERCATION")
        print("Bounding box colour: RED")
    else:
        print("Result: NO PHYSICAL ALTERCATION")
        print("Bounding box colour: BLUE")

    print("--------------------------------------")

    # ---------------- OPEN VIDEO ----------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Unable to open video.")
        return

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # YOLO person detection (class 0 = person)
            results = yolo_model(
                frame,
                conf=0.35,
                classes=[0],
                device="cpu",
                verbose=False
            )

            # Choose bounding-box colour
            # OpenCV uses BGR format
            if prediction == 1:
                box_color = (0, 0, 255)       # RED
                message = "POSSIBLE PHYSICAL ALTERCATION"
                banner_color = (0, 0, 255)
            else:
                box_color = (255, 0, 0)       # BLUE
                message = "NO PHYSICAL ALTERCATION"
                banner_color = (255, 0, 0)

            # ---------------- DRAW PERSON BOXES ----------------

            for box in results[0].boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                confidence = float(box.conf[0])

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )

                # Person label
                cv2.putText(
                    frame,
                    f"Person {confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2
                )

            # ---------------- TOP ALERT BANNER ----------------

            cv2.rectangle(
                frame,
                (0, 0),
                (frame.shape[1], 60),
                banner_color,
                -1
            )

            cv2.putText(
                frame,
                message,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # ---------------- FIGHT PROBABILITY ----------------

            cv2.putText(
                frame,
                f"Fight Probability: {fight_probability:.2f}%",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Display video
            cv2.imshow(
                "Physical Altercation Detection",
                frame
            )

            # Press Q to exit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ---------------- RUN PROGRAM ----------------

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: python fight-detection/src/fight_detection_demo.py <video_path>"
        )

    else:

        run_demo(sys.argv[1])
