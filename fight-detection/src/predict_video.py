import cv2
import numpy as np
import joblib
import pandas as pd
import sys
import os

# Configuration
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FRAME_SKIP = 5


MODEL_PATH = "models/fight_detection_final.pkl"



def extract_video_features(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return None

    previous_frame = None

    motion_values = []
    horizontal_values = []
    vertical_values = []
    energy_values = []

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        if previous_frame is not None:

            flow = cv2.calcOpticalFlowFarneback(
                previous_frame,
                gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0
            )

            magnitude, angle = cv2.cartToPolar(
                flow[..., 0],
                flow[..., 1]
            )

            motion_values.append(np.mean(magnitude))

            horizontal_values.append(
                np.mean(np.abs(flow[..., 0]))
            )

            vertical_values.append(
                np.mean(np.abs(flow[..., 1]))
            )

            energy_values.append(
                np.mean(magnitude ** 2)
            )

        previous_frame = gray

    cap.release()

    if len(motion_values) == 0:
        return None

    motion_values = np.array(motion_values)

    threshold = np.mean(motion_values) + np.std(motion_values)

    features = {
        "mean_motion": np.mean(motion_values),
        "max_motion": np.max(motion_values),
        "std_motion": np.std(motion_values),
        "mean_horizontal": np.mean(horizontal_values),
        "mean_vertical": np.mean(vertical_values),
        "motion_variation": np.std(motion_values),
        "motion_energy": np.mean(energy_values),
        "high_motion_ratio": np.mean(motion_values > threshold),
        "median_motion": np.median(motion_values),
        "min_motion": np.min(motion_values)
    }

    return features


def predict_video(video_path):

    if not os.path.exists(video_path):
        print("Error: Video file not found.")
        return

    print("\nExtracting video features...")

    features = extract_video_features(video_path)

    if features is None:
        print("Error: Could not extract features.")
        return

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Convert features into DataFrame
    X = pd.DataFrame([features])

    # Ensure correct feature order
    X = X[model.feature_names_in_]

    # Predict
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    print("\n========== FIGHT DETECTION RESULT ==========")

    print("Video:", os.path.basename(video_path))

    print("NonFight probability:", round(probabilities[0] * 100, 2), "%")
    print("Fight probability:", round(probabilities[1] * 100, 2), "%")

    if prediction == 1:
        print("\nALERT: Possible fight detected!")
        print("Human verification is required.")
    else:
        print("\nNo fight detected by the model.")

    print("\n============================================")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python src/predict_video.py <video_path>")
    else:
        predict_video(sys.argv[1])