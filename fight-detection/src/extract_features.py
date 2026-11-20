
import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

DATASET_PATH = "dataset/raw"
OUTPUT_FILE = "features/motion_features.csv"

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FRAME_SKIP = 5


def extract_video_features(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
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

    return {
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


# Dataset processing runs only when this file is executed directly.
# Importing extract_video_features() will not process the dataset.

if __name__ == "__main__":

    all_features = []

    categories = {
        "Fight": 1,
        "NonFight": 0
    }

    for category, label in categories.items():

        folder = os.path.join(DATASET_PATH, category)

        videos = [
            file for file in os.listdir(folder)
            if file.lower().endswith(".mp4")
        ]

        print(f"\nProcessing {category} videos...")

        for video in tqdm(videos):

            video_path = os.path.join(folder, video)

            features = extract_video_features(video_path)

            if features is not None:

                features["video_name"] = video
                features["label"] = label

                all_features.append(features)

            else:
                print(f"Could not process: {video}")

    df = pd.DataFrame(all_features)

    columns = [
        "video_name",
        "mean_motion",
        "max_motion",
        "std_motion",
        "mean_horizontal",
        "mean_vertical",
        "motion_variation",
        "motion_energy",
        "high_motion_ratio",
        "median_motion",
        "min_motion",
        "label"
    ]

    df = df[columns]

    os.makedirs("features", exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nImproved feature extraction completed!")
    print("Total processed videos:", len(df))
    print("Fight videos:", len(df[df["label"] == 1]))
    print("NonFight videos:", len(df[df["label"] == 0]))
    print("Total features:", len(columns) - 2)
    print("Saved to:", OUTPUT_FILE)
