import os
import sys
import pandas as pd

# Import the feature extraction function
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from extract_features import extract_video_features

folders = {
    "Fight": "dataset/recorded/Fight",
    "NonFight": "dataset/recorded/NonFight"
}

results = []

print("\nAnalyzing motion features...\n")

for actual_label, folder in folders.items():

    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        continue

    for filename in os.listdir(folder):

        if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        video_path = os.path.join(folder, filename)

        print(f"Extracting: {filename}")

        features = extract_video_features(video_path)

        if features:
            features["video_name"] = filename
            features["actual_label"] = actual_label
            results.append(features)

if results:

    df = pd.DataFrame(results)

    # Save individual video feature values
    df.to_csv("features/college_feature_analysis.csv", index=False)

    print("\n========== FEATURE VALUES ==========\n")
    print(df.to_string(index=False))

    # Calculate average feature values for each group
    feature_columns = [
        col for col in df.columns
        if col not in ["video_name", "actual_label", "label"]
    ]

    summary = df.groupby("actual_label")[feature_columns].mean()

    print("\n========== GROUP AVERAGES ==========\n")
    print(summary.to_string())

    summary.to_csv("features/college_feature_summary.csv")

    print("\nSaved:")
    print("features/college_feature_analysis.csv")
    print("features/college_feature_summary.csv")

else:
    print("No videos analyzed.")