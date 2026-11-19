import os
import sys
import joblib
import pandas as pd

# Allow Python to import files from the src folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_features import extract_video_features

# Model paths
ORIGINAL_MODEL_PATH = "models/fight_detection_final.pkl"
AUGMENTED_MODEL_PATH = "models/fight_detection_augmented_candidate.pkl"

# College recording folders
VIDEO_FOLDERS = {
    "Fight": "dataset/recorded/Fight",
    "NonFight": "dataset/recorded/NonFight"
}

# Load both models
original_model = joblib.load(ORIGINAL_MODEL_PATH)
augmented_model = joblib.load(AUGMENTED_MODEL_PATH)

results = []

print("\nComparing Original and Augmented Models...\n")

for actual_label, folder in VIDEO_FOLDERS.items():

    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        continue

    for filename in os.listdir(folder):

        if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        video_path = os.path.join(folder, filename)

        print(f"Processing: {filename}")

        # Extract features once
        features = extract_video_features(video_path)

        if not features:
            print("Could not extract features. Skipping.")
            continue

        feature_df = pd.DataFrame([features])

        # Predict using the exact features expected by each model
        original_X = feature_df[original_model.feature_names_in_]
        augmented_X = feature_df[augmented_model.feature_names_in_]

        original_prediction = original_model.predict(original_X)[0]
        augmented_prediction = augmented_model.predict(augmented_X)[0]

        original_probability = original_model.predict_proba(original_X)[0][1]
        augmented_probability = augmented_model.predict_proba(augmented_X)[0][1]

        results.append({
            "Video": filename,
            "Actual": actual_label,
            "Original Prediction": "Fight" if original_prediction == 1 else "NonFight",
            "Original Fight Probability (%)": round(original_probability * 100, 2),
            "Augmented Prediction": "Fight" if augmented_prediction == 1 else "NonFight",
            "Augmented Fight Probability (%)": round(augmented_probability * 100, 2)
        })

# Display and save results
if results:
    df = pd.DataFrame(results)

    print("\n========== MODEL COMPARISON ==========\n")
    print(df.to_string(index=False))

    os.makedirs("features", exist_ok=True)
    df.to_csv("features/college_model_comparison.csv", index=False)

    print("\nResults saved to:")
    print("features/college_model_comparison.csv")
else:
    print("No videos were processed.")