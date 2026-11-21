
import joblib
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

MODEL_PATH = PROJECT_DIR / "models" / "fight_detection_final.pkl"
INPUT_FILE = PROJECT_DIR / "features" / "college_feature_analysis.csv"
OUTPUT_FILE = PROJECT_DIR / "features" / "college_prediction_results.csv"


# --------------------------------------------------
# LOAD MODEL AND FEATURES
# --------------------------------------------------

print("\nLoading existing final model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")
print("Model classes:", model.classes_)

df = pd.read_csv(INPUT_FILE)

# Select features in the exact order expected by the model.
X = df[list(model.feature_names_in_)]

# --------------------------------------------------
# PREDICT
# --------------------------------------------------

predictions = model.predict(X)
probabilities = model.predict_proba(X)

# Find the correct Fight probability column.
fight_index = list(model.classes_).index(1)

df["predicted_label"] = predictions
df["fight_probability"] = probabilities[:, fight_index] * 100

df["predicted_label"] = df["predicted_label"].map({
    0: "NonFight",
    1: "Fight"
})

# --------------------------------------------------
# DISPLAY INDIVIDUAL RESULTS
# --------------------------------------------------

print("\n========== COLLEGE VIDEO PREDICTIONS ==========\n")

for _, row in df.iterrows():

    print("Video:", row["video_name"])
    print("Actual:", row["actual_label"])
    print("Predicted:", row["predicted_label"])
    print("Fight probability:", round(row["fight_probability"], 2), "%")

    if row["actual_label"] == "NonFight" and row["predicted_label"] == "Fight":
        print("Observation: FALSE POSITIVE")

    elif row["actual_label"] == "Fight" and row["predicted_label"] == "NonFight":
        print("Observation: MISSED FIGHT")

    print("----------------------------------------")


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

df["correct"] = df["actual_label"] == df["predicted_label"]

accuracy = df["correct"].mean() * 100

print("\n========== SUMMARY ==========")
print("Total videos:", len(df))
print("Correct predictions:", df["correct"].sum())
print("Incorrect predictions:", len(df) - df["correct"].sum())
print("Accuracy:", round(accuracy, 2), "%")

false_positives = df[
    (df["actual_label"] == "NonFight") &
    (df["predicted_label"] == "Fight")
]

print("\n========== FALSE POSITIVES ==========")

if false_positives.empty:
    print("No false positives.")
else:
    print(
        false_positives[
            ["video_name", "fight_probability"]
        ].to_string(index=False)
    )


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\nResults saved to:")
print(OUTPUT_FILE)
