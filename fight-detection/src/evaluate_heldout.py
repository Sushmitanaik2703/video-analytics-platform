import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load trained model
model = joblib.load("models/fight_detection_model.pkl")

# Load held-out dataset
df = pd.read_csv("features/heldout_features.csv")

# Separate features and actual labels
X = df.drop(columns=["video_name", "label"])
y_true = df["label"]

# Predict
y_pred = model.predict(X)

print("\nHELD-OUT MODEL EVALUATION")
print("=========================")

print("Total videos:", len(df))

print("\nAccuracy:", accuracy_score(y_true, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["NonFight", "Fight"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nIndividual Predictions:")
for name, actual, predicted in zip(
    df["video_name"], y_true, y_pred
):
    actual_label = "Fight" if actual == 1 else "NonFight"
    predicted_label = "Fight" if predicted == 1 else "NonFight"

    print(
        f"{name}: Actual={actual_label}, Predicted={predicted_label}"
    )