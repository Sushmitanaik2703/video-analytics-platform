import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# Project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "features", "development_features.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "fight_detection_final.pkl"
)

# Load development dataset
print("Loading development dataset...")

df = pd.read_csv(DATA_PATH)

# Remove video name, label and duplicate feature
X = df.drop(columns=["video_name", "label", "motion_variation"])
y = df["label"]

print("Total videos:", len(df))
print("Fight videos:", sum(y == 1))
print("NonFight videos:", sum(y == 0))
print("Features:", list(X.columns))

# Best parameters obtained from hyperparameter tuning
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=42,
    class_weight="balanced"
)

# Train using all development videos
print("\nTraining final model...")

model.fit(X, y)

# Save final model
joblib.dump(model, MODEL_PATH)

print("\nFinal model trained successfully!")
print("Model saved at:")
print(MODEL_PATH)

print("\nTotal training videos:", len(X))
print("Total features:", X.shape[1])