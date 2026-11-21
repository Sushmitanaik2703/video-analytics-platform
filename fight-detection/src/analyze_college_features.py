
import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FEATURES_DIR = PROJECT_DIR / "features"

# Original videos are read from your backup/development folder.
# We are NOT moving or modifying these videos.
VIDEO_FOLDERS = {
    "Fight": Path(
        r"C:\Users\sushm\fight-detection-ml\dataset\recorded\fight"
    ),
    "NonFight": Path(
        r"C:\Users\sushm\fight-detection-ml\dataset\recorded\nonfight"
    )
}

# Import the existing feature extraction function.
sys.path.append(str(SCRIPT_DIR))
from extract_features import extract_video_features


# --------------------------------------------------
# CREATE OUTPUT DIRECTORY
# --------------------------------------------------

FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# EXTRACT FEATURES FROM COLLEGE VIDEOS
# --------------------------------------------------

results = []

SUPPORTED_FORMATS = (".mp4", ".avi", ".mov", ".mkv")

print("\n========== COLLEGE VIDEO FEATURE ANALYSIS ==========\n")

for actual_label, folder in VIDEO_FOLDERS.items():

    if not folder.exists():
        print(f"Folder not found: {folder}")
        continue

    video_files = sorted(
        file for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS
    )

    print(f"\nAnalyzing {actual_label} videos...")
    print(f"Videos found: {len(video_files)}")

    for video_path in video_files:

        print(f"Extracting: {video_path.name}")

        try:
            features = extract_video_features(str(video_path))

            if features:
                features["video_name"] = video_path.name
                features["actual_label"] = actual_label
                results.append(features)
            else:
                print(f"Warning: No features extracted from {video_path.name}")

        except Exception as error:
            print(f"Error processing {video_path.name}: {error}")


# --------------------------------------------------
# SAVE INDIVIDUAL FEATURE VALUES AND GROUP AVERAGES
# --------------------------------------------------

if results:

    df = pd.DataFrame(results)

    # Keep identification columns separate from numeric features.
    identification_columns = ["video_name", "actual_label", "label"]

    feature_columns = [
        column for column in df.columns
        if column not in identification_columns
    ]

    # Save individual video feature values.
    analysis_file = FEATURES_DIR / "college_feature_analysis.csv"
    df.to_csv(analysis_file, index=False)

    # Calculate average motion features for Fight and NonFight videos.
    summary = df.groupby("actual_label")[feature_columns].mean()

    summary_file = FEATURES_DIR / "college_feature_summary.csv"
    summary.to_csv(summary_file)

    # Display results.
    print("\n========== INDIVIDUAL VIDEO FEATURES ==========\n")
    print(df.to_string(index=False))

    print("\n========== GROUP AVERAGES ==========\n")
    print(summary.to_string())

    print("\n========== ANALYSIS COMPLETED ==========\n")
    print(f"Videos analyzed: {len(df)}")
    print(f"Fight videos: {(df['actual_label'] == 'Fight').sum()}")
    print(f"NonFight videos: {(df['actual_label'] == 'NonFight').sum()}")

    print("\nFiles saved:")
    print(analysis_file)
    print(summary_file)

else:
    print("\nNo videos were analyzed. Check the video folder paths.")
