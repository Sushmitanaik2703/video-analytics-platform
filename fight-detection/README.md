# Fight Detection Module

## 1. Overview

The Fight Detection module is a component of the **Video Analytics Platform for Automated CCTV Monitoring** mini project.

Its purpose is to analyse surveillance video footage and identify possible fighting activities using computer vision and machine learning.

The module uses **Farneback Optical Flow** to extract motion-based features from video frames and a **Random Forest Classifier** to classify the video as Fight or NonFight.

The system is intended as a prototype to assist human monitoring. Its predictions require human verification.

## 2. Technologies Used

- Python 3.11
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- Farneback Optical Flow
- Random Forest Classifier

## 3. Working Methodology

1. **Video Input:** Accepts a surveillance video as input.
2. **Frame Processing:** OpenCV reads video frames and resizes them to 320 × 240 pixels.
3. **Optical Flow:** Farneback Optical Flow estimates movement between consecutive sampled frames.
4. **Feature Extraction:** Calculates motion-based features such as mean motion, maximum motion, motion energy, horizontal movement, vertical movement and motion variation.
5. **Classification:** Extracted features are passed to the trained Random Forest model.
6. **Prediction:** The model predicts Fight or NonFight and provides class probabilities.
7. **Alert:** A possible-fight alert is generated when the video is classified as Fight. Human verification is required.

## 4. Dataset

The development dataset was prepared using selected videos from the UCF-Crime dataset.

Two classes were used:

- **Fight (1):** Videos showing fighting activities.
- **NonFight (0):** Normal activities without fighting.

The development dataset contains 90 videos:

- 45 Fight videos
- 45 NonFight videos

A separate set of 10 videos was reserved for preliminary held-out evaluation.

Additional college-recorded videos and unattended-object detection clips were used for exploratory testing.

## 5. Model Details

- **Feature Extraction:** Farneback Optical Flow
- **Classification Algorithm:** Random Forest
- **Number of Estimators:** 100
- **Maximum Tree Depth:** 5
- **Minimum Samples Split:** 5
- **Class Weight:** Balanced

The final model was trained using all 90 development videos and saved as:

`models/fight_detection_final.pkl`

The model uses nine motion-based features for classification.

## 6. Project Structure

```text
fight-detection/
├── src/
│   ├── extract_features.py
│   ├── predict_video.py
│   ├── train_final_model.py
│   ├── evaluate_heldout.py
│   ├── compare_college_models.py
│   ├── generate_college_report.py
│   └── analyze_college_features.py
├── models/
│   └── fight_detection_final.pkl
├── features/
│   ├── development_features.csv
│   ├── heldout_features.csv
│   ├── college_model_comparison.csv
│   ├── college_feature_analysis.csv
│   └── college_feature_summary.csv
├── reports/
│   ├── college_evaluation_report.txt
│   ├── college_accuracy_comparison.png
│   ├── original_confusion_matrix.png
│   └── augmented_confusion_matrix.png
├── .gitignore
└── README.md
```

## 7. Setup Instructions

### Step 1: Create a virtual environment

```bash
python -m venv venv
```

### Step 2: Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install opencv-python numpy pandas scikit-learn matplotlib seaborn joblib tqdm scipy
```

### Step 4: Prepare the dataset

Place the dataset videos in this structure:

```text
dataset/
└── raw/
    ├── Fight/
    └── NonFight/
```

### Step 5: Extract features

From the project root, run:

```bash
python src/extract_features.py
```

### Step 6: Train the model

```bash
python src/train_final_model.py
```

### Step 7: Predict a video

```bash
python src/predict_video.py path/to/video.mp4
```

The prediction script loads the trained model from `models/fight_detection_final.pkl`.

## 8. Testing and Evaluation

The module was tested using UCF-Crime videos and additional college-recorded videos.

### Preliminary held-out evaluation

An earlier model evaluation on 10 held-out videos produced:

- Accuracy: 70%
- Fight recall: 80%
- NonFight recall: 60%

Confusion matrix:

```text
[[3, 2],
 [1, 4]]
```

This result belongs to the earlier model evaluated at that stage, not the final model trained on all 90 development videos.

### College and unattended-object video comparison

An exploratory comparison using 10 videos (3 staged Fight and 7 NonFight) produced:

| Metric | Original Model | Augmented Candidate |
|---|---:|---:|
| Accuracy | 30% | 60% |
| Fight Recall | 100% | 33.33% |
| NonFight Recall | 0% | 71.43% |

These results are based on a small exploratory sample and should not be treated as reliable real-world performance measurements.

The evaluation plots and detailed results are available in the `reports/` directory.

## 9. Limitations

- The model relies on motion-based features and does not understand the context of an activity.
- Running, dancing, sports and sudden movements may cause false predictions.
- Lighting changes and camera movement can affect motion estimation.
- Performance may vary between the UCF-Crime dataset and locally recorded videos.
- The current implementation classifies an entire video rather than identifying the exact time and location of a fight.
- Predictions require human verification before any action is taken.

## 10. Future Improvements

- Improve feature extraction using additional motion and spatial information.
- Expand the dataset with more diverse normal and fighting activities.
- Reduce false-positive and false-negative predictions.
- Integrate the model with the common backend and dashboard.
- Add timestamp-based detection and visualisation of suspicious segments.

## 11. Team Project

Project: Video Analytics Platform for Automated CCTV Monitoring
Institution: Sahyadri College of Engineering & Management
Department: Computer Science and Engineering
Module: Fight Detection