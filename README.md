# Smart CCTV Monitoring System

## Overview

A Smart CCTV Monitoring System using YOLOv8, ByteTrack and OSNet for unattended baggage detection, owner re-identification and retrieval verification.

## Features

- Person and baggage detection
- Object tracking using ByteTrack
- Unattended baggage detection
- Owner identification using OSNet
- Owner return and retrieval verification
- Annotated video and event screenshot generation

## 1. Clone the Repository

```powershell
git clone -b member-2 https://github.com/Sushmitanaik2703/video-analytics-platform.git

cd video-analytics-platform
```

## 2. Prerequisites

Install Python and Git.

## 3. Create Virtual Environment

```powershell
python -m venv venv
venv\Scripts\activate
```

## 4. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU acceleration, install a PyTorch and Torchvision combination compatible with your CUDA version.

## 5. Download Model Weights

Download the following three files from the shared Google Drive folder:

**Google Drive:** [INSERT GOOGLE DRIVE LINK]

| Model | Purpose |
|---|---|
| `best.pt` | Custom-trained YOLO model |
| `yolov8s.pt` | Standard YOLO fallback |
| `osnet_x0_25_msmt17.pth` | OSNet person re-identification |

Place them in these locations:

```text
video-analytics-platform/
├── yolov8s.pt
├── runs/
│   └── unattended_yolov8n_targeted_v2/
│       └── weights/
│           └── best.pt
└── experimental/
    └── owner_reidentification/
        └── weights/
            └── osnet_x0_25_msmt17.pth
```

Do not rename the files.

## 6. Prepare Input Video

Place your CCTV video in the project directory or provide its full path in the command.

## 7. Run the System

From the project root:

```powershell
python -m experimental.owner_reidentification.main --video "datasets/samples/real_world_tests/aboda_video1.avi" --output-video "output/aboda_result.mp4" --save-screenshots "output/screenshots"
```

The system selects `best.pt` when the input video filename contains `video5`. Otherwise, it selects `yolov8s.pt`.

## 8. Output

The system generates:

- Annotated CCTV video
- Event screenshots
- Owner re-identification and retrieval verification results

## Important Notes

- Download all three model weights before execution.
- Model weights are shared through Google Drive, not committed to GitHub.
- Preserve the directory structure and model filenames.
- Run commands from the repository root.

## Repository

https://github.com/Sushmitanaik2703/video-analytics-platform

Branch: `member-2`
