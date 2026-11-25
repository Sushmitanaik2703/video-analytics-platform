# Smart CCTV Monitoring System

A computer vision project for CCTV surveillance.

## Project Features
- Physical Altercation Detection
- Unattended Object Detection

## Team Responsibilities
- Member 1: Object Detection
- Member 2: Unattended Object Detection
- Member 3: Fight Detection
- Member 4: Backend and Dashboard

## Unattended Object Detection (Member 2)

Run `unattended_object.py` to detect people, backpacks, handbags, and suitcases.
It uses the existing `yolov8n.pt` file in this folder and Ultralytics ByteTrack
to maintain IDs across frames. The faster YOLOv8n model is selected by default;
use `--model yolov8s.pt` if you prefer the larger model. Inference uses CUDA
when available and otherwise runs on the CPU. It does not modify Member 1's
`detect_objects.py`.

### Install dependencies in PowerShell

From the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Incident database

The desktop dashboard stores incidents in the local MongoDB Community Server,
database `smart_cctv`, collection `incidents`. Install MongoDB Community Server
and ensure its local service is running before starting the application.
`pymongo` is installed with `requirements.txt`. Dashboard history loads
asynchronously; if MongoDB is unavailable, detection and video playback continue
and failed database operations are logged and retried.

### Run with the default webcam

```powershell
py unattended_object.py --webcam
```

### Run with a video file

```powershell
py unattended_object.py --video "C:\path\to\sample.mp4"
```

Press `q` to close the video window.
For faster processing, detection runs on every third frame by default. Change
the sampling rate with `--frame-stride` (for example, `--frame-stride 4` for
even faster processing or `--frame-stride 1` to process every frame).

### Detection settings

The default settings are a confidence threshold of `0.50`, stationary time of
`15` seconds, nearby-person distance of `150` pixels, stationary movement
tolerance of `25` pixels, and an alert cooldown of `30` seconds. Override them
on the command line, for example:

```powershell
py unattended_object.py --video "C:\path\to\sample.mp4" --confidence 0.45 --stationary-time 20 --person-distance 180 --stationary-distance 30 --alert-cooldown 45
```

### How the feature works

1. YOLOv8n detects people and the three COCO bag classes: backpack, handbag,
   and suitcase.
2. ByteTrack assigns tracking IDs. The module keeps each bag's class, bounding
   box, center position, first/last detection times, and stationary duration.
3. If a person is within the configured pixel distance when a bag is observed,
   the closest person track is associated with that bag.
4. The bag must stay within its stationary movement tolerance for the
   stationary-time threshold, with no detected person within the configured
   nearby-person distance. An owner does not have to be identified first.
5. Once triggered, the red alert remains active if a person returns or the bag
   is temporarily missed. It clears after significant movement is detected for
   several consecutive frames; the cooldown rate-limits a later alert.

“POSSIBLE UNATTENDED OBJECT” is a warning, not a determination that an item
has been abandoned. Person-to-bag association is a simple nearest-center
approximation; overlapping people, missed detections, occlusion, camera
perspective, and tracking-ID changes can lead to missed or false alerts. Test
with consent-appropriate footage containing bags that are picked up, left
stationary while a person remains nearby, and left stationary with no person
nearby. Try different distances and camera views; this college-project
prototype is not production-grade.
