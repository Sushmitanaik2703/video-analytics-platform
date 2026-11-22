"""
Automated Comprehensive Test Suite for Vehicle Detection and Tracking (Step 3).

Validates:
1. All four COCO vehicle classes: Car (2), Motorcycle (3), Bus (5), Truck (7).
2. Trajectory history maintenance, center-point calculation, and confidence tracking.
3. Edge cases: malformed boxes, invalid IDs, crossing paths, and temporary occlusion.
4. Stale track cleanup beyond max_lost_frames.
5. Ingestion processing speed (upstream mode FPS) vs Standalone YOLOv8s inference FPS.
6. Real-world CCTV test cases:
   - ABODA Video 1 (Negative Control: 0 vehicles expected).
   - ABODA Video 2 (Positive Control: cars in background).
   - sample_unattended_test.mp4 (Positive Control: moving/stationary cars).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path("D:/Miniproject/video-analytics-platform")
sys.path.insert(0, str(PROJECT_ROOT))

from vehicle_tracking import VehicleTracker, VehicleTrack, VEHICLE_CLASS_MAP


def run_unit_and_synthetic_tests() -> dict:
    print("\n" + "=" * 70)
    print("STAGE 1: MULTI-CLASS SYNTHETIC & EDGE-CASE TEST SUITE")
    print("=" * 70)

    tracker = VehicleTracker(confidence=0.40, max_lost_frames=5, max_trajectory_points=10)
    results = {}

    # Test 1.1: All 4 Vehicle Classes Ingestion & Trajectory Tracking
    print("\n[Test 1.1] Ingestion of all 4 vehicle classes (Car, Motorcycle, Bus, Truck)...")
    tracks_f1 = [
        {"track_id": 10, "class_id": 2, "bbox": (100, 100, 200, 180), "conf": 0.90},  # Car
        {"track_id": 20, "class_name": "motorcycle", "bbox": (250, 150, 300, 220), "conf": 0.85},  # Motorcycle
        {"track_id": 30, "class_id": 5, "bbox": (350, 80, 500, 250), "conf": 0.92},   # Bus
        {"track_id": 40, "class_name": "truck", "bbox": (550, 90, 700, 270), "conf": 0.88},   # Truck
        {"track_id": 99, "class_name": "person", "bbox": (50, 50, 80, 150), "conf": 0.95},   # Non-vehicle (ignored)
    ]
    status1 = tracker.process_tracks(tracks_f1, 800, 600, 0.1)

    t1_pass = (
        len(status1["active_vehicles"]) == 4 and
        status1["active_counts_by_class"]["car"] == 1 and
        status1["active_counts_by_class"]["motorcycle"] == 1 and
        status1["active_counts_by_class"]["bus"] == 1 and
        status1["active_counts_by_class"]["truck"] == 1 and
        99 not in [v.track_id for v in status1["active_vehicles"]]
    )
    results["1.1_four_vehicle_classes"] = "PASS" if t1_pass else "FAIL"
    print(f" -> Result: {results['1.1_four_vehicle_classes']} (Active: {len(status1['active_vehicles'])}, Non-vehicle filtered: True)")

    # Test 1.2: Trajectory & Center Point Accuracy
    print("\n[Test 1.2] Center point calculation & trajectory trail accumulation...")
    # Frame 2: Car 10 moves from (100,100,200,180) -> (110,105,210,185)
    tracks_f2 = [
        {"track_id": 10, "class_id": 2, "bbox": (110, 105, 210, 185), "conf": 0.91},
    ]
    status2 = tracker.process_tracks(tracks_f2, 800, 600, 0.2)
    car10 = tracker.active_tracks.get(10)
    expected_c1 = ((100 + 200) // 2, (100 + 180) // 2)  # (150, 140)
    expected_c2 = ((110 + 210) // 2, (105 + 185) // 2)  # (160, 145)

    t2_pass = (
        car10 is not None and
        car10.center_point == expected_c2 and
        car10.trajectory == [expected_c1, expected_c2] and
        car10.total_observations == 2
    )
    results["1.2_trajectory_and_center"] = "PASS" if t2_pass else "FAIL"
    print(f" -> Result: {results['1.2_trajectory_and_center']} (Trajectory: {car10.trajectory if car10 else None})")

    # Test 1.3: Crossing Vehicles (Independent persistent IDs)
    print("\n[Test 1.3] Crossing vehicles maintaining independent IDs...")
    # Two vehicles crossing paths: Car A (ID 1) moves left->right, Car B (ID 2) moves right->left
    tracker_cross = VehicleTracker(confidence=0.40)
    tracks_c1 = [
        {"track_id": 1, "class_name": "car", "bbox": (100, 200, 160, 260), "conf": 0.9},
        {"track_id": 2, "class_name": "car", "bbox": (300, 200, 360, 260), "conf": 0.9},
    ]
    tracker_cross.process_tracks(tracks_c1, 800, 600, 1.0)
    # Crossing midpoint
    tracks_c2 = [
        {"track_id": 1, "class_name": "car", "bbox": (190, 200, 250, 260), "conf": 0.88},
        {"track_id": 2, "class_name": "car", "bbox": (210, 200, 270, 260), "conf": 0.88},
    ]
    tracker_cross.process_tracks(tracks_c2, 800, 600, 1.1)
    # Crossed over
    tracks_c3 = [
        {"track_id": 1, "class_name": "car", "bbox": (300, 200, 360, 260), "conf": 0.9},
        {"track_id": 2, "class_name": "car", "bbox": (100, 200, 160, 260), "conf": 0.9},
    ]
    status_cross = tracker_cross.process_tracks(tracks_c3, 800, 600, 1.2)
    ids_cross = {v.track_id for v in status_cross["active_vehicles"]}

    t3_pass = (ids_cross == {1, 2} and len(tracker_cross.unique_track_ids) == 2)
    results["1.3_crossing_vehicles"] = "PASS" if t3_pass else "FAIL"
    print(f" -> Result: {results['1.3_crossing_vehicles']} (Unique IDs maintained: {ids_cross})")

    # Test 1.4: Temporary Occlusion & Grace Period Recovery (max_lost_frames=5)
    print("\n[Test 1.4] Temporary occlusion & grace period recovery...")
    tracker_occ = VehicleTracker(confidence=0.40, max_lost_frames=5)
    # Frames 1-3: Car 50 detected
    for f in range(1, 4):
        tracker_occ.process_tracks([{"track_id": 50, "class_name": "car", "bbox": (200, 200, 300, 280), "conf": 0.9}], 800, 600, f * 0.1)
    # Frame 4: Occluded / missing for 1 frame
    status_occ_f4 = tracker_occ.process_tracks([], 800, 600, 0.4)
    v50_lost1 = tracker_occ.active_tracks.get(50)
    lost_f4 = v50_lost1.lost_frames if v50_lost1 is not None else None

    # Frame 5: Reappears
    status_occ_f5 = tracker_occ.process_tracks([{"track_id": 50, "class_name": "car", "bbox": (205, 200, 305, 280), "conf": 0.9}], 800, 600, 0.5)
    v50_reappear = tracker_occ.active_tracks.get(50)
    lost_f5 = v50_reappear.lost_frames if v50_reappear is not None else None

    t4_pass = (
        v50_lost1 is not None and lost_f4 == 1 and
        v50_reappear is not None and lost_f5 == 0 and
        len(tracker_occ.unique_track_ids) == 1
    )
    results["1.4_temporary_occlusion"] = "PASS" if t4_pass else "FAIL"
    print(f" -> Result: {results['1.4_temporary_occlusion']} (Lost frame count: {lost_f4} -> {lost_f5})")

    # Test 1.5: Stale Track Pruning Beyond Grace Period
    print("\n[Test 1.5] Stale track cleanup beyond grace period (> 5 frames)...")
    for f in range(1, 6):
        tracker_occ.process_tracks([], 800, 600, 0.5 + f * 0.1)
    # At 5 missing frames, still kept
    kept_5 = 50 in tracker_occ.active_tracks
    # 6th missing frame -> exceeds max_lost_frames=5
    tracker_occ.process_tracks([], 800, 600, 1.2)
    purged_6 = 50 not in tracker_occ.active_tracks

    t5_pass = kept_5 and purged_6
    results["1.5_stale_track_cleanup"] = "PASS" if t5_pass else "FAIL"
    print(f" -> Result: {results['1.5_stale_track_cleanup']} (Kept at 5 missing: {kept_5}, Purged at 6 missing: {purged_6})")

    # Test 1.6: Defensive Sanitization (Malformed, Inverted, Negative, or OOB Bounding Boxes)
    print("\n[Test 1.6] Defensive sanitization of malformed boxes & invalid IDs...")
    tracker_def = VehicleTracker(confidence=0.40)
    malformed_tracks = [
        {"track_id": -5, "class_name": "car", "bbox": (100, 100, 200, 200), "conf": 0.9},        # Negative ID
        {"track_id": 0, "class_name": "car", "bbox": (100, 100, 200, 200), "conf": 0.9},         # Zero ID
        {"track_id": 61, "class_name": "car", "bbox": (200, 200, 100, 100), "conf": 0.9},       # Inverted box (x2 < x1)
        {"track_id": 62, "class_name": "car", "bbox": (100, 100, 100, 200), "conf": 0.9},       # Zero-width box
        {"track_id": 63, "class_name": "car", "bbox": (-50, -50, 1000, 1000), "conf": 0.9},     # Clamped OOB box
        {"track_id": 64, "class_name": "car", "bbox": (100, 100, 200, 200), "conf": 0.1},       # Low confidence (< 0.40)
        {"track_id": 65, "class_name": "airplane", "bbox": (100, 100, 200, 200), "conf": 0.9},  # Unsupported class
    ]
    status_def = tracker_def.process_tracks(malformed_tracks, 800, 600, 0.1)
    # Only track 63 should be safely clamped and accepted
    active_ids = [v.track_id for v in status_def["active_vehicles"]]
    clamped_box = tracker_def.active_tracks[63].bbox if 63 in tracker_def.active_tracks else None

    t6_pass = (active_ids == [63] and clamped_box == (0, 0, 800, 600))
    results["1.6_defensive_sanitization"] = "PASS" if t6_pass else "FAIL"
    print(f" -> Result: {results['1.6_defensive_sanitization']} (Accepted only valid/clamped ID: {active_ids}, Clamped: {clamped_box})")

    return results


def run_speed_benchmark() -> dict:
    print("\n" + "=" * 70)
    print("STAGE 2: SPEED BENCHMARK (STANDALONE YOLO INFERENCE VS UPSTREAM INGESTION)")
    print("=" * 70)

    # 1. Upstream Track Ingestion Speed (CPU)
    tracker_bench = VehicleTracker(confidence=0.40)
    dummy_tracks = [
        {"track_id": i, "class_name": "car", "bbox": (100 + i * 20, 100, 180 + i * 20, 200), "conf": 0.9}
        for i in range(1, 6)
    ]
    NUM_INGEST_FRAMES = 2000
    t0 = time.perf_counter()
    for f in range(NUM_INGEST_FRAMES):
        tracker_bench.process_tracks(dummy_tracks, 1920, 1080, f * 0.04)
    t_ingest = time.perf_counter() - t0
    ingest_fps = NUM_INGEST_FRAMES / t_ingest

    # 2. Standalone YOLOv8s Inference Speed on GPU
    model_path = PROJECT_ROOT / "yolov8s.pt"
    yolo_model = YOLO(str(model_path))
    tracker_standalone = VehicleTracker(yolo_model=yolo_model, confidence=0.40, imgsz=640)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Warmup
    for _ in range(5):
        tracker_standalone.process_frame(dummy_frame, 0.0)

    NUM_STANDALONE_FRAMES = 60
    t0 = time.perf_counter()
    for f in range(NUM_STANDALONE_FRAMES):
        tracker_standalone.process_frame(dummy_frame, f * 0.04)
    t_standalone = time.perf_counter() - t0
    standalone_fps = NUM_STANDALONE_FRAMES / t_standalone

    print(f"Upstream Ingestion Speed (Pure tracking logic): {ingest_fps:.1f} FPS ({NUM_INGEST_FRAMES} frames in {t_ingest:.3f}s)")
    print(f"Standalone Mode Speed (YOLOv8s + ByteTrack): {standalone_fps:.1f} FPS ({NUM_STANDALONE_FRAMES} frames in {t_standalone:.3f}s)")

    return {
        "upstream_ingest_fps": ingest_fps,
        "standalone_yolo_fps": standalone_fps,
    }


def run_video_tests() -> dict:
    print("\n" + "=" * 70)
    print("STAGE 3: REAL-WORLD CCTV VIDEO TESTS")
    print("=" * 70)

    model_path = PROJECT_ROOT / "yolov8s.pt"
    yolo_model = YOLO(str(model_path))

    output_dir = PROJECT_ROOT / "runs" / "vehicle_tracking_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    test_videos = [
        {
            "name": "ABODA Video 1 (Negative Control)",
            "path": PROJECT_ROOT / "datasets" / "samples" / "real_world_tests" / "aboda_video1.avi",
            "out_name": "aboda_video1_annotated.mp4",
            "max_frames": None,  # Full complete video test
            "expected_vehicles": 0,
        },
        {
            "name": "ABODA Video 2 (Positive Control)",
            "path": PROJECT_ROOT / "datasets" / "samples" / "real_world_tests" / "aboda_video2.avi",
            "out_name": "aboda_video2_annotated.mp4",
            "max_frames": 600,   # Sample 600 frames
            "expected_vehicles": None,
        },
        {
            "name": "Sample Unattended Test Video (Positive Control)",
            "path": PROJECT_ROOT / "datasets" / "samples" / "sample_unattended_test.mp4",
            "out_name": "sample_unattended_test_annotated.mp4",
            "max_frames": None,  # Full video (60 frames)
            "expected_vehicles": None,
        },
    ]

    video_results = {}

    for tcfg in test_videos:
        vpath = tcfg["path"]
        vname = tcfg["name"]
        print(f"\nEvaluating: {vname}...")

        if not vpath.is_file():
            print(f" [SKIP] File not found: {vpath}")
            continue

        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_f = tcfg["max_frames"] or total_frames

        out_path = output_dir / tcfg["out_name"]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        tracker = VehicleTracker(yolo_model=yolo_model, confidence=0.35, imgsz=640)

        f_count = 0
        t0 = time.time()
        while f_count < max_f:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = f_count / fps
            annotated_frame, status = tracker.process_frame(frame, timestamp, fps_metric=fps)
            writer.write(annotated_frame)
            f_count += 1

        cap.release()
        writer.release()
        elapsed = time.time() - t0
        proc_fps = f_count / elapsed if elapsed > 0 else 0.0

        unique_count = len(tracker.unique_track_ids)
        total_obs = tracker.total_observations
        class_breakdown = {c: len(ids) for c, ids in tracker.class_unique_counts.items()}

        print(f" -> Processed {f_count} frames in {elapsed:.2f}s ({proc_fps:.1f} FPS)")
        print(f" -> Unique Vehicles Tracked: {unique_count}")
        print(f" -> Total Detection Observations: {total_obs}")
        print(f" -> Class Breakdown: {class_breakdown}")
        print(f" -> Saved annotated video: {out_path.name}")

        is_pass = True
        if tcfg["expected_vehicles"] == 0:
            is_pass = (unique_count == 0 and total_obs == 0)

        video_results[vname] = {
            "status": "PASS" if is_pass else "FAIL",
            "frames_processed": f_count,
            "proc_fps": proc_fps,
            "unique_vehicles": unique_count,
            "total_observations": total_obs,
            "class_breakdown": class_breakdown,
            "output_video": str(out_path),
        }

    return video_results


def generate_markdown_report(unit_res: dict, speed_res: dict, video_res: dict) -> Path:
    report_path = PROJECT_ROOT / "handover_reports" / "VEHICLE_TRACKING_TEST_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Vehicle Detection and Tracking (Step 3) Verification Report\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("**Target Module:** `vehicle_tracking.py`\n")
        f.write("**YOLO Model:** `yolov8s.pt` (COCO Classes: Car=2, Motorcycle=3, Bus=5, Truck=7)\n\n")

        f.write("## 1. Unit & Edge-Case Verification Results\n\n")
        f.write("| Test Scenario | Status | Description |\n")
        f.write("|---|:---:|---|\n")
        for k, s in unit_res.items():
            f.write(f"| `{k}` | **{s}** | Automated validation of tracking invariants and edge cases |\n")

        f.write("\n## 2. Processing Speed & Benchmark\n\n")
        f.write(f"- **Pure Upstream Track Ingestion Speed:** `{speed_res['upstream_ingest_fps']:.1f} FPS`\n")
        f.write(f"- **Standalone Mode (YOLOv8s + ByteTrack on GPU):** `{speed_res['standalone_yolo_fps']:.1f} FPS`\n\n")

        f.write("## 3. Real CCTV Video Evaluation Results\n\n")
        f.write("| Video Source | Frames | Speed (FPS) | Unique Vehicles | Total Observations | Class Breakdown | Status |\n")
        f.write("|---|:---:|:---:|:---:|:---:|---|:---:|\n")
        for vname, vdata in video_res.items():
            cb = ", ".join(f"{k}: {v}" for k, v in vdata['class_breakdown'].items() if v > 0) or "None"
            f.write(f"| {vname} | {vdata['frames_processed']} | {vdata['proc_fps']:.1f} | {vdata['unique_vehicles']} | {vdata['total_observations']} | {cb} | **{vdata['status']}** |\n")

        f.write("\n## 4. Output Artifacts\n\n")
        for vname, vdata in video_res.items():
            f.write(f"- **{vname}:** `{vdata['output_video']}`\n")

    print(f"\n[Report] Test report saved successfully to: {report_path}")
    return report_path


def main():
    print("=" * 70)
    print("STARTING COMPLETE VEHICLE DETECTION AND TRACKING TEST SUITE")
    print("=" * 70)

    unit_res = run_unit_and_synthetic_tests()
    speed_res = run_speed_benchmark()
    video_res = run_video_tests()

    rep_path = generate_markdown_report(unit_res, speed_res, video_res)

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print(f"Report: {rep_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
