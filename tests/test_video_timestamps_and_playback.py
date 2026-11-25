from __future__ import annotations

import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from queue import Queue
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from cctv_app import SmartCCTVApp
from integrated_pipeline import IntegratedPipeline, process_integrated_stream
from intrusion_detection import (
    IntrusionDetector,
    RestrictedZone,
    format_video_timestamp,
    process_video_stream,
)


class _PlaybackPipeline:
    enable_intrusion = False
    enable_vehicle_tracking = False
    enable_unattended_bag = False

    def __init__(self, frame_delay: float = 0.0):
        self.frame_delay = frame_delay
        self.timestamps: list[float] = []
        self.incident_history: list[dict] = []

    def process_frame(self, frame, timestamp):
        if self.frame_delay:
            time.sleep(self.frame_delay)
        self.timestamps.append(timestamp)
        return frame, {"all_active_alerts": 0}

    def get_incident_history(self):
        return list(self.incident_history)

    def get_summary(self):
        return {
            "total_incidents": 0,
            "person_incidents": 0,
            "vehicle_incidents": 0,
            "unattended_bag_incidents": 0,
            "incident_history": [],
        }


class _StandaloneDetector:
    def __init__(self, frame_delay: float = 0.0):
        self.frame_delay = frame_delay
        self.timestamps: list[float] = []

    def process_frame(self, frame, timestamp):
        if self.frame_delay:
            time.sleep(self.frame_delay)
        self.timestamps.append(timestamp)
        return frame, {
            "notifications": [],
            "confirmed_intruders": [],
            "time_str": "",
        }

    def get_incident_history(self):
        return []


def _create_sample_video(path: Path, frame_count: int = 30, fps: float = 30.0):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (64, 48),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create the sample video for playback tests")
    for frame_index in range(frame_count):
        frame = np.full((48, 64, 3), frame_index % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestVideoTimestampsAndPlayback(unittest.TestCase):
    def test_video_timestamp_format_and_vehicle_incident_duration(self):
        self.assertEqual(format_video_timestamp(83.984), "00:01:24")
        self.assertEqual(format_video_timestamp(84.9), "00:01:25")
        self.assertEqual(format_video_timestamp(3661.2), "01:01:01")

        zone = RestrictedZone(
            zone_id="restricted",
            name="Restricted Zone",
            raw_points=[[10, 10], [90, 10], [90, 90], [10, 90]],
            classes={"car"},
            is_normalized=False,
        )
        detector = IntrusionDetector(zones=[zone], confirm_frames=1)
        inside = [{"track_id": 12, "class_name": "car", "bbox": (40, 30, 60, 55)}]
        outside = [{"track_id": 12, "class_name": "car", "bbox": (0, 0, 10, 10)}]

        detector.process_tracks(inside, 100, 100, 83.984)
        detector.process_tracks(outside, 100, 100, 89.6)

        incident = detector.incident_history[0]
        self.assertEqual(incident["entry_time"], "00:01:24")
        self.assertEqual(incident["exit_time"], "00:01:30")
        self.assertEqual(incident["duration_seconds"], 6)

    def test_video_frame_at_01_24_records_vehicle_entry_at_00_01_24(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_path = Path(temp_dir) / "sample_85_seconds_30fps.avi"
            _create_sample_video(sample_path, frame_count=2550)
            pipeline = _PlaybackPipeline()
            with redirect_stdout(StringIO()):
                result = process_integrated_stream(sample_path, pipeline)

        self.assertEqual(result["total_frames"], 2550)
        video_timestamp = pipeline.timestamps[2520]
        self.assertAlmostEqual(video_timestamp, 84.0, delta=0.03)

        zone = RestrictedZone(
            zone_id="restricted",
            name="Restricted Zone",
            raw_points=[[10, 10], [90, 10], [90, 90], [10, 90]],
            classes={"car"},
            is_normalized=False,
        )
        detector = IntrusionDetector(zones=[zone], confirm_frames=1)
        detector.process_tracks(
            [{"track_id": 21, "class_name": "car", "bbox": (40, 30, 60, 55)}],
            100,
            100,
            video_timestamp,
        )

        self.assertEqual(detector.incident_history[0]["entry_time"], "00:01:24")

    def test_person_intrusion_uses_elapsed_video_timestamps(self):
        zone = RestrictedZone(
            zone_id="restricted",
            name="Restricted Zone",
            raw_points=[[10, 10], [90, 10], [90, 90], [10, 90]],
            classes={"person"},
            is_normalized=False,
        )
        detector = IntrusionDetector(zones=[zone], confirm_frames=1)
        inside = [{"track_id": 8, "class_name": "person", "bbox": (40, 30, 60, 55)}]
        outside = [{"track_id": 8, "class_name": "person", "bbox": (0, 0, 10, 10)}]

        detector.process_tracks(inside, 100, 100, 84.0)
        detector.process_tracks(outside, 100, 100, 87.0)

        incident = detector.incident_history[0]
        self.assertEqual(incident["entry_time"], "00:01:24")
        self.assertEqual(incident["exit_time"], "00:01:27")
        self.assertEqual(incident["duration_seconds"], 3)

    def test_unattended_bag_uses_elapsed_video_timestamps(self):
        bag = SimpleNamespace(
            track_id=7,
            class_name="backpack",
            alert_active=True,
            stationary_since=70.0,
            associated_person_id=None,
        )
        pipeline = SimpleNamespace(
            enable_intrusion=False,
            enable_vehicle_tracking=False,
            enable_unattended_bag=True,
            yolo_bag_model=object(),
            unattended_bags={7: bag},
            bag_cfg=SimpleNamespace(),
            _logged_bag_alert_ids=set(),
            incident_history=[],
            event_log=[],
            camera_name="Test Camera",
        )
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        with patch(
            "integrated_pipeline.uo.process_frame",
            side_effect=lambda **kwargs: kwargs["frame"],
        ):
            IntegratedPipeline.process_frame(pipeline, frame, 84.2)
            bag.alert_active = False
            IntegratedPipeline.process_frame(pipeline, frame, 90.2)

        incident = pipeline.incident_history[0]
        self.assertEqual(incident["entry_time"], "00:01:24")
        self.assertEqual(incident["exit_time"], "00:01:30")
        self.assertEqual(incident["duration_seconds"], 6)

    def test_live_camera_keeps_wall_clock_time(self):
        pipeline = SimpleNamespace(
            enable_intrusion=False,
            intrusion_detector=None,
            enable_vehicle_tracking=False,
            vehicle_tracker=None,
            enable_unattended_bag=False,
            yolo_bag_model=None,
        )
        frame = np.zeros((48, 64, 3), dtype=np.uint8)

        with patch("integrated_pipeline.datetime.datetime") as mocked_datetime:
            mocked_datetime.now.return_value.strftime.return_value = "12:34:56"
            _, video_status = IntegratedPipeline.process_frame(pipeline, frame, 84.0)
            _, live_status = IntegratedPipeline.process_frame(
                pipeline,
                frame,
                84.0,
                live_camera=True,
            )

        self.assertEqual(video_status["time_str"], "00:01:24")
        self.assertEqual(live_status["time_str"], "12:34:56")

    def test_sample_video_plays_at_source_fps_without_dropping_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_path = Path(temp_dir) / "sample_30fps.avi"
            _create_sample_video(sample_path)

            capture = cv2.VideoCapture(str(sample_path))
            self.assertTrue(capture.isOpened())
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 30.0, delta=0.1)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 30)
            capture.release()

            pipeline = _PlaybackPipeline()
            callback_frames: list[tuple[int, float]] = []
            with redirect_stdout(StringIO()):
                result = process_integrated_stream(
                    sample_path,
                    pipeline,
                    progress_callback=lambda data: callback_frames.append(
                        (data["frame_idx"], time.perf_counter())
                    ),
                    realtime_playback=True,
                )

        self.assertEqual(result["total_frames"], 30)
        self.assertEqual([frame for frame, _ in callback_frames], list(range(30)))
        self.assertAlmostEqual(pipeline.timestamps[0], 0.0, delta=0.01)
        self.assertAlmostEqual(pipeline.timestamps[-1], 29 / 30, delta=0.03)
        playback_fps = 29 / (
            callback_frames[-1][1] - callback_frames[0][1]
        )
        self.assertAlmostEqual(playback_fps, 30.0, delta=8.0)
        self.assertAlmostEqual(result["elapsed_seconds"], 1.0, delta=0.5)

    def test_non_30_fps_video_uses_its_reported_frame_rate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_path = Path(temp_dir) / "sample_24fps.avi"
            _create_sample_video(sample_path, frame_count=24, fps=24.0)
            pipeline = _PlaybackPipeline()
            callback_times: list[float] = []
            with redirect_stdout(StringIO()):
                result = process_integrated_stream(
                    sample_path,
                    pipeline,
                    progress_callback=lambda _data: callback_times.append(
                        time.perf_counter()
                    ),
                    realtime_playback=True,
                )

        measured_fps = 23 / (callback_times[-1] - callback_times[0])
        self.assertEqual(result["total_frames"], 24)
        self.assertAlmostEqual(measured_fps, 24.0, delta=7.0)
        self.assertAlmostEqual(result["elapsed_seconds"], 1.0, delta=0.5)

    def test_slow_processing_still_processes_every_source_frame(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_path = Path(temp_dir) / "sample_slow_processing.avi"
            _create_sample_video(sample_path, frame_count=10)
            pipeline = _PlaybackPipeline(frame_delay=0.04)
            with redirect_stdout(StringIO()):
                result = process_integrated_stream(
                    sample_path,
                    pipeline,
                    realtime_playback=True,
                )

        self.assertEqual(result["total_frames"], 10)
        self.assertEqual(len(pipeline.timestamps), 10)
        self.assertEqual(
            pipeline.timestamps,
            sorted(pipeline.timestamps),
        )
        self.assertGreater(result["elapsed_seconds"], 10 / 30)

    def test_standalone_intrusion_stream_does_not_skip_slow_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_path = Path(temp_dir) / "sample_standalone_slow.avi"
            _create_sample_video(sample_path, frame_count=10)
            detector = _StandaloneDetector(frame_delay=0.04)
            with redirect_stdout(StringIO()):
                result = process_video_stream(
                    sample_path,
                    detector,
                    realtime=True,
                )

        self.assertEqual(result["total_frames"], 10)
        self.assertEqual(len(detector.timestamps), 10)
        self.assertGreater(result["elapsed_seconds"], 10 / 30)

    def test_stop_during_pacing_counts_the_frame_already_processed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_path = Path(temp_dir) / "sample_stop.avi"
            _create_sample_video(sample_path)
            pipeline = _PlaybackPipeline()
            stop_event = threading.Event()
            with redirect_stdout(StringIO()):
                result = process_integrated_stream(
                    sample_path,
                    pipeline,
                    progress_callback=lambda _data: stop_event.set(),
                    stop_event=stop_event,
                    realtime_playback=True,
                )

        self.assertEqual(result["total_frames"], 1)
        self.assertEqual(len(pipeline.timestamps), 1)

    def test_ui_consumes_queued_frames_without_coalescing(self):
        app = SimpleNamespace(
            ui_queue=Queue(maxsize=2),
            is_processing=True,
            fps=30.0,
            root=SimpleNamespace(after=Mock()),
            _update_live_ui=Mock(),
            _poll_ui_queue=Mock(),
        )
        first_frame = {
            "frame_idx": 0,
            "total_frames": 2,
            "fps": 30.0,
            "timestamp": 0.0,
            "status": {},
            "annotated_frame": np.zeros((1, 1, 3), dtype=np.uint8),
            "incident_history": [],
        }
        second_frame = {**first_frame, "frame_idx": 1, "timestamp": 1 / 30}
        app.ui_queue.put(("FRAME", first_frame))
        app.ui_queue.put(("FRAME", second_frame))

        SmartCCTVApp._poll_ui_queue(app)

        app._update_live_ui.assert_called_once_with(
            0,
            2,
            30.0,
            0.0,
            {},
            first_frame["annotated_frame"],
            [],
        )
        self.assertEqual(app.ui_queue.qsize(), 1)


if __name__ == "__main__":
    unittest.main()
