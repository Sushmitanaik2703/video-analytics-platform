"""
Unit and Integration Tests for Integrated Multi-Feature AI Pipeline & GUI Integration.
Tests independent and combined execution of:
1. Unattended Bag Detection (automatic, zero zones)
2. Restricted-Zone Intrusion Detection
3. Vehicle Detection & Tracking
4. All Three Features Combined
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrated_pipeline import IntegratedPipeline, process_integrated_stream
from intrusion_detection import RestrictedZone


class TestIntegratedPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = PROJECT_ROOT / "yolov8n.pt"
        if not cls.model_path.is_file():
            cls.model_path = PROJECT_ROOT / "yolov8s.pt"
        cls.dummy_frame = np.zeros((480, 720, 3), dtype=np.uint8)

    def test_01_unattended_bag_detection_only_no_zones(self):
        """Unattended bag detection runs automatically across full frame without requiring zones."""
        pipeline = IntegratedPipeline(
            yolo_model_path=self.model_path,
            enable_intrusion=False,
            enable_vehicle_tracking=False,
            enable_unattended_bag=True,
            zones=[],
            stationary_time=2.0,
            person_distance=150.0,
        )
        self.assertFalse(pipeline.enable_intrusion)
        self.assertFalse(pipeline.enable_vehicle_tracking)
        self.assertTrue(pipeline.enable_unattended_bag)
        self.assertIsNone(pipeline.intrusion_detector)
        self.assertIsNotNone(pipeline.yolo_bag_model)

        annotated, status = pipeline.process_frame(self.dummy_frame, 0.0)
        self.assertEqual(annotated.shape, self.dummy_frame.shape)
        self.assertIn("unattended_bags", status)
        self.assertEqual(status["banner_level"], "NORMAL")

    def test_02_restricted_zone_intrusion_only(self):
        """Restricted zone intrusion runs with defined polygon zones."""
        zone = RestrictedZone("z1", "Test Zone", [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)], is_normalized=True)
        pipeline = IntegratedPipeline(
            yolo_model_path=self.model_path,
            enable_intrusion=True,
            enable_vehicle_tracking=False,
            enable_unattended_bag=False,
            zones=[zone],
        )
        self.assertTrue(pipeline.enable_intrusion)
        self.assertFalse(pipeline.enable_vehicle_tracking)
        self.assertFalse(pipeline.enable_unattended_bag)
        self.assertIsNotNone(pipeline.intrusion_detector)
        self.assertIsNone(pipeline.yolo_bag_model)

        annotated, status = pipeline.process_frame(self.dummy_frame, 0.0)
        self.assertEqual(annotated.shape, self.dummy_frame.shape)
        self.assertIn("confirmed_intruders", status)

    def test_03_vehicle_tracking_only(self):
        """Vehicle tracking runs standalone without intrusion polygons or bag models."""
        pipeline = IntegratedPipeline(
            yolo_model_path=self.model_path,
            enable_intrusion=False,
            enable_vehicle_tracking=True,
            enable_unattended_bag=False,
        )
        self.assertFalse(pipeline.enable_intrusion)
        self.assertTrue(pipeline.enable_vehicle_tracking)
        self.assertFalse(pipeline.enable_unattended_bag)
        self.assertIsNotNone(pipeline.vehicle_tracker)
        self.assertIsNone(pipeline.intrusion_detector)
        self.assertIsNone(pipeline.yolo_bag_model)

        annotated, status = pipeline.process_frame(self.dummy_frame, 0.0)
        self.assertEqual(annotated.shape, self.dummy_frame.shape)
        self.assertIn("active_vehicles", status)

    def test_04_all_three_features_combined(self):
        """All three features run together in a unified pipeline."""
        zone = RestrictedZone("z1", "Test Zone", [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)], is_normalized=True)
        pipeline = IntegratedPipeline(
            yolo_model_path=self.model_path,
            enable_intrusion=True,
            enable_vehicle_tracking=True,
            enable_unattended_bag=True,
            zones=[zone],
            stationary_time=2.0,
            person_distance=150.0,
        )
        self.assertTrue(pipeline.enable_intrusion)
        self.assertTrue(pipeline.enable_vehicle_tracking)
        self.assertTrue(pipeline.enable_unattended_bag)

        annotated, status = pipeline.process_frame(self.dummy_frame, 0.0)
        self.assertEqual(annotated.shape, self.dummy_frame.shape)
        self.assertIn("confirmed_intruders", status)
        self.assertIn("unattended_bags", status)
        self.assertIn("banner_text", status)

        summary = pipeline.get_summary()
        self.assertIn("total_incidents", summary)
        self.assertIn("person_incidents", summary)
        self.assertIn("vehicle_incidents", summary)
        self.assertIn("unattended_bag_incidents", summary)


if __name__ == "__main__":
    unittest.main()
