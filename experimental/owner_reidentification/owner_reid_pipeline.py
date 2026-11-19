"""
Hybrid Original Owner Re-Identification and Return Detection Pipeline (Phase 1).

Maintains compatibility with Member 2 baseline parameters and logic, while adding:
1. Owner appearance gallery accumulation during ownership association.
2. OSNet re-identification matching upon person proximity.
3. Multi-frame persistence verification (>= 5 frames) before declaring OWNER RETURNED.
4. Bag-person interaction verification (bbox overlap + displacement/pickup) to transition to
   ORIGINAL OWNER RETRIEVED BAG and resolve alert.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .reid_extractor import OSNetExtractor

# Inherit baseline constants
PERSON_CLASS_ID = 0
BAG_CLASS_IDS = {24, 26, 28}
TRACK_EXPIRY_SECONDS = 2.0
MAX_CLASS_HISTORY = 30

# Re-ID specific thresholds
REID_SIMILARITY_THRESHOLD = 0.65
CONFIRMATION_FRAMES = 5
MAX_GALLERY_SIZE = 10
INTERACTION_IOU_THRESHOLD = 0.05
INTERACTION_PROXIMITY_THRESHOLD = 80.0

# Safe carried-to-deposited handoff thresholds
HANDOFF_MAX_TIME_DELTA = 1.5
HANDOFF_MAX_DISTANCE = 80.0
HANDOFF_ALERTING_MAX_DISTANCE = 160.0
HANDOFF_ABSENT_TRIGGER = 0.05
MIN_CARRIED_FRAMES = 3

# Dormant luggage recovery thresholds
DORMANT_MAX_TIME_DELTA = 60.0
DORMANT_MAX_DISTANCE = 50.0
DORMANT_MIN_IOU = 0.25


@dataclass
class TrackedPerson:
    track_id: int
    position: tuple[float, float]
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray | None = None


@dataclass
class OwnerGallery:
    owner_track_id: int
    embeddings: list[np.ndarray] = field(default_factory=list)
    last_updated_time: float = 0.0

    def add_embedding(self, emb: np.ndarray, max_size: int = MAX_GALLERY_SIZE):
        if len(self.embeddings) < max_size:
            self.embeddings.append(emb)
        else:
            # Replace embedding that is most similar to keep diversity
            sims = [float(np.dot(emb, e)) for e in self.embeddings]
            max_idx = int(np.argmax(sims))
            if sims[max_idx] < 0.98:  # Only update if sufficiently distinct
                self.embeddings[max_idx] = emb

    def match_similarity(self, candidate_emb: np.ndarray) -> float:
        if not self.embeddings or candidate_emb is None:
            return 0.0
        sims = [float(np.dot(candidate_emb, e)) for e in self.embeddings]
        # Max cosine similarity across gallery
        return max(sims)


@dataclass
class DormantLuggageRecord:
    original_track_id: int
    class_name: str
    class_history: list[str]
    last_bbox: tuple[int, int, int, int]
    last_position: tuple[float, float]
    stationary_anchor: tuple[float, float]
    stationary_anchor_bbox: tuple[int, int, int, int]
    stationary_since: float
    associated_person_id: int
    owner_gallery: OwnerGallery
    alert_active: bool
    last_alert_at: float | None
    bag_retrieved: bool
    became_dormant_at: float
    bag_removed_unverified: bool = False
    last_owner_seen_at: float | None = None


@dataclass
class TrackedBag:
    track_id: int
    class_name: str
    bbox: tuple[int, int, int, int]
    position: tuple[float, float]
    first_detected_at: datetime
    last_detected_at: datetime
    stationary_since: float
    stationary_anchor: tuple[float, float]
    associated_person_id: int | None = None
    owner_gallery: OwnerGallery | None = None
    last_alert_at: float | None = None
    alert_active: bool = False
    last_seen_time: float = 0.0
    class_history: list[str] = field(default_factory=list)
    anchor_id: int = 0
    # Phase 1 Return & Retrieval states
    owner_returned: bool = False
    returned_person_id: int | None = None
    owner_match_counter: int = 0
    bag_retrieved: bool = False
    retrieval_confirmed_at: float | None = None
    initial_pickup_pos: tuple[float, float] | None = None
    is_static_background: bool = False
    nearby_person_frames: int = 0
    # Safe carried-to-deposited handoff state
    transferred_handoff: bool = False
    handoff_candidate_created: bool = False
    last_known_bbox: tuple[int, int, int, int] | None = None
    # Jitter-resistant stationary anchor footprint
    stationary_anchor_bbox: tuple[int, int, int, int] | None = None
    displacement_frames: int = 0
    # Two-State Physical Removal & Owner Retrieval
    bag_removed_unverified: bool = False
    unverified_removal_frames: int = 0
    last_owner_seen_at: float | None = None


def compute_iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    boxBArea = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return float(iou)


def box_details(box: object) -> tuple[int, tuple[int, int, int, int], tuple[float, float]]:
    coordinates = box.xyxy[0].cpu().tolist()
    bbox = tuple(int(value) for value in coordinates)
    center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
    return int(box.cls[0].item()), bbox, center


def get_model_class_ids(model: YOLO) -> tuple[int, set[int], list[int]]:
    names = model.names
    person_id = 0
    bag_ids = set()
    for cid, cname in names.items():
        if cname == "person":
            person_id = int(cid)
        elif cname in {"backpack", "handbag", "suitcase"}:
            bag_ids.add(int(cid))
    if not bag_ids:
        person_id = PERSON_CLASS_ID
        bag_ids = BAG_CLASS_IDS
    tracked_ids = [person_id, *sorted(bag_ids)]
    return person_id, bag_ids, tracked_ids


def draw_box(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


class HybridOwnerReIDPipeline:
    def __init__(
        self,
        yolo_model: YOLO,
        reid_extractor: OSNetExtractor | None = None,
        confidence: float = 0.50,
        bag_confidence: float = 0.40,
        stationary_time: float = 15.0,
        person_distance: float = 150.0,
        stationary_distance: float = 25.0,
        alert_cooldown: float = 30.0,
        imgsz: int = 640,
        reid_threshold: float = REID_SIMILARITY_THRESHOLD,
        confirm_frames: int = CONFIRMATION_FRAMES,
    ):
        self.model = yolo_model
        self.reid = reid_extractor or OSNetExtractor()
        self.confidence = confidence
        self.bag_confidence = bag_confidence
        self.stationary_time = stationary_time
        self.person_distance = person_distance
        self.stationary_distance = stationary_distance
        self.alert_cooldown = alert_cooldown
        self.imgsz = imgsz
        self.reid_threshold = reid_threshold
        self.confirm_frames = confirm_frames

        self.bags: dict[int, TrackedBag] = {}
        self.person_cls_id, self.bag_cls_ids, self.tracked_cls_ids = get_model_class_ids(
            self.model
        )
        self.background_anchors: set[tuple[int, int]] = set()
        self.baseline_frames: int = 0
        self.initial_bag_locations: dict[tuple[int, int], int] = {}
        # Safe carried-to-deposited handoff pool
        self.recent_departed_bags: list[dict] = []
        self.handoff_events: list[dict] = []
        # Dormant luggage pool for long-term occlusion recovery
        self.dormant_luggage: list[DormantLuggageRecord] = []
        self.dormant_recovery_events: list[dict] = []

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
    ) -> tuple[np.ndarray, dict]:
        """
        Process a single frame:
        1. YOLO inference + ByteTrack.
        2. Extract person embeddings via OSNet.
        3. Associate owner and update appearance gallery.
        4. Detect unattended bags.
        5. Verify returning owner via Re-ID matching.
        6. Verify interaction/retrieval when owner reaches bag.
        7. Render bounding boxes and alert banners.
        """
        inference_confidence = min(self.confidence, self.bag_confidence)
        results = self.model.track(
            frame,
            conf=inference_confidence,
            classes=self.tracked_cls_ids,
            imgsz=self.imgsz,
            device=0 if torch.cuda.is_available() else "cpu",
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes

        people: dict[int, TrackedPerson] = {}
        detected_bags: list[
            tuple[int, str, tuple[int, int, int, int], tuple[float, float]]
        ] = []

        track_ids = (
            boxes.id.int().cpu().tolist()
            if boxes is not None and boxes.id is not None
            else []
        )

        if boxes is not None:
            for index, box in enumerate(boxes):
                class_id, bbox, position = box_details(box)
                conf = float(box.conf[0].item())
                class_name = str(result.names[class_id])
                track_id = int(track_ids[index]) if index < len(track_ids) else None

                if class_id == self.person_cls_id and conf >= self.confidence:
                    if track_id is not None:
                        # Extract embedding using OSNet
                        emb = self.reid.extract_crop(frame, bbox)
                        people[track_id] = TrackedPerson(track_id, position, bbox, emb)
                elif class_id in self.bag_cls_ids and conf >= self.bag_confidence:
                    if track_id is not None:
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        b_area = bw * bh
                        b_aspect = bw / max(1.0, bh)
                        # Geometry filter: reject extreme thin floor seams / door strips and excessive patches
                        if 0.35 <= b_aspect <= 2.80 and 300 <= b_area <= 25000:
                            detected_bags.append((track_id, class_name, bbox, position))

        # Static Environmental Artifact Learning (First 2.0 seconds):
        # Learn persistent static patches present at the start of video with no moving person deposited.
        if timestamp <= 2.0:
            self.baseline_frames += 1
            for _, _, _, pos in detected_bags:
                grid_key = (int(round(pos[0] / 30.0) * 30), int(round(pos[1] / 30.0) * 30))
                self.initial_bag_locations[grid_key] = self.initial_bag_locations.get(grid_key, 0) + 1
                if self.initial_bag_locations[grid_key] >= 12:
                    self.background_anchors.add(grid_key)

        # 0. Cancel pending handoff snapshots for tracks that reappear in this frame
        current_detected_bag_ids = {t_id for t_id, _, _, _ in detected_bags}
        self.recent_departed_bags = [
            snap for snap in self.recent_departed_bags
            if snap["track_id"] not in current_detected_bag_ids
            and (timestamp - snap["departed_at"]) <= HANDOFF_MAX_TIME_DELTA
        ]

        # Register un-updated / departed bag candidates before new track ingestion
        # so that when a track is replaced in the same frame, the handoff snapshot is available
        for track_id, b_cand in self.bags.items():
            if (
                track_id not in current_detected_bag_ids
                and not b_cand.handoff_candidate_created
                and not b_cand.transferred_handoff
                and b_cand.associated_person_id is not None
                and b_cand.owner_gallery is not None
                and len(b_cand.owner_gallery.embeddings) > 0
                and (b_cand.nearby_person_frames >= MIN_CARRIED_FRAMES or b_cand.alert_active)
                and not b_cand.is_static_background
                and not b_cand.bag_retrieved
            ):
                b_cand.handoff_candidate_created = True
                snap = {
                    "track_id": track_id,
                    "class_name": b_cand.class_name,
                    "bbox": b_cand.bbox,
                    "position": b_cand.position,
                    "associated_person_id": b_cand.associated_person_id,
                    "owner_gallery": b_cand.owner_gallery,
                    "departed_at": b_cand.last_seen_time,
                    "stationary_since": b_cand.stationary_since,
                    "stationary_anchor": b_cand.stationary_anchor,
                    "stationary_anchor_bbox": b_cand.stationary_anchor_bbox or b_cand.bbox,
                    "alert_active": b_cand.alert_active,
                    "last_alert_at": b_cand.last_alert_at,
                    "bag_removed_unverified": b_cand.bag_removed_unverified,
                    "owner_returned": b_cand.owner_returned,
                    "returned_person_id": b_cand.returned_person_id,
                    "last_owner_seen_at": b_cand.last_owner_seen_at,
                    "nearby_person_frames": b_cand.nearby_person_frames,
                }
                self.recent_departed_bags.append(snap)

        # Ingestion & Direct ByteTrack Association:
        # Match each detected bag directly by track_id.
        for track_id, class_name, bbox, position in detected_bags:
            pos_key = (int(round(position[0] / 30.0) * 30), int(round(position[1] / 30.0) * 30))
            is_env_artifact = pos_key in self.background_anchors

            bag = self.bags.get(track_id)
            if bag is None:
                # 1. Check Dormant Luggage Recovery (for stationary/alerting bags recovered after occlusion):
                # Look for an eligible dormant record matching class, distance <= 50px, IoU >= 0.25, age <= 60s
                dormant_match_idx = None
                best_dormant_dist = float("inf")

                for d_idx, d_rec in enumerate(self.dormant_luggage):
                    age = timestamp - d_rec.became_dormant_at
                    if age > DORMANT_MAX_TIME_DELTA:
                        continue
                    if class_name != d_rec.class_name and (class_name not in {"backpack", "handbag", "suitcase"} or d_rec.class_name not in {"backpack", "handbag", "suitcase"}):
                        continue
                    c_dist = (
                        (position[0] - d_rec.last_position[0]) ** 2
                        + (position[1] - d_rec.last_position[1]) ** 2
                    ) ** 0.5
                    b_iou = compute_iou(bbox, d_rec.last_bbox)

                    if c_dist <= DORMANT_MAX_DISTANCE and b_iou >= DORMANT_MIN_IOU and c_dist < best_dormant_dist:
                        best_dormant_dist = c_dist
                        dormant_match_idx = d_idx

                if dormant_match_idx is not None:
                    recovered_rec = self.dormant_luggage.pop(dormant_match_idx)
                    recovery_record = {
                        "frame_timestamp": timestamp,
                        "old_track_id": recovered_rec.original_track_id,
                        "new_track_id": track_id,
                        "class_name": class_name,
                        "distance": round(best_dormant_dist, 2),
                        "age": round(timestamp - recovered_rec.became_dormant_at, 2),
                        "owner_id": recovered_rec.associated_person_id,
                        "gallery_size": len(recovered_rec.owner_gallery.embeddings),
                        "alert_active": recovered_rec.alert_active,
                    }
                    self.dormant_recovery_events.append(recovery_record)
                    print(
                        f"[DORMANT RECOVER] Time {timestamp:.2f}s: Bag {track_id} recovered dormant state "
                        f"from Track {recovered_rec.original_track_id} | Dist: {best_dormant_dist:.1f}px | "
                        f"Age: {timestamp - recovered_rec.became_dormant_at:.2f}s | "
                        f"Owner: {recovered_rec.associated_person_id} | Alert: {recovered_rec.alert_active}"
                    )

                    bag = TrackedBag(
                        track_id=track_id,
                        class_name=class_name,
                        bbox=bbox,
                        position=position,
                        first_detected_at=datetime.now(),
                        last_detected_at=datetime.now(),
                        stationary_since=recovered_rec.stationary_since,
                        stationary_anchor=recovered_rec.last_position,
                        last_seen_time=timestamp,
                        class_history=recovered_rec.class_history + [class_name],
                        anchor_id=track_id,
                        associated_person_id=recovered_rec.associated_person_id,
                        owner_gallery=recovered_rec.owner_gallery,
                        last_alert_at=recovered_rec.last_alert_at,
                        alert_active=recovered_rec.alert_active,
                        bag_retrieved=recovered_rec.bag_retrieved,
                        transferred_handoff=True,
                        handoff_candidate_created=False,
                        last_known_bbox=bbox,
                        stationary_anchor_bbox=recovered_rec.last_bbox,
                        displacement_frames=0,
                        bag_removed_unverified=recovered_rec.bag_removed_unverified,
                        last_owner_seen_at=recovered_rec.last_owner_seen_at,
                        nearby_person_frames=MIN_CARRIED_FRAMES,
                    )
                    self.bags[track_id] = bag

                else:
                    # 2. Active Co-Located Luggage Track Continuity:
                    # When a new luggage ByteTrack ID appears, check existing un-retrieved luggage tracks
                    # with an established owner and non-empty OSNet gallery that are absent from current frame.
                    active_match = None
                    best_active_dist = float("inf")
                    for other_id, other_b in list(self.bags.items()):
                        if (
                            other_id != track_id
                            and other_id not in current_detected_bag_ids
                            and not other_b.bag_retrieved
                            and other_b.associated_person_id is not None
                            and other_b.owner_gallery is not None
                            and len(other_b.owner_gallery.embeddings) > 0
                            and (other_b.nearby_person_frames >= MIN_CARRIED_FRAMES or other_b.alert_active)
                            and (timestamp - other_b.last_seen_time) <= TRACK_EXPIRY_SECONDS
                        ):
                            # Class compatibility check
                            if (
                                class_name not in {"backpack", "handbag", "suitcase"}
                                or other_b.class_name not in {"backpack", "handbag", "suitcase"}
                            ):
                                continue
                            dist = (
                                (position[0] - other_b.position[0]) ** 2
                                + (position[1] - other_b.position[1]) ** 2
                            ) ** 0.5
                            iou = compute_iou(bbox, other_b.bbox)
                            if (dist <= DORMANT_MAX_DISTANCE or iou >= DORMANT_MIN_IOU) and dist < best_active_dist:
                                best_active_dist = dist
                                active_match = other_b

                    if active_match is not None:
                        bag = TrackedBag(
                            track_id=track_id,
                            class_name=class_name,
                            bbox=bbox,
                            position=position,
                            first_detected_at=active_match.first_detected_at,
                            last_detected_at=datetime.now(),
                            stationary_since=active_match.stationary_since,
                            stationary_anchor=active_match.stationary_anchor,
                            last_seen_time=timestamp,
                            class_history=list(active_match.class_history),
                            anchor_id=track_id,
                            associated_person_id=active_match.associated_person_id,
                            owner_gallery=active_match.owner_gallery,
                            last_alert_at=active_match.last_alert_at,
                            alert_active=active_match.alert_active,
                            bag_retrieved=active_match.bag_retrieved,
                            transferred_handoff=True,
                            handoff_candidate_created=False,
                            last_known_bbox=bbox,
                            stationary_anchor_bbox=active_match.stationary_anchor_bbox or bbox,
                            displacement_frames=0,
                            bag_removed_unverified=active_match.bag_removed_unverified,
                            owner_returned=active_match.owner_returned,
                            returned_person_id=active_match.returned_person_id,
                            last_owner_seen_at=active_match.last_owner_seen_at,
                            nearby_person_frames=max(active_match.nearby_person_frames, MIN_CARRIED_FRAMES),
                        )
                        active_match.transferred_handoff = True
                        del self.bags[active_match.track_id]
                        self.bags[track_id] = bag
                        print(
                            f"[CO-LOCATED CONTINUITY] Time {timestamp:.2f}s: Active Bag {active_match.track_id} "
                            f"-> Replaced by Track {track_id} | Dist: {best_active_dist:.1f}px | "
                            f"Owner: {bag.associated_person_id} (Gallery: {len(bag.owner_gallery.embeddings)} embs)"
                        )

                    else:
                        # 3. Safe Carried-to-Deposited Track Handoff check (short window <= 1.5s):
                        # Look for an eligible departed carried bag snapshot to transfer ownership
                        inherited_owner_id = None
                        inherited_gallery = None
                        matched_snap = None
                        matched_snap_idx = None
                        best_distance = float("inf")

                        for s_idx, snap in enumerate(self.recent_departed_bags):
                            time_diff = timestamp - snap["departed_at"]
                            if time_diff > HANDOFF_MAX_TIME_DELTA:
                                continue
                            # Check class compatibility (both must be valid luggage classes)
                            if class_name not in {"backpack", "handbag", "suitcase"} or snap["class_name"] not in {"backpack", "handbag", "suitcase"}:
                                continue
                            # Check center distance
                            c_dist = (
                                (position[0] - snap["position"][0]) ** 2
                                + (position[1] - snap["position"][1]) ** 2
                            ) ** 0.5
                            # Check bbox IoU or containment
                            snap_box = snap["bbox"]
                            overlap_iou = compute_iou(bbox, snap_box)
                            center_contained = (
                                snap_box[0] <= position[0] <= snap_box[2]
                                and snap_box[1] <= position[1] <= snap_box[3]
                            )
                            # Candidate must satisfy proximity OR direct bbox overlap/containment
                            max_allowed_dist = HANDOFF_ALERTING_MAX_DISTANCE if snap.get("alert_active") else HANDOFF_MAX_DISTANCE
                            if (c_dist <= max_allowed_dist or overlap_iou >= 0.10 or center_contained) and c_dist < best_distance:
                                best_distance = c_dist
                                matched_snap_idx = s_idx

                        if matched_snap_idx is not None:
                            matched_snap = self.recent_departed_bags.pop(matched_snap_idx)
                            inherited_owner_id = matched_snap["associated_person_id"]
                            inherited_gallery = matched_snap["owner_gallery"]
                            # Mark source bag if still in self.bags as having handed off
                            if matched_snap["track_id"] in self.bags:
                                self.bags[matched_snap["track_id"]].transferred_handoff = True

                            handoff_record = {
                                "frame_timestamp": timestamp,
                                "old_track_id": matched_snap["track_id"],
                                "new_track_id": track_id,
                                "old_class": matched_snap["class_name"],
                                "new_class": class_name,
                                "distance": round(best_distance, 2),
                                "time_diff": round(timestamp - matched_snap["departed_at"], 2),
                                "transferred_owner_id": inherited_owner_id,
                                "gallery_size": len(inherited_gallery.embeddings) if inherited_gallery else 0,
                            }
                            self.handoff_events.append(handoff_record)
                            print(
                                f"[HANDOFF] Time {timestamp:.2f}s: Carried Bag {matched_snap['track_id']} "
                                f"-> Deposited Bag {track_id} | Dist: {best_distance:.1f}px | "
                                f"dt: {timestamp - matched_snap['departed_at']:.2f}s | "
                                f"Owner: {inherited_owner_id} (Gallery: {len(inherited_gallery.embeddings)} embs)"
                            )

                        # For alerting bags that hand off, preserve original stationary anchor footprint and alert state.
                        # For normal carried bags, establish anchor at the new deposit position.
                        is_alerting_handoff = bool(matched_snap and matched_snap.get("alert_active"))
                        bag = TrackedBag(
                            track_id=track_id,
                            class_name=class_name,
                            bbox=bbox,
                            position=position,
                            first_detected_at=datetime.now(),
                            last_detected_at=datetime.now(),
                            stationary_since=matched_snap["stationary_since"] if is_alerting_handoff else timestamp,
                            stationary_anchor=matched_snap["stationary_anchor"] if is_alerting_handoff else position,
                            last_seen_time=timestamp,
                            class_history=[class_name],
                            anchor_id=track_id,
                            associated_person_id=inherited_owner_id,
                            owner_gallery=inherited_gallery,
                            last_alert_at=matched_snap.get("last_alert_at") if is_alerting_handoff else None,
                            alert_active=matched_snap.get("alert_active", False) if is_alerting_handoff else False,
                            bag_retrieved=False,
                            transferred_handoff=False,
                            handoff_candidate_created=False,
                            last_known_bbox=bbox,
                            stationary_anchor_bbox=matched_snap["stationary_anchor_bbox"] if is_alerting_handoff else bbox,
                            displacement_frames=0,
                            bag_removed_unverified=matched_snap.get("bag_removed_unverified", False) if is_alerting_handoff else False,
                            owner_returned=matched_snap.get("owner_returned", False) if matched_snap else False,
                            returned_person_id=matched_snap.get("returned_person_id") if matched_snap else None,
                            last_owner_seen_at=matched_snap.get("last_owner_seen_at") if matched_snap else None,
                            nearby_person_frames=matched_snap.get("nearby_person_frames", MIN_CARRIED_FRAMES) if matched_snap else 0,
                        )
                        self.bags[track_id] = bag

            # Update bag attributes
            bag.bbox = bbox
            bag.last_known_bbox = bbox
            bag.position = position
            bag.last_detected_at = datetime.now()
            bag.last_seen_time = timestamp

            bag.class_history.append(class_name)
            if len(bag.class_history) > MAX_CLASS_HISTORY:
                bag.class_history.pop(0)
            bag.class_name = Counter(bag.class_history).most_common(1)[0][0]

            # Displacement & Jitter Check
            movement = (
                (position[0] - bag.stationary_anchor[0]) ** 2
                + (position[1] - bag.stationary_anchor[1]) ** 2
            ) ** 0.5

            if bag.stationary_anchor_bbox is None:
                bag.stationary_anchor_bbox = bbox

            anchor_iou = compute_iou(bbox, bag.stationary_anchor_bbox)

            # Distinguish genuine luggage movement from detector aspect/centroid jitter:
            # If centroid moved beyond threshold, check if current box still substantially overlaps anchor footprint
            if movement > self.stationary_distance:
                if not bag.bag_retrieved:
                    # Genuine movement requires departing the anchor footprint (IoU < 0.35)
                    if anchor_iou < 0.35:
                        bag.displacement_frames += 1
                        # Require sustained displacement across 5 consecutive frames before resetting timer
                        if bag.displacement_frames >= 5:
                            # Alert persistence: anchor recalibration only applies to non-alerting bags
                            # An alerting bag's stationary anchor represents its fixed deposited footprint!
                            if not bag.alert_active:
                                bag.stationary_anchor = position
                                bag.stationary_anchor_bbox = bbox
                                bag.stationary_since = timestamp
                            bag.displacement_frames = 0
                    else:
                        # Bounding box still shares >= 0.35 IoU with anchor: centroid jitter, do not reset timer
                        bag.displacement_frames = 0
            else:
                # Luggage is well within stationary threshold: reset any transient displacement counter
                bag.displacement_frames = 0

            # Environmental artifact and person association check
            if is_env_artifact:
                bag.is_static_background = True

            close_people = [
                (
                    ((p.position[0] - position[0]) ** 2 + (p.position[1] - position[1]) ** 2) ** 0.5,
                    p.track_id,
                )
                for p in people.values()
                if ((p.position[0] - position[0]) ** 2 + (p.position[1] - position[1]) ** 2) ** 0.5 <= INTERACTION_PROXIMITY_THRESHOLD
            ]

            if not is_env_artifact:
                if close_people:
                    bag.nearby_person_frames += 1
                    bag.is_static_background = False
                elif bag.associated_person_id is None:
                    # If no person has ever been close to this bag candidate, flag as static background
                    if bag.nearby_person_frames == 0:
                        bag.is_static_background = True

            # Owner Association & Appearance Gallery Accumulation
            if not bag.is_static_background:
                if bag.associated_person_id is None:
                    if close_people:
                        owner_id = min(close_people)[1]
                        bag.associated_person_id = owner_id
                        bag.owner_gallery = OwnerGallery(owner_track_id=owner_id)
                        if people[owner_id].embedding is not None:
                            bag.owner_gallery.add_embedding(people[owner_id].embedding)

                elif bag.owner_gallery is not None:
                    # If original owner is still in frame and nearby, continue updating gallery
                    if bag.associated_person_id in people:
                        owner_p = people[bag.associated_person_id]
                        dist_to_owner = (
                            (owner_p.position[0] - position[0]) ** 2
                            + (owner_p.position[1] - position[1]) ** 2
                        ) ** 0.5
                        if dist_to_owner <= self.person_distance and owner_p.embedding is not None:
                            bag.owner_gallery.add_embedding(owner_p.embedding)
                            bag.owner_gallery.last_updated_time = timestamp

        # Check for bags absent for >= HANDOFF_ABSENT_TRIGGER (0.05s) to create handoff snapshot
        # before the 2.0s track expiry deletes them
        for track_id, bag in self.bags.items():
            if (
                not bag.handoff_candidate_created
                and not bag.transferred_handoff
                and (timestamp - bag.last_seen_time) >= HANDOFF_ABSENT_TRIGGER
                and bag.associated_person_id is not None
                and bag.owner_gallery is not None
                and len(bag.owner_gallery.embeddings) > 0
                and (bag.nearby_person_frames >= MIN_CARRIED_FRAMES or bag.alert_active)
                and not bag.is_static_background
                and not bag.bag_retrieved
            ):
                # Verify that the previous owner was close to the luggage immediately before track disappeared
                # or that the bag was genuinely in an interaction zone or already alerting
                bag.handoff_candidate_created = True
                snap = {
                    "track_id": track_id,
                    "class_name": bag.class_name,
                    "bbox": bag.bbox,
                    "position": bag.position,
                    "associated_person_id": bag.associated_person_id,
                    "owner_gallery": bag.owner_gallery,
                    "departed_at": bag.last_seen_time,
                    "stationary_since": bag.stationary_since,
                    "stationary_anchor": bag.stationary_anchor,
                    "stationary_anchor_bbox": bag.stationary_anchor_bbox or bag.bbox,
                    "alert_active": bag.alert_active,
                    "last_alert_at": bag.last_alert_at,
                    "bag_removed_unverified": bag.bag_removed_unverified,
                    "owner_returned": bag.owner_returned,
                    "returned_person_id": bag.returned_person_id,
                    "last_owner_seen_at": bag.last_owner_seen_at,
                    "nearby_person_frames": bag.nearby_person_frames,
                }
                self.recent_departed_bags.append(snap)

        # Prune dead bag tracks beyond 2.0s expiry period
        for track_id in list(self.bags):
            if timestamp - self.bags[track_id].last_seen_time > TRACK_EXPIRY_SECONDS:
                dead_bag = self.bags[track_id]
                dead_stat_dur = max(0.0, timestamp - dead_bag.stationary_since)

                # Preserve dormant state for eligible owner-associated luggage:
                # Must have valid owner, valid gallery, and be alerting or have reached unattended threshold
                if (
                    dead_bag.associated_person_id is not None
                    and dead_bag.owner_gallery is not None
                    and len(dead_bag.owner_gallery.embeddings) > 0
                    and not dead_bag.is_static_background
                    and not dead_bag.bag_retrieved
                    and (dead_bag.alert_active or dead_stat_dur >= self.stationary_time)
                ):
                    dormant_rec = DormantLuggageRecord(
                        original_track_id=dead_bag.track_id,
                        class_name=dead_bag.class_name,
                        class_history=list(dead_bag.class_history),
                        last_bbox=dead_bag.bbox,
                        last_position=dead_bag.position,
                        stationary_anchor=dead_bag.stationary_anchor,
                        stationary_anchor_bbox=dead_bag.stationary_anchor_bbox or dead_bag.bbox,
                        stationary_since=dead_bag.stationary_since,
                        associated_person_id=dead_bag.associated_person_id,
                        owner_gallery=dead_bag.owner_gallery,
                        alert_active=dead_bag.alert_active,
                        last_alert_at=dead_bag.last_alert_at,
                        bag_retrieved=dead_bag.bag_retrieved,
                        became_dormant_at=timestamp,
                        bag_removed_unverified=dead_bag.bag_removed_unverified,
                        last_owner_seen_at=dead_bag.last_owner_seen_at,
                    )
                    self.dormant_luggage.append(dormant_rec)
                    print(
                        f"[DORMANT ENTER] Time {timestamp:.2f}s: Bag {dead_bag.track_id} "
                        f"entered dormant state | Owner: {dead_bag.associated_person_id} | "
                        f"Alert: {dead_bag.alert_active} | Pos: {dead_bag.position}"
                    )

                del self.bags[track_id]

        # Evaluate unattended status, Re-ID return, and retrieval
        active_alert = False
        active_retrieval = False
        dormant_alert = False   # True when any dormant record is still alerting (Option A)
        returning_owner_ids: set[int] = set()

        for bag in self.bags.values():
            stationary_duration = max(0.0, timestamp - bag.stationary_since)

            # Owner distance
            associated_person = people.get(bag.associated_person_id)
            if associated_person is not None:
                dist_to_owner = (
                    (associated_person.position[0] - bag.position[0]) ** 2
                    + (associated_person.position[1] - bag.position[1]) ** 2
                ) ** 0.5
                person_moved_away = dist_to_owner > self.person_distance
            elif bag.associated_person_id is not None:
                # Owner was previously identified but has now walked out of view/off-screen
                min_dist_to_anyone = min(
                    [
                        (
                            (p.position[0] - bag.position[0]) ** 2
                            + (p.position[1] - bag.position[1]) ** 2
                        )
                        ** 0.5
                        for p in people.values()
                    ],
                    default=float("inf"),
                )
                person_moved_away = min_dist_to_anyone > self.person_distance
            else:
                # Unassociated luggage with no known owner: MUST NEVER TRIGGER ALERT
                person_moved_away = False

            # Scenario 1: Bag left unattended
            if not bag.bag_retrieved and not bag.is_static_background:
                conditions_met = (
                    stationary_duration >= self.stationary_time and person_moved_away
                )
                if not conditions_met and not bag.owner_returned:
                    # Alert persistence: an already-active alert must remain active until confirmed retrieval
                    if not bag.alert_active:
                        bag.alert_active = False
                elif conditions_met:
                    if not bag.alert_active and (
                        bag.last_alert_at is None
                        or timestamp - bag.last_alert_at >= self.alert_cooldown
                    ):
                        bag.alert_active = True
                        bag.last_alert_at = timestamp
                        # Calibrate stationary anchor footprint to the confirmed resting position at alert activation
                        bag.stationary_anchor = bag.position
                        bag.stationary_anchor_bbox = bag.bbox

            # Scenario 2: Return Detection (Only if alert was triggered / bag unattended)
            if bag.alert_active and not bag.bag_retrieved and bag.owner_gallery is not None:
                # Find approaching candidates
                best_sim = 0.0
                best_candidate_id = None

                for p_id, p in people.items():
                    dist = (
                        (p.position[0] - bag.position[0]) ** 2
                        + (p.position[1] - bag.position[1]) ** 2
                    ) ** 0.5
                    # Candidate must be in reasonable proximity to be considered approaching
                    if dist <= self.person_distance * 1.5:
                        sim = bag.owner_gallery.match_similarity(p.embedding) if p.embedding is not None else -1.0
                        if p.embedding is not None and sim > best_sim:
                            best_sim = sim
                            best_candidate_id = p_id

                if best_sim >= self.reid_threshold and best_candidate_id is not None:
                    bag.owner_match_counter += 1
                    if bag.owner_match_counter >= self.confirm_frames:
                        bag.owner_returned = True
                        bag.returned_person_id = best_candidate_id
                        bag.last_owner_seen_at = timestamp
                        returning_owner_ids.add(best_candidate_id)
                else:
                    if bag.owner_match_counter > 0:
                        bag.owner_match_counter -= 1

                # If recognized owner is currently visible, update their last seen timestamp
                if bag.owner_returned and bag.returned_person_id in people:
                    bag.last_owner_seen_at = timestamp

                # State Evaluation: Physical Movement vs. Owner Retrieval
                bag_movement = (
                    (bag.position[0] - bag.stationary_anchor[0]) ** 2
                    + (bag.position[1] - bag.stationary_anchor[1]) ** 2
                ) ** 0.5
                curr_anchor_iou = compute_iou(bag.bbox, bag.stationary_anchor_bbox) if bag.stationary_anchor_bbox else 0.0

                # 1. State B: OWNER_RETRIEVAL_CONFIRMED
                # Requires recognized owner detection to be available, interaction conditions satisfied,
                # and interaction occurs within 2.0s of the bag's departure/movement
                owner_simultaneous_interaction = False
                if bag.owner_returned and bag.returned_person_id in people:
                    ret_person = people[bag.returned_person_id]
                    p_box = ret_person.bbox
                    b_box = bag.bbox

                    iou = compute_iou(p_box, b_box)
                    b_center = bag.position
                    p_contains_center = (
                        p_box[0] <= b_center[0] <= p_box[2]
                        and p_box[1] <= b_center[1] <= p_box[3]
                    )
                    dist_to_ret = (
                        (ret_person.position[0] - bag.position[0]) ** 2
                        + (ret_person.position[1] - bag.position[1]) ** 2
                    ) ** 0.5

                    # Interaction criteria: overlap or severe proximity
                    if (iou >= INTERACTION_IOU_THRESHOLD or p_contains_center or dist_to_ret < INTERACTION_PROXIMITY_THRESHOLD):
                        time_since_owner = timestamp - (bag.last_owner_seen_at or timestamp)
                        if time_since_owner <= 2.0:
                            owner_simultaneous_interaction = True
                            if bag_movement > self.stationary_distance / 2.0 or iou >= 0.15:
                                bag.bag_retrieved = True
                                bag.alert_active = False
                                bag.bag_removed_unverified = False
                                bag.retrieval_confirmed_at = timestamp
                                # Clear any corresponding dormant alert records upon confirmed retrieval
                                for drec in self.dormant_luggage:
                                    if drec.associated_person_id == bag.associated_person_id:
                                        drec.bag_retrieved = True
                                        drec.alert_active = False

                # 2. State A: BAG_REMOVED_UNVERIFIED
                # Trigger when:
                # - The bag has an active unattended alert
                # - Displacement from its stationary anchor exceeds 50 pixels
                # - Anchor IoU is below 0.10
                # - Persists for at least 5 consecutive frames
                # - No simultaneous verified interaction with the recognized owner
                if not bag.bag_retrieved and bag.alert_active:
                    if bag_movement > 50.0 and curr_anchor_iou < 0.10 and not owner_simultaneous_interaction:
                        bag.unverified_removal_frames += 1
                        if bag.unverified_removal_frames >= 5:
                            bag.bag_removed_unverified = True
                            # Security requirement: Keep the security alert active!
                            # Do not clear alert_active, do not display blue resolved state
                    else:
                        bag.unverified_removal_frames = 0

            if bag.alert_active:
                active_alert = True
            if bag.bag_retrieved:
                active_retrieval = True

        # Option A: Check dormant records for active alerts that are no longer in self.bags.
        # A dormant record keeps alert_active=True when the bag disappeared without confirmed retrieval.
        # We render the red banner for these records so the security alert stays visible.
        # Dormant records that have been retrieved (bag_retrieved=True) are excluded — retrieval
        # is only set on active tracks via confirmed owner interaction, so this guard is for safety.
        for drec in self.dormant_luggage:
            if drec.alert_active and not drec.bag_retrieved:
                dormant_alert = True
                break

        # Render Visuals
        # 1. Render People
        for person in people.values():
            if person.track_id in returning_owner_ids and not any(
                b.bag_retrieved for b in self.bags.values() if b.returned_person_id == person.track_id
            ):
                # Scenario 2: Original owner returned, alert still active
                draw_box(
                    frame,
                    person.bbox,
                    f"ORIGINAL OWNER RETURNED | ID {person.track_id}",
                    (0, 0, 255),  # RED
                    thickness=2,
                )
            else:
                # Normal person (Yellow/Orange)
                draw_box(
                    frame,
                    person.bbox,
                    f"person ID {person.track_id}",
                    (255, 180, 0),
                    thickness=2,
                )

        # 2. Render Luggage (skip static background artifacts)
        for bag in self.bags.values():
            if bag.is_static_background:
                continue
            # For non-alerted, non-retrieved luggage, only render if actively detected in recent frames
            if not bag.alert_active and not bag.bag_retrieved:
                if timestamp - bag.last_seen_time > 0.4:
                    continue

            stat_sec = max(0.0, timestamp - bag.stationary_since)
            person_text = (
                f"person {bag.associated_person_id}"
                if bag.associated_person_id is not None
                else "person unknown"
            )

            if bag.bag_retrieved:
                # Alert Resolution: Bag retrieved by original owner -> BLUE
                label = f"ORIGINAL OWNER RETRIEVED BAG | ID: {bag.track_id} | {person_text}"
                draw_box(frame, bag.bbox, label, (255, 0, 0), thickness=2)  # BLUE (BGR: 255, 0, 0)
                cv2.putText(
                    frame,
                    "ORIGINAL OWNER RETRIEVED BAG",
                    (bag.bbox[0], min(frame.shape[0] - 10, bag.bbox[3] + 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 0, 0),
                    2,
                )
            elif bag.bag_removed_unverified:
                # State A: Bag moved/removed without verified owner interaction -> YELLOW / AMBER
                label = f"{bag.class_name} ID {bag.track_id} | MOVED / UNVERIFIED CARRIER"
                draw_box(frame, bag.bbox, label, (0, 215, 255), thickness=2)  # YELLOW/AMBER (BGR: 0, 215, 255)
                cv2.putText(
                    frame,
                    "UNVERIFIED REMOVAL",
                    (bag.bbox[0], min(frame.shape[0] - 10, bag.bbox[3] + 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 215, 255),
                    2,
                )
            elif bag.alert_active:
                # Unattended Bag -> RED
                label = (
                    f"UNATTENDED BAG | ID: {bag.track_id} | "
                    f"{stat_sec:.1f}s | "
                    f"pos ({int(bag.position[0])}, {int(bag.position[1])}) | {person_text}"
                )
                draw_box(frame, bag.bbox, label, (0, 0, 255), thickness=2)
                cv2.putText(
                    frame,
                    "UNATTENDED OBJECT",
                    (bag.bbox[0], min(frame.shape[0] - 10, bag.bbox[3] + 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                )
            else:
                # Normal luggage -> GREEN
                label = (
                    f"{bag.class_name} ID {bag.track_id} | "
                    f"{stat_sec:.1f}s | "
                    f"pos ({int(bag.position[0])}, {int(bag.position[1])}) | {person_text}"
                )
                draw_box(frame, bag.bbox, label, (0, 220, 0), thickness=2)

        # Top Banner Alerts
        any_unverified_removal = any(b.bag_removed_unverified for b in self.bags.values())

        if active_retrieval and not active_alert and not dormant_alert:
            # Resolved banner: BLUE — only when NO active or dormant alert remains
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (200, 50, 0), -1)
            cv2.putText(
                frame,
                "RESOLVED: ORIGINAL OWNER RETRIEVED BAG",
                (20, 32),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
        elif any_unverified_removal:
            # Caution banner: YELLOW/AMBER with black text
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (0, 215, 255), -1)
            cv2.putText(
                frame,
                "CAUTION: UNATTENDED BAG MOVED / REMOVED (UNVERIFIED CARRIER)",
                (20, 32),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 0, 0),
                2,
            )
        elif active_alert:
            # Active alert banner: RED (bag actively tracked)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (0, 0, 220), -1)
            cv2.putText(
                frame,
                "ALERT: UNATTENDED BAG DETECTED",
                (20, 32),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
        elif dormant_alert:
            # Option A: Dormant alert banner — bag temporarily untracked but alert not cleared.
            # Same RED colour as active alert; text distinguishes the temporarily-lost state.
            # No bounding box ghost is drawn; the alert is banner-only.
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (0, 0, 220), -1)
            cv2.putText(
                frame,
                "ALERT: UNATTENDED BAG DETECTED (TEMPORARILY UNTRACKED)",
                (20, 32),
                cv2.FONT_HERSHEY_DUPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

        status_info = {
            "timestamp": timestamp,
            "active_alert": active_alert or dormant_alert,
            "actively_tracked_alert": active_alert,
            "dormant_alert": dormant_alert,
            "active_retrieval": active_retrieval,
            "any_unverified_removal": any_unverified_removal,
            "bags": {
                b_id: {
                    "alert_active": b.alert_active,
                    "owner_returned": b.owner_returned,
                    "bag_retrieved": b.bag_retrieved,
                    "bag_removed_unverified": b.bag_removed_unverified,
                    "stationary_duration": max(0.0, timestamp - b.stationary_since),
                }
                for b_id, b in self.bags.items()
            },
        }

        return frame, status_info
