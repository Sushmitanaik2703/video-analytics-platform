"""
Extractor for Person Re-Identification features using OSNet.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from .osnet import osnet_x0_25


class OSNetExtractor:
    """Extracts 512-dimensional L2-normalized feature embeddings from person image crops."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str | None = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if weights_path is None:
            weights_path = (
                Path(__file__).resolve().parent / "weights" / "osnet_x0_25_msmt17.pth"
            )
        self.weights_path = Path(weights_path)

        self.model = osnet_x0_25(num_classes=4101, pretrained=False)
        if self.weights_path.is_file():
            state_dict = torch.load(self.weights_path, map_location="cpu")
            # Filter classifier keys if shape mismatch
            model_dict = self.model.state_dict()
            filtered_dict = {
                k: v
                for k, v in state_dict.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }
            model_dict.update(filtered_dict)
            self.model.load_state_dict(model_dict)
        else:
            raise FileNotFoundError(f"OSNet weights not found at {self.weights_path}")

        self.model.to(self.device)
        self.model.eval()

        # ImageNet normalization standard for OSNet
        self.transform = T.Compose(
            [
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def extract_crop(self, frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        """Extract a normalized 512-d feature vector from a bounding box crop."""
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 - x1 < 10 or y2 - y1 < 20:
            return None

        crop_bgr = frame_bgr[y1:y2, x1:x2]
        crop_rgb = cv2_to_pil(crop_bgr)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

        embedding = self.model(tensor)
        # L2-normalize
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()[0]

    @torch.no_grad()
    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity between two unit vectors."""
        # Since features are L2 normalized, dot product is cosine similarity
        return float(np.dot(feat1, feat2))


def cv2_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """Convert BGR OpenCV image to PIL RGB Image."""
    import cv2

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)
