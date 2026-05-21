from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b4


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = Path(
    os.getenv("VERITAI_CNN_CHECKPOINT", str(BASE_DIR / "checkpoints" / "cnn_model.pt"))
)
MAX_IMAGE_WIDTH = int(os.getenv("VERITAI_MAX_IMAGE_WIDTH", "1280"))
FAKE_THRESHOLD = float(os.getenv("VERITAI_CNN_FAKE_THRESHOLD", "0.5"))
# train_colab.py: class 0=real, class 1=fake
REAL_CLASS_INDEX = 0
FAKE_CLASS_INDEX = 1
# Service-aligned training zip (crop_all --pipeline service) includes __full.jpg by default.
INCLUDE_FULL_VIEW = os.getenv("VERITAI_CNN_INCLUDE_FULL_VIEW", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class CnnView:
    name: str
    bgr: np.ndarray
    bbox: Optional[Dict[str, int]] = None


class DeepfakeDetector(nn.Module):
    """Same checkpoint structure used by colab/train_colab.py."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        backbone = efficientnet_b4(weights=None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def normalize_bgr_width(bgr: np.ndarray, max_width: int = MAX_IMAGE_WIDTH) -> np.ndarray:
    if bgr is None or bgr.size == 0:
        return bgr
    h, w = bgr.shape[:2]
    if w <= max_width:
        return bgr
    ratio = max_width / float(w)
    return cv2.resize(bgr, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)


def normalize_bbox(source: Dict[str, Any]) -> Optional[Dict[str, int]]:
    if "bbox" in source and isinstance(source["bbox"], dict):
        bbox = source["bbox"]
        return {
            "x": int(bbox.get("x", 0)),
            "y": int(bbox.get("y", 0)),
            "w": int(bbox.get("w", 0)),
            "h": int(bbox.get("h", 0)),
        }
    if "box" in source:
        x, y, w, h = source["box"]
        return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
    return None


def crop_bgr_from_bbox(bgr: np.ndarray, bbox: Dict[str, int]) -> Optional[np.ndarray]:
    x = int(bbox.get("x", 0))
    y = int(bbox.get("y", 0))
    w = int(bbox.get("w", 0))
    h = int(bbox.get("h", 0))
    if w <= 0 or h <= 0:
        return None
    image_h, image_w = bgr.shape[:2]
    x = max(0, min(x, image_w - 1))
    y = max(0, min(y, image_h - 1))
    w = max(1, min(w, image_w - x))
    h = max(1, min(h, image_h - y))
    crop = bgr[y : y + h, x : x + w]
    if crop.size == 0:
        return None
    return crop


def build_cnn_views_bgr(bgr: np.ndarray, faces: Sequence[Dict[str, Any]]) -> List[CnnView]:
    """
    CNN inputs aligned with POST /predict (crop_all --pipeline service).

    - face_N: build_face_output() bbox on MAX_IMAGE_WIDTH resized image
    - full: included when VERITAI_CNN_INCLUDE_FULL_VIEW=1 (default; matches service zip)
    """
    full = normalize_bgr_width(bgr)
    views: List[CnnView] = []
    if INCLUDE_FULL_VIEW:
        views.append(CnnView("full", full))
    for index, face in enumerate(faces):
        bbox = normalize_bbox(face)
        if bbox is None:
            continue
        crop = crop_bgr_from_bbox(full, bbox)
        if crop is None:
            continue
        views.append(
            CnnView(
                f"face_{index + 1}",
                crop,
                bbox,
            )
        )
    return views


class CnnRuntime:
    def __init__(
        self,
        checkpoint_path: Path = CHECKPOINT_PATH,
        *,
        device: Optional[str] = None,
        threshold: float = FAKE_THRESHOLD,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.threshold = threshold
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.model: Optional[DeepfakeDetector] = None
        self.load_error: Optional[str] = None
        self._load()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def _load(self) -> None:
        if not self.checkpoint_path.is_file():
            self.load_error = f"checkpoint not found: {self.checkpoint_path}"
            return
        try:
            payload = torch.load(self.checkpoint_path, map_location=self.device)
            if isinstance(payload, dict) and "model" in payload:
                payload = payload["model"]
            elif isinstance(payload, dict) and "state_dict" in payload:
                payload = payload["state_dict"]
            model = DeepfakeDetector().to(self.device)
            model.load_state_dict(payload, strict=True)
            model.eval()
            self.model = model
            self.load_error = None
        except Exception as exc:
            self.model = None
            self.load_error = str(exc)

    def _tensor_from_bgr(self, bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.transform(Image.fromarray(rgb))

    @torch.no_grad()
    def predict_views(self, views: Sequence[CnnView]) -> Dict[str, Any]:
        if not self.loaded or self.model is None:
            return {
                "modelLoaded": False,
                "checkpoint": str(self.checkpoint_path),
                "error": self.load_error,
                "fakeProbability": 0.0,
                "isDeepfake": False,
                "views": [],
            }
        valid_views = [view for view in views if view.bgr is not None and view.bgr.size > 0]
        if not valid_views:
            return {
                "modelLoaded": True,
                "checkpoint": str(self.checkpoint_path),
                "fakeProbability": 0.0,
                "isDeepfake": False,
                "views": [],
            }
        batch = torch.stack([self._tensor_from_bgr(view.bgr) for view in valid_views]).to(self.device)
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1)[:, FAKE_CLASS_INDEX].detach().cpu().numpy()
        view_results = []
        best_prob = 0.0
        for view, prob in zip(valid_views, probs):
            fake_prob = float(prob)
            best_prob = max(best_prob, fake_prob)
            view_results.append(
                {
                    "name": view.name,
                    "fakeProbability": round(fake_prob, 6),
                    "bbox": view.bbox,
                }
            )
        return {
            "modelLoaded": True,
            "checkpoint": str(self.checkpoint_path),
            "fakeProbability": round(float(best_prob), 6),
            "isDeepfake": bool(best_prob >= self.threshold),
            "threshold": self.threshold,
            "views": view_results,
        }

    def predict_image_faces(self, bgr: np.ndarray, faces: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return self.predict_views(build_cnn_views_bgr(bgr, faces))


_RUNTIME: Optional[CnnRuntime] = None


def get_runtime() -> CnnRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = CnnRuntime()
    return _RUNTIME


def predict_image_faces(bgr: np.ndarray, faces: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return get_runtime().predict_image_faces(bgr, faces)
