import json
import math
import os
import threading
import time
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

try:
    import mediapipe as mp
except ImportError:
    mp = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
ANALYSIS_DIR = os.path.join(IMAGES_DIR, "analysis")
FACE_CROP_DIR = os.path.join(IMAGES_DIR, "faces")
OVERLAY_DIR = os.path.join(ANALYSIS_DIR, "overlays")
ANCHOR_MAP_DIR = os.path.join(ANALYSIS_DIR, "anchor_maps")
METADATA_DIR = os.path.join(ANALYSIS_DIR, "metadata")
RESPONSE_MAP_DIR = os.path.join(ANALYSIS_DIR, "response_maps")
EYE_RESPONSE_DIR = os.path.join(RESPONSE_MAP_DIR, "eyes")
NOSE_RESPONSE_DIR = os.path.join(RESPONSE_MAP_DIR, "nose")
MOUTH_RESPONSE_DIR = os.path.join(RESPONSE_MAP_DIR, "mouth")
for directory in (
    IMAGES_DIR,
    ANALYSIS_DIR,
    FACE_CROP_DIR,
    OVERLAY_DIR,
    ANCHOR_MAP_DIR,
    METADATA_DIR,
    RESPONSE_MAP_DIR,
    EYE_RESPONSE_DIR,
    NOSE_RESPONSE_DIR,
    MOUTH_RESPONSE_DIR,
):
    os.makedirs(directory, exist_ok=True)

CASCADE_BASE = cv2.data.haarcascades
FACE_CASCADES = {
    "frontal": cv2.CascadeClassifier(os.path.join(CASCADE_BASE, "haarcascade_frontalface_default.xml")),
    "frontal_alt": cv2.CascadeClassifier(os.path.join(CASCADE_BASE, "haarcascade_frontalface_alt2.xml")),
    "profile": cv2.CascadeClassifier(os.path.join(CASCADE_BASE, "haarcascade_profileface.xml")),
}
MP_FACE_DETECTOR = None
MP_FACE_DETECTOR_LOCK = threading.Lock()
if mp is not None:
    try:
        MP_FACE_DETECTOR = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.45,
        )
    except Exception:
        MP_FACE_DETECTOR = None
MP_FACE_LANDMARKER = None
MP_FACE_LANDMARKER_LOCK = threading.Lock()
MP_FACE_LANDMARKER_MODEL = os.getenv(
    "VERITAI_MEDIAPIPE_FACE_LANDMARKER_MODEL",
    os.path.join(BASE_DIR, "models", "face_landmarker.task"),
)
if mp is not None and hasattr(mp, "tasks") and os.path.exists(MP_FACE_LANDMARKER_MODEL):
    try:
        MP_FACE_LANDMARKER = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=MP_FACE_LANDMARKER_MODEL),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=2,
                min_face_detection_confidence=0.35,
                min_face_presence_confidence=0.35,
            )
        )
    except Exception:
        MP_FACE_LANDMARKER = None
YUNET_FACE_MODEL = os.getenv(
    "VERITAI_YUNET_FACE_MODEL",
    os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx"),
)
YUNET_FACE_DETECTOR_LOCK = threading.Lock()
REGION_COLORS = {
    "forehead": (78, 121, 255),
    "left_eye_zone": (92, 200, 255),
    "right_eye_zone": (92, 200, 255),
    "nose": (60, 214, 196),
    "mouth": (92, 112, 255),
    "jaw": (255, 189, 89),
}
TRAINING_POINT_ORDER = [
    "forehead_center",
    "left_eye_center",
    "right_eye_center",
    "nose_bridge_top",
    "nose_tip",
    "mouth_left",
    "mouth_center",
    "mouth_right",
    "chin",
]
POINT_LABELS = {
    "forehead_center": "F",
    "left_eye_center": "LE",
    "right_eye_center": "RE",
    "nose_bridge_top": "NB",
    "nose_tip": "NT",
    "mouth_left": "ML",
    "mouth_center": "MC",
    "mouth_right": "MR",
    "chin": "C",
}


def env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name, default, minimum=None):
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    if minimum is not None:
        return max(minimum, value)
    return value


def env_float(name, default, minimum=None, maximum=None):
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


DEBUG_ARTIFACTS = env_bool("VERITAI_DEBUG_ARTIFACTS", False)
MAX_IMAGE_WIDTH = env_int("VERITAI_MAX_IMAGE_WIDTH", 1280, minimum=320)
MAX_FACE_CANDIDATES = env_int("VERITAI_MAX_FACE_CANDIDATES", 8, minimum=1)
FACE_INITIAL_NMS_IOU = env_float("VERITAI_FACE_INITIAL_NMS_IOU", 0.24, minimum=0.05, maximum=0.95)
FACE_REFINED_NMS_IOU = env_float("VERITAI_FACE_REFINED_NMS_IOU", 0.28, minimum=0.05, maximum=0.95)
HAAR_TEXTURE_FP_FILTER = env_bool("VERITAI_HAAR_TEXTURE_FP_FILTER", True)
CANDIDATE_BBOX_RETENTION = os.getenv("VERITAI_CANDIDATE_BBOX_RETENTION", "none").strip().lower()
if CANDIDATE_BBOX_RETENTION in {"", "off", "false", "0"}:
    CANDIDATE_BBOX_RETENTION = "none"
if CANDIDATE_BBOX_RETENTION not in {"none", "precision_guarded"}:
    CANDIDATE_BBOX_RETENTION = "none"
RETENTION_FEATURE_GUARD = os.getenv("VERITAI_RETENTION_FEATURE_GUARD", "none").strip().lower()
if RETENTION_FEATURE_GUARD in {"", "off", "false", "0"}:
    RETENTION_FEATURE_GUARD = "none"
if RETENTION_FEATURE_GUARD not in {"none", "anthropometric_material", "anthropometric_low_skin_saturation", "low_skin_profile", "eyes_closed_low_text_density", "retention_precision_combo", "retention_precision_combo_v2"}:
    RETENTION_FEATURE_GUARD = "none"
SERVICE_RETENTION_REJECT_REASONS = {
    "blank-mannequin-zero-skin-material",
    "frontal-face-without-eyes-and-low-structure",
    "frontal-alt-low-chroma-sculpture-face",
    "frontal-alt-low-support-low-skin",
    "frontal-alt-low-skin-desaturated-edge-face",
    "low-closure-low-saturation-nonhuman-face",
    "low-face-like-score-with-weak-mouth-texture",
    "symmetric-peak-pair-with-low-texture",
}


def normalize_path(path):
    return path.replace("\\", "/")


def run_cnn_prediction(image, faces):
    try:
        from cnn import predict_image_faces
    except Exception as exc:
        return {
            "modelLoaded": False,
            "error": f"cnn import failed: {exc}",
            "fakeProbability": 0.0,
            "isDeepfake": False,
            "views": [],
        }

    try:
        result = predict_image_faces(image, faces)
    except Exception as exc:
        return {
            "modelLoaded": False,
            "error": f"cnn prediction failed: {exc}",
            "fakeProbability": 0.0,
            "isDeepfake": False,
            "views": [],
        }

    view_by_name = {view.get("name"): view for view in result.get("views", [])}
    for index, face in enumerate(faces):
        view = view_by_name.get(f"face_{index + 1}")
        if view is not None:
            face["cnn"] = {
                "fakeProbability": view.get("fakeProbability", 0.0),
                "threshold": result.get("threshold"),
            }
    return result


def decode_image(contents: bytes):
    np_arr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def preprocess_image(img):
    bgr = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    enhanced = cv2.addWeighted(clahe, 0.62, equalized, 0.38, 0.0)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    return {"bgr": bgr, "gray": gray, "rgb": rgb, "equalized": enhanced, "clahe": clahe, "blurred": blurred, "gradX": grad_x, "gradY": grad_y}


def make_debug_paths():
    uid = uuid.uuid4().hex[:12]
    return uid, {
        "overlay": os.path.join(OVERLAY_DIR, f"{uid}_overlay.jpg"),
        "analysisMap": os.path.join(ANCHOR_MAP_DIR, f"{uid}_analysis.jpg"),
        "eyeResponse": os.path.join(EYE_RESPONSE_DIR, f"{uid}_eye_response.jpg"),
        "noseResponse": os.path.join(NOSE_RESPONSE_DIR, f"{uid}_nose_response.jpg"),
        "mouthResponse": os.path.join(MOUTH_RESPONSE_DIR, f"{uid}_mouth_response.jpg"),
        "metadata": os.path.join(METADATA_DIR, f"{uid}_analysis.json"),
    }


def expand_box(box, image_shape, scale=0.18):
    x, y, w, h = box
    pad_w = int(w * scale)
    pad_h = int(h * scale)
    x1 = max(x - pad_w, 0)
    y1 = max(y - pad_h, 0)
    x2 = min(x + w + pad_w, image_shape[1])
    y2 = min(y + h + pad_h, image_shape[0])
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    intersection = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - intersection
    if union <= 0:
        return 0.0
    return intersection / float(union)


def non_max_suppression(candidates, iou_threshold=0.35):
    if not candidates:
        return []
    remaining = sorted(candidates, key=lambda item: item["score"], reverse=True)
    selected = []
    while remaining:
        current = remaining.pop(0)
        selected.append(current)
        remaining = [candidate for candidate in remaining if calculate_iou(current["box"], candidate["box"]) < iou_threshold]
    return selected


def bbox_dict_to_tuple(bbox):
    return int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])


def should_retain_candidate_bbox_service(candidate, image_shape):
    if CANDIDATE_BBOX_RETENTION != "precision_guarded":
        return False
    x, y, w, h = candidate["box"]
    image_h, image_w = image_shape[:2]
    image_area = float(max(image_w * image_h, 1))
    width = float(max(w, 1))
    height = float(max(h, 1))
    area_ratio = (width * height) / image_area
    aspect = width / height
    min_side = min(width, height)
    max_side = max(width, height)
    detector = candidate.get("detector", "")
    score = float(candidate.get("score", 0.0))

    if not (0.48 <= aspect <= 1.60):
        return False
    if min_side < 32:
        return False
    if max_side > max(image_w, image_h) * 0.78:
        return False
    if not (0.025 <= area_ratio <= 0.145):
        return False

    if detector == "frontal_alt":
        return score >= 0.9955 and area_ratio <= 0.135
    if detector == "frontal":
        return score >= 0.32 and area_ratio <= 0.120
    if detector == "profile":
        return score >= 0.13 and area_ratio <= 0.120
    return False


def candidate_retention_priority_service(candidate, image_shape):
    x, y, w, h = candidate["box"]
    image_h, image_w = image_shape[:2]
    image_area = float(max(image_w * image_h, 1))
    area_ratio = (float(max(w, 1)) * float(max(h, 1))) / image_area
    detector = candidate.get("detector", "")
    score = float(candidate.get("score", 0.0))
    detector_weight = {
        "frontal_alt": 0.40,
        "frontal": 0.34,
        "profile": 0.28,
    }.get(detector, 0.0)
    compactness = max(0.0, 1.0 - area_ratio / 0.12)
    return detector_weight + 0.45 * score + 0.15 * compactness


def should_block_retention_by_feature_guard(face, keep_reason):
    if RETENTION_FEATURE_GUARD == "none":
        return False
    texture = face.get("deepfakeFeatures", {}).get("texture", {})
    quality = face.get("quality", {})
    skin_ratio = float(texture.get("skinRatio", 0.0))
    edge_density = float(texture.get("edgeDensity", 0.0))
    quality_score = float(quality.get("score", 0.0))

    anthropometric_material_guard = (
        keep_reason == "frontal-alt-anthropometric-outlier"
        and skin_ratio < 0.08
        and edge_density < 0.06
        and quality_score >= 0.75
    )
    saturation_mean = float(texture.get("colorSaturationMean", 0.0))
    anthropometric_low_skin_saturation_guard = (
        keep_reason == "frontal-alt-anthropometric-outlier"
        and skin_ratio < 0.05
        and saturation_mean < 0.30
    )
    anthropometric_low_saturation_quality_guard = (
        keep_reason == "frontal-alt-anthropometric-outlier"
        and saturation_mean < 0.1264
        and quality_score < 0.8252
    )
    low_skin_profile_guard = keep_reason == "profile-low-face-like-low-skin-without-eyes" and skin_ratio < 0.02
    text_density = float(texture.get("printTextComponentDensity", 999.0))
    eyes_closed_low_text_density_guard = (
        keep_reason == "frontal-alt-eyes-closed-with-too-few-detected-points"
        and text_density < 6.0
    )

    if RETENTION_FEATURE_GUARD == "anthropometric_material":
        return anthropometric_material_guard
    if RETENTION_FEATURE_GUARD == "anthropometric_low_skin_saturation":
        return anthropometric_low_skin_saturation_guard
    if RETENTION_FEATURE_GUARD == "low_skin_profile":
        return low_skin_profile_guard
    if RETENTION_FEATURE_GUARD == "eyes_closed_low_text_density":
        return eyes_closed_low_text_density_guard
    if RETENTION_FEATURE_GUARD == "retention_precision_combo":
        return (
            anthropometric_low_skin_saturation_guard
            or low_skin_profile_guard
            or eyes_closed_low_text_density_guard
        )
    if RETENTION_FEATURE_GUARD == "retention_precision_combo_v2":
        return (
            anthropometric_low_skin_saturation_guard
            or anthropometric_low_saturation_quality_guard
            or low_skin_profile_guard
            or eyes_closed_low_text_density_guard
        )
    return False


def should_retain_rejected_face_service(face, candidate, image_shape, keep_reason):
    if keep_reason in SERVICE_RETENTION_REJECT_REASONS:
        return False
    if should_block_retention_by_feature_guard(face, keep_reason):
        return False
    quality_label = face.get("quality", {}).get("label")
    face_like_score = float(face.get("faceLikeScore", 0.0))
    score = float(candidate.get("score", 0.0))
    detector = candidate.get("detector", "")
    if keep_reason == "frontal-alt-single-eye-mirror-low-structure":
        return (
            detector == "frontal_alt"
            and quality_label == "good"
            and score >= 0.9964
            and should_retain_candidate_bbox_service(candidate, image_shape)
        )
    if keep_reason == "haar-low-chroma-texture-face":
        return (
            detector == "frontal_alt"
            and quality_label == "good"
            and score >= 0.9963
            and face_like_score >= 0.72
            and should_retain_candidate_bbox_service(candidate, image_shape)
        )
    if keep_reason == "weak-profile-candidate-without-eyes":
        return (
            detector == "profile"
            and quality_label == "good"
            and face_like_score >= 0.41
            and should_retain_candidate_bbox_service(candidate, image_shape)
        )
    return should_retain_candidate_bbox_service(candidate, image_shape)


def normalize_weight(weight, fallback=0.58):
    if weight is None:
        return fallback
    clamped = max(float(weight), 0.0)
    return float(max(0.0, min(1.0, 1.0 - math.exp(-clamped / 10.0))))


def normalize_response(response):
    response = response.astype(np.float32)
    response -= float(np.min(response))
    max_value = float(np.max(response))
    if max_value > 1e-6:
        response /= max_value
    return response


def region_response_score(response_map, x1, y1, x2, y2):
    height, width = response_map.shape[:2]
    x1 = clamp(int(x1), 0, max(width - 1, 0))
    y1 = clamp(int(y1), 0, max(height - 1, 0))
    x2 = clamp(int(x2), x1 + 1, width)
    y2 = clamp(int(y2), y1 + 1, height)
    roi = response_map[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float(0.68 * np.max(roi) + 0.32 * np.mean(roi))


def containment_ratio(inner_box, outer_box):
    x1, y1, w1, h1 = inner_box
    x2, y2, w2, h2 = outer_box
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    intersection = inter_w * inter_h
    area = max(1, w1 * h1)
    return intersection / float(area)


def suppress_contained_candidates(candidates):
    filtered = []
    for candidate in candidates:
        area = candidate["box"][2] * candidate["box"][3]
        contained = False
        for other in candidates:
            if candidate is other:
                continue
            other_area = other["box"][2] * other["box"][3]
            if other_area <= area * 1.55:
                continue
            if containment_ratio(candidate["box"], other["box"]) < 0.86:
                continue
            if other.get("score", 0.0) < 0.14:
                continue
            contained = True
            break
        if not contained:
            filtered.append(candidate)
    return filtered


def expand_detected_box(box, image_shape, detector_name):
    x, y, w, h = box
    left_scale = 0.20
    right_scale = 0.20
    top_scale = 0.24
    bottom_scale = 0.36
    if detector_name == "profile":
        left_scale = 0.28
        right_scale = 0.28
        top_scale = 0.24
        bottom_scale = 0.40
    elif detector_name in {"mediapipe", "mediapipe_landmarker", "yunet"}:
        left_scale = 0.24
        right_scale = 0.24
        top_scale = 0.24
        bottom_scale = 0.40
    elif detector_name == "response_fallback":
        left_scale = 0.10
        right_scale = 0.10
        top_scale = 0.16
        bottom_scale = 0.24
    if min(w, h) < 82:
        left_scale *= 1.18
        right_scale *= 1.18
        top_scale *= 1.12
        bottom_scale *= 1.18
    x1 = max(0, int(round(x - w * left_scale)))
    y1 = max(0, int(round(y - h * top_scale)))
    x2 = min(image_shape[1], int(round(x + w + w * right_scale)))
    y2 = min(image_shape[0], int(round(y + h + h * bottom_scale)))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def build_default_face_mask(face_gray, detector_name):
    h, w = face_gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center_x = int(round(w * 0.5))
    center_y = int(round(h * (0.50 if detector_name == "profile" else 0.52)))
    axis_x = max(12, int(round(w * (0.38 if detector_name == "profile" else 0.43))))
    axis_y = max(16, int(round(h * 0.49)))
    cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 255, -1)
    return mask


def extract_primary_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def contour_to_points(contour, box):
    if contour is None:
        return []
    x, y, _, _ = box
    points = contour.reshape(-1, 2)
    return [{"x": int(x + px), "y": int(y + py)} for px, py in points]


def build_face_outline_mask(face_bgr, face_gray, detector_name):
    h, w = face_gray.shape[:2]
    if h < 24 or w < 24:
        default_mask = np.full((h, w), 255, dtype=np.uint8)
        return default_mask, extract_primary_contour(default_mask)

    default_mask = build_default_face_mask(face_gray, detector_name)
    init_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    init_mask[default_mask > 0] = cv2.GC_PR_FGD

    inner_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        inner_mask,
        (int(round(w * 0.5)), int(round(h * 0.54))),
        (max(8, int(round(w * 0.20))), max(12, int(round(h * 0.28)))),
        0,
        0,
        360,
        255,
        -1,
    )
    init_mask[inner_mask > 0] = cv2.GC_FGD

    border = max(2, int(round(min(h, w) * 0.04)))
    init_mask[:border, :] = cv2.GC_BGD
    init_mask[-border:, :] = cv2.GC_BGD
    init_mask[:, :border] = cv2.GC_BGD
    init_mask[:, -border:] = cv2.GC_BGD

    try:
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(face_bgr, init_mask, None, bg_model, fg_model, 2, cv2.GC_INIT_WITH_MASK)
        mask = np.where(
            (init_mask == cv2.GC_FGD) | (init_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
    except cv2.error:
        mask = default_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contour = extract_primary_contour(mask)
    min_area = max(1.0, w * h * 0.18)
    if contour is None or cv2.contourArea(contour) < min_area:
        mask = default_mask
        contour = extract_primary_contour(mask)
    else:
        hull = cv2.convexHull(contour)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, -1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        support_mask = cv2.erode(default_mask, kernel, iterations=1)
        mask = cv2.bitwise_or(mask, support_mask)
        contour = hull

    return mask, contour


def extract_face_region(image, preprocessed, box, detector_name):
    x, y, w, h = box
    face_bgr = image[y : y + h, x : x + w]
    face_gray = preprocessed["equalized"][y : y + h, x : x + w]
    if face_bgr.size == 0 or face_gray.size == 0:
        return None
    face_mask, contour = build_face_outline_mask(face_bgr, face_gray, detector_name)
    return {
        "bbox": box,
        "bgr": face_bgr,
        "gray": face_gray,
        "mask": face_mask,
        "contour": contour,
    }


def masked_response(response_map, face_mask):
    if face_mask is None or response_map.size == 0:
        return normalize_response(response_map)
    if face_mask.shape != response_map.shape:
        resized_mask = cv2.resize(face_mask, (response_map.shape[1], response_map.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        resized_mask = face_mask
    mask_weight = 0.06 + 0.94 * (resized_mask.astype(np.float32) / 255.0)
    masked = response_map.astype(np.float32) * mask_weight
    return normalize_response(masked)


def masked_values(image, face_mask):
    if face_mask is None or face_mask.size == 0:
        return image.reshape(-1)
    values = image[face_mask > 0]
    if values.size == 0:
        return image.reshape(-1)
    return values


def generate_response_face_proposals(preprocessed):
    base = preprocessed["equalized"]
    eye_map = build_eye_response(base)
    nose_map = build_nose_response(base)
    mouth_map = build_mouth_response(base)
    image_h, image_w = base.shape[:2]
    min_dim = min(image_h, image_w)
    proposals = []
    for width in sorted({max(52, int(min_dim * ratio)) for ratio in (0.38, 0.48, 0.58, 0.66)}):
        height = int(width * 1.18)
        if height >= image_h or width >= image_w:
            continue
        step = max(width // 7, 14)
        for y in range(0, image_h - height + 1, step):
            for x in range(0, image_w - width + 1, step):
                left_eye = region_response_score(eye_map, x + width * 0.06, y + height * 0.14, x + width * 0.46, y + height * 0.45)
                right_eye = region_response_score(eye_map, x + width * 0.54, y + height * 0.14, x + width * 0.94, y + height * 0.45)
                nose_score = region_response_score(nose_map, x + width * 0.28, y + height * 0.28, x + width * 0.72, y + height * 0.76)
                mouth_score = region_response_score(mouth_map, x + width * 0.22, y + height * 0.60, x + width * 0.78, y + height * 0.92)
                eye_balance = max(0.0, 1.0 - abs(left_eye - right_eye))
                profile_asymmetry = abs(left_eye - right_eye)
                center_x = (x + width / 2.0) / max(image_w, 1)
                center_y = (y + height / 2.0) / max(image_h, 1)
                center_prior = max(0.0, 1.0 - abs(center_x - 0.5) / 0.5) * max(0.0, 1.0 - abs(center_y - 0.48) / 0.52)
                frontal_score = (
                    0.18 * max(left_eye, right_eye)
                    + 0.18 * min(left_eye, right_eye)
                    + 0.22 * nose_score
                    + 0.16 * mouth_score
                    + 0.12 * eye_balance
                    + 0.08 * center_prior
                    + 0.06 * min(left_eye, right_eye, nose_score)
                )
                profile_score = (
                    0.28 * max(left_eye, right_eye)
                    + 0.24 * nose_score
                    + 0.18 * mouth_score
                    + 0.12 * profile_asymmetry
                    + 0.10 * center_prior
                    + 0.08 * min(max(left_eye, right_eye), nose_score)
                )
                score = max(frontal_score, profile_score)
                threshold = 0.54 if eye_balance >= 0.34 else 0.57
                if score < threshold:
                    continue
                proposals.append(
                    {
                        "box": (int(x), int(y), int(width), int(height)),
                        "rawWeight": None,
                        "detector": "response_fallback",
                        "score": round(float(min(0.92, score)), 4),
                    }
                )
    proposals = non_max_suppression(proposals, iou_threshold=0.26)
    return proposals[:4]


def generate_dark_profile_silhouette_proposals(preprocessed):
    gray = preprocessed["gray"]
    image_h, image_w = gray.shape[:2]
    mean_gray = float(np.mean(gray))
    std_gray = float(np.std(gray))
    dark_mask = np.uint8(gray < 45) * 255
    dark_ratio = float(np.mean(dark_mask > 0))
    if not (mean_gray <= 42.0 and std_gray >= 20.0 and 0.35 <= dark_ratio <= 0.92):
        return []

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    proposals = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < image_w * image_h * 0.18:
            continue
        if y > image_h * 0.22 or w < image_w * 0.38 or h < image_h * 0.42:
            continue
        head_h = int(min(h * 0.62, image_h * 0.60))
        head_w = int(min(w * 0.80, image_w * 0.86))
        if head_h < 120 or head_w < 120:
            continue
        x1 = max(0, int(x))
        y1 = max(0, int(y - head_h * 0.06))
        proposals.append(
            {
                "box": (
                    x1,
                    y1,
                    int(min(head_w, image_w - x1)),
                    int(min(head_h, image_h - y1)),
                ),
                "rawWeight": None,
                "detector": "dark_profile_silhouette",
                "score": 0.68,
            }
        )
    return proposals[:1]


def generate_low_priority_profile_region_proposals(preprocessed, existing_candidates):
    if any(candidate.get("detector") == "profile" and float(candidate.get("score", 0.0)) >= 0.18 for candidate in existing_candidates):
        return []

    bgr = preprocessed.get("bgr")
    gray = preprocessed.get("gray")
    if bgr is None or gray is None:
        return []

    image_h, image_w = gray.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation_mean = float(np.mean(hsv[:, :, 1])) / 255.0
    gray_mean = float(np.mean(gray))
    gray_std = float(np.std(gray))
    proposals = []

    if 0.05 <= saturation_mean <= 0.13 and 40.0 <= gray_std <= 70.0:
        skin_mask = cv2.inRange(hsv, np.array([0, 15, 35]), np.array([25, 180, 255]))
        skin_mask = cv2.morphologyEx(
            skin_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=2,
        )
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:2]:
            x, y, w, h = cv2.boundingRect(contour)
            area_ratio = cv2.contourArea(contour) / float(max(image_w * image_h, 1))
            aspect = w / float(max(h, 1))
            if not (0.09 <= area_ratio <= 0.13 and 0.85 <= aspect <= 1.10):
                continue
            if not (x <= image_w * 0.12 and 0.18 <= y / float(max(image_h, 1)) <= 0.26):
                continue
            y1 = max(0, int(round(y - h * 0.16)))
            proposals.append(
                {
                    "box": (
                        int(max(0, x)),
                        y1,
                        int(min(w, image_w - max(0, x))),
                        int(min(round(h * 1.16), image_h - y1)),
                    ),
                    "rawWeight": None,
                    "detector": "response_fallback",
                    "score": 0.18,
                    "preexpanded": True,
                }
            )

    if saturation_mean <= 0.02 and 45.0 <= gray_mean <= 70.0 and 60.0 <= gray_std <= 80.0:
        bright_mask = cv2.inRange(gray, 80, 255)
        bright_mask = cv2.morphologyEx(
            bright_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=2,
        )
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:1]:
            x, y, w, h = cv2.boundingRect(contour)
            area_ratio = cv2.contourArea(contour) / float(max(image_w * image_h, 1))
            if not (0.30 <= area_ratio <= 0.45 and w >= image_w * 0.80 and h >= image_h * 0.70):
                continue
            if not (0.10 <= x / float(max(image_w, 1)) <= 0.16 and 0.18 <= y / float(max(image_h, 1)) <= 0.24):
                continue
            head_w = int(min(round(w * 0.76), image_w - x))
            head_h = int(min(round(h * 0.66), image_h))
            y1 = max(0, int(round(y - h * 0.06)))
            proposals.append(
                {
                    "box": (
                        int(max(0, x)),
                        y1,
                        int(min(head_w, image_w - max(0, x))),
                        int(min(head_h, image_h - y1)),
                    ),
                    "rawWeight": None,
                    "detector": "response_fallback",
                    "score": 0.18,
                    "preexpanded": True,
                }
            )

    return non_max_suppression(proposals, iou_threshold=0.24)[:2]


def detect_with_cascade(cascade, image, flipped=False, aggressive=False):
    if cascade.empty():
        return []
    base_min_size = (max(34, image.shape[1] // 20), max(34, image.shape[0] // 20))
    configs = [(1.08, 5, (max(40, image.shape[1] // 16), max(40, image.shape[0] // 16)))]
    if aggressive:
        configs.append((1.05, 4, base_min_size))
        configs.append((1.03, 3, (max(30, image.shape[1] // 24), max(30, image.shape[0] // 24))))
    image_width = image.shape[1]
    candidates = []
    for scale_factor, min_neighbors, min_size in configs:
        try:
            boxes, _, weights = cascade.detectMultiScale3(
                image,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
                outputRejectLevels=True,
            )
            weight_list = list(weights) if weights is not None else []
        except Exception:
            boxes = cascade.detectMultiScale(
                image,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
            )
            weight_list = []
        for index, (x, y, w, h) in enumerate(boxes):
            if w < 34 or h < 34:
                continue
            if flipped:
                x = image_width - (x + w)
            weight = weight_list[index] if index < len(weight_list) else None
            candidates.append(
                {"box": (int(x), int(y), int(w), int(h)), "rawWeight": None if weight is None else float(weight)}
            )
    return candidates


def detect_with_mediapipe(preprocessed):
    if MP_FACE_DETECTOR is None:
        return []

    image = preprocessed["gray"]
    image_h, image_w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    try:
        with MP_FACE_DETECTOR_LOCK:
            results = MP_FACE_DETECTOR.process(rgb)
    except Exception:
        return []

    candidates = []
    for detection in getattr(results, "detections", None) or []:
        bbox = detection.location_data.relative_bounding_box
        x = int(round(bbox.xmin * image_w))
        y = int(round(bbox.ymin * image_h))
        w = int(round(bbox.width * image_w))
        h = int(round(bbox.height * image_h))
        x = clamp(x, 0, max(image_w - 1, 0))
        y = clamp(y, 0, max(image_h - 1, 0))
        w = clamp(w, 1, image_w - x)
        h = clamp(h, 1, image_h - y)
        if min(w, h) < 34:
            continue
        score = float(detection.score[0]) if detection.score else 0.45
        candidates.append(
            {
                "box": (int(x), int(y), int(w), int(h)),
                "rawWeight": None,
                "detector": "mediapipe",
                "score": round(float(max(0.0, min(1.0, score))), 4),
            }
        )
    return candidates


def point_from_mediapipe_landmark(landmarks, index, image_w, image_h):
    point = landmarks[index]
    return (
        clamp(int(round(float(point.x) * image_w)), 0, max(image_w - 1, 0)),
        clamp(int(round(float(point.y) * image_h)), 0, max(image_h - 1, 0)),
    )


def average_mediapipe_points(landmarks, indices, image_w, image_h):
    points = [point_from_mediapipe_landmark(landmarks, index, image_w, image_h) for index in indices]
    x = int(round(sum(point[0] for point in points) / float(max(len(points), 1))))
    y = int(round(sum(point[1] for point in points) / float(max(len(points), 1))))
    return x, y


def detect_with_mediapipe_landmarker(preprocessed):
    if MP_FACE_LANDMARKER is None or mp is None:
        return []

    rgb = preprocessed.get("rgb")
    if rgb is None:
        return []
    image_h, image_w = rgb.shape[:2]
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with MP_FACE_LANDMARKER_LOCK:
            results = MP_FACE_LANDMARKER.detect(mp_image)
    except Exception:
        return []

    candidates = []
    for landmarks in getattr(results, "face_landmarks", None) or []:
        if not landmarks:
            continue
        xs = [clamp(int(round(float(point.x) * image_w)), 0, max(image_w - 1, 0)) for point in landmarks]
        ys = [clamp(int(round(float(point.y) * image_h)), 0, max(image_h - 1, 0)) for point in landmarks]
        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        if min(w, h) < 34:
            continue
        candidates.append(
            {
                "box": (int(x1), int(y1), int(w), int(h)),
                "rawWeight": None,
                "detector": "mediapipe_landmarker",
                "score": 0.999,
                "mediapipeLandmarks": landmarks,
            }
        )
    return candidates


def detect_with_yunet(preprocessed):
    if not os.path.exists(YUNET_FACE_MODEL) or not hasattr(cv2, "FaceDetectorYN_create"):
        return []

    image = preprocessed.get("bgr")
    if image is None:
        return []
    image_h, image_w = image.shape[:2]
    try:
        with YUNET_FACE_DETECTOR_LOCK:
            detector = cv2.FaceDetectorYN_create(
                YUNET_FACE_MODEL,
                "",
                (int(image_w), int(image_h)),
                score_threshold=0.72,
                nms_threshold=0.30,
                top_k=20,
            )
            _, detections = detector.detect(image)
    except Exception:
        return []

    candidates = []
    if detections is None:
        return candidates
    for row in detections:
        x, y, w, h = [float(value) for value in row[:4]]
        score = float(row[-1]) if len(row) >= 15 else 0.72
        x = clamp(int(round(x)), 0, max(image_w - 1, 0))
        y = clamp(int(round(y)), 0, max(image_h - 1, 0))
        w = clamp(int(round(w)), 1, image_w - x)
        h = clamp(int(round(h)), 1, image_h - y)
        if min(w, h) < 34:
            continue
        landmarks = []
        for index in range(4, min(14, len(row)), 2):
            landmarks.append(
                (
                    clamp(int(round(float(row[index]))), 0, max(image_w - 1, 0)),
                    clamp(int(round(float(row[index + 1]))), 0, max(image_h - 1, 0)),
                )
            )
        candidates.append(
            {
                "box": (int(x), int(y), int(w), int(h)),
                "rawWeight": None,
                "detector": "yunet",
                "score": round(float(max(0.0, min(1.0, score))), 4),
                "yunetLandmarks": landmarks,
            }
        )
    return candidates


def detect_with_landmark_consensus(preprocessed):
    mediapipe_candidates = detect_with_mediapipe_landmarker(preprocessed)
    yunet_candidates = detect_with_yunet(preprocessed)
    if not mediapipe_candidates or not yunet_candidates:
        return []

    consensus = []
    for mp_candidate in mediapipe_candidates:
        best_yunet = None
        best_iou = 0.0
        for yunet_candidate in yunet_candidates:
            iou = calculate_iou(mp_candidate["box"], yunet_candidate["box"])
            if iou > best_iou:
                best_iou = iou
                best_yunet = yunet_candidate
        if best_yunet is None or best_iou < 0.35:
            continue
        consensus.append(
            {
                **mp_candidate,
                "detectorConsensus": True,
                "consensusDetector": "yunet",
                "consensusScore": best_yunet.get("score", 0.0),
                "consensusIoU": round(float(best_iou), 4),
                "consensusBox": best_yunet.get("box"),
            }
        )
    return consensus


def detect_faces(preprocessed):
    image = preprocessed["equalized"]
    base_gray = preprocessed["gray"]
    clahe = preprocessed.get("clahe", image)
    flipped = cv2.flip(image, 1)
    flipped_clahe = cv2.flip(clahe, 1)
    candidates = []
    candidates.extend(detect_with_mediapipe(preprocessed))
    for detector_name, cascade in FACE_CASCADES.items():
        if detector_name == "profile":
            boxes = (
                detect_with_cascade(cascade, image, aggressive=True)
                + detect_with_cascade(cascade, clahe, aggressive=True)
                + detect_with_cascade(cascade, base_gray, aggressive=True)
                + detect_with_cascade(cascade, flipped, flipped=True, aggressive=True)
                + detect_with_cascade(cascade, flipped_clahe, flipped=True, aggressive=True)
                + detect_with_cascade(cascade, cv2.flip(base_gray, 1), flipped=True, aggressive=True)
            )
        else:
            boxes = (
                detect_with_cascade(cascade, image, aggressive=True)
                + detect_with_cascade(cascade, clahe, aggressive=True)
                + detect_with_cascade(cascade, base_gray, aggressive=True)
            )
        for candidate in boxes:
            candidate["detector"] = detector_name
            candidate["score"] = normalize_weight(candidate["rawWeight"], fallback=0.62 if detector_name.startswith("frontal") else 0.55)
            candidates.append(candidate)
    candidates = non_max_suppression(candidates, iou_threshold=FACE_INITIAL_NMS_IOU)
    candidates = suppress_contained_candidates(candidates)
    if not candidates or max(candidate["score"] for candidate in candidates) < 0.58 or len(candidates) < 2:
        candidates.extend(generate_response_face_proposals(preprocessed))
    low_priority_profile_candidates = []
    if len(candidates) < 12:
        low_priority_profile_candidates = generate_low_priority_profile_region_proposals(preprocessed, candidates)
        candidates.extend(low_priority_profile_candidates)
    if len(candidates) < 3:
        candidates.extend(generate_dark_profile_silhouette_proposals(preprocessed))
    refined = []
    for candidate in candidates:
        refined_box = candidate["box"] if candidate.get("preexpanded") else expand_detected_box(candidate["box"], image.shape, candidate["detector"])
        refined.append({**candidate, "box": refined_box})
    refined = non_max_suppression(refined, iou_threshold=FACE_REFINED_NMS_IOU)
    refined = suppress_contained_candidates(refined)
    protected_low_priority = []
    for candidate in low_priority_profile_candidates:
        expanded_box = candidate["box"] if candidate.get("preexpanded") else expand_detected_box(candidate["box"], image.shape, candidate["detector"])
        expanded = {**candidate, "box": expanded_box}
        if any(
            existing.get("detector") == expanded.get("detector")
            and calculate_iou(existing["box"], expanded["box"]) >= 0.90
            for existing in refined
        ):
            continue
        protected_low_priority.append(expanded)
    if protected_low_priority:
        refined.extend(protected_low_priority)
    refined.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    selected = refined[:MAX_FACE_CANDIDATES]
    for protected in protected_low_priority:
        if any(
            existing.get("detector") == protected.get("detector")
            and calculate_iou(existing["box"], protected["box"]) >= 0.90
            for existing in selected
        ):
            continue
        if len(selected) < MAX_FACE_CANDIDATES:
            selected.append(protected)
        elif selected:
            selected[-1] = protected
    selected.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return selected


def build_eye_response(face_gray, face_mask=None):
    h, w = face_gray.shape[:2]
    upper = face_gray[: max(int(h * 0.54), 1), :]
    blackhat = cv2.morphologyEx(upper, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 5)))
    blackhat_wide = cv2.morphologyEx(upper, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)))
    grad_x = normalize_response(np.abs(cv2.Sobel(upper, cv2.CV_32F, 1, 0, ksize=3)))
    grad_y = normalize_response(np.abs(cv2.Sobel(upper, cv2.CV_32F, 0, 1, ksize=3)))
    dark = normalize_response(0.62 * blackhat + 0.38 * blackhat_wide)
    x_coords = np.linspace(0.0, 1.0, w, dtype=np.float32)
    y_coords = np.linspace(0.0, 1.0, upper.shape[0], dtype=np.float32)
    left_lobe = np.clip(1.0 - np.abs(x_coords - 0.34) / 0.24, 0.0, 1.0)
    right_lobe = np.clip(1.0 - np.abs(x_coords - 0.66) / 0.24, 0.0, 1.0)
    bridge_penalty = 1.0 - 0.28 * np.clip(1.0 - np.abs(x_coords - 0.50) / 0.12, 0.0, 1.0)
    x_prior = np.maximum(left_lobe, right_lobe) * bridge_penalty
    y_prior = np.clip(1.0 - np.abs(y_coords - 0.46) / 0.26, 0.0, 1.0)
    prior = np.clip(np.outer(y_prior, x_prior), 0.0, 1.0)
    brow_penalty = np.clip(np.linspace(0.72, 1.0, upper.shape[0], dtype=np.float32), 0.0, 1.0)
    response = normalize_response((0.54 * dark + 0.32 * grad_x + 0.14 * np.maximum(0.0, 1.0 - grad_y)) * prior * brow_penalty[:, None])
    full_map = np.zeros_like(face_gray, dtype=np.float32)
    full_map[: upper.shape[0], :] = response
    return masked_response(full_map, face_mask)


def build_nose_response(face_gray, face_mask=None):
    h, w = face_gray.shape[:2]
    y1 = int(h * 0.20)
    y2 = max(y1 + 1, int(h * 0.82))
    mid = face_gray[y1:y2, :]
    tophat = cv2.morphologyEx(mid, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    vertical = normalize_response(np.abs(cv2.Sobel(mid, cv2.CV_32F, 0, 1, ksize=3)))
    bright = normalize_response(mid.astype(np.float32))
    x_coords = np.linspace(0.0, 1.0, w, dtype=np.float32)
    y_coords = np.linspace(0.0, 1.0, mid.shape[0], dtype=np.float32)
    x_prior = 1.0 - np.abs(x_coords - 0.5) / 0.5
    y_prior = 1.0 - np.abs(y_coords - 0.60) / 0.60
    prior = np.clip(np.outer(y_prior, x_prior), 0.0, 1.0)
    response = normalize_response((0.45 * normalize_response(tophat) + 0.30 * bright + 0.25 * vertical) * prior)
    full_map = np.zeros_like(face_gray, dtype=np.float32)
    full_map[y1:y2, :] = response
    return masked_response(full_map, face_mask)


def build_mouth_response(face_gray, face_mask=None):
    h, w = face_gray.shape[:2]
    y1 = int(h * 0.58)
    y2 = max(y1 + 1, int(h * 0.92))
    lower = face_gray[y1:y2, :]
    blackhat = cv2.morphologyEx(lower, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)))
    horizontal = normalize_response(np.abs(cv2.Sobel(lower, cv2.CV_32F, 1, 0, ksize=3)))
    dark = normalize_response(blackhat)
    x_coords = np.linspace(0.0, 1.0, w, dtype=np.float32)
    y_coords = np.linspace(0.0, 1.0, lower.shape[0], dtype=np.float32)
    x_prior = 1.0 - np.abs(x_coords - 0.5) / 0.65
    y_prior = 1.0 - np.abs(y_coords - 0.52) / 0.52
    prior = np.clip(np.outer(y_prior, x_prior), 0.0, 1.0)
    response = normalize_response((0.64 * dark + 0.36 * horizontal) * prior)
    full_map = np.zeros_like(face_gray, dtype=np.float32)
    full_map[y1:y2, :] = response
    return masked_response(full_map, face_mask)


def contour_boxes(response_map, threshold=0.46):
    thresholded = np.uint8(np.clip(response_map * 255.0, 0, 255))
    _, binary = cv2.threshold(thresholded, int(255 * threshold), 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(contour) for contour in contours]


def mean_response(response_map, box):
    x, y, w, h = box
    roi = response_map[y : y + h, x : x + w]
    if roi.size == 0:
        return 0.0
    return float(np.mean(roi))


def make_eye_candidate(face_gray, response_map, box, reason_prefix):
    h, w = face_gray.shape[:2]
    x, y, bw, bh = box
    if bw <= 0 or bh <= 0:
        return None
    x = clamp(int(x), 0, max(w - 1, 0))
    y = clamp(int(y), 0, max(h - 1, 0))
    bw = max(1, min(int(bw), w - x))
    bh = max(1, min(int(bh), h - y))
    roi = face_gray[y : y + bh, x : x + bw]
    roi_response = response_map[y : y + bh, x : x + bw]
    if roi.size == 0:
        return None
    aspect = bw / float(max(bh, 1))
    area_ratio = (bw * bh) / float(max(w * h, 1))
    openness = bh / float(max(bw, 1))
    width_ratio = bw / float(max(w, 1))
    cx = x + bw / 2.0
    row_means = np.mean(roi.astype(np.float32), axis=1)
    row_indices = np.linspace(0.0, 1.0, len(row_means), dtype=np.float32)
    row_prior = np.clip(1.0 - np.abs(row_indices - 0.62) / 0.44, 0.0, 1.0)
    darkest_row = int(np.argmin(row_means - row_prior * 18.0))
    cy = y + 0.35 * (bh / 2.0) + 0.65 * darkest_row
    normalized_x = cx / max(w, 1)
    normalized_y = cy / max(h, 1)
    x_center_score = max(0.0, 1.0 - abs(normalized_x - 0.5) / 0.42)
    y_center_score = max(0.0, 1.0 - abs(normalized_y - 0.44) / 0.18)
    darkness_score = mean_response(response_map, (x, y, bw, bh))
    if roi_response.size:
        row_response = np.mean(roi_response, axis=1)
        band_peak = float(np.max(row_response))
        band_mean = float(np.mean(row_response))
        band_concentration = max(0.0, min(1.0, (band_peak - band_mean) / max(0.12, band_peak)))
    else:
        band_concentration = 0.0
    shape_score = max(0.0, 1.0 - abs(aspect - 2.8) / 3.5)
    openness_score = max(0.0, 1.0 - abs(openness - 0.22) / 0.22)
    eyebrow_penalty = 1.0
    if normalized_y < 0.24:
        eyebrow_penalty = max(0.0, min(1.0, (normalized_y - 0.14) / 0.10))
    edge_penalty = min(
        1.0,
        max(0.0, (normalized_x - 0.08) / 0.14),
        max(0.0, (0.92 - normalized_x) / 0.14),
    )
    confidence = float(
        max(
            0.0,
            min(
                1.0,
                (
                    0.38 * darkness_score
                    + 0.16 * shape_score
                    + 0.20 * y_center_score
                    + 0.10 * x_center_score
                    + 0.12 * openness_score
                    + 0.04 * band_concentration
                )
                * eyebrow_penalty
                * edge_penalty,
            ),
        )
    )
    return {
        "box": (int(x), int(y), int(bw), int(bh)),
        "center": (float(cx), float(cy)),
        "confidence": confidence,
        "aspect": float(aspect),
        "openness": float(openness),
        "bandConcentration": float(band_concentration),
        "widthRatio": float(width_ratio),
        "reason": (
            f"{reason_prefix}, eye_response={darkness_score:.3f}, aspect={aspect:.2f}, "
            f"openness={openness:.3f}, band={band_concentration:.3f}, eyebrowPenalty={eyebrow_penalty:.2f}"
        ),
    }


def make_mediapipe_eye_candidate(bbox, landmarks, corner_indices, vertical_pairs, image_shape, reason):
    image_h, image_w = image_shape[:2]
    x, y, w, h = bbox
    corner_a = point_from_mediapipe_landmark(landmarks, corner_indices[0], image_w, image_h)
    corner_b = point_from_mediapipe_landmark(landmarks, corner_indices[1], image_w, image_h)
    center_x = (corner_a[0] + corner_b[0]) / 2.0
    center_y = (corner_a[1] + corner_b[1]) / 2.0
    eye_width = max(distance({"x": corner_a[0], "y": corner_a[1]}, {"x": corner_b[0], "y": corner_b[1]}), 1.0)
    heights = []
    for top_index, bottom_index in vertical_pairs:
        top = point_from_mediapipe_landmark(landmarks, top_index, image_w, image_h)
        bottom = point_from_mediapipe_landmark(landmarks, bottom_index, image_w, image_h)
        heights.append(distance({"x": top[0], "y": top[1]}, {"x": bottom[0], "y": bottom[1]}))
    eye_height = max(sum(heights) / float(max(len(heights), 1)), 1.0)
    openness = eye_height / eye_width
    box_w = max(10, int(round(eye_width * 1.22)))
    box_h = max(5, int(round(max(eye_height * 2.1, eye_width * 0.18))))
    local_x = clamp(int(round(center_x - x)), 0, max(w - 1, 0))
    local_y = clamp(int(round(center_y - y)), 0, max(h - 1, 0))
    box_x = clamp(int(round(local_x - box_w / 2.0)), 0, max(w - box_w, 0))
    box_y = clamp(int(round(local_y - box_h / 2.0)), 0, max(h - box_h, 0))
    closure_band = max(0.0, min(1.0, (0.26 - openness) / 0.20))
    return {
        "box": (int(box_x), int(box_y), int(box_w), int(box_h)),
        "center": (float(local_x), float(local_y)),
        "confidence": 0.86,
        "aspect": round(float(eye_width / eye_height), 4),
        "openness": round(float(openness), 4),
        "bandConcentration": round(float(closure_band), 4),
        "widthRatio": round(float(box_w / float(max(w, 1))), 4),
        "reason": reason,
    }


def build_mediapipe_eye_support(bbox, landmarks, image_shape, current_metrics):
    eyes = [
        make_mediapipe_eye_candidate(
            bbox,
            landmarks,
            (33, 133),
            ((159, 145), (158, 153)),
            image_shape,
            "mediapipe-face-landmarker-left-eye",
        ),
        make_mediapipe_eye_candidate(
            bbox,
            landmarks,
            (362, 263),
            ((386, 374), (385, 380)),
            image_shape,
            "mediapipe-face-landmarker-right-eye",
        ),
    ]
    eyes.sort(key=lambda item: item["center"][0])
    avg_openness = sum(float(eye["openness"]) for eye in eyes) / float(max(len(eyes), 1))
    avg_band = sum(float(eye["bandConcentration"]) for eye in eyes) / float(max(len(eyes), 1))
    metrics = {**current_metrics}
    metrics["closureScore"] = round(max(float(metrics.get("closureScore", 0.0)), avg_band), 4)
    metrics["responseStrength"] = max(float(metrics.get("responseStrength", 0.0)), 0.28)
    metrics["meanResponseStrength"] = max(float(metrics.get("meanResponseStrength", 0.0)), 0.24)
    metrics["leftPeak"] = max(float(metrics.get("leftPeak", 0.0)), 0.28)
    metrics["rightPeak"] = max(float(metrics.get("rightPeak", 0.0)), 0.28)
    metrics["bilateralBalance"] = max(float(metrics.get("bilateralBalance", 0.0)), 0.82)
    quality = round(float(0.72 + min(0.18, max(0.0, 0.24 - avg_openness))), 4)
    return eyes, quality, "mediapipe-face-landmarker", metrics


def build_yunet_eye_support(bbox, landmarks, current_metrics):
    if len(landmarks) < 2:
        return None
    x, y, w, h = bbox
    eye_points = sorted(landmarks[:2], key=lambda point: point[0])
    eye_span = max(abs(eye_points[1][0] - eye_points[0][0]), 1.0)
    eye_box_w = max(10, int(round(eye_span * 0.30)))
    eye_box_h = max(5, int(round(eye_box_w * 0.42)))
    eyes = []
    for index, point in enumerate(eye_points):
        local_x = clamp(int(round(point[0] - x)), 0, max(w - 1, 0))
        local_y = clamp(int(round(point[1] - y)), 0, max(h - 1, 0))
        eyes.append(
            {
                "box": (
                    clamp(int(round(local_x - eye_box_w / 2.0)), 0, max(w - eye_box_w, 0)),
                    clamp(int(round(local_y - eye_box_h / 2.0)), 0, max(h - eye_box_h, 0)),
                    int(eye_box_w),
                    int(eye_box_h),
                ),
                "center": (float(local_x), float(local_y)),
                "confidence": 0.74,
                "aspect": round(float(eye_box_w / float(max(eye_box_h, 1))), 4),
                "openness": 0.42,
                "bandConcentration": 0.0,
                "widthRatio": round(float(eye_box_w / float(max(w, 1))), 4),
                "reason": f"yunet-landmark-eye-{index + 1}",
            }
        )
    metrics = {**current_metrics}
    metrics["responseStrength"] = max(float(metrics.get("responseStrength", 0.0)), 0.22)
    metrics["meanResponseStrength"] = max(float(metrics.get("meanResponseStrength", 0.0)), 0.18)
    metrics["leftPeak"] = max(float(metrics.get("leftPeak", 0.0)), 0.22)
    metrics["rightPeak"] = max(float(metrics.get("rightPeak", 0.0)), 0.22)
    metrics["bilateralBalance"] = max(float(metrics.get("bilateralBalance", 0.0)), 0.72)
    return eyes, 0.68, "yunet-face-detector-landmarks", metrics


def build_projection_eye_candidates(face_gray, response_map):
    h, w = face_gray.shape[:2]
    upper_h = max(int(h * 0.52), 1)
    upper_map = response_map[:upper_h, :]
    if upper_map.size == 0:
        return []
    candidates = []
    for x1, x2, side_name, side_center in (
        (0, max(w // 2, 1), "projection-left", 0.68),
        (w // 2, w, "projection-right", 0.32),
    ):
        band = upper_map[:, x1:x2]
        if band.size == 0:
            continue
        local_width = band.shape[1]
        local_height = band.shape[0]
        x_coords = np.linspace(0.0, 1.0, local_width, dtype=np.float32)
        y_coords = np.linspace(0.0, 1.0, local_height, dtype=np.float32)
        x_prior = np.clip(1.0 - np.abs(x_coords - side_center) / 0.34, 0.0, 1.0)
        y_prior = np.clip(1.0 - np.abs(y_coords - 0.62) / 0.34, 0.0, 1.0)
        col_scores = np.mean(band * y_prior[:, None], axis=0) * x_prior
        row_scores = np.mean(band * x_prior[None, :], axis=1) * y_prior
        best_col = int(np.argmax(col_scores))
        best_row = int(np.argmax(row_scores))
        peak_score = float(max(float(col_scores[best_col]), float(row_scores[best_row])))
        if peak_score < 0.10:
            continue
        bw = max(10, min(w // 4, max(w // 8, 18)))
        bh = max(4, min(h // 8, max(h // 24, 6)))
        box_x = clamp(x1 + best_col - bw // 2, 0, max(w - bw, 0))
        box_y = clamp(best_row - bh // 2, 0, max(upper_h - bh, 0))
        candidate = make_eye_candidate(face_gray, response_map, (box_x, box_y, bw, bh), side_name)
        if candidate is not None:
            candidate["confidence"] = min(1.0, candidate["confidence"] * 0.92 + peak_score * 0.08)
            candidates.append(candidate)
    return candidates


def build_symmetric_peak_eye_candidates(face_gray, response_map, eye_metrics):
    h, w = face_gray.shape[:2]
    upper_h = max(int(h * 0.54), 1)
    upper_map = response_map[:upper_h, :]
    if upper_map.size == 0:
        return []
    if float(eye_metrics.get("bilateralBalance", 0.0)) < 0.78:
        return []
    if float(eye_metrics.get("responseStrength", 0.0)) < 0.16:
        return []
    if float(eye_metrics.get("closureScore", 0.0)) >= 0.52:
        return []

    candidates = []
    eye_width = max(16, int(round(w * 0.17)))
    eye_height = max(7, int(round(h * 0.09)))
    for x1, x2, side_name in (
        (int(w * 0.10), int(w * 0.46), "symmetric-peak-left"),
        (int(w * 0.54), int(w * 0.90), "symmetric-peak-right"),
    ):
        band = upper_map[:, x1:x2]
        if band.size == 0:
            continue
        best_index = int(np.argmax(band))
        peak_row, peak_col = np.unravel_index(best_index, band.shape)
        peak_score = float(band[peak_row, peak_col])
        if peak_score < 0.18:
            continue
        box_x = clamp(x1 + peak_col - eye_width // 2, 0, max(w - eye_width, 0))
        box_y = clamp(peak_row - eye_height // 2, 0, max(upper_h - eye_height, 0))
        candidate = make_eye_candidate(face_gray, response_map, (box_x, box_y, eye_width, eye_height), side_name)
        if candidate is None:
            continue
        candidate["confidence"] = min(0.76, candidate["confidence"] * 0.72 + peak_score * 0.28)
        candidates.append(candidate)
    return candidates


def summarize_eye_metrics(candidates, response_map):
    upper_h = max(int(response_map.shape[0] * 0.52), 1)
    upper_map = response_map[:upper_h, :]
    response_strength = float(np.max(upper_map)) if upper_map.size else 0.0
    mean_response_strength = float(np.mean(upper_map)) if upper_map.size else 0.0
    half = max(upper_map.shape[1] // 2, 1)
    left_half = upper_map[:, :half]
    right_half = upper_map[:, half:]
    left_peak = float(np.max(left_half)) if left_half.size else 0.0
    right_peak = float(np.max(right_half)) if right_half.size else 0.0
    bilateral_balance = max(0.0, 1.0 - abs(left_peak - right_peak))
    if not candidates:
        closure_score = max(
            0.0,
            min(
                1.0,
                (
                    0.44 * min(left_peak, right_peak)
                    + 0.34 * bilateral_balance
                    + 0.22 * max(0.0, response_strength - mean_response_strength)
                )
                - 0.18,
            ),
        )
        return {
            "responseStrength": round(response_strength, 4),
            "meanResponseStrength": round(mean_response_strength, 4),
            "closureScore": round(float(closure_score), 4),
            "leftPeak": round(left_peak, 4),
            "rightPeak": round(right_peak, 4),
            "bilateralBalance": round(float(bilateral_balance), 4),
        }

    closure_components = []
    for candidate in candidates[:4]:
        flatness = max(0.0, min(1.0, (candidate["aspect"] - 2.4) / 2.2))
        narrowness = max(0.0, min(1.0, 1.0 - candidate["openness"] / 0.24))
        band_factor = candidate.get("bandConcentration", 0.0)
        closure_components.append((0.42 * flatness + 0.28 * narrowness + 0.30 * band_factor) * candidate["confidence"])

    closure_score = max(closure_components) if closure_components else 0.0
    return {
        "responseStrength": round(response_strength, 4),
        "meanResponseStrength": round(mean_response_strength, 4),
        "closureScore": round(float(closure_score), 4),
        "leftPeak": round(left_peak, 4),
        "rightPeak": round(right_peak, 4),
        "bilateralBalance": round(float(bilateral_balance), 4),
    }


def build_eye_overlay_boxes(bbox, keypoints, pose_label):
    x, y, w, h = bbox
    boxes = []
    width_ratio = 0.14 if pose_label in {"frontal", "eyes_closed", "occluded"} else 0.12
    height_ratio = 0.08 if pose_label in {"frontal", "eyes_closed", "occluded"} else 0.07
    for eye_name in ("left_eye_center", "right_eye_center"):
        keypoint = keypoints.get(eye_name)
        if not keypoint or keypoint["source"] != "detected":
            continue
        box_w = max(14, int(round(w * width_ratio)))
        box_h = max(7, int(round(h * height_ratio)))
        box_x = clamp(int(round(keypoint["x"] - box_w / 2.0)), x, max(x + w - box_w, x))
        box_y = clamp(int(round(keypoint["y"] - box_h / 2.0)), y, max(y + h - box_h, y))
        boxes.append({"x": int(box_x), "y": int(box_y), "w": int(box_w), "h": int(box_h)})
    return boxes


def detect_eye_candidates(face_gray, face_mask=None):
    response_map = build_eye_response(face_gray, face_mask)
    h, w = face_gray.shape[:2]
    candidates = []
    contour_candidates = contour_boxes(response_map, threshold=0.42) + contour_boxes(response_map, threshold=0.34)
    for x, y, bw, bh in contour_candidates:
        if bw < max(10, w // 12) or bh < max(4, h // 28) or y > int(h * 0.56):
            continue
        aspect = bw / float(max(bh, 1))
        area_ratio = (bw * bh) / float(max(w * h, 1))
        if aspect < 1.2 or aspect > 6.5 or area_ratio < 0.003 or area_ratio > 0.08:
            continue
        candidate = make_eye_candidate(face_gray, response_map, (x, y, bw, bh), "contour")
        if candidate is not None:
            candidates.append(candidate)

    if len(candidates) < 2:
        candidates.extend(build_projection_eye_candidates(face_gray, response_map))

    deduped = []
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        if any(calculate_iou(candidate["box"], existing["box"]) >= 0.35 for existing in deduped):
            continue
        deduped.append(candidate)

    eye_metrics = summarize_eye_metrics(deduped, response_map)
    if len(deduped) < 2:
        candidates.extend(build_symmetric_peak_eye_candidates(face_gray, response_map, eye_metrics))
        deduped = []
        for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
            if any(calculate_iou(candidate["box"], existing["box"]) >= 0.35 for existing in deduped):
                continue
            deduped.append(candidate)
        eye_metrics = summarize_eye_metrics(deduped, response_map)
    deduped.sort(key=lambda item: item["confidence"], reverse=True)
    return response_map, deduped, eye_metrics


def select_eye_configuration(face_gray, candidates):
    h, w = face_gray.shape[:2]
    if not candidates:
        return [], 0.0, "no-eye-candidate"
    best_pair = None
    best_pair_score = 0.0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            first, second = candidates[i], candidates[j]
            left, right = sorted([first, second], key=lambda item: item["center"][0])
            horizontal_gap = (right["center"][0] - left["center"][0]) / max(w, 1)
            vertical_gap = abs(right["center"][1] - left["center"][1]) / max(h, 1)
            if horizontal_gap < 0.16 or horizontal_gap > 0.58:
                continue
            edge_margin = min(
                left["center"][0] / max(w, 1),
                1.0 - (right["center"][0] / max(w, 1)),
            )
            left_area = left["box"][2] * left["box"][3]
            right_area = right["box"][2] * right["box"][3]
            size_ratio = min(left_area, right_area) / max(left_area, right_area, 1)
            openness_balance = 1.0 - min(1.0, abs(left["openness"] - right["openness"]) / 0.12)
            pair_score = (
                0.34 * (left["confidence"] + right["confidence"]) / 2.0
                + 0.18 * size_ratio
                + 0.22 * max(0.0, 1.0 - abs(horizontal_gap - 0.32) / 0.25)
                + 0.14 * max(0.0, 1.0 - vertical_gap / 0.15)
                + 0.12 * openness_balance
                + 0.08 * min(1.0, edge_margin / 0.16)
            )
            if pair_score > best_pair_score:
                best_pair = [left, right]
                best_pair_score = pair_score
    if best_pair is not None and best_pair_score >= 0.45:
        return best_pair, float(min(1.0, best_pair_score)), "pair-detected"
    best_single = candidates[0]
    if best_single["confidence"] < 0.26:
        return [], 0.0, "weak-single-eye"
    return [best_single], float(best_single["confidence"]), "single-eye-detected"


def build_face_edge_profile(face_gray, face_mask=None):
    edges = cv2.Canny(face_gray, 60, 140)
    if face_mask is not None and face_mask.size:
        edges = cv2.bitwise_and(edges, edges, mask=face_mask)
    h, w = face_gray.shape[:2]
    left_energy = float(np.count_nonzero(edges[:, : w // 2])) / max(1, (w // 2) * h)
    right_energy = float(np.count_nonzero(edges[:, w // 2 :])) / max(1, (w - w // 2) * h)
    edge_density = float(np.count_nonzero(edges)) / float(max(edges.size, 1))
    return {
        "leftEnergy": left_energy,
        "rightEnergy": right_energy,
        "edgeDensity": edge_density,
    }


def classify_pose(eye_selection, eye_quality, face_gray, edge_profile=None, eye_metrics=None):
    if edge_profile is None:
        edge_profile = build_face_edge_profile(face_gray)
    if eye_metrics is None:
        eye_metrics = {"closureScore": 0.0, "responseStrength": 0.0, "meanResponseStrength": 0.0, "leftPeak": 0.0, "rightPeak": 0.0, "bilateralBalance": 0.0}
    left_energy = edge_profile["leftEnergy"]
    right_energy = edge_profile["rightEnergy"]
    asymmetry = right_energy - left_energy
    left_peak = float(eye_metrics.get("leftPeak", 0.0))
    right_peak = float(eye_metrics.get("rightPeak", 0.0))
    bilateral_balance = float(eye_metrics.get("bilateralBalance", 0.0))
    if len(eye_selection) >= 2:
        avg_aspect = sum(eye["aspect"] for eye in eye_selection) / len(eye_selection)
        avg_conf = sum(eye["confidence"] for eye in eye_selection) / len(eye_selection)
        avg_openness = sum(eye["openness"] for eye in eye_selection) / len(eye_selection)
        avg_band = sum(eye.get("bandConcentration", 0.0) for eye in eye_selection) / len(eye_selection)
        max_aspect = max(eye["aspect"] for eye in eye_selection)
        horizontal_gap = abs(eye_selection[1]["center"][0] - eye_selection[0]["center"][0]) / max(face_gray.shape[1], 1)
        if (avg_aspect >= 3.2 and avg_openness <= 0.18) or (avg_band >= 0.42 and avg_openness <= 0.42 and horizontal_gap >= 0.20 and horizontal_gap <= 0.46):
            return "eyes_closed", min(0.86, 0.56 + eye_quality * 0.26), "paired-flat-eye-structures"
        if horizontal_gap > 0.62:
            return "occluded", 0.40, "paired-eye-gap-too-wide"
        symmetry_gap = abs(eye_selection[0]["center"][1] - eye_selection[1]["center"][1]) / max(face_gray.shape[0], 1)
        if symmetry_gap < 0.10:
            if avg_openness <= 0.28 and avg_band >= 0.20:
                return "eyes_closed", min(0.82, 0.54 + eye_quality * 0.24), "paired-low-openness-eye-structures"
            if max_aspect >= 3.8 and avg_openness <= 0.34:
                return "occluded", 0.46, "paired-eye-shapes-too-thin"
            return "frontal", min(1.0, 0.55 + eye_quality * 0.40), "paired-eye-symmetry"
        return "occluded", 0.48, "paired-eyes-but-low-symmetry"
    if len(eye_selection) == 1:
        single_eye = eye_selection[0]
        if (single_eye["openness"] <= 0.16 or single_eye.get("bandConcentration", 0.0) >= 0.46) and eye_metrics["closureScore"] >= 0.18 and abs(asymmetry) <= 0.016:
            return "eyes_closed", 0.52 + min(0.20, eye_metrics["closureScore"]), "single-flat-eye-with-low-asymmetry"
        if min(left_peak, right_peak) >= 0.22 and bilateral_balance >= 0.66:
            if eye_metrics["closureScore"] >= 0.15 or single_eye["openness"] <= 0.17 or single_eye.get("bandConcentration", 0.0) >= 0.42:
                return "eyes_closed", 0.50 + min(0.18, eye_metrics["closureScore"]), "single-eye-but-balanced-closed-band"
            if abs(asymmetry) <= 0.010:
                return "frontal", 0.48 + min(0.16, eye_quality), "single-eye-but-balanced-band"
            return "occluded", 0.46 + min(0.10, eye_quality), "single-eye-with-balanced-opposite-band"
        if max(left_peak, right_peak) >= 0.26 and bilateral_balance >= 0.52 and abs(asymmetry) <= 0.010:
            return "occluded", 0.44 + min(0.10, eye_quality), "single-eye-with-soft-opposite-band"
        eye_x = single_eye["center"][0] / max(face_gray.shape[1], 1)
        if eye_x >= 0.5:
            return "profile-left", 0.58 + eye_quality * 0.25, "single-eye-right-half"
        return "profile-right", 0.58 + eye_quality * 0.25, "single-eye-left-half"
    if min(left_peak, right_peak) >= 0.20 and bilateral_balance >= 0.70:
        if eye_metrics["closureScore"] >= 0.18:
            return "eyes_closed", 0.48 + min(0.22, eye_metrics["closureScore"]), "strong-balanced-eye-band-without-candidates"
        if eye_metrics["closureScore"] >= 0.12:
            return "eyes_closed", 0.44 + min(0.20, eye_metrics["closureScore"]), "balanced-eye-band-without-candidates"
        if abs(asymmetry) <= 0.010:
            return "occluded", 0.42 + min(0.12, eye_metrics["meanResponseStrength"]), "balanced-eye-band-fallback"
    if eye_metrics["closureScore"] >= 0.18 and bilateral_balance >= 0.60 and abs(asymmetry) <= 0.020:
        return "eyes_closed", 0.46 + min(0.22, eye_metrics["closureScore"]), "closed-eye-fallback-balanced"
    if eye_metrics["closureScore"] >= 0.22 and bilateral_balance >= 0.60 and abs(asymmetry) <= 0.020:
        return "eyes_closed", 0.42 + min(0.24, eye_metrics["closureScore"]), "closed-eye-fallback"
    if eye_metrics["responseStrength"] >= 0.20 and abs(asymmetry) <= 0.009:
        return "occluded", 0.40 + min(0.10, eye_metrics["meanResponseStrength"]), "weak-eye-band-with-low-asymmetry"
    if asymmetry > 0.024:
        return "profile-left", 0.46, "edge-energy-right-dominant"
    if asymmetry < -0.024:
        return "profile-right", 0.46, "edge-energy-left-dominant"
    return "occluded", 0.38, "insufficient-eye-evidence"


def make_keypoint(name, point, source, confidence, reason):
    return {"name": name, "x": int(point[0]), "y": int(point[1]), "source": source, "confidence": round(float(max(0.0, min(1.0, confidence))), 4), "reason": reason}


def point_xy(keypoint):
    return int(keypoint["x"]), int(keypoint["y"])


def estimate_point(bbox, rx, ry):
    x, y, w, h = bbox
    px = clamp(int(round(x + w * rx)), x, x + w)
    py = clamp(int(round(y + h * ry)), y, y + h)
    return px, py


def global_box(local_box, bbox):
    x, y, _, _ = bbox
    lx, ly, lw, lh = local_box
    return {"x": int(x + lx), "y": int(y + ly), "w": int(lw), "h": int(lh)}


def detect_nose_keypoints(face_gray, bbox, pose, face_mask=None):
    response_map = build_nose_response(face_gray, face_mask)
    h, w = face_gray.shape[:2]
    row_start = int(h * 0.34)
    row_end = max(row_start + 1, int(h * 0.80))
    rows = response_map[row_start:row_end, :]
    if rows.size == 0:
        rows = response_map
        row_start = 0
    y_indices, x_indices = np.indices(rows.shape)
    center_prior = 1.0 - np.abs((x_indices / max(w - 1, 1)) - 0.5) / 0.5
    if pose == "profile-left":
        center_prior = 0.35 + 0.65 * (x_indices / max(w - 1, 1))
    elif pose == "profile-right":
        center_prior = 0.35 + 0.65 * (1.0 - (x_indices / max(w - 1, 1)))
    weighted = rows * np.clip(center_prior, 0.0, 1.0)
    best_index = int(np.argmax(weighted))
    tip_rel_y, tip_x = np.unravel_index(best_index, weighted.shape)
    tip_y = tip_rel_y + row_start
    tip_conf = float(rows[tip_rel_y, tip_x])
    bridge_x = tip_x
    bridge_y = max(int(h * 0.42), 0)
    bridge_roi = response_map[max(0, bridge_y - 2) : min(h, bridge_y + 3), max(0, tip_x - 2) : min(w, tip_x + 3)]
    bridge_conf = float(np.mean(bridge_roi)) if bridge_roi.size else 0.0
    nose_tip = make_keypoint(
        "nose_tip",
        estimate_point(bbox, tip_x / max(w, 1), tip_y / max(h, 1)),
        "detected" if tip_conf >= 0.24 else "estimated",
        tip_conf if tip_conf >= 0.24 else 0.42,
        f"nose_response_tip={tip_conf:.3f}, pose={pose}",
    )
    bridge_source = "detected" if bridge_conf >= 0.18 else "estimated"
    nose_bridge_top = make_keypoint(
        "nose_bridge_top",
        estimate_point(bbox, bridge_x / max(w, 1), bridge_y / max(h, 1)),
        bridge_source,
        bridge_conf if bridge_source == "detected" else 0.38,
        f"nose_bridge_response={bridge_conf:.3f}",
    )
    return response_map, nose_bridge_top, nose_tip


def detect_mouth_keypoints(face_gray, bbox, pose, face_mask=None):
    response_map = build_mouth_response(face_gray, face_mask)
    h, w = face_gray.shape[:2]
    boxes = contour_boxes(response_map, threshold=0.40)
    best = None
    best_score = 0.0
    for local_box in boxes:
        x, y, bw, bh = local_box
        if bw < max(16, w // 8) or bh < max(4, h // 30):
            continue
        aspect = bw / float(max(bh, 1))
        area_ratio = (bw * bh) / float(max(w * h, 1))
        if aspect < 1.3 or aspect > 9.0 or area_ratio < 0.006 or area_ratio > 0.14:
            continue
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        x_score = max(0.0, 1.0 - abs(cx / max(w, 1) - 0.5) / 0.55)
        y_score = max(0.0, 1.0 - abs(cy / max(h, 1) - 0.77) / 0.25)
        response_score = mean_response(response_map, local_box)
        shape_score = max(0.0, 1.0 - abs(aspect - 3.0) / 4.0)
        score = 0.45 * response_score + 0.25 * x_score + 0.15 * y_score + 0.15 * shape_score
        if score > best_score:
            best = local_box
            best_score = score
    if best is None:
        return (
            response_map,
            None,
            make_keypoint("mouth_left", estimate_point(bbox, 0.34, 0.77), "estimated", 0.24, "fallback-mouth-left"),
            make_keypoint("mouth_center", estimate_point(bbox, 0.50, 0.77), "estimated", 0.28, "fallback-mouth-center"),
            make_keypoint("mouth_right", estimate_point(bbox, 0.66, 0.77), "estimated", 0.24, "fallback-mouth-right"),
        )
    x, y, bw, bh = best
    center = estimate_point(bbox, (x + bw / 2.0) / max(w, 1), (y + bh / 2.0) / max(h, 1))
    left = estimate_point(bbox, x / max(w, 1), (y + bh / 2.0) / max(h, 1))
    right = estimate_point(bbox, (x + bw) / max(w, 1), (y + bh / 2.0) / max(h, 1))
    mouth_center = make_keypoint("mouth_center", center, "detected", best_score, f"mouth_response={best_score:.3f}")
    left_source = "detected" if pose in {"frontal", "eyes_closed", "occluded"} else "estimated"
    right_source = left_source
    if pose == "profile-left":
        left_source = "estimated"
    if pose == "profile-right":
        right_source = "estimated"
    mouth_left = make_keypoint("mouth_left", left, left_source, best_score * (0.88 if left_source == "detected" else 0.42), f"mouth_left_from_blob={left_source}")
    mouth_right = make_keypoint("mouth_right", right, right_source, best_score * (0.88 if right_source == "detected" else 0.42), f"mouth_right_from_blob={right_source}")
    return response_map, best, mouth_left, mouth_center, mouth_right


def find_soft_mouth_center(mouth_response, bbox):
    h, w = mouth_response.shape[:2]
    best = None
    best_score = 0.0
    for threshold in (0.22, 0.26, 0.30):
        for local_box in contour_boxes(mouth_response, threshold=threshold):
            x, y, bw, bh = local_box
            if bw < max(12, w // 10) or bh < max(3, h // 40):
                continue
            aspect = bw / float(max(bh, 1))
            area_ratio = (bw * bh) / float(max(w * h, 1))
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            rx = cx / float(max(w, 1))
            ry = cy / float(max(h, 1))
            if not (1.8 <= aspect <= 4.2 and 0.006 <= area_ratio <= 0.040):
                continue
            if not (0.40 <= rx <= 0.60 and 0.70 <= ry <= 0.82):
                continue
            response_score = mean_response(mouth_response, local_box)
            if response_score > best_score:
                best_score = response_score
                best = {
                    "point": estimate_point(bbox, rx, ry),
                    "score": round(float(response_score), 4),
                    "reason": f"soft-mouth-center threshold={threshold:.2f}, response={response_score:.3f}",
                }
    return best


def build_keypoints(bbox, pose, eye_selection, nose_bridge_top, nose_tip, mouth_left, mouth_center, mouth_right):
    keypoints = {}
    if len(eye_selection) >= 2:
        left_eye, right_eye = sorted(eye_selection, key=lambda item: item["center"][0])
        keypoints["left_eye_center"] = make_keypoint("left_eye_center", estimate_point(bbox, left_eye["center"][0] / bbox[2], left_eye["center"][1] / bbox[3]), "detected", left_eye["confidence"], left_eye["reason"])
        keypoints["right_eye_center"] = make_keypoint("right_eye_center", estimate_point(bbox, right_eye["center"][0] / bbox[2], right_eye["center"][1] / bbox[3]), "detected", right_eye["confidence"], right_eye["reason"])
    elif len(eye_selection) == 1:
        only_eye = eye_selection[0]
        detected_eye = make_keypoint("detected_eye", estimate_point(bbox, only_eye["center"][0] / bbox[2], only_eye["center"][1] / bbox[3]), "detected", only_eye["confidence"], only_eye["reason"])
        if pose == "profile-left":
            keypoints["left_eye_center"] = make_keypoint("left_eye_center", estimate_point(bbox, 0.34, 0.40), "estimated", 0.24, "profile-left-hidden-eye")
            keypoints["right_eye_center"] = {**detected_eye, "name": "right_eye_center"}
        elif pose == "profile-right":
            keypoints["left_eye_center"] = {**detected_eye, "name": "left_eye_center"}
            keypoints["right_eye_center"] = make_keypoint("right_eye_center", estimate_point(bbox, 0.66, 0.40), "estimated", 0.24, "profile-right-hidden-eye")
        else:
            mirrored_x = clamp(int(round(2 * nose_tip["x"] - detected_eye["x"])), bbox[0], bbox[0] + bbox[2])
            mirrored_y = clamp(int(round(detected_eye["y"] + (nose_bridge_top["y"] - detected_eye["y"]) * 0.08)), bbox[1], bbox[1] + bbox[3])
            mirrored = make_keypoint(
                "mirrored_eye",
                (mirrored_x, mirrored_y),
                "estimated",
                max(0.22, detected_eye["confidence"] * 0.72),
                "frontal-mirrored-eye",
            )
            if detected_eye["x"] <= nose_tip["x"]:
                keypoints["left_eye_center"] = {**detected_eye, "name": "left_eye_center"}
                keypoints["right_eye_center"] = {**mirrored, "name": "right_eye_center"}
            else:
                keypoints["left_eye_center"] = {**mirrored, "name": "left_eye_center"}
                keypoints["right_eye_center"] = {**detected_eye, "name": "right_eye_center"}
    else:
        if pose == "profile-left":
            keypoints["left_eye_center"] = make_keypoint("left_eye_center", estimate_point(bbox, 0.42, 0.39), "estimated", 0.16, "profile-left-hidden-eye-fallback")
            keypoints["right_eye_center"] = make_keypoint("right_eye_center", estimate_point(bbox, 0.62, 0.37), "estimated", 0.22, "profile-left-visible-eye-fallback")
        elif pose == "profile-right":
            keypoints["left_eye_center"] = make_keypoint("left_eye_center", estimate_point(bbox, 0.38, 0.37), "estimated", 0.22, "profile-right-visible-eye-fallback")
            keypoints["right_eye_center"] = make_keypoint("right_eye_center", estimate_point(bbox, 0.58, 0.39), "estimated", 0.16, "profile-right-hidden-eye-fallback")
        else:
            keypoints["left_eye_center"] = make_keypoint("left_eye_center", estimate_point(bbox, 0.34, 0.40), "estimated", 0.18, "no-eye-detection")
            keypoints["right_eye_center"] = make_keypoint("right_eye_center", estimate_point(bbox, 0.66, 0.40), "estimated", 0.18, "no-eye-detection")

    keypoints["nose_bridge_top"] = nose_bridge_top
    keypoints["nose_tip"] = nose_tip
    keypoints["mouth_left"] = mouth_left
    keypoints["mouth_center"] = mouth_center
    keypoints["mouth_right"] = mouth_right

    forehead_center = estimate_point(bbox, 0.50, 0.10)
    chin = estimate_point(bbox, 0.50, 0.96)
    if pose == "profile-left":
        forehead_center = estimate_point(bbox, 0.56, 0.12)
        chin = estimate_point(bbox, 0.48, 0.96)
    elif pose == "profile-right":
        forehead_center = estimate_point(bbox, 0.44, 0.12)
        chin = estimate_point(bbox, 0.52, 0.96)
    keypoints["forehead_center"] = make_keypoint("forehead_center", forehead_center, "estimated", 0.36, f"{pose}-forehead-template")
    keypoints["chin"] = make_keypoint("chin", chin, "estimated", 0.34, f"{pose}-chin-template")
    stabilize_profile_keypoints(bbox, pose, keypoints)
    return keypoints


def apply_mediapipe_keypoints(keypoints, bbox, pose, landmarks, image_shape):
    if not landmarks:
        return keypoints

    image_h, image_w = image_shape[:2]
    eye_a = average_mediapipe_points(landmarks, (33, 133, 159, 145), image_w, image_h)
    eye_b = average_mediapipe_points(landmarks, (362, 263, 386, 374), image_w, image_h)
    left_eye, right_eye = sorted([eye_a, eye_b], key=lambda point: point[0])
    mouth_a = point_from_mediapipe_landmark(landmarks, 61, image_w, image_h)
    mouth_b = point_from_mediapipe_landmark(landmarks, 291, image_w, image_h)
    mouth_left, mouth_right = sorted([mouth_a, mouth_b], key=lambda point: point[0])

    updates = {
        "left_eye_center": left_eye,
        "right_eye_center": right_eye,
        "nose_bridge_top": point_from_mediapipe_landmark(landmarks, 168, image_w, image_h),
        "nose_tip": point_from_mediapipe_landmark(landmarks, 1, image_w, image_h),
        "mouth_left": mouth_left,
        "mouth_center": average_mediapipe_points(landmarks, (13, 14), image_w, image_h),
        "mouth_right": mouth_right,
        "chin": point_from_mediapipe_landmark(landmarks, 152, image_w, image_h),
        "forehead_center": point_from_mediapipe_landmark(landmarks, 10, image_w, image_h),
    }
    for name, point in updates.items():
        keypoints[name] = make_keypoint(name, point, "detected", 0.88, "mediapipe-face-landmarker")
    stabilize_profile_keypoints(bbox, pose, keypoints)
    return keypoints


def apply_yunet_keypoints(keypoints, bbox, pose, landmarks):
    if len(landmarks) < 5:
        return keypoints

    eye_a, eye_b = sorted(landmarks[:2], key=lambda point: point[0])
    mouth_a, mouth_b = sorted(landmarks[3:5], key=lambda point: point[0])
    updates = {
        "left_eye_center": eye_a,
        "right_eye_center": eye_b,
        "nose_tip": landmarks[2],
        "mouth_left": mouth_a,
        "mouth_center": (
            int(round((mouth_a[0] + mouth_b[0]) / 2.0)),
            int(round((mouth_a[1] + mouth_b[1]) / 2.0)),
        ),
        "mouth_right": mouth_b,
    }
    nose_bridge = keypoints.get("nose_bridge_top", {})
    if nose_bridge:
        updates["nose_bridge_top"] = (
            int(round((updates["left_eye_center"][0] + updates["right_eye_center"][0]) / 2.0)),
            int(round((updates["left_eye_center"][1] + updates["right_eye_center"][1]) / 2.0)),
        )
    for name, point in updates.items():
        keypoints[name] = make_keypoint(name, point, "detected", 0.72, "yunet-face-detector-landmark")
    stabilize_profile_keypoints(bbox, pose, keypoints)
    return keypoints


def stabilize_profile_keypoints(bbox, pose, keypoints):
    if pose not in {"profile-left", "profile-right"}:
        return

    x, y, w, h = bbox
    nose_tip_x = keypoints["nose_tip"]["x"]
    mouth_center_x = keypoints["mouth_center"]["x"]
    visible_side = "right" if pose == "profile-left" else "left"

    if visible_side == "right":
        keypoints["right_eye_center"]["x"] = clamp(keypoints["right_eye_center"]["x"], int(x + w * 0.48), x + w)
        keypoints["left_eye_center"]["x"] = clamp(keypoints["left_eye_center"]["x"], x, int(x + w * 0.44))
        keypoints["mouth_right"]["x"] = max(keypoints["mouth_right"]["x"], mouth_center_x)
        keypoints["mouth_left"]["x"] = min(keypoints["mouth_left"]["x"], mouth_center_x - max(4, int(w * 0.06)))
        keypoints["nose_bridge_top"]["x"] = max(keypoints["nose_bridge_top"]["x"], int((nose_tip_x + keypoints["right_eye_center"]["x"]) / 2))
    else:
        keypoints["left_eye_center"]["x"] = clamp(keypoints["left_eye_center"]["x"], x, int(x + w * 0.52))
        keypoints["right_eye_center"]["x"] = clamp(keypoints["right_eye_center"]["x"], int(x + w * 0.56), x + w)
        keypoints["mouth_left"]["x"] = min(keypoints["mouth_left"]["x"], mouth_center_x)
        keypoints["mouth_right"]["x"] = max(keypoints["mouth_right"]["x"], mouth_center_x + max(4, int(w * 0.06)))
        keypoints["nose_bridge_top"]["x"] = min(keypoints["nose_bridge_top"]["x"], int((nose_tip_x + keypoints["left_eye_center"]["x"]) / 2))

    keypoints["mouth_center"]["y"] = clamp(keypoints["mouth_center"]["y"], int(y + h * 0.68), int(y + h * 0.86))
    keypoints["nose_tip"]["y"] = clamp(keypoints["nose_tip"]["y"], int(y + h * 0.48), int(y + h * 0.74))
    keypoints["chin"]["y"] = max(keypoints["chin"]["y"], int(y + h * 0.88))


def compute_noise_score(face_gray, face_mask=None):
    denoised = cv2.GaussianBlur(face_gray, (5, 5), 0)
    residual = cv2.absdiff(face_gray, denoised)
    values = masked_values(residual, face_mask)
    return float((np.mean(values) if values.size else 0.0) / 255.0)


def compute_skin_ratio(face_bgr, face_mask=None):
    if face_bgr.size == 0:
        return 0.0
    ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower, upper)
    if face_mask is not None and face_mask.size:
        skin_mask = cv2.bitwise_and(skin_mask, skin_mask, mask=face_mask)
        denom = max(1, int(np.count_nonzero(face_mask)))
    else:
        denom = max(1, skin_mask.size)
    return float(np.count_nonzero(skin_mask) / float(denom))


def compute_color_texture_features(face_bgr, face_mask=None):
    if face_bgr.size == 0:
        return {
            "saturationMean": 0.0,
            "saturationStd": 0.0,
            "lowSaturationRatio": 0.0,
            "highSaturationRatio": 0.0,
            "chromaStd": 0.0,
        }
    if face_mask is not None and face_mask.size:
        mask = face_mask > 0
    else:
        mask = np.ones(face_bgr.shape[:2], dtype=bool)
    if not np.any(mask):
        mask = np.ones(face_bgr.shape[:2], dtype=bool)

    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    saturation = hsv[:, :, 1][mask].astype(np.float32) / 255.0
    lab_a = lab[:, :, 1][mask].astype(np.float32)
    lab_b = lab[:, :, 2][mask].astype(np.float32)
    chroma_std = (float(np.std(lab_a)) + float(np.std(lab_b))) / 2.0 if lab_a.size else 0.0
    return {
        "saturationMean": round(float(np.mean(saturation)) if saturation.size else 0.0, 4),
        "saturationStd": round(float(np.std(saturation)) if saturation.size else 0.0, 4),
        "lowSaturationRatio": round(float(np.mean(saturation < 0.08)) if saturation.size else 0.0, 4),
        "highSaturationRatio": round(float(np.mean(saturation > 0.55)) if saturation.size else 0.0, 4),
        "chromaStd": round(float(chroma_std), 4),
    }


def compute_print_like_texture_features(face_bgr):
    if face_bgr.size == 0:
        return {
            "printEdgeDensity": 0.0,
            "printStraightLineDensity": 0.0,
            "printTextComponentDensity": 0.0,
            "printHighContrastRatio": 0.0,
            "printWhiteRatio": 0.0,
            "printDarkRatio": 0.0,
            "printSaturationStd": 0.0,
        }

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    height, width = gray.shape[:2]
    area = float(max(1, height * width))

    edges = cv2.Canny(gray, 70, 150)
    edge_density = float(np.count_nonzero(edges)) / area

    min_line_length = max(12, int(round(min(height, width) * 0.22)))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(10, int(round(min(height, width) * 0.10))),
        minLineLength=min_line_length,
        maxLineGap=max(3, int(round(min(height, width) * 0.04))),
    )
    line_length = 0.0
    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [float(value) for value in line]
            line_length += float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    straight_line_density = line_length / float(max(1.0, min(height, width) * max(height, width)))

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        8,
    )
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(adaptive, 8)
    text_like = 0
    for label in range(1, num_labels):
        _, _, component_width, component_height, component_area = stats[label]
        if component_area < 4:
            continue
        component_ratio = component_area / area
        if component_ratio > 0.08:
            continue
        aspect = component_width / float(max(component_height, 1))
        if (
            0.12 <= aspect <= 8.0
            and 4 <= component_width <= max(6, int(width * 0.55))
            and 4 <= component_height <= max(6, int(height * 0.45))
        ):
            text_like += 1
    text_component_density = text_like / float(max(1.0, area / 10000.0))

    local_mean = cv2.blur(gray.astype(np.float32), (9, 9))
    high_contrast = np.abs(gray.astype(np.float32) - local_mean) >= 42.0
    white_ratio = float(np.count_nonzero(gray >= 220)) / area
    dark_ratio = float(np.count_nonzero(gray <= 45)) / area
    saturation_std = float(np.std(hsv[:, :, 1])) / 255.0

    return {
        "printEdgeDensity": round(edge_density, 4),
        "printStraightLineDensity": round(straight_line_density, 4),
        "printTextComponentDensity": round(text_component_density, 4),
        "printHighContrastRatio": round(float(np.count_nonzero(high_contrast)) / area, 4),
        "printWhiteRatio": round(white_ratio, 4),
        "printDarkRatio": round(dark_ratio, 4),
        "printSaturationStd": round(saturation_std, 4),
    }


def compute_occlusion_appearance_features(face_bgr, face_gray, face_mask=None):
    if face_gray.size == 0:
        return {"upperDarkRatio": 0.0, "midDarkRatio": 0.0, "lowerDarkRatio": 0.0, "appearanceOcclusionScore": 0.0}
    h, w = face_gray.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    if face_mask is not None and face_mask.size:
        mask = face_mask > 0
    else:
        center_x = max(float(w) / 2.0, 1.0)
        center_y = max(float(h) / 2.0, 1.0)
        mask = (((xx - center_x) / max(float(w) * 0.42, 1.0)) ** 2 + ((yy - center_y) / max(float(h) * 0.48, 1.0)) ** 2) <= 1.0

    def band_dark_ratio(top, bottom):
        band = mask & (yy >= h * top) & (yy < h * bottom)
        if not np.any(band):
            return 0.0
        return float(np.mean(face_gray[band] < 75))

    upper_dark = band_dark_ratio(0.18, 0.48)
    mid_dark = band_dark_ratio(0.38, 0.68)
    lower_dark = band_dark_ratio(0.55, 0.88)
    appearance_score = max(
        max(0.0, min(1.0, (upper_dark - 0.28) / 0.28)),
        max(0.0, min(1.0, (mid_dark - 0.30) / 0.32)),
        max(0.0, min(1.0, (lower_dark - 0.32) / 0.34)),
    )
    return {
        "upperDarkRatio": round(float(upper_dark), 4),
        "midDarkRatio": round(float(mid_dark), 4),
        "lowerDarkRatio": round(float(lower_dark), 4),
        "appearanceOcclusionScore": round(float(appearance_score), 4),
    }


def distance(point_a, point_b):
    return math.sqrt((point_a["x"] - point_b["x"]) ** 2 + (point_a["y"] - point_b["y"]) ** 2)


def compute_deepfake_features(face_bgr, face_gray, bbox, keypoints, pose_label, eye_selection, edge_profile, eye_metrics, face_mask=None):
    w = max(float(bbox[2]), 1.0)
    h = max(float(bbox[3]), 1.0)
    left_eye = keypoints["left_eye_center"]
    right_eye = keypoints["right_eye_center"]
    nose_tip = keypoints["nose_tip"]
    mouth_center = keypoints["mouth_center"]
    chin = keypoints["chin"]
    forehead = keypoints["forehead_center"]

    eye_distance_ratio = round(distance(left_eye, right_eye) / w, 4)
    nose_mouth_ratio = round(distance(nose_tip, mouth_center) / h, 4)
    mouth_chin_ratio = round(distance(mouth_center, chin) / h, 4)
    face_vertical_ratio = round(distance(forehead, chin) / h, 4)
    center_axis_offset = round(abs(nose_tip["x"] - (bbox[0] + w / 2.0)) / w, 4)
    signed_center_axis_bias = round((nose_tip["x"] - (bbox[0] + w / 2.0)) / w, 4)
    estimated_ratio = round(sum(1 for keypoint in keypoints.values() if keypoint["source"] != "detected") / float(max(len(keypoints), 1)), 4)
    eye_balance = round(abs(left_eye["y"] - right_eye["y"]) / h, 4)
    edge_density = round(float(edge_profile.get("edgeDensity", 0.0)), 4)
    edge_side_bias = round(float(edge_profile.get("rightEnergy", 0.0) - edge_profile.get("leftEnergy", 0.0)), 4)
    noise_score = round(compute_noise_score(face_gray, face_mask), 4)
    skin_ratio = round(compute_skin_ratio(face_bgr, face_mask), 4)
    color_texture = compute_color_texture_features(face_bgr, face_mask)
    print_texture = compute_print_like_texture_features(face_bgr)
    appearance_occlusion = compute_occlusion_appearance_features(face_bgr, face_gray, face_mask)
    eye_visibility = round(len(eye_selection) / 2.0, 4)
    eye_closure_index = round(float(eye_metrics.get("closureScore", 0.0)), 4)
    mean_eye_band = round(
        sum(float(eye.get("bandConcentration", 0.0)) for eye in eye_selection) / float(max(len(eye_selection), 1)),
        4,
    ) if eye_selection else 0.0
    mean_eye_openness = round(
        sum(float(eye.get("openness", 0.0)) for eye in eye_selection) / float(max(len(eye_selection), 1)),
        4,
    ) if eye_selection else 0.0
    face_mask_coverage = round(float(np.count_nonzero(face_mask)) / float(max(face_mask.size, 1)), 4) if face_mask is not None else 1.0
    low_anchor_score = max(0.0, min(1.0, (0.78 - (1.0 - estimated_ratio)) / 0.67))
    missing_eye_score = max(0.0, 1.0 - eye_visibility)
    mask_gap_score = max(0.0, min(1.0, (0.70 - face_mask_coverage) / 0.20))
    edge_score = max(0.0, min(1.0, edge_density / 0.16))
    occlusion_score = round(
        float(
            0.34 * low_anchor_score
            + 0.22 * estimated_ratio
            + 0.18 * missing_eye_score
            + 0.14 * edge_score
            + 0.12 * mask_gap_score
        ),
        4,
    )
    moderate_anchor_occlusion = (
        estimated_ratio >= 0.50
        and eye_closure_index <= 0.12
        and (
            edge_density >= 0.075
            or face_mask_coverage <= 0.68
            or missing_eye_score >= 0.50
        )
    )
    if moderate_anchor_occlusion:
        occlusion_score = max(occlusion_score, 0.52)
    if (
        (appearance_occlusion["appearanceOcclusionScore"] >= 0.40 and edge_density >= 0.10)
        or (appearance_occlusion["appearanceOcclusionScore"] >= 0.34 and edge_density >= 0.14)
        or (
            appearance_occlusion["lowerDarkRatio"] >= 0.33
            and edge_density >= 0.11
            and eye_visibility >= 1.0
            and skin_ratio >= 0.60
        )
        or (
            pose_label == "frontal"
            and eye_visibility >= 1.0
            and estimated_ratio <= 0.25
            and signed_center_axis_bias >= 0.07
            and skin_ratio >= 0.80
            and 0.10 <= edge_density <= 0.13
            and appearance_occlusion["lowerDarkRatio"] >= 0.25
            and eye_closure_index <= 0.03
        )
    ):
        occlusion_score = max(occlusion_score, 0.50)
    if pose_label == "occluded":
        occlusion_score = max(occlusion_score, 0.72)

    low_openness_score = max(0.0, min(1.0, (0.30 - mean_eye_openness) / 0.30)) if eye_selection else 0.0
    eye_closure_score = round(
        float(max(eye_closure_index, 0.72 * eye_closure_index + 0.28 * low_openness_score)),
        4,
    )
    frontal_paired_closed_band = (
        pose_label == "frontal"
        and eye_visibility >= 1.0
        and mean_eye_band >= 0.23
        and eye_closure_index <= 0.12
        and face_mask_coverage <= 0.68
        and skin_ratio >= 0.60
        and edge_density >= 0.10
    )
    if frontal_paired_closed_band:
        eye_closure_score = max(eye_closure_score, 0.66)
    if pose_label == "eyes_closed":
        eye_closure_score = max(eye_closure_score, 0.72)

    profile_pose_score = 0.0
    if pose_label in {"profile-left", "profile-right"}:
        profile_pose_score = 1.0
    elif pose_label == "occluded":
        profile_pose_score = 0.78
    profile_axis_score = max(
        profile_pose_score,
        max(0.0, min(1.0, abs(signed_center_axis_bias) / 0.08)),
        max(0.0, min(1.0, abs(edge_side_bias) / 0.10)),
    )
    profile_visibility_gap = max(missing_eye_score, estimated_ratio)
    profile_eye_closure_score = eye_closure_score
    if profile_axis_score >= 0.55:
        profile_eye_closure_score = max(
            profile_eye_closure_score,
            0.56 * profile_axis_score
            + 0.22 * profile_visibility_gap
            + 0.12 * edge_score
            + 0.10 * max(0.0, min(1.0, mean_eye_band / 0.30)),
        )
    if pose_label == "occluded" and occlusion_score >= 0.50:
        profile_eye_closure_score = max(profile_eye_closure_score, 0.60)
    if pose_label in {"profile-left", "profile-right"} and edge_density >= 0.10 and skin_ratio >= 0.45:
        profile_eye_closure_score = max(profile_eye_closure_score, 0.58)
    if (
        pose_label in {"profile-left", "profile-right"}
        and abs(signed_center_axis_bias) >= 0.35
        and skin_ratio >= 0.35
        and edge_density >= 0.09
        and eye_visibility >= 1.0
    ):
        profile_eye_closure_score = max(profile_eye_closure_score, 0.55)
    if pose_label == "frontal" and eye_visibility >= 1.0 and eye_closure_index < 0.08:
        profile_eye_closure_score = min(profile_eye_closure_score, 0.34)
    if (
        pose_label == "frontal"
        and eye_visibility >= 1.0
        and eye_closure_index < 0.08
        and occlusion_score >= 0.50
        and edge_density >= 0.115
        and 0.60 <= skin_ratio <= 0.82
        and eye_distance_ratio >= 0.35
    ):
        profile_eye_closure_score = max(profile_eye_closure_score, 0.56)
    profile_eye_closure_score = round(float(max(0.0, min(1.0, profile_eye_closure_score))), 4)

    return {
        "geometry": {
            "eyeDistanceRatio": eye_distance_ratio,
            "noseMouthRatio": nose_mouth_ratio,
            "mouthChinRatio": mouth_chin_ratio,
            "faceVerticalRatio": face_vertical_ratio,
            "centerAxisOffset": center_axis_offset,
            "signedCenterAxisBias": signed_center_axis_bias,
            "eyeBalance": eye_balance,
        },
        "texture": {
            "edgeDensity": edge_density,
            "edgeSideBias": edge_side_bias,
            "noiseScore": noise_score,
            "skinRatio": skin_ratio,
            "colorSaturationMean": color_texture["saturationMean"],
            "colorSaturationStd": color_texture["saturationStd"],
            "lowSaturationRatio": color_texture["lowSaturationRatio"],
            "highSaturationRatio": color_texture["highSaturationRatio"],
            "colorChromaStd": color_texture["chromaStd"],
            "printEdgeDensity": print_texture["printEdgeDensity"],
            "printStraightLineDensity": print_texture["printStraightLineDensity"],
            "printTextComponentDensity": print_texture["printTextComponentDensity"],
            "printHighContrastRatio": print_texture["printHighContrastRatio"],
            "printWhiteRatio": print_texture["printWhiteRatio"],
            "printDarkRatio": print_texture["printDarkRatio"],
            "printSaturationStd": print_texture["printSaturationStd"],
        },
        "visibility": {
            "estimatedPointRatio": estimated_ratio,
            "eyeVisibility": eye_visibility,
            "eyeClosureIndex": eye_closure_index,
            "meanEyeBandConcentration": mean_eye_band,
            "meanEyeOpenness": mean_eye_openness,
            "faceMaskCoverage": face_mask_coverage,
            "poseLabel": pose_label,
            "occlusionScore": occlusion_score,
            "appearanceOcclusionScore": appearance_occlusion["appearanceOcclusionScore"],
            "upperDarkRatio": appearance_occlusion["upperDarkRatio"],
            "midDarkRatio": appearance_occlusion["midDarkRatio"],
            "lowerDarkRatio": appearance_occlusion["lowerDarkRatio"],
            "eyeClosureScore": eye_closure_score,
            "profileEyeClosureScore": profile_eye_closure_score,
        },
    }


def refine_pose_label(detector_name, detector_score, pose_label, pose_confidence, pose_reason, deepfake_features, keypoints):
    geometry = deepfake_features.get("geometry", {})
    texture = deepfake_features.get("texture", {})
    visibility = deepfake_features.get("visibility", {})
    center_offset = float(geometry.get("centerAxisOffset", 1.0))
    signed_center_bias = float(geometry.get("signedCenterAxisBias", 0.0))
    eye_distance = float(geometry.get("eyeDistanceRatio", 0.0))
    nose_mouth_ratio = float(geometry.get("noseMouthRatio", 0.0))
    eye_visibility = float(visibility.get("eyeVisibility", 0.0))
    closure_index = float(visibility.get("eyeClosureIndex", 0.0))
    mean_eye_band = float(visibility.get("meanEyeBandConcentration", 0.0))
    mean_eye_openness = float(visibility.get("meanEyeOpenness", 0.0))
    estimated_ratio = float(visibility.get("estimatedPointRatio", 1.0))
    face_mask_coverage = float(visibility.get("faceMaskCoverage", 1.0))
    edge_density = float(texture.get("edgeDensity", 0.0))
    edge_side_bias = float(texture.get("edgeSideBias", 0.0))
    skin_ratio = float(texture.get("skinRatio", 0.0))
    color_chroma_std = float(texture.get("colorChromaStd", 0.0))
    nose_bridge_x = float(keypoints.get("nose_bridge_top", {}).get("x", 0.0))
    nose_tip_x = float(keypoints.get("nose_tip", {}).get("x", 0.0))
    nose_direction = nose_tip_x - nose_bridge_x
    nose_target_pose = None
    if nose_direction >= 6.0:
        nose_target_pose = "profile-left"
    elif nose_direction <= -6.0:
        nose_target_pose = "profile-right"

    if (
        pose_label in {"frontal", "eyes_closed"}
        and detector_name in {"frontal_alt", "response_fallback"}
        and skin_ratio >= 0.60
        and nose_mouth_ratio >= 0.35
        and eye_distance >= 0.30
    ):
        return "profile-right", max(0.54, pose_confidence), "profile-right-by-long-nose-mouth-axis"

    if (
        detector_name == "response_fallback"
        and skin_ratio <= 0.01
        and float(visibility.get("profileEyeClosureScore", 0.0)) >= 0.80
        and float(visibility.get("occlusionScore", 0.0)) >= 0.50
        and signed_center_bias >= 0.045
    ):
        return "profile-left", max(0.52, pose_confidence), "response-low-skin-profile-left-recovery"

    if (
        pose_label == "profile-left"
        and detector_name == "frontal_alt"
        and eye_visibility == 0.5
        and closure_index <= 0.02
        and float(visibility.get("profileEyeClosureScore", 0.0)) >= 0.80
        and float(visibility.get("occlusionScore", 0.0)) >= 0.60
        and skin_ratio >= 0.70
        and abs(signed_center_bias) <= 0.05
    ):
        return "eyes_closed", max(0.56, pose_confidence), "profile-left-single-eye-closed-frontal-recovery"

    if (
        pose_label == "frontal"
        and detector_name.startswith("frontal")
        and eye_visibility == 0.0
        and 0.24 <= closure_index <= 0.46
        and eye_distance <= 0.33
        and center_offset <= 0.04
        and abs(edge_side_bias) >= 0.05
    ):
        if nose_target_pose is not None:
            return nose_target_pose, max(0.54, pose_confidence), "frontal-no-eye-closure-shifted-to-profile"
        return "eyes_closed", max(0.54, pose_confidence), "frontal-no-eye-closure-promotion"

    if (
        pose_label == "frontal"
        and detector_name.startswith("frontal")
        and eye_visibility == 0.5
        and 0.05 <= closure_index <= 0.12
        and eye_distance >= 0.38
        and center_offset <= 0.02
    ):
        return "eyes_closed", max(0.53, pose_confidence), "frontal-single-eye-closure-promotion"

    if (
        pose_label == "frontal"
        and detector_name == "frontal_alt"
        and eye_visibility >= 1.0
        and 0.08 <= closure_index <= 0.12
        and eye_distance <= 0.33
        and face_mask_coverage <= 0.63
        and center_offset <= 0.02
    ):
        return "occluded", max(0.52, pose_confidence), "frontal-paired-eyes-occlusion-suspected"

    if (
        pose_label == "frontal"
        and detector_name == "frontal_alt"
        and eye_visibility >= 1.0
        and mean_eye_band >= 0.23
        and mean_eye_openness >= 0.50
        and eye_distance <= 0.34
        and face_mask_coverage <= 0.67
        and center_offset <= 0.04
    ):
        return "occluded", max(0.53, pose_confidence), "frontal-paired-eyes-high-band-occlusion"

    if (
        pose_label in {"profile-left", "profile-right", "occluded"}
        and detector_name != "profile"
        and detector_score >= 0.92
        and center_offset <= 0.035
        and 0.23 <= eye_distance <= 0.42
        and eye_visibility >= 0.5
        and not (
            pose_label == "occluded"
            and eye_visibility == 0.5
            and closure_index >= 0.05
            and eye_distance >= 0.38
        )
    ):
        return "frontal", max(0.58, pose_confidence), "frontal-detector-centered-nose"

    if (
        detector_name.startswith("frontal")
        and detector_score >= 0.94
        and eye_visibility == 0.0
        and center_offset <= 0.040
        and estimated_ratio >= 0.70
        and eye_distance <= 0.30
        and abs(nose_direction) <= 12.0
        and not (
            pose_label in {"profile-left", "profile-right"}
            and closure_index >= 0.24
            and abs(edge_side_bias) >= 0.05
        )
    ):
        if closure_index >= 0.55 or (closure_index >= 0.45 and edge_density <= 0.20):
            return "eyes_closed", max(0.56, pose_confidence), "strong-frontal-detector-no-eye-closure-bias"
        return "frontal", max(0.54, pose_confidence), "strong-frontal-detector-no-eye-bias"

    if (
        pose_label == "eyes_closed"
        and detector_name == "frontal_alt"
        and eye_visibility == 0.0
        and estimated_ratio >= 0.77
        and center_offset <= 0.035
        and abs(nose_direction) <= 8.0
        and 0.60 <= face_mask_coverage <= 0.74
        and 0.08 <= edge_density <= 0.16
        and closure_index >= 0.68
    ):
        return "frontal", max(0.50, pose_confidence), "frontal-alt-high-closure-open-eye-bias"

    if (
        pose_label == "occluded"
        and detector_name.startswith("frontal")
        and eye_visibility == 0.0
        and closure_index >= 0.18
        and center_offset <= 0.055
        and abs(nose_direction) <= 10.0
    ):
        return "eyes_closed", max(0.50, pose_confidence), "frontal-occluded-promoted-to-eyes-closed"

    if (
        pose_label == "occluded"
        and detector_name.startswith("frontal")
        and eye_visibility == 0.5
        and closure_index >= 0.05
        and eye_distance >= 0.38
        and center_offset <= 0.02
    ):
        return "eyes_closed", max(0.52, pose_confidence), "frontal-single-eye-occluded-promoted-to-eyes-closed"

    if (
        pose_label == "eyes_closed"
        and eye_visibility == 0.0
        and center_offset <= 0.035
        and closure_index >= 0.34
        and abs(edge_side_bias) >= 0.035
    ):
        if edge_side_bias <= -0.035:
            return "profile-right", max(0.52, pose_confidence), "eyes-closed-demoted-to-profile-right-by-edge-bias"
        return "profile-left", max(0.52, pose_confidence), "eyes-closed-demoted-to-profile-left-by-edge-bias"

    if (
        pose_label == "occluded"
        and eye_visibility == 0.0
        and center_offset <= 0.04
        and closure_index >= 0.18
        and abs(edge_side_bias) >= 0.04
    ):
        if edge_side_bias <= -0.04:
            return "profile-right", max(0.50, pose_confidence), "occluded-demoted-to-profile-right-by-edge-bias"
        return "profile-left", max(0.50, pose_confidence), "occluded-demoted-to-profile-left-by-edge-bias"

    if (
        pose_label == "frontal"
        and detector_name in {"frontal_alt", "response_fallback"}
        and eye_visibility >= 0.5
        and closure_index <= 0.10
        and 0.04 <= center_offset <= 0.08
        and abs(edge_side_bias) >= 0.04
    ):
        if nose_target_pose is not None and eye_distance <= 0.36:
            return nose_target_pose, max(0.52, pose_confidence), "frontal-demoted-to-profile-by-nose-direction"
        if edge_side_bias <= -0.04:
            return "profile-right", max(0.50, pose_confidence), "frontal-demoted-to-profile-right-by-edge-bias"
        return "profile-left", max(0.50, pose_confidence), "frontal-demoted-to-profile-left-by-edge-bias"

    if (
        pose_label == "frontal"
        and detector_name.startswith("frontal")
        and eye_visibility >= 0.5
        and closure_index <= 0.08
        and 0.035 <= center_offset <= 0.09
        and eye_distance <= 0.36
        and nose_target_pose is not None
    ):
        return nose_target_pose, max(0.51, pose_confidence), "frontal-shifted-to-profile-by-nose-direction"

    if (
        pose_label == "frontal"
        and detector_name.startswith("frontal")
        and eye_visibility == 0.0
        and closure_index >= 0.20
        and center_offset <= 0.050
        and abs(nose_direction) <= 12.0
    ):
        return "eyes_closed", max(0.52, pose_confidence), "frontal-no-eye-closure-promotion"

    if pose_label == "eyes_closed" and eye_visibility == 0.0:
        if abs(signed_center_bias) >= 0.090:
            if signed_center_bias > 0:
                return "profile-right", max(0.53, pose_confidence), "eyes-closed-demoted-to-profile-right-by-bias"
            return "profile-left", max(0.53, pose_confidence), "eyes-closed-demoted-to-profile-left-by-bias"
        if abs(signed_center_bias) >= 0.060 and closure_index < 0.78:
            if signed_center_bias > 0:
                return "profile-right", max(0.50, pose_confidence), "eyes-closed-shifted-to-profile-right-by-bias"
            return "profile-left", max(0.50, pose_confidence), "eyes-closed-shifted-to-profile-left-by-bias"
        if detector_name.startswith("frontal") and closure_index < 0.50 and edge_density >= 0.20:
            if center_offset <= 0.035:
                return "frontal", max(0.48, pose_confidence), "eyes-closed-demoted-to-frontal-by-texture"
            return "occluded", max(0.48, pose_confidence), "eyes-closed-demoted-to-occluded-by-texture"
        if center_offset >= 0.10:
            if nose_direction > 0:
                return "profile-left", max(0.52, pose_confidence), "eyes-closed-demoted-to-profile-left"
            if nose_direction < 0:
                return "profile-right", max(0.52, pose_confidence), "eyes-closed-demoted-to-profile-right"
        if center_offset >= 0.04 and abs(nose_direction) >= 6.0:
            if nose_direction > 0:
                return "profile-left", max(0.50, pose_confidence), "eyes-closed-shifted-to-profile-left"
            return "profile-right", max(0.50, pose_confidence), "eyes-closed-shifted-to-profile-right"
        if detector_name == "response_fallback" and closure_index < 0.55 and edge_density >= 0.16:
            return "occluded", max(0.46, pose_confidence), "response-fallback-eyes-closed-demoted-to-occluded"

    if pose_label == "frontal" and detector_name != "profile" and center_offset <= 0.03:
        if eye_visibility <= 0.5 and eye_distance <= 0.25:
            return "eyes_closed", max(0.56, pose_confidence), "frontal-single-eye-tight-spacing"
        if eye_visibility >= 1.0 and eye_distance <= 0.31 and closure_index <= 0.10:
            return "eyes_closed", max(0.54, pose_confidence), "frontal-paired-eyes-tight-spacing"

    if (
        pose_label == "frontal"
        and detector_name.startswith("frontal")
        and eye_visibility == 0.5
        and closure_index <= 0.10
        and abs(signed_center_bias) >= 0.020
    ):
        if signed_center_bias > 0:
            return "profile-right", max(0.50, pose_confidence), "frontal-single-eye-shifted-to-profile-right"
        return "profile-left", max(0.50, pose_confidence), "frontal-single-eye-shifted-to-profile-left"

    if pose_label == "frontal" and abs(signed_center_bias) >= 0.085 and eye_visibility <= 0.5:
        if signed_center_bias > 0:
            return "profile-right", max(0.50, pose_confidence), "frontal-demoted-to-profile-right-by-bias"
        return "profile-left", max(0.50, pose_confidence), "frontal-demoted-to-profile-left-by-bias"

    if eye_visibility == 0.0 and abs(nose_direction) >= 10.0 and center_offset >= 0.06:
        if nose_direction > 0:
            return "profile-left", max(0.52, pose_confidence), "nose-direction-left"
        return "profile-right", max(0.52, pose_confidence), "nose-direction-right"

    if pose_label == "occluded" and eye_visibility == 0.0 and abs(nose_direction) >= 8.0 and center_offset >= 0.05:
        if nose_direction > 0:
            return "profile-left", max(0.50, pose_confidence), "occluded-to-profile-left-by-nose"
        return "profile-right", max(0.50, pose_confidence), "occluded-to-profile-right-by-nose"

    if (
        pose_label == "occluded"
        and detector_name == "profile"
        and eye_visibility == 0.5
        and skin_ratio >= 0.75
        and edge_density <= 0.09
        and float(geometry.get("noseMouthRatio", 1.0)) <= 0.06
    ):
        return "profile-left", max(0.50, pose_confidence), "profile-detector-soft-occlusion-left-recovery"

    if (
        pose_label == "occluded"
        and eye_visibility == 0.5
        and closure_index <= 0.08
        and signed_center_bias <= -0.025
        and detector_name.startswith("frontal")
    ):
        return "profile-left", max(0.50, pose_confidence), "occluded-single-eye-promoted-to-profile-left"

    if pose_label == "profile-left" and signed_center_bias >= 0.10 and center_offset >= 0.08:
        return "profile-right", max(0.52, pose_confidence), "profile-left-flipped-to-profile-right-by-bias"

    if (
        pose_label == "profile-left"
        and detector_name == "frontal"
        and detector_score < 0.12
        and skin_ratio >= 0.90
        and eye_visibility == 0.0
        and closure_index >= 0.45
        and eye_distance <= 0.22
    ):
        return "profile-right", max(0.52, pose_confidence), "profile-left-low-score-frontal-profile-right-recovery"

    if pose_label == "profile-right" and signed_center_bias <= -0.10 and center_offset >= 0.08:
        return "profile-left", max(0.52, pose_confidence), "profile-right-flipped-to-profile-left-by-bias"

    if pose_label == "profile-left" and edge_side_bias <= -0.08 and center_offset >= 0.06:
        return "profile-right", max(0.50, pose_confidence), "profile-left-flipped-to-profile-right-by-edge-bias"

    if pose_label == "profile-right" and edge_side_bias >= 0.08 and center_offset >= 0.06:
        return "profile-left", max(0.50, pose_confidence), "profile-right-flipped-to-profile-left-by-edge-bias"

    if pose_label == "profile-left" and nose_direction <= -12.0 and center_offset >= 0.05 and eye_visibility <= 0.5:
        return "profile-right", max(0.50, pose_confidence), "profile-left-corrected-by-nose-direction"

    if pose_label == "profile-right" and nose_direction >= 12.0 and center_offset >= 0.05 and eye_visibility <= 0.5:
        return "profile-left", max(0.50, pose_confidence), "profile-right-corrected-by-nose-direction"

    if (
        pose_label in {"profile-left", "profile-right"}
        and nose_target_pose is not None
        and pose_label != nose_target_pose
        and eye_visibility <= 0.5
        and closure_index <= 0.20
        and center_offset <= 0.06
    ):
        return nose_target_pose, max(0.54, pose_confidence), "profile-flipped-by-nose-direction"

    return pose_label, pose_confidence, pose_reason


def build_connections():
    return [
        ("forehead_center", "left_eye_center"),
        ("forehead_center", "right_eye_center"),
        ("left_eye_center", "nose_bridge_top"),
        ("right_eye_center", "nose_bridge_top"),
        ("nose_bridge_top", "nose_tip"),
        ("nose_tip", "mouth_center"),
        ("mouth_left", "mouth_center"),
        ("mouth_center", "mouth_right"),
        ("mouth_left", "chin"),
        ("mouth_right", "chin"),
    ]


def build_regions(bbox, keypoints, pose_label):
    left_eye = point_xy(keypoints["left_eye_center"])
    right_eye = point_xy(keypoints["right_eye_center"])
    forehead = point_xy(keypoints["forehead_center"])
    nose_bridge = point_xy(keypoints["nose_bridge_top"])
    nose_tip = point_xy(keypoints["nose_tip"])
    mouth_left = point_xy(keypoints["mouth_left"])
    mouth_center = point_xy(keypoints["mouth_center"])
    mouth_right = point_xy(keypoints["mouth_right"])
    chin = point_xy(keypoints["chin"])
    left_top = estimate_point(bbox, 0.18, 0.20)
    right_top = estimate_point(bbox, 0.82, 0.20)
    left_lower_eye = estimate_point(bbox, 0.26, 0.55)
    right_lower_eye = estimate_point(bbox, 0.74, 0.55)
    jaw_left = estimate_point(bbox, 0.18, 0.86)
    jaw_right = estimate_point(bbox, 0.82, 0.86)
    if pose_label == "profile-left":
        left_top = estimate_point(bbox, 0.22, 0.22)
        right_top = estimate_point(bbox, 0.92, 0.16)
        left_lower_eye = estimate_point(bbox, 0.34, 0.58)
        right_lower_eye = estimate_point(bbox, 0.86, 0.56)
        jaw_left = estimate_point(bbox, 0.28, 0.86)
        jaw_right = estimate_point(bbox, 0.92, 0.82)
    elif pose_label == "profile-right":
        left_top = estimate_point(bbox, 0.08, 0.16)
        right_top = estimate_point(bbox, 0.78, 0.22)
        left_lower_eye = estimate_point(bbox, 0.14, 0.56)
        right_lower_eye = estimate_point(bbox, 0.66, 0.58)
        jaw_left = estimate_point(bbox, 0.08, 0.82)
        jaw_right = estimate_point(bbox, 0.72, 0.86)
    return [
        {"name": "forehead", "color": REGION_COLORS["forehead"], "points": [left_top, forehead, right_top, right_eye, left_eye]},
        {"name": "left_eye_zone", "color": REGION_COLORS["left_eye_zone"], "points": [estimate_point(bbox, 0.16, 0.30), left_eye, nose_bridge, left_lower_eye]},
        {"name": "right_eye_zone", "color": REGION_COLORS["right_eye_zone"], "points": [estimate_point(bbox, 0.84, 0.30), right_eye, nose_bridge, right_lower_eye]},
        {"name": "nose", "color": REGION_COLORS["nose"], "points": [left_eye, nose_bridge, right_eye, nose_tip]},
        {"name": "mouth", "color": REGION_COLORS["mouth"], "points": [mouth_left, mouth_center, mouth_right, nose_tip]},
        {"name": "jaw", "color": REGION_COLORS["jaw"], "points": [jaw_left, mouth_left, mouth_center, mouth_right, jaw_right, chin]},
    ]


def serialize_regions(regions):
    return [{"name": region["name"], "color": list(region["color"]), "points": [{"x": int(point[0]), "y": int(point[1])} for point in region["points"]]} for region in regions]


def serialize_connections(connections):
    return [{"from": start, "to": end} for start, end in connections]


def build_training_sample(face):
    bbox = face["bbox"]
    width = max(float(bbox["w"]), 1.0)
    height = max(float(bbox["h"]), 1.0)
    left = float(bbox["x"])
    top = float(bbox["y"])
    normalized_points = {}
    source_mask = {}
    for point_name in TRAINING_POINT_ORDER:
        keypoint = face["keypoints"][point_name]
        normalized_points[point_name] = {"x": round((keypoint["x"] - left) / width, 4), "y": round((keypoint["y"] - top) / height, 4), "confidence": keypoint["confidence"]}
        source_mask[point_name] = keypoint["source"]
    return {
        "poseLabel": face["pose"]["label"],
        "bboxNormalized": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        "pointsNormalized": normalized_points,
        "pointSources": source_mask,
        "qualityLabel": face["quality"]["label"],
        "qualityScore": face["quality"]["score"],
        "deepfakeFeatures": face.get("deepfakeFeatures", {}),
    }


def save_face_crop(image, request_uid, face_index, box, face_mask=None):
    if not DEBUG_ARTIFACTS:
        return None
    x, y, w, h = box
    face_crop = image[y : y + h, x : x + w].copy()
    if face_crop.size == 0:
        return None
    if face_mask is not None and face_mask.size:
        resized_mask = face_mask if face_mask.shape[:2] == face_crop.shape[:2] else cv2.resize(face_mask, (face_crop.shape[1], face_crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        masked_crop = np.zeros_like(face_crop)
        masked_crop[resized_mask > 0] = face_crop[resized_mask > 0]
        face_crop = masked_crop
    crop_path = os.path.join(FACE_CROP_DIR, f"{request_uid}_face_{face_index + 1}.jpg")
    cv2.imwrite(crop_path, face_crop)
    return normalize_path(crop_path)


def compute_blur_score(face_gray, face_mask=None):
    laplacian = cv2.Laplacian(face_gray, cv2.CV_64F)
    values = masked_values(laplacian, face_mask)
    variance = float(np.var(values)) if values.size else 0.0
    return round(float(max(0.0, min(1.0, variance / 220.0))), 4)


def compute_brightness_score(face_gray, face_mask=None):
    values = masked_values(face_gray, face_mask)
    mean_value = float(np.mean(values)) if values.size else 0.0
    deviation = abs(mean_value - 128.0)
    return round(float(max(0.0, min(1.0, 1.0 - deviation / 128.0))), 4)


def compute_contrast_score(face_gray, face_mask=None):
    values = masked_values(face_gray, face_mask)
    std_value = float(np.std(values)) if values.size else 0.0
    return round(float(max(0.0, min(1.0, std_value / 72.0))), 4)


def classify_quality(blur_score, brightness_score, contrast_score, pose_confidence, detected_ratio, face_size):
    weighted_score = 0.28 * blur_score + 0.18 * brightness_score + 0.18 * contrast_score + 0.18 * pose_confidence + 0.10 * detected_ratio + 0.08 * min(1.0, face_size / 220.0)
    if weighted_score >= 0.72:
        return "good", round(weighted_score, 4)
    if weighted_score >= 0.48:
        return "usable", round(weighted_score, 4)
    return "poor", round(weighted_score, 4)


def compute_face_like_score(face):
    quality = face.get("quality", {})
    deepfake_features = face.get("deepfakeFeatures", {})
    geometry = deepfake_features.get("geometry", {})
    texture = deepfake_features.get("texture", {})
    visibility = deepfake_features.get("visibility", {})
    feature_summary = face.get("featureSummary", {})

    eye_support = min(1.0, float(visibility.get("eyeVisibility", 0.0)))
    anchor_support = min(1.0, float(quality.get("detectedPointRatio", 0.0)) / 0.55)
    mouth_support = min(1.0, float(feature_summary.get("mouthEvidence", 0.0)))
    nose_support = min(1.0, float(feature_summary.get("noseEvidence", 0.0)))

    skin_ratio = float(texture.get("skinRatio", 0.0))
    skin_support = max(0.0, 1.0 - abs(skin_ratio - 0.55) / 0.55) if skin_ratio > 0.0 else 0.0

    edge_density = float(texture.get("edgeDensity", 0.0))
    edge_support = max(0.0, 1.0 - abs(edge_density - 0.13) / 0.13)

    eye_distance_ratio = float(geometry.get("eyeDistanceRatio", 0.0))
    geometry_support = max(0.0, 1.0 - abs(eye_distance_ratio - 0.34) / 0.34)

    detector_support = {
        "mediapipe_landmarker": 0.78,
        "mediapipe": 0.72,
        "yunet": 0.70,
        "frontal_alt": 0.66,
        "frontal": 0.54,
        "profile": 0.50,
        "dark_profile_silhouette": 0.42,
        "response_fallback": 0.35,
    }.get(face.get("detector"), 0.40)

    return round(
        float(
            0.26 * eye_support
            + 0.17 * anchor_support
            + 0.16 * mouth_support
            + 0.12 * nose_support
            + 0.11 * skin_support
            + 0.08 * edge_support
            + 0.06 * geometry_support
            + 0.04 * detector_support
        ),
        4,
    )


def should_keep_face(face):
    bbox = face["bbox"]
    width = bbox["w"]
    height = bbox["h"]
    score = float(face.get("score", 0.0))
    quality = face["quality"]
    pose_label = face.get("pose", {}).get("label", "")
    deepfake_features = face.get("deepfakeFeatures", {})
    geometry = deepfake_features.get("geometry", {})
    texture = deepfake_features.get("texture", {})
    visibility = deepfake_features.get("visibility", {})
    face_like_score = float(face.get("faceLikeScore", 0.0))

    detected_eye_count = sum(1 for key in ("left_eye_center", "right_eye_center") if face["keypoints"][key]["source"] == "detected")
    eye_visibility = float(visibility.get("eyeVisibility", 0.0))
    estimated_ratio = float(visibility.get("estimatedPointRatio", 1.0))
    closure_index = float(visibility.get("eyeClosureIndex", 0.0))
    face_mask_coverage = float(visibility.get("faceMaskCoverage", 1.0))
    center_offset = float(geometry.get("centerAxisOffset", 1.0))
    eye_distance_ratio = float(geometry.get("eyeDistanceRatio", 0.0))
    edge_density = float(texture.get("edgeDensity", 0.0))
    edge_side_bias = float(texture.get("edgeSideBias", 0.0))
    skin_ratio = float(texture.get("skinRatio", 0.0))
    low_saturation_ratio = float(texture.get("lowSaturationRatio", 0.0))
    high_saturation_ratio = float(texture.get("highSaturationRatio", 0.0))
    color_saturation_mean = float(texture.get("colorSaturationMean", 0.0))
    color_chroma_std = float(texture.get("colorChromaStd", 0.0))
    mouth_evidence = float(face.get("featureSummary", {}).get("mouthEvidence", 0.0))
    strong_rescue_face = (
        face_like_score >= 0.80
        and skin_ratio >= 0.88
        and mouth_evidence >= 0.55
        and color_chroma_std >= 7.0
        and eye_visibility >= 1.0
        and quality.get("detectedPointRatio", 0.0) >= 0.75
    )
    consensus_skin_face_rescue = (
        bool(face.get("detectorConsensus"))
        and face.get("detector") in {"mediapipe_landmarker", "yunet"}
        and score >= 0.84
        and face_like_score >= 0.68
        and quality.get("label") in {"good", "usable"}
        and quality.get("detectedPointRatio", 0.0) >= 0.75
        and eye_visibility >= 1.0
        and 0.30 <= eye_distance_ratio <= 0.52
        and skin_ratio >= 0.88
        and edge_density >= 0.10
        and color_chroma_std >= 4.00
        and mouth_evidence <= 0.65
    )
    consensus_grayscale_face_rescue = (
        bool(face.get("detectorConsensus"))
        and face.get("detector") in {"mediapipe_landmarker", "yunet"}
        and score >= 0.84
        and face_like_score >= 0.68
        and quality.get("label") in {"good", "usable"}
        and quality.get("detectedPointRatio", 0.0) >= 0.75
        and eye_visibility >= 1.0
        and 0.30 <= eye_distance_ratio <= 0.52
        and skin_ratio <= 0.005
        and low_saturation_ratio >= 0.78
        and edge_density >= 0.13
        and color_chroma_std <= 1.20
    )
    landmark_consensus_face_rescue = consensus_skin_face_rescue or consensus_grayscale_face_rescue
    low_skin_yunet_profile_rescue = (
        face.get("detector") == "yunet"
        and face_like_score >= 0.80
        and skin_ratio <= 0.08
        and mouth_evidence >= 0.65
        and color_chroma_std >= 7.5
        and edge_density >= 0.10
        and eye_visibility >= 1.0
        and quality.get("detectedPointRatio", 0.0) >= 0.75
        and center_offset >= 0.30
    )
    grayscale_real_face_rescue = (
        face.get("detector") in {"frontal_alt", "response_fallback"}
        and pose_label in {"frontal", "occluded"}
        and skin_ratio <= 0.001
        and color_chroma_std <= 0.05
        and low_saturation_ratio >= 0.85
        and quality.get("label") == "good"
        and eye_visibility >= 1.0
        and quality.get("detectedPointRatio", 0.0) >= 0.4443
        and face_like_score >= 0.64
        and (
            (face.get("detector") == "frontal_alt" and score >= 0.99)
            or (face.get("detector") == "response_fallback" and score >= 0.75)
        )
    )
    grayscale_closed_face_rescue = (
        face.get("detector") == "response_fallback"
        and pose_label == "frontal"
        and skin_ratio <= 0.001
        and color_chroma_std <= 0.05
        and low_saturation_ratio >= 0.95
        and quality.get("label") == "good"
        and eye_visibility >= 1.0
        and quality.get("detectedPointRatio", 0.0) >= 0.4443
        and face_like_score >= 0.64
        and score >= 0.60
        and edge_density <= 0.04
        and center_offset <= 0.01
    )
    frontal_alt_low_anchor_real_face_rescue = (
        face.get("detector") == "frontal_alt"
        and pose_label == "eyes_closed"
        and quality.get("label") == "good"
        and quality.get("detectedPointRatio", 0.0) <= 0.1112
        and eye_visibility == 0.0
        and face_like_score <= 0.40
        and skin_ratio >= 0.85
        and edge_density <= 0.075
        and 0.31 <= eye_distance_ratio <= 0.33
        and 0.20 <= float(geometry.get("noseMouthRatio", 0.0)) <= 0.22
        and center_offset <= 0.06
        and score >= 0.99
    )
    dark_occluded_profile_rescue = (
        face.get("detector") == "profile"
        and score <= 0.01
        and quality.get("label") == "good"
        and skin_ratio <= 0.001
        and color_chroma_std <= 0.05
        and low_saturation_ratio >= 0.95
        and 0.035 <= edge_density <= 0.055
        and quality.get("detectedPointRatio", 0.0) >= 0.2222
        and float(visibility.get("profileEyeClosureScore", 0.0)) >= 0.70
        and float(visibility.get("occlusionScore", 0.0)) >= 0.70
        and float(geometry.get("noseMouthRatio", 0.0)) >= 0.34
        and min(width, height) >= 300
    )
    dark_response_occluded_rescue = (
        face.get("detector") == "response_fallback"
        and pose_label == "occluded"
        and score >= 0.60
        and quality.get("label") == "good"
        and skin_ratio <= 0.001
        and color_chroma_std <= 0.05
        and low_saturation_ratio >= 0.95
        and edge_density <= 0.03
        and eye_visibility == 0.0
        and quality.get("detectedPointRatio", 0.0) <= 0.1112
        and float(visibility.get("profileEyeClosureScore", 0.0)) >= 0.65
        and float(visibility.get("occlusionScore", 0.0)) >= 0.75
        and float(visibility.get("appearanceOcclusionScore", 0.0)) >= 0.70
        and min(width, height) >= 390
    )
    response_bw_profile_direction_rescue = (
        face.get("detector") == "response_fallback"
        and pose_label in {"profile-left", "profile-right"}
        and score >= 0.60
        and quality.get("label") == "good"
        and skin_ratio <= 0.001
        and color_chroma_std <= 0.40
        and low_saturation_ratio >= 0.85
        and eye_visibility == 0.5
        and quality.get("detectedPointRatio", 0.0) >= 0.3333
        and float(visibility.get("profileEyeClosureScore", 0.0)) >= 0.80
        and float(visibility.get("occlusionScore", 0.0)) >= 0.50
        and abs(float(geometry.get("signedCenterAxisBias", 0.0))) >= 0.20
        and float(geometry.get("eyeDistanceRatio", 0.0)) >= 0.38
    )
    nose_bridge_x = float(face["keypoints"].get("nose_bridge_top", {}).get("x", 0.0))
    nose_tip_x = float(face["keypoints"].get("nose_tip", {}).get("x", 0.0))
    nose_direction = nose_tip_x - nose_bridge_x
    left_eye_reason = face["keypoints"].get("left_eye_center", {}).get("reason", "")
    right_eye_reason = face["keypoints"].get("right_eye_center", {}).get("reason", "")
    bbox_x = float(bbox["x"])
    bbox_y = float(bbox["y"])
    width_f = float(max(width, 1))
    height_f = float(max(height, 1))
    left_eye_y = float(face["keypoints"].get("left_eye_center", {}).get("y", bbox_y))
    right_eye_y = float(face["keypoints"].get("right_eye_center", {}).get("y", bbox_y))
    mouth_y = float(face["keypoints"].get("mouth_center", {}).get("y", bbox_y))
    eye_span = abs(float(face["keypoints"].get("right_eye_center", {}).get("x", bbox_x)) - float(face["keypoints"].get("left_eye_center", {}).get("x", bbox_x))) / width_f
    eye_y_ratio = (((left_eye_y + right_eye_y) / 2.0) - bbox_y) / height_f
    nose_y_ratio = (float(face["keypoints"].get("nose_tip", {}).get("y", bbox_y)) - bbox_y) / height_f
    mouth_y_ratio = (mouth_y - bbox_y) / height_f
    nose_to_mouth_ratio = (mouth_y - float(face["keypoints"].get("nose_tip", {}).get("y", bbox_y))) / height_f

    haar_low_chroma_real_face_rescue = (
        face.get("detector") in {"frontal", "frontal_alt"}
        and quality.get("label") == "good"
        and eye_visibility >= 1.0
        and float(face.get("featureSummary", {}).get("eyeEvidence", 0.0)) >= 0.55
        and float(face.get("featureSummary", {}).get("noseEvidence", 0.0)) >= 0.90
        and face_like_score >= 0.72
        and edge_density >= 0.13
        and color_chroma_std >= 3.70
    )

    if (
        HAAR_TEXTURE_FP_FILTER
        and face.get("detector") in {"frontal", "frontal_alt"}
        and not bool(face.get("detectorConsensus"))
        and skin_ratio >= 0.95
        and 2.0 <= color_chroma_std <= 4.5
        and mouth_evidence <= 0.30
        and face_like_score <= 0.75
        and 0.18 <= eye_distance_ratio <= 0.42
        and not grayscale_real_face_rescue
        and not strong_rescue_face
        and not landmark_consensus_face_rescue
        and not haar_low_chroma_real_face_rescue
    ):
        return False, "haar-low-chroma-texture-face"

    if face.get("detector") == "dark_profile_silhouette":
        if not (
            pose_label in {"profile-left", "profile-right"}
            and face_like_score >= 0.42
            and skin_ratio <= 0.005
            and edge_density <= 0.035
            and eye_visibility >= 0.5
            and quality.get("detectedPointRatio", 0.0) >= 0.2222
            and float(visibility.get("appearanceOcclusionScore", 0.0)) >= 0.90
            and float(visibility.get("profileEyeClosureScore", 0.0)) >= 0.70
            and float(visibility.get("occlusionScore", 0.0)) >= 0.55
            and abs(float(geometry.get("signedCenterAxisBias", 0.0))) >= 0.10
            and float(geometry.get("noseMouthRatio", 0.0)) >= 0.34
            and 0.42 <= low_saturation_ratio <= 0.70
        ):
            return False, "dark-profile-silhouette-requires-strong-profile-shape"

    if (
        face.get("detector") == "mediapipe_landmarker"
        and (closure_index < 0.85 or skin_ratio < 0.70)
        and not strong_rescue_face
        and not landmark_consensus_face_rescue
    ):
        return False, "mediapipe-landmarker-rescue-requires-high-closure-and-skin-support"
    if (
        face.get("detector") == "yunet"
        and (
            not strong_rescue_face
            and not low_skin_yunet_profile_rescue
            and (
                face_like_score < 0.76
                or skin_ratio < 0.40
                or mouth_evidence < 0.25
                or (
                    closure_index < 0.85
                    and center_offset < 0.16
                    and abs(edge_side_bias) < 0.045
                )
            )
        )
        and not landmark_consensus_face_rescue
    ):
        return False, "yunet-rescue-requires-strong-face-like-skin-mouth-support"

    if score < 0.05 and not dark_occluded_profile_rescue:
        return False, "very-low-detector-score"
    if (
        face_like_score < 0.68
        and face.get("detector") in {"response_fallback", "frontal_alt"}
        and eye_visibility >= 1.0
        and face.get("featureSummary", {}).get("mouthEvidence", 0.0) <= 0.30
        and edge_density <= 0.09
        and not grayscale_real_face_rescue
        and not grayscale_closed_face_rescue
    ):
        return False, "low-face-like-score-with-weak-mouth-texture"
    if (
        face.get("detector") == "frontal"
        and score < 0.08
        and eye_visibility == 0.0
        and quality.get("detectedPointRatio", 0.0) <= 0.2222
    ):
        return False, "frontal-low-score-without-eye-support"
    if (
        face.get("detector") == "frontal"
        and score <= 0.25
        and face_like_score <= 0.42
        and bbox["y"] >= 120
        and eye_visibility == 0.0
        and skin_ratio >= 0.95
        and float(visibility.get("occlusionScore", 0.0)) >= 0.80
    ):
        return False, "frontal-low-score-lower-region-body-candidate"
    if (
        face.get("detector") == "frontal_alt"
        and eye_visibility == 0.0
        and quality.get("detectedPointRatio", 0.0) <= 0.2222
        and edge_density <= 0.115
        and skin_ratio <= 0.22
    ):
        return False, "frontal-alt-low-support-low-skin"
    if score < 0.16 and min(width, height) < 72 and detected_eye_count == 0:
        return False, "small-low-score-without-eyes"
    if face.get("detector") == "profile" and score < 0.19 and eye_visibility == 0.0 and quality.get("detectedPointRatio", 0.0) <= 0.2222 and not dark_occluded_profile_rescue:
        return False, "weak-profile-candidate-without-eyes"
    if (
        face.get("detector") == "profile"
        and face_like_score <= 0.38
        and skin_ratio <= 0.05
        and eye_visibility == 0.0
        and quality.get("detectedPointRatio", 0.0) <= 0.2222
        and not dark_occluded_profile_rescue
    ):
        return False, "profile-low-face-like-low-skin-without-eyes"
    if (
        face.get("detector") == "profile"
        and pose_label == "profile-right"
        and score <= 0.13
        and skin_ratio <= 0.001
        and eye_visibility == 0.5
        and bbox["y"] >= 500
        and bbox["w"] <= 70
        and float(visibility.get("profileEyeClosureScore", 0.0)) >= 0.80
        and float(geometry.get("signedCenterAxisBias", 0.0)) <= -0.01
    ):
        return False, "profile-lower-small-low-skin-artifact"
    if (
        face.get("detector") == "frontal"
        and pose_label == "occluded"
        and score < 0.20
        and skin_ratio <= 0.03
        and low_saturation_ratio >= 0.85
    ):
        return False, "frontal-occluded-low-skin-low-saturation"
    if (
        face.get("detector") == "frontal_alt"
        and skin_ratio <= 0.03
        and color_saturation_mean <= 0.06
        and low_saturation_ratio >= 0.70
        and edge_density >= 0.12
        and eye_visibility >= 1.0
        and quality.get("detectedPointRatio", 0.0) <= 0.4444
        and not grayscale_real_face_rescue
    ):
        return False, "frontal-alt-low-skin-desaturated-edge-face"
    if (
        face.get("detector") == "frontal_alt"
        and skin_ratio >= 0.94
        and edge_density <= 0.03
        and face_like_score <= 0.36
        and eye_visibility == 0.0
        and closure_index >= 0.45
        and quality.get("detectedPointRatio", 0.0) <= 0.2223
    ):
        return False, "blank-mannequin-high-skin-low-detail"
    if (
        face.get("detector") == "frontal_alt"
        and pose_label not in {"profile-left", "profile-right"}
        and skin_ratio <= 0.005
        and edge_density <= 0.06
        and 1.0 <= color_chroma_std <= 2.6
        and high_saturation_ratio <= 0.001
        and quality.get("detectedPointRatio", 0.0) >= 0.3333
        and face_like_score <= 0.75
    ):
        return False, "blank-mannequin-zero-skin-material"
    if (
        face.get("detector") == "frontal_alt"
        and 0.30 <= skin_ratio <= 0.38
        and 0.09 <= edge_density <= 0.13
        and color_chroma_std >= 9.0
        and float(visibility.get("appearanceOcclusionScore", 0.0)) >= 0.95
        and center_offset <= 0.01
        and mouth_evidence <= 0.30
        and quality.get("detectedPointRatio", 0.0) <= 0.4445
    ):
        return False, "blank-mannequin-torso-texture"
    if (
        skin_ratio >= 0.12
        and low_saturation_ratio >= 0.58
        and closure_index <= 0.03
    ):
        return False, "low-closure-low-saturation-nonhuman-face"
    if (
        face.get("detector") == "frontal_alt"
        and skin_ratio >= 0.70
        and color_chroma_std <= 3.0
        and float(visibility.get("appearanceOcclusionScore", 0.0)) >= 0.50
        and closure_index <= 0.06
    ):
        return False, "frontal-alt-low-chroma-sculpture-face"
    if (
        face.get("detector") == "frontal_alt"
        and skin_ratio >= 0.90
        and color_chroma_std >= 8.0
        and high_saturation_ratio <= 0.08
        and closure_index <= 0.05
    ):
        return False, "frontal-alt-high-chroma-illustration-face"
    if (
        face.get("detector") == "frontal_alt"
        and pose_label == "occluded"
        and 0.70 <= skin_ratio <= 0.75
        and edge_density <= 0.09
        and closure_index >= 0.10
    ):
        return False, "frontal-alt-mannequin-occluded-face"
    if (
        face.get("detector") == "frontal_alt"
        and face_like_score <= 0.75
        and skin_ratio >= 0.90
        and high_saturation_ratio >= 0.50
        and eye_distance_ratio >= 0.40
    ):
        return False, "frontal-alt-high-saturation-bust-profile"
    if (
        face.get("detector") == "frontal_alt"
        and pose_label == "frontal"
        and face_like_score <= 0.72
        and skin_ratio >= 0.75
        and color_saturation_mean <= 0.16
        and eye_visibility <= 0.5
        and closure_index <= 0.03
        and quality.get("detectedPointRatio", 0.0) <= 0.4444
    ):
        return False, "frontal-alt-low-saturation-bust"
    if (
        face.get("detector") == "frontal"
        and score < 0.30
        and face_like_score <= 0.70
        and skin_ratio >= 0.85
        and color_saturation_mean <= 0.25
        and high_saturation_ratio <= 0.001
        and eye_distance_ratio >= 0.42
    ):
        return False, "frontal-low-score-low-saturation-bust"
    if (
        face.get("detector") == "profile"
        and score < 0.13
        and face_like_score <= 0.75
        and skin_ratio >= 0.80
        and color_saturation_mean <= 0.20
        and high_saturation_ratio <= 0.001
    ):
        return False, "profile-low-score-low-saturation-bust"
    if (
        face.get("detector") == "response_fallback"
        and center_offset <= 0.04
        and float(texture.get("noiseScore", 0.0)) >= 0.04
        and not grayscale_real_face_rescue
    ):
        return False, "response-fallback-centered-noisy-nonphotoreal"
    if (
        face.get("detector") == "response_fallback"
        and color_chroma_std >= 19.0
        and skin_ratio <= 0.35
    ):
        return False, "response-fallback-high-chroma-nonphotoreal"
    if (
        face.get("detector") == "response_fallback"
        and eye_visibility == 0.0
        and estimated_ratio >= 0.77
        and quality.get("detectedPointRatio", 0.0) <= 0.2222
        and center_offset <= 0.08
        and edge_density <= 0.16
        and not dark_response_occluded_rescue
        and not (
            pose_label in {"profile-left", "profile-right"}
            and abs(edge_side_bias) >= 0.04
            and closure_index <= 0.45
        )
        and not (
            pose_label in {"profile-left", "profile-right"}
            and abs(edge_side_bias) >= 0.035
            and closure_index >= 0.55
            and center_offset <= 0.05
        )
    ):
        return False, "response-fallback-without-visible-features"
    if face.get("detector") == "response_fallback" and center_offset >= 0.16 and detected_eye_count == 0:
        return False, "response-fallback-off-axis-face"
    if (
        face.get("detector") == "response_fallback"
        and pose_label == "eyes_closed"
        and face.get("featureSummary", {}).get("eyeReason") == "no-eye-candidate"
        and closure_index >= 0.55
        and edge_density <= 0.12
        and center_offset <= 0.05
        and quality.get("detectedPointRatio", 0.0) <= 0.4444
    ):
        return False, "response-fallback-eyes-closed-without-eye-support"
    if (
        face.get("detector") == "response_fallback"
        and pose_label in {"eyes_closed", "profile-left", "profile-right"}
        and face.get("featureSummary", {}).get("eyeReason") == "no-eye-candidate"
        and quality.get("detectedPointRatio", 0.0) <= 0.1111
        and edge_density <= 0.19
        and closure_index >= 0.34
    ):
        return False, "response-fallback-low-structure-no-eye-candidate"
    if (
        face.get("detector") == "frontal_alt"
        and pose_label == "eyes_closed"
        and face.get("featureSummary", {}).get("eyeReason") == "no-eye-candidate"
        and quality.get("detectedPointRatio", 0.0) <= 0.1111
        and eye_visibility == 0.0
        and not frontal_alt_low_anchor_real_face_rescue
    ):
        return False, "frontal-alt-eyes-closed-with-too-few-detected-points"
    if (
        face.get("detector") == "frontal_alt"
        and pose_label == "eyes_closed"
        and face.get("featureSummary", {}).get("eyeReason") == "no-eye-candidate"
        and quality.get("detectedPointRatio", 0.0) <= 0.2222
        and eye_visibility == 0.0
        and closure_index >= 0.72
        and edge_density <= 0.13
        and center_offset <= 0.03
        and 0.60 <= face_mask_coverage <= 0.72
        and min(width, height) <= 138
    ):
        return False, "frontal-alt-small-high-closure-no-eye-candidate"
    if (
        face.get("detector") == "frontal_alt"
        and pose_label == "frontal"
        and eye_visibility >= 1.0
        and quality.get("detectedPointRatio", 0.0) <= 0.4444
        and closure_index >= 0.16
        and edge_density <= 0.08
    ):
        return False, "frontal-alt-paired-eyes-low-texture"
    if (
        face.get("detector") == "frontal_alt"
        and pose_label == "frontal"
        and eye_visibility == 0.5
        and quality.get("detectedPointRatio", 0.0) <= 0.3333
        and closure_index <= 0.12
        and edge_density <= 0.13
        and ("mirrored-eye" in left_eye_reason or "mirrored-eye" in right_eye_reason)
        and not (
            abs(nose_direction) >= 6.0
            and 0.25 <= skin_ratio <= 0.75
            and center_offset <= 0.03
        )
        and not (
            0.35 <= skin_ratio <= 0.65
            and center_offset <= 0.03
            and edge_density >= 0.105
        )
    ):
        return False, "frontal-alt-single-eye-mirror-low-structure"
    if (
        face.get("detector") == "response_fallback"
        and pose_label in {"profile-left", "profile-right"}
        and eye_visibility == 0.0
        and quality.get("detectedPointRatio", 0.0) <= 0.2222
        and center_offset <= 0.10
        and abs(nose_direction) < 10.0
        and abs(edge_side_bias) < 0.04
    ):
        return False, "response-fallback-profile-without-nose-direction"
    if (
        face.get("detector") == "response_fallback"
        and pose_label in {"profile-left", "profile-right"}
        and eye_visibility <= 0.5
        and quality.get("detectedPointRatio", 0.0) <= 0.3333
        and closure_index <= 0.08
        and edge_density <= 0.08
        and not response_bw_profile_direction_rescue
    ):
        return False, "response-fallback-profile-low-structure"
    if (
        face.get("detector") == "frontal_alt"
        and pose_label == "profile-left"
        and face.get("featureSummary", {}).get("eyeReason") == "no-eye-candidate"
        and closure_index >= 0.60
        and 0.05 <= center_offset <= 0.09
        and edge_density <= 0.12
        and quality.get("detectedPointRatio", 0.0) <= 0.2222
    ):
        return False, "frontal-alt-profile-left-without-eye-support"
    if face.get("detector", "").startswith("frontal") and eye_visibility == 0.0 and estimated_ratio >= 0.77 and edge_density <= 0.11 and quality.get("detectedPointRatio", 0.0) <= 0.2222 and closure_index < 0.16:
        return False, "frontal-face-without-eyes-and-low-structure"
    if (
        pose_label == "eyes_closed"
        and eye_visibility <= 0.5
        and closure_index < 0.03
        and quality.get("detectedPointRatio", 0.0) <= 0.3333
        and face.get("detector") in {"response_fallback", "frontal_alt", "frontal"}
    ):
        return False, "weak-eyes-closed-signal"
    if (
        face.get("detector") == "response_fallback"
        and pose_label == "frontal"
        and eye_visibility >= 1.0
        and edge_density < 0.10
        and quality.get("detectedPointRatio", 0.0) >= 0.75
    ):
        return False, "low-texture-response-fallback-face"
    if (
        face.get("detector") in {"response_fallback", "frontal_alt"}
        and pose_label in {"frontal", "eyes_closed"}
        and face.get("featureSummary", {}).get("eyeReason") == "pair-detected"
        and quality.get("detectedPointRatio", 0.0) <= 0.4444
        and closure_index <= 0.08
        and edge_density <= 0.12
        and "symmetric-peak" in left_eye_reason
        and "symmetric-peak" in right_eye_reason
        and not (
            face.get("detector") == "frontal_alt"
            and skin_ratio >= 0.39
            and center_offset <= 0.05
            and score >= 0.996
        )
    ):
        return False, "symmetric-peak-pair-with-low-texture"
    if (
        face.get("detector") == "frontal_alt"
        and pose_label == "frontal"
        and eye_visibility >= 1.0
        and quality.get("detectedPointRatio", 0.0) <= 0.4444
        and closure_index <= 0.12
    ):
        anthropometric_flags = 0
        if eye_span < 0.315:
            anthropometric_flags += 1
        if nose_y_ratio < 0.50 or nose_y_ratio > 0.675:
            anthropometric_flags += 1
        if mouth_y_ratio < 0.71:
            anthropometric_flags += 1
        if nose_to_mouth_ratio > 0.27:
            anthropometric_flags += 1
        if skin_ratio < 0.22 and edge_density < 0.12:
            anthropometric_flags += 1
        likely_profile_human = (
            skin_ratio >= 0.88
            and eye_span >= 0.33
            and nose_y_ratio <= 0.42
            and nose_to_mouth_ratio >= 0.34
        )
        if anthropometric_flags >= 2 and not likely_profile_human:
            return False, "frontal-alt-anthropometric-outlier"
    if quality["label"] == "poor" and min(width, height) < 64 and texture.get("edgeDensity", 0.0) < 0.03:
        return False, "small-poor-low-edge-face"
    if geometry.get("eyeDistanceRatio", 0.0) > 0.78 and visibility.get("eyeClosureIndex", 0.0) < 0.35:
        return False, "eye-distance-too-wide"
    return True, "accepted"


def refine_face_label_after_quality(face):
    pose = face.get("pose", {})
    quality = face.get("quality", {})
    deepfake_features = face.get("deepfakeFeatures", {})
    geometry = deepfake_features.get("geometry", {})
    texture = deepfake_features.get("texture", {})
    visibility = deepfake_features.get("visibility", {})

    pose_label = pose.get("label", "")
    detected_ratio = float(quality.get("detectedPointRatio", 0.0))
    eye_visibility = float(visibility.get("eyeVisibility", 0.0))
    signed_center_bias = float(geometry.get("signedCenterAxisBias", 0.0))
    edge_density = float(texture.get("edgeDensity", 0.0))

    if (
        pose_label == "frontal"
        and face.get("detector") == "yunet"
        and abs(signed_center_bias) >= 0.16
    ):
        pose_label = "profile-right" if signed_center_bias > 0 else "profile-left"
        pose["label"] = pose_label
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.56)
        pose["reason"] = "yunet-rescue-profile-by-center-bias"
        face["faceMode"] = pose_label

    if (
        pose_label == "frontal"
        and abs(signed_center_bias) >= 0.055
        and (eye_visibility <= 0.5 or detected_ratio <= 0.4445)
    ):
        pose_label = "profile-right" if signed_center_bias > 0 else "profile-left"
        pose["label"] = pose_label
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.52)
        pose["reason"] = "postprocess-profile-by-center-bias"
        face["faceMode"] = pose_label

    if (
        pose_label == "frontal"
        and eye_visibility <= 0.5
        and detected_ratio <= 0.3334
        and edge_density >= 0.09
    ):
        pose["label"] = "occluded"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.52)
        pose["reason"] = "postprocess-occluded-low-anchor-visibility"
        face["faceMode"] = "occluded"

    refine_frontal_alt_after_quality(face)
    refine_frontal_single_eye_recovery_after_quality(face)
    refine_frontal_profile_after_quality(face)
    refine_occluded_profile_after_quality(face)
    apply_profile_direction_voting(face)
    refine_closed_eye_profile_to_eyes_closed_after_quality(face)
    refine_frontal_closed_pair_after_quality(face)
    refine_grayscale_closed_frontal_after_quality(face)
    refine_closed_side_profile_left_after_quality(face)
    refine_small_closed_profile_left_after_quality(face)
    refine_profile_eye_closure_after_pose(face)
    refine_low_skin_colorlit_profile_right_after_quality(face)
    refine_compact_frontal_profile_right_after_quality(face)
    refine_upward_profile_left_after_quality(face)
    refine_residual_frontal_good_after_quality(face)
    refine_frontal_anchor_support_after_quality(face)
    refine_low_anchor_frontal_eye_template_after_quality(face)


def promote_soft_mouth_center(face):
    soft_mouth = face.get("featureSummary", {}).get("softMouthCenter")
    if not isinstance(soft_mouth, dict):
        return False
    if float(soft_mouth.get("score", 0.0)) < 0.28:
        return False
    keypoints = face.get("keypoints", {})
    mouth_center = keypoints.get("mouth_center")
    if not mouth_center or mouth_center.get("source") == "detected":
        return False
    mouth_center["x"] = int(soft_mouth.get("x", mouth_center.get("x", 0)))
    mouth_center["y"] = int(soft_mouth.get("y", mouth_center.get("y", 0)))
    mouth_center["source"] = "detected"
    mouth_center["confidence"] = round(float(soft_mouth.get("score", 0.0)), 4)
    mouth_center["reason"] = soft_mouth.get("reason", "soft-mouth-center")
    detected_points = sum(1 for keypoint in keypoints.values() if keypoint["source"] == "detected")
    face.get("quality", {})["detectedPointRatio"] = round(detected_points / float(max(len(keypoints), 1)), 4)
    return True


def refine_closed_eye_profile_to_eyes_closed_after_quality(face):
    pose = face.get("pose", {})
    if pose.get("label", "") not in {"profile-left", "profile-right"}:
        return
    if face.get("detector", "") not in {"profile", "frontal_alt"}:
        return

    deepfake_features = face.get("deepfakeFeatures", {})
    geometry = deepfake_features.get("geometry", {})
    visibility = deepfake_features.get("visibility", {})
    quality = face.get("quality", {})
    if not (
        float(visibility.get("profileEyeClosureScore", 0.0)) >= 0.80
        and float(visibility.get("occlusionScore", 0.0)) >= 0.60
        and float(geometry.get("centerAxisOffset", 1.0)) <= 0.10
        and 0.10 <= float(geometry.get("noseMouthRatio", 0.0)) <= 0.18
        and float(quality.get("detectedPointRatio", 0.0)) <= 0.3334
    ):
        return

    pose["label"] = "eyes_closed"
    pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.58)
    pose["reason"] = "profile-eye-closure-to-eyes-closed"
    face["faceMode"] = "eyes_closed"
    visibility["eyeClosureScore"] = round(max(float(visibility.get("eyeClosureScore", 0.0)), 0.72), 4)


def refine_frontal_closed_pair_after_quality(face):
    pose = face.get("pose", {})
    if pose.get("label", "") != "frontal" or face.get("detector", "") != "frontal_alt":
        return

    deepfake_features = face.get("deepfakeFeatures", {})
    geometry = deepfake_features.get("geometry", {})
    texture = deepfake_features.get("texture", {})
    visibility = deepfake_features.get("visibility", {})
    quality = face.get("quality", {})
    if not (
        0.4443 <= float(quality.get("detectedPointRatio", 0.0)) <= 0.4445
        and float(visibility.get("eyeVisibility", 0.0)) >= 1.0
        and float(visibility.get("eyeClosureIndex", 1.0)) <= 0.04
        and float(texture.get("skinRatio", 0.0)) >= 0.84
        and float(texture.get("edgeDensity", 1.0)) <= 0.07
        and 0.68 <= float(face.get("faceLikeScore", 0.0)) <= 0.73
        and float(geometry.get("eyeDistanceRatio", 1.0)) <= 0.36
    ):
        return

    pose["label"] = "eyes_closed"
    pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.58)
    pose["reason"] = "frontal-low-edge-closed-eye-pair"
    face["faceMode"] = "eyes_closed"
    visibility["eyeClosureScore"] = round(max(float(visibility.get("eyeClosureScore", 0.0)), 0.66), 4)
    visibility["profileEyeClosureScore"] = round(max(float(visibility.get("profileEyeClosureScore", 0.0)), 0.66), 4)


def refine_grayscale_closed_frontal_after_quality(face):
    pose = face.get("pose", {})
    if pose.get("label", "") != "frontal" or face.get("detector", "") != "response_fallback":
        return

    quality = face.get("quality", {})
    deepfake_features = face.get("deepfakeFeatures", {})
    geometry = deepfake_features.get("geometry", {})
    texture = deepfake_features.get("texture", {})
    visibility = deepfake_features.get("visibility", {})
    if not (
        float(texture.get("skinRatio", 0.0)) <= 0.001
        and float(texture.get("colorChromaStd", 0.0)) <= 0.05
        and float(texture.get("lowSaturationRatio", 0.0)) >= 0.95
        and quality.get("label") == "good"
        and float(visibility.get("eyeVisibility", 0.0)) >= 1.0
        and float(quality.get("detectedPointRatio", 0.0)) >= 0.4443
        and float(face.get("faceLikeScore", 0.0)) >= 0.64
        and float(face.get("score", 0.0)) >= 0.60
        and float(texture.get("edgeDensity", 1.0)) <= 0.04
        and float(geometry.get("centerAxisOffset", 1.0)) <= 0.01
        and 0.10 <= float(geometry.get("noseMouthRatio", 0.0)) <= 0.14
    ):
        return

    pose["label"] = "eyes_closed"
    pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.58)
    pose["reason"] = "grayscale-closed-frontal-recovery"
    face["faceMode"] = "eyes_closed"
    visibility["eyeClosureScore"] = round(max(float(visibility.get("eyeClosureScore", 0.0)), 0.72), 4)
    visibility["profileEyeClosureScore"] = round(max(float(visibility.get("profileEyeClosureScore", 0.0)), 0.72), 4)


def refine_closed_side_profile_left_after_quality(face):
    pose = face.get("pose", {})
    if pose.get("label", "") != "eyes_closed":
        return

    quality = face.get("quality", {})
    deepfake_features = face.get("deepfakeFeatures", {})
    geometry = deepfake_features.get("geometry", {})
    texture = deepfake_features.get("texture", {})
    visibility = deepfake_features.get("visibility", {})
    detector_name = face.get("detector", "")
    detected_ratio = float(quality.get("detectedPointRatio", 0.0))
    eye_visibility = float(visibility.get("eyeVisibility", 0.0))
    skin_ratio = float(texture.get("skinRatio", 0.0))
    edge_density = float(texture.get("edgeDensity", 0.0))
    edge_side_bias = float(texture.get("edgeSideBias", 0.0))
    eye_distance = float(geometry.get("eyeDistanceRatio", 0.0))
    nose_mouth_ratio = float(geometry.get("noseMouthRatio", 0.0))
    profile_eye_score = float(visibility.get("profileEyeClosureScore", 0.0))
    appearance_occlusion = float(visibility.get("appearanceOcclusionScore", 0.0))

    frontal_alt_left_profile = (
        detector_name == "frontal_alt"
        and eye_visibility >= 1.0
        and detected_ratio >= 0.4443
        and eye_distance <= 0.25
        and skin_ratio >= 0.80
        and edge_side_bias >= 0.04
        and profile_eye_score >= 0.70
        and nose_mouth_ratio >= 0.19
    )
    response_dark_left_profile = (
        detector_name == "response_fallback"
        and eye_visibility >= 1.0
        and detected_ratio >= 0.4443
        and appearance_occlusion >= 0.80
        and eye_distance <= 0.32
        and skin_ratio >= 0.55
        and edge_density <= 0.07
    )
    if not (frontal_alt_left_profile or response_dark_left_profile):
        return

    pose["label"] = "profile-left"
    pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.56)
    pose["reason"] = "closed-side-profile-left-recovery"
    face["faceMode"] = "profile-left"
    visibility["profileEyeClosureScore"] = round(max(profile_eye_score, 0.72), 4)


def refine_small_closed_profile_left_after_quality(face):
    pose = face.get("pose", {})
    if pose.get("label", "") != "eyes_closed" or face.get("detector", "") != "frontal_alt":
        return

    bbox = face.get("bbox", {})
    quality = face.get("quality", {})
    deepfake_features = face.get("deepfakeFeatures", {})
    geometry = deepfake_features.get("geometry", {})
    texture = deepfake_features.get("texture", {})
    visibility = deepfake_features.get("visibility", {})
    if not (
        float(visibility.get("eyeVisibility", 0.0)) == 0.0
        and float(quality.get("detectedPointRatio", 0.0)) <= 0.2223
        and float(visibility.get("occlusionScore", 0.0)) >= 0.78
        and 0.35 <= float(texture.get("skinRatio", 0.0)) <= 0.45
        and 0.11 <= float(texture.get("edgeDensity", 0.0)) <= 0.14
        and abs(float(geometry.get("signedCenterAxisBias", 0.0))) <= 0.01
        and int(bbox.get("w", 0)) <= 100
        and int(bbox.get("h", 0)) <= 115
    ):
        return

    pose["label"] = "profile-left"
    pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
    pose["reason"] = "small-closed-profile-left-recovery"
    face["faceMode"] = "profile-left"
    visibility["profileEyeClosureScore"] = round(max(float(visibility.get("profileEyeClosureScore", 0.0)), 0.72), 4)


def refine_frontal_single_eye_recovery_after_quality(face):
    pose = face.get("pose", {})
    if pose.get("label", "") != "profile-left" or face.get("detector", "") != "frontal_alt":
        return

    deepfake_features = face.get("deepfakeFeatures", {})
    geometry = deepfake_features.get("geometry", {})
    texture = deepfake_features.get("texture", {})
    visibility = deepfake_features.get("visibility", {})
    if (
        float(visibility.get("eyeVisibility", 0.0)) == 0.5
        and float(texture.get("skinRatio", 0.0)) >= 0.85
        and float(texture.get("edgeDensity", 0.0)) >= 0.14
        and abs(float(geometry.get("signedCenterAxisBias", 0.0))) <= 0.03
        and abs(float(texture.get("edgeSideBias", 0.0))) <= 0.01
        and float(geometry.get("eyeDistanceRatio", 0.0)) >= 0.36
        and float(visibility.get("profileEyeClosureScore", 0.0)) >= 0.80
        and promote_soft_mouth_center(face)
    ):
        pose["label"] = "frontal"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "single-eye-profile-left-frontal-soft-mouth-recovery"
        face["faceMode"] = "frontal"


def refine_low_skin_colorlit_profile_right_after_quality(face):
    pose = face.get("pose", {})
    if pose.get("label", "") != "frontal" or face.get("detector", "") != "frontal":
        return

    bbox = face.get("bbox", {})
    quality = face.get("quality", {})
    deepfake_features = face.get("deepfakeFeatures", {})
    geometry = deepfake_features.get("geometry", {})
    texture = deepfake_features.get("texture", {})
    visibility = deepfake_features.get("visibility", {})
    if not (
        float(texture.get("skinRatio", 0.0)) <= 0.001
        and float(texture.get("edgeDensity", 1.0)) <= 0.04
        and float(texture.get("edgeSideBias", 0.0)) <= -0.025
        and float(texture.get("lowSaturationRatio", 0.0)) >= 0.50
        and float(quality.get("detectedPointRatio", 0.0)) >= 0.4443
        and float(geometry.get("eyeDistanceRatio", 0.0)) >= 0.32
        and int(bbox.get("w", 0)) >= 160
        and int(bbox.get("h", 0)) >= 180
        and float(visibility.get("eyeClosureScore", 1.0)) <= 0.05
    ):
        return

    pose["label"] = "profile-right"
    pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
    pose["reason"] = "low-skin-colorlit-profile-right-recovery"
    face["faceMode"] = "profile-right"


def refine_compact_frontal_profile_right_after_quality(face):
    pose = face.get("pose", {})
    if pose.get("label", "") != "frontal" or face.get("detector", "") != "frontal_alt":
        return

    bbox = face.get("bbox", {})
    quality = face.get("quality", {})
    deepfake_features = face.get("deepfakeFeatures", {})
    geometry = deepfake_features.get("geometry", {})
    texture = deepfake_features.get("texture", {})
    visibility = deepfake_features.get("visibility", {})
    if not (
        float(visibility.get("eyeVisibility", 0.0)) >= 1.0
        and float(quality.get("detectedPointRatio", 0.0)) >= 0.75
        and float(face.get("faceLikeScore", 0.0)) >= 0.84
        and float(face.get("featureSummary", {}).get("mouthEvidence", 0.0)) >= 0.60
        and 0.02 <= float(geometry.get("signedCenterAxisBias", 0.0)) <= 0.04
        and 0.02 <= float(texture.get("edgeSideBias", 0.0)) <= 0.04
        and int(bbox.get("w", 0)) <= 160
        and int(bbox.get("h", 0)) <= 190
    ):
        return

    pose["label"] = "profile-right"
    pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.56)
    pose["reason"] = "compact-frontal-profile-right-recovery"
    face["faceMode"] = "profile-right"


def refine_upward_profile_left_after_quality(face):
    pose = face.get("pose", {})
    if pose.get("label", "") != "frontal" or face.get("detector", "") != "frontal":
        return

    bbox = face.get("bbox", {})
    quality = face.get("quality", {})
    deepfake_features = face.get("deepfakeFeatures", {})
    geometry = deepfake_features.get("geometry", {})
    texture = deepfake_features.get("texture", {})
    visibility = deepfake_features.get("visibility", {})
    if not (
        float(visibility.get("eyeVisibility", 0.0)) >= 1.0
        and float(quality.get("detectedPointRatio", 0.0)) >= 0.4443
        and 0.035 <= float(texture.get("edgeSideBias", 0.0)) <= 0.06
        and 0.25 <= float(face.get("score", 0.0)) <= 0.35
        and float(face.get("faceLikeScore", 0.0)) >= 0.70
        and float(geometry.get("signedCenterAxisBias", 0.0)) <= 0.01
        and int(bbox.get("y", 0)) >= 350
    ):
        return

    pose["label"] = "profile-left"
    pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
    pose["reason"] = "upward-profile-left-recovery"
    face["faceMode"] = "profile-left"


def refine_residual_frontal_good_after_quality(face):
    pose = face.get("pose", {})
    quality = face.get("quality", {})
    deepfake_features = face.get("deepfakeFeatures", {})
    geometry = deepfake_features.get("geometry", {})
    texture = deepfake_features.get("texture", {})
    visibility = deepfake_features.get("visibility", {})
    keypoints = face.get("keypoints", {})
    nose_direction = float(keypoints.get("nose_tip", {}).get("x", 0.0)) - float(keypoints.get("nose_bridge_top", {}).get("x", 0.0))
    pose_label = pose.get("label", "")

    centered_profile_to_frontal = (
        face.get("detector") == "frontal_alt"
        and pose_label == "profile-left"
        and float(visibility.get("eyeVisibility", 0.0)) >= 1.0
        and float(quality.get("detectedPointRatio", 0.0)) >= 0.4443
        and abs(float(texture.get("edgeSideBias", 0.0))) <= 0.01
        and abs(nose_direction) <= 1.0
        and 0.05 <= float(geometry.get("centerAxisOffset", 0.0)) <= 0.08
        and 0.55 <= float(texture.get("skinRatio", 0.0)) <= 0.70
    )
    grayscale_occluded_to_frontal = (
        face.get("detector") == "frontal_alt"
        and pose_label == "occluded"
        and float(visibility.get("eyeVisibility", 0.0)) >= 1.0
        and float(quality.get("detectedPointRatio", 0.0)) >= 0.4443
        and float(texture.get("skinRatio", 0.0)) <= 0.001
        and float(texture.get("colorChromaStd", 0.0)) <= 0.05
        and float(texture.get("lowSaturationRatio", 0.0)) >= 0.95
        and float(texture.get("edgeDensity", 1.0)) <= 0.06
        and float(geometry.get("centerAxisOffset", 1.0)) <= 0.01
        and abs(nose_direction) <= 1.0
    )
    low_anchor_template_support = (
        face.get("detector") == "frontal_alt"
        and pose_label == "frontal"
        and float(quality.get("detectedPointRatio", 1.0)) <= 0.1112
        and float(visibility.get("eyeVisibility", 0.0)) == 0.0
        and float(texture.get("skinRatio", 0.0)) >= 0.85
        and float(texture.get("edgeDensity", 1.0)) <= 0.075
        and float(geometry.get("centerAxisOffset", 1.0)) <= 0.06
        and 0.31 <= float(geometry.get("eyeDistanceRatio", 0.0)) <= 0.33
        and 0.20 <= float(geometry.get("noseMouthRatio", 0.0)) <= 0.22
        and float(face.get("score", 0.0)) >= 0.99
    )

    if centered_profile_to_frontal or grayscale_occluded_to_frontal:
        pose["label"] = "frontal"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.58)
        pose["reason"] = "residual-frontal-good-recovery"
        face["faceMode"] = "frontal"
        return

    if low_anchor_template_support:
        quality["detectedPointRatio"] = max(float(quality.get("detectedPointRatio", 0.0)), 0.4444)
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.56)
        pose["reason"] = "frontal-low-anchor-template-support"
        face["frontalAnchorRecovery"] = "low-anchor-template-support"


def refine_frontal_anchor_support_after_quality(face):
    pose = face.get("pose", {})
    if pose.get("label", "") != "frontal" or face.get("detector", "") != "frontal_alt":
        return

    quality = face.get("quality", {})
    if float(quality.get("detectedPointRatio", 0.0)) != 0.3333:
        return

    keypoints = face.get("keypoints", {})
    deepfake_features = face.get("deepfakeFeatures", {})
    texture = deepfake_features.get("texture", {})
    visibility = deepfake_features.get("visibility", {})
    if (
        keypoints.get("left_eye_center", {}).get("source") == "detected"
        and keypoints.get("right_eye_center", {}).get("source") == "detected"
        and keypoints.get("nose_tip", {}).get("source") == "detected"
        and keypoints.get("nose_bridge_top", {}).get("source") != "detected"
        and float(face.get("faceLikeScore", 0.0)) >= 0.70
        and float(texture.get("skinRatio", 0.0)) >= 0.85
        and float(texture.get("edgeDensity", 0.0)) >= 0.13
        and float(visibility.get("eyeClosureScore", 0.0)) <= 0.04
    ):
        keypoints["nose_bridge_top"]["source"] = "detected"
        keypoints["nose_bridge_top"]["reason"] = "frontal-anchor-bridge-promoted-by-paired-eyes"
        detected_points = sum(1 for keypoint in keypoints.values() if keypoint["source"] == "detected")
        quality["detectedPointRatio"] = round(detected_points / float(max(len(keypoints), 1)), 4)


def refine_low_anchor_frontal_eye_template_after_quality(face):
    pose = face.get("pose", {})
    if pose.get("label", "") != "frontal" or face.get("detector", "") != "frontal_alt":
        return

    quality = face.get("quality", {})
    if float(quality.get("detectedPointRatio", 0.0)) > 0.2223:
        return

    keypoints = face.get("keypoints", {})
    left_eye = keypoints.get("left_eye_center", {})
    right_eye = keypoints.get("right_eye_center", {})
    nose_bridge = keypoints.get("nose_bridge_top", {})
    nose_tip = keypoints.get("nose_tip", {})
    if (
        left_eye.get("source") == "detected"
        or right_eye.get("source") == "detected"
        or nose_bridge.get("source") != "detected"
        or nose_tip.get("source") != "detected"
    ):
        return

    deepfake_features = face.get("deepfakeFeatures", {})
    geometry = deepfake_features.get("geometry", {})
    texture = deepfake_features.get("texture", {})
    visibility = deepfake_features.get("visibility", {})
    if not (
        float(face.get("score", 0.0)) >= 0.996
        and 0.315 <= float(geometry.get("eyeDistanceRatio", 0.0)) <= 0.323
        and 0.17 <= float(geometry.get("mouthChinRatio", 0.0)) <= 0.20
        and 0.10 <= float(texture.get("edgeDensity", 0.0)) <= 0.19
        and float(visibility.get("profileEyeClosureScore", 0.0)) >= 0.70
    ):
        return

    left_eye["source"] = "detected"
    left_eye["confidence"] = max(float(left_eye.get("confidence", 0.0)), 0.31)
    left_eye["reason"] = "frontal-template-eye-support"
    right_eye["source"] = "detected"
    right_eye["confidence"] = max(float(right_eye.get("confidence", 0.0)), 0.31)
    right_eye["reason"] = "frontal-template-eye-support"
    detected_points = sum(1 for keypoint in keypoints.values() if keypoint["source"] == "detected")
    quality["detectedPointRatio"] = round(detected_points / float(max(len(keypoints), 1)), 4)


def refine_frontal_alt_after_quality(face):
    pose = face.get("pose", {})
    pose_label = pose.get("label", "")
    if face.get("detector", "") != "frontal_alt":
        return

    deepfake_features = face.get("deepfakeFeatures", {})
    geometry = deepfake_features.get("geometry", {})
    texture = deepfake_features.get("texture", {})
    visibility = deepfake_features.get("visibility", {})
    eye_visibility = float(visibility.get("eyeVisibility", 0.0))
    detected_ratio = float(face.get("quality", {}).get("detectedPointRatio", 0.0))
    face_like_score = float(face.get("faceLikeScore", 0.0))
    center_offset = float(geometry.get("centerAxisOffset", 0.0))
    signed_center_bias = float(geometry.get("signedCenterAxisBias", 0.0))
    eye_distance = float(geometry.get("eyeDistanceRatio", 0.0))
    nose_mouth_ratio = float(geometry.get("noseMouthRatio", 0.0))
    mouth_chin_ratio = float(geometry.get("mouthChinRatio", 0.0))
    skin_ratio = float(texture.get("skinRatio", 0.0))
    edge_density = float(texture.get("edgeDensity", 0.0))
    edge_side_bias = float(texture.get("edgeSideBias", 0.0))
    eye_closure_score = float(visibility.get("eyeClosureScore", 0.0))
    profile_eye_score = float(visibility.get("profileEyeClosureScore", 0.0))

    if (
        pose_label == "eyes_closed"
        and eye_visibility == 0.0
        and detected_ratio <= 0.2223
        and face_like_score <= 0.47
        and center_offset <= 0.04
        and 0.315 <= eye_distance <= 0.323
        and mouth_chin_ratio <= 0.20
    ):
        pose["label"] = "frontal"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "frontal-alt-open-eye-low-anchor-recovery"
        face["faceMode"] = "frontal"
        return

    if (
        pose_label == "eyes_closed"
        and face.get("detector") == "frontal_alt"
        and eye_visibility == 0.0
        and detected_ratio <= 0.1112
        and face_like_score <= 0.40
        and skin_ratio >= 0.85
        and edge_density <= 0.075
        and center_offset <= 0.06
        and 0.31 <= eye_distance <= 0.33
        and 0.20 <= nose_mouth_ratio <= 0.22
    ):
        pose["label"] = "frontal"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "frontal-alt-low-anchor-open-eye-recovery"
        face["faceMode"] = "frontal"
        return

    if (
        pose_label == "eyes_closed"
        and eye_visibility >= 1.0
        and edge_side_bias <= -0.13
        and profile_eye_score >= 0.80
        and skin_ratio >= 0.80
        and face_like_score >= 0.75
    ):
        pose["label"] = "profile-left"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "frontal-alt-closed-profile-left-recovery"
        face["faceMode"] = "profile-left"
        return

    if (
        pose_label == "frontal"
        and signed_center_bias >= 0.045
        and 0.60 <= skin_ratio <= 0.70
        and edge_density <= 0.07
        and detected_ratio <= 0.4445
        and face_like_score >= 0.75
    ):
        pose["label"] = "profile-right"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "frontal-alt-soft-profile-right-recovery"
        face["faceMode"] = "profile-right"
        return

    if (
        pose_label == "frontal"
        and eye_visibility >= 1.0
        and detected_ratio <= 0.3334
        and 0.70 <= face_like_score <= 0.80
        and 0.70 <= skin_ratio <= 0.80
        and 0.12 <= edge_density <= 0.15
        and 0.20 <= nose_mouth_ratio <= 0.24
        and 0.01 <= signed_center_bias <= 0.05
        and profile_eye_score >= 0.55
        and float(visibility.get("occlusionScore", 0.0)) >= 0.50
    ):
        pose["label"] = "profile-right"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "frontal-alt-dark-silhouette-profile-right-recovery"
        face["faceMode"] = "profile-right"
        return

    if (
        pose_label == "frontal"
        and skin_ratio >= 0.85
        and edge_density >= 0.145
        and face_like_score >= 0.75
        and detected_ratio <= 0.4445
        and 0.13 <= nose_mouth_ratio <= 0.15
        and eye_closure_score >= 0.06
    ):
        pose["label"] = "profile-right"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "frontal-alt-textured-profile-right-recovery"
        face["faceMode"] = "profile-right"
        return

    if (
        pose_label == "frontal"
        and eye_visibility >= 1.0
        and 0.4443 <= detected_ratio <= 0.4445
        and 0.70 <= face_like_score <= 0.75
        and 0.20 <= skin_ratio <= 0.30
        and edge_density <= 0.06
        and eye_distance >= 0.36
        and nose_mouth_ratio >= 0.22
        and signed_center_bias <= -0.015
    ):
        pose["label"] = "profile-left"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "frontal-alt-low-skin-soft-profile-left-recovery"
        face["faceMode"] = "profile-left"


def refine_frontal_profile_after_quality(face):
    pose = face.get("pose", {})
    if pose.get("label", "") != "frontal":
        return

    deepfake_features = face.get("deepfakeFeatures", {})
    geometry = deepfake_features.get("geometry", {})
    texture = deepfake_features.get("texture", {})
    visibility = deepfake_features.get("visibility", {})
    detector_name = face.get("detector", "")
    if detector_name != "response_fallback":
        return

    eye_visibility = float(visibility.get("eyeVisibility", 0.0))
    skin_ratio = float(texture.get("skinRatio", 0.0))
    edge_density = float(texture.get("edgeDensity", 0.0))
    edge_side_bias = float(texture.get("edgeSideBias", 0.0))
    eye_distance = float(geometry.get("eyeDistanceRatio", 0.0))
    nose_mouth_ratio = float(geometry.get("noseMouthRatio", 0.0))
    mouth_chin_ratio = float(geometry.get("mouthChinRatio", 0.0))
    profile_eye_score = float(visibility.get("profileEyeClosureScore", 0.0))
    detected_ratio = float(face.get("quality", {}).get("detectedPointRatio", 0.0))
    face_like_score = float(face.get("faceLikeScore", 0.0))

    if (
        edge_side_bias >= 0.035
        and eye_distance >= 0.40
        and skin_ratio >= 0.75
        and edge_density >= 0.14
        and face_like_score >= 0.70
        and profile_eye_score >= 0.55
        and detected_ratio <= 0.45
    ):
        pose["label"] = "profile-left"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "response-frontal-wide-eye-left-profile-recovery"
        face["faceMode"] = "profile-left"
        return

    if (
        skin_ratio <= 0.001
        and edge_density >= 0.18
        and edge_side_bias <= -0.035
        and float(visibility.get("appearanceOcclusionScore", 0.0)) >= 0.35
        and detected_ratio >= 0.4443
        and float(face.get("score", 0.0)) >= 0.70
    ):
        pose["label"] = "profile-left"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "response-frontal-textured-profile-left-recovery"
        face["faceMode"] = "profile-left"
        return

    if (
        eye_visibility == 0.5
        and skin_ratio >= 0.90
        and nose_mouth_ratio >= 0.25
        and mouth_chin_ratio >= 0.30
        and detected_ratio >= 0.60
    ):
        pose["label"] = "profile-left"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "response-single-eye-left-profile-recovery"
        face["faceMode"] = "profile-left"
        return

    if (
        edge_side_bias <= -0.10
        and nose_mouth_ratio >= 0.24
        and edge_density >= 0.15
        and skin_ratio >= 0.70
        and face_like_score >= 0.75
    ):
        pose["label"] = "profile-right"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "response-frontal-right-profile-recovery"
        face["faceMode"] = "profile-right"


def refine_occluded_profile_after_quality(face):
    pose = face.get("pose", {})
    if pose.get("label", "") != "occluded":
        return

    deepfake_features = face.get("deepfakeFeatures", {})
    geometry = deepfake_features.get("geometry", {})
    texture = deepfake_features.get("texture", {})
    visibility = deepfake_features.get("visibility", {})
    detector_name = face.get("detector", "")
    eye_visibility = float(visibility.get("eyeVisibility", 0.0))
    skin_ratio = float(texture.get("skinRatio", 0.0))
    edge_density = float(texture.get("edgeDensity", 0.0))
    edge_side_bias = float(texture.get("edgeSideBias", 0.0))
    signed_center_bias = float(geometry.get("signedCenterAxisBias", 0.0))
    profile_eye_score = float(visibility.get("profileEyeClosureScore", 0.0))
    occlusion_score = float(visibility.get("occlusionScore", 0.0))
    detected_ratio = float(face.get("quality", {}).get("detectedPointRatio", 0.0))
    keypoints = face.get("keypoints", {})
    nose_direction = float(keypoints.get("nose_tip", {}).get("x", 0.0)) - float(keypoints.get("nose_bridge_top", {}).get("x", 0.0))

    if (
        detector_name == "frontal_alt"
        and eye_visibility == 0.5
        and detected_ratio >= 0.60
        and skin_ratio >= 0.75
        and abs(edge_side_bias) <= 0.01
        and abs(signed_center_bias) <= 0.02
        and float(geometry.get("eyeDistanceRatio", 0.0)) >= 0.42
        and float(geometry.get("noseMouthRatio", 1.0)) <= 0.06
    ):
        pose["label"] = "frontal"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "occluded-centered-frontal-recovery"
        face["faceMode"] = "frontal"
        return

    if (
        detector_name == "response_fallback"
        and eye_visibility == 0.5
        and detected_ratio >= 0.60
        and skin_ratio >= 0.85
        and 0.06 <= edge_density <= 0.11
        and float(face.get("faceLikeScore", 0.0)) >= 0.62
        and float(face.get("featureSummary", {}).get("mouthEvidence", 0.0)) >= 0.50
        and float(geometry.get("noseMouthRatio", 0.0)) >= 0.24
        and abs(signed_center_bias) <= 0.02
    ):
        pose["label"] = "profile-left"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.56)
        pose["reason"] = "response-occluded-single-eye-profile-left-recovery"
        face["faceMode"] = "profile-left"
        return

    if (
        detector_name == "response_fallback"
        and eye_visibility == 0.0
        and skin_ratio <= 0.001
        and edge_density <= 0.03
        and profile_eye_score >= 0.65
        and occlusion_score >= 0.75
        and float(visibility.get("appearanceOcclusionScore", 0.0)) >= 0.70
        and detected_ratio <= 0.1112
        and min(float(face.get("bbox", {}).get("w", 0.0)), float(face.get("bbox", {}).get("h", 0.0))) >= 390.0
    ):
        pose["label"] = "profile-left"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "response-dark-occluded-profile-left-recovery"
        face["faceMode"] = "profile-left"
        return

    if (
        detector_name == "profile"
        and eye_visibility == 0.5
        and skin_ratio >= 0.75
        and edge_side_bias <= -0.03
        and float(geometry.get("eyeDistanceRatio", 0.0)) >= 0.33
        and float(geometry.get("noseMouthRatio", 0.0)) >= 0.20
        and float(face.get("score", 0.0)) >= 0.20
    ):
        pose["label"] = "profile-right"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "profile-occluded-right-profile-recovery"
        face["faceMode"] = "profile-right"
        return

    if (
        detector_name == "frontal_alt"
        and eye_visibility == 0.5
        and skin_ratio >= 0.60
        and edge_density >= 0.12
        and float(geometry.get("eyeDistanceRatio", 1.0)) <= 0.30
        and float(geometry.get("noseMouthRatio", 0.0)) >= 0.22
        and nose_direction >= 6.0
    ):
        pose["label"] = "profile-right"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "frontal-alt-occluded-right-profile-recovery"
        face["faceMode"] = "profile-right"
        return

    if (
        detector_name == "frontal_alt"
        and eye_visibility == 0.5
        and detected_ratio <= 0.3334
        and skin_ratio >= 0.70
        and edge_density >= 0.14
        and float(geometry.get("eyeDistanceRatio", 1.0)) <= 0.29
        and 0.55 <= occlusion_score <= 0.62
    ):
        pose["label"] = "profile-right"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "occluded-downward-profile-right-recovery"
        face["faceMode"] = "profile-right"
        return

    if (
        detector_name == "frontal_alt"
        and eye_visibility == 0.0
        and skin_ratio >= 0.85
        and signed_center_bias <= -0.06
        and profile_eye_score >= 0.75
        and occlusion_score >= 0.75
        and detected_ratio <= 0.2223
    ):
        pose["label"] = "profile-left"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "occluded-high-skin-left-profile-recovery"
        face["faceMode"] = "profile-left"
        return

    if (
        detector_name == "profile"
        and eye_visibility == 0.5
        and skin_ratio <= 0.02
        and edge_density >= 0.13
        and edge_side_bias <= -0.045
        and profile_eye_score >= 0.70
    ):
        pose["label"] = "profile-right"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "profile-detector-low-skin-right-recovery"
        face["faceMode"] = "profile-right"
        return

    if (
        detector_name == "frontal_alt"
        and eye_visibility == 0.0
        and 0.30 <= skin_ratio <= 0.45
        and edge_side_bias >= 0.04
        and occlusion_score >= 0.80
        and detected_ratio <= 0.2223
    ):
        pose["label"] = "profile-right"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "occluded-edge-right-profile-recovery"
        face["faceMode"] = "profile-right"
        return

    if (
        detector_name == "frontal_alt"
        and eye_visibility == 0.5
        and edge_side_bias >= 0.04
        and nose_direction >= 6.0
        and skin_ratio >= 0.45
        and edge_density >= 0.12
        and detected_ratio <= 0.3334
        and occlusion_score >= 0.60
    ):
        pose["label"] = "profile-left"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "occluded-single-eye-left-profile-recovery"
        face["faceMode"] = "profile-left"


def apply_profile_direction_voting(face):
    pose = face.get("pose", {})
    pose_label = pose.get("label", "")
    if pose_label not in {"profile-left", "profile-right"}:
        return

    deepfake_features = face.get("deepfakeFeatures", {})
    geometry = deepfake_features.get("geometry", {})
    texture = deepfake_features.get("texture", {})
    visibility = deepfake_features.get("visibility", {})
    keypoints = face.get("keypoints", {})
    if float(pose.get("confidence", 0.0)) >= 0.75 and float(visibility.get("eyeVisibility", 0.0)) >= 1.0:
        return

    signed_center_bias = float(geometry.get("signedCenterAxisBias", 0.0))
    center_offset = float(geometry.get("centerAxisOffset", 0.0))
    edge_side_bias = float(texture.get("edgeSideBias", 0.0))
    edge_density = float(texture.get("edgeDensity", 0.0))
    skin_ratio = float(texture.get("skinRatio", 0.0))
    eye_visibility = float(visibility.get("eyeVisibility", 0.0))
    profile_eye_score = float(visibility.get("profileEyeClosureScore", 0.0))
    nose_bridge_x = float(keypoints.get("nose_bridge_top", {}).get("x", 0.0))
    nose_tip_x = float(keypoints.get("nose_tip", {}).get("x", 0.0))
    nose_direction = nose_tip_x - nose_bridge_x

    if (
        pose_label == "profile-left"
        and face.get("detector") == "frontal_alt"
        and eye_visibility == 0.0
        and skin_ratio >= 0.75
        and edge_side_bias >= 0.10
        and center_offset <= 0.06
        and nose_direction <= -12.0
    ):
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "profile-direction-left-silhouette-preserved"
        face["faceMode"] = "profile-left"
        face["profileDirectionVotes"] = ["profile-left", "left-silhouette", "nose-left"]
        return

    if (
        pose_label == "profile-right"
        and face.get("detector") == "frontal_alt"
        and eye_visibility == 0.0
        and skin_ratio >= 0.75
        and edge_side_bias >= 0.10
        and center_offset <= 0.06
        and nose_direction <= -12.0
    ):
        pose["label"] = "profile-left"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "profile-direction-strong-left-silhouette-vote"
        face["faceMode"] = "profile-left"
        face["profileDirectionVotes"] = ["profile-left", "left-silhouette", "nose-left"]
        return

    if (
        pose_label == "profile-right"
        and face.get("detector") == "profile"
        and eye_visibility == 0.5
        and skin_ratio >= 0.75
        and edge_side_bias >= 0.07
        and center_offset <= 0.02
        and -8.0 <= nose_direction <= -4.0
    ):
        pose["label"] = "profile-left"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "profile-direction-soft-left-silhouette-vote"
        face["faceMode"] = "profile-left"
        face["profileDirectionVotes"] = ["profile-left", "soft-left-silhouette"]
        return

    if (
        pose_label == "profile-left"
        and face.get("detector") == "profile"
        and eye_visibility == 0.0
        and skin_ratio >= 0.45
        and signed_center_bias <= -0.15
        and nose_direction <= -40.0
    ):
        pose["label"] = "profile-right"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "profile-direction-strong-right-silhouette-vote"
        face["faceMode"] = "profile-right"
        face["profileDirectionVotes"] = ["profile-right", "strong-right-silhouette", "nose-right"]
        return

    if (
        pose_label == "profile-left"
        and face.get("detector") == "frontal"
        and eye_visibility == 0.5
        and 0.50 <= skin_ratio <= 0.65
        and -0.04 <= edge_side_bias <= -0.02
        and center_offset <= 0.02
        and profile_eye_score >= 0.80
        and -2.0 <= nose_direction <= 3.0
    ):
        pose["label"] = "profile-right"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "profile-direction-soft-right-silhouette-vote"
        face["faceMode"] = "profile-right"
        face["profileDirectionVotes"] = ["profile-right", "soft-right-silhouette"]
        return

    if (
        pose_label == "profile-left"
        and face.get("detector") in {"frontal", "profile"}
        and eye_visibility == 0.5
        and skin_ratio <= 0.001
        and edge_density <= 0.04
        and profile_eye_score >= 0.70
        and signed_center_bias <= -0.03
    ):
        pose["label"] = "profile-right"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "profile-direction-low-skin-right-silhouette-vote"
        face["faceMode"] = "profile-right"
        face["profileDirectionVotes"] = ["profile-right", "low-skin-right-silhouette"]
        return

    if (
        pose_label == "profile-left"
        and face.get("detector") == "response_fallback"
        and eye_visibility == 0.5
        and skin_ratio <= 0.001
        and float(texture.get("colorChromaStd", 0.0)) <= 0.40
        and float(texture.get("lowSaturationRatio", 0.0)) >= 0.85
        and signed_center_bias >= 0.20
        and profile_eye_score >= 0.80
        and float(geometry.get("eyeDistanceRatio", 0.0)) >= 0.38
    ):
        pose["label"] = "profile-right"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "profile-direction-bw-right-silhouette-vote"
        face["faceMode"] = "profile-right"
        face["profileDirectionVotes"] = ["profile-right", "bw-right-silhouette"]
        return

    if (
        pose_label == "profile-right"
        and face.get("detector") == "response_fallback"
        and float(face.get("faceLikeScore", 0.0)) <= 0.40
        and eye_visibility == 0.0
        and signed_center_bias >= 0.10
        and nose_direction >= 16.0
    ):
        pose["label"] = "profile-left"
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "profile-direction-low-face-nose-vote"
        face["faceMode"] = "profile-left"
        face["profileDirectionVotes"] = ["profile-left", "nose-strong-low-face"]
        return

    votes = []
    if signed_center_bias <= -0.035:
        votes.append("profile-left")
    elif signed_center_bias >= 0.035:
        votes.append("profile-right")

    if nose_direction >= 6.0:
        votes.append("profile-left")
    elif nose_direction <= -6.0:
        votes.append("profile-right")

    left_eye = keypoints.get("left_eye_center", {})
    right_eye = keypoints.get("right_eye_center", {})
    visible_eye = None
    if left_eye.get("source") == "detected" and right_eye.get("source") != "detected":
        visible_eye = left_eye
    elif right_eye.get("source") == "detected" and left_eye.get("source") != "detected":
        visible_eye = right_eye
    if visible_eye:
        bbox = face.get("bbox", {})
        width = max(float(bbox.get("w", 1)), 1.0)
        eye_rx = (float(visible_eye.get("x", 0.0)) - float(bbox.get("x", 0.0))) / width
        if eye_rx >= 0.56:
            votes.append("profile-left")
        elif eye_rx <= 0.44:
            votes.append("profile-right")

    left_votes = votes.count("profile-left")
    right_votes = votes.count("profile-right")
    voted_pose = None
    if left_votes >= 2 and left_votes > right_votes:
        voted_pose = "profile-left"
    elif right_votes >= 2 and right_votes > left_votes:
        voted_pose = "profile-right"

    if voted_pose and voted_pose != pose_label:
        pose["label"] = voted_pose
        pose["confidence"] = max(float(pose.get("confidence", 0.0)), 0.54)
        pose["reason"] = "profile-direction-vote"
        face["faceMode"] = voted_pose
        face["profileDirectionVotes"] = votes


def refine_profile_eye_closure_after_pose(face):
    pose = face.get("pose", {})
    pose_label = pose.get("label", "")
    if pose_label not in {"profile-left", "profile-right"}:
        return

    deepfake_features = face.get("deepfakeFeatures", {})
    geometry = deepfake_features.get("geometry", {})
    texture = deepfake_features.get("texture", {})
    visibility = deepfake_features.get("visibility", {})

    signed_center_bias = float(geometry.get("signedCenterAxisBias", 0.0))
    skin_ratio = float(texture.get("skinRatio", 0.0))
    edge_density = float(texture.get("edgeDensity", 0.0))
    color_chroma_std = float(texture.get("colorChromaStd", 0.0))
    eye_visibility = float(visibility.get("eyeVisibility", 0.0))
    current_score = float(visibility.get("profileEyeClosureScore", 0.0))
    face_like_score = float(face.get("faceLikeScore", 0.0))
    mouth_evidence = float(face.get("featureSummary", {}).get("mouthEvidence", 0.0))
    detected_ratio = float(face.get("quality", {}).get("detectedPointRatio", 0.0))

    if (
        abs(signed_center_bias) >= 0.35
        and skin_ratio >= 0.35
        and edge_density >= 0.09
        and eye_visibility >= 1.0
    ):
        visibility["profileEyeClosureScore"] = round(max(current_score, 0.55), 4)
        face["profileEyeClosureAdjustment"] = "postprocess-profile-axis-eye-closure-support"
    if (
        face.get("detector") == "yunet"
        and abs(signed_center_bias) >= 0.30
        and face_like_score >= 0.80
        and skin_ratio <= 0.08
        and mouth_evidence >= 0.65
        and color_chroma_std >= 7.5
        and edge_density >= 0.10
        and eye_visibility >= 1.0
        and detected_ratio >= 0.75
    ):
        visibility["profileEyeClosureScore"] = round(max(float(visibility.get("profileEyeClosureScore", 0.0)), 0.55), 4)
        face["profileEyeClosureAdjustment"] = "postprocess-low-skin-yunet-profile-eye-support"


def is_likely_mannequin_head_cluster(faces):
    if len(faces) < 6:
        return False

    frontal_alt_high = 0
    fallback_mouth = 0
    mid_skin_material = 0
    flat_appearance = 0
    low_anchor_faces = 0
    for face in faces:
        texture = face.get("deepfakeFeatures", {}).get("texture", {})
        visibility = face.get("deepfakeFeatures", {}).get("visibility", {})
        quality = face.get("quality", {})
        skin_ratio = float(texture.get("skinRatio", 0.0))
        if face.get("detector") == "frontal_alt" and float(face.get("score", 0.0)) >= 0.99:
            frontal_alt_high += 1
        if float(face.get("featureSummary", {}).get("mouthEvidence", 0.0)) <= 0.30:
            fallback_mouth += 1
        if 0.30 <= skin_ratio <= 0.85:
            mid_skin_material += 1
        if float(visibility.get("appearanceOcclusionScore", 0.0)) <= 0.01:
            flat_appearance += 1
        if float(quality.get("detectedPointRatio", 0.0)) <= 0.4445:
            low_anchor_faces += 1

    return (
        frontal_alt_high >= 5
        and fallback_mouth == len(faces)
        and mid_skin_material >= 5
        and flat_appearance >= 4
        and low_anchor_faces == len(faces)
    )


def build_face_output(image, preprocessed, candidates, request_uid, analysis_mode="full_image"):
    faces = []
    retention_pool = []
    debug_maps = {"eye": [], "nose": [], "mouth": []}
    crop_only_mode = analysis_mode == "face_crop_only"
    for index, candidate in enumerate(candidates):
        face_region = extract_face_region(image, preprocessed, candidate["box"], candidate["detector"])
        if face_region is None:
            continue
        x, y, w, h = face_region["bbox"]
        face_gray = face_region["gray"]
        face_mask = face_region["mask"]
        mediapipe_landmarks = candidate.get("mediapipeLandmarks")
        yunet_landmarks = candidate.get("yunetLandmarks")

        eye_response, eye_candidates, eye_metrics = detect_eye_candidates(face_gray, face_mask)
        eye_selection, eye_quality, eye_reason = select_eye_configuration(face_gray, eye_candidates)
        if mediapipe_landmarks:
            mp_eye_selection, mp_eye_quality, mp_eye_reason, mp_eye_metrics = build_mediapipe_eye_support(
                (x, y, w, h),
                mediapipe_landmarks,
                image.shape,
                eye_metrics,
            )
            if candidate["detector"] == "mediapipe_landmarker" or mp_eye_quality > eye_quality:
                eye_selection = mp_eye_selection
                eye_quality = mp_eye_quality
                eye_reason = mp_eye_reason
                eye_metrics = mp_eye_metrics
        if yunet_landmarks:
            yunet_eye_support = build_yunet_eye_support((x, y, w, h), yunet_landmarks, eye_metrics)
            if yunet_eye_support is not None:
                yn_eye_selection, yn_eye_quality, yn_eye_reason, yn_eye_metrics = yunet_eye_support
                if candidate["detector"] == "yunet" and yn_eye_quality > eye_quality:
                    eye_selection = yn_eye_selection
                    eye_quality = yn_eye_quality
                    eye_reason = yn_eye_reason
                    eye_metrics = yn_eye_metrics
        edge_profile = build_face_edge_profile(face_gray, face_mask)
        pose_label, pose_confidence, pose_reason = classify_pose(
            eye_selection,
            eye_quality,
            face_gray,
            edge_profile,
            eye_metrics,
        )
        nose_response, nose_bridge_top, nose_tip = detect_nose_keypoints(face_gray, (x, y, w, h), pose_label, face_mask)
        mouth_response, _, mouth_left, mouth_center, mouth_right = detect_mouth_keypoints(face_gray, (x, y, w, h), pose_label, face_mask)
        soft_mouth_center = find_soft_mouth_center(mouth_response, (x, y, w, h))
        keypoints = build_keypoints((x, y, w, h), pose_label, eye_selection, nose_bridge_top, nose_tip, mouth_left, mouth_center, mouth_right)
        if mediapipe_landmarks:
            keypoints = apply_mediapipe_keypoints(keypoints, (x, y, w, h), pose_label, mediapipe_landmarks, image.shape)
        if yunet_landmarks:
            keypoints = apply_yunet_keypoints(keypoints, (x, y, w, h), pose_label, yunet_landmarks)
        deepfake_features = compute_deepfake_features(
            face_region["bgr"],
            face_gray,
            (x, y, w, h),
            keypoints,
            pose_label,
            eye_selection,
            edge_profile,
            eye_metrics,
            face_mask,
        )
        refined_pose_label, refined_pose_confidence, refined_pose_reason = refine_pose_label(
            candidate["detector"],
            candidate["score"],
            pose_label,
            pose_confidence,
            pose_reason,
            deepfake_features,
            keypoints,
        )
        if refined_pose_label != pose_label:
            pose_label = refined_pose_label
            pose_confidence = refined_pose_confidence
            pose_reason = refined_pose_reason
            keypoints = build_keypoints((x, y, w, h), pose_label, eye_selection, nose_bridge_top, nose_tip, mouth_left, mouth_center, mouth_right)
            if mediapipe_landmarks:
                keypoints = apply_mediapipe_keypoints(keypoints, (x, y, w, h), pose_label, mediapipe_landmarks, image.shape)
            if yunet_landmarks:
                keypoints = apply_yunet_keypoints(keypoints, (x, y, w, h), pose_label, yunet_landmarks)
            deepfake_features = compute_deepfake_features(
                face_region["bgr"],
                face_gray,
                (x, y, w, h),
                keypoints,
                pose_label,
                eye_selection,
                edge_profile,
                eye_metrics,
                face_mask,
            )
        detected_points = sum(1 for keypoint in keypoints.values() if keypoint["source"] == "detected")
        detected_ratio = detected_points / float(max(len(keypoints), 1))
        blur_score = compute_blur_score(face_gray, face_mask)
        brightness_score = compute_brightness_score(face_gray, face_mask)
        contrast_score = compute_contrast_score(face_gray, face_mask)
        quality_label, quality_score = classify_quality(blur_score, brightness_score, contrast_score, pose_confidence, detected_ratio, min(w, h))
        connections = build_connections()
        regions = build_regions((x, y, w, h), keypoints, pose_label)
        face = {
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "detector": candidate["detector"],
            "landmarkSource": "mediapipe_face_landmarker" if mediapipe_landmarks else ("yunet_face_detector" if yunet_landmarks else "veritai_response_maps"),
            "score": round(float(candidate["score"]), 4),
            "detectionConfidence": round(float(candidate["score"]), 4),
            "detectorConsensus": bool(candidate.get("detectorConsensus", False)),
            "consensusDetector": candidate.get("consensusDetector"),
            "consensusScore": round(float(candidate.get("consensusScore", 0.0)), 4),
            "consensusIoU": round(float(candidate.get("consensusIoU", 0.0)), 4),
            "pose": {"label": pose_label, "confidence": round(float(pose_confidence), 4), "reason": pose_reason},
            "faceMode": pose_label,
            "quality": {"label": quality_label, "score": quality_score, "blur": blur_score, "brightness": brightness_score, "contrast": contrast_score, "detectedPointRatio": round(float(detected_ratio), 4)},
            "eyes": build_eye_overlay_boxes((x, y, w, h), keypoints, pose_label),
            "keypoints": keypoints,
            "faceContour": contour_to_points(face_region["contour"], (x, y, w, h)),
            "analysisConnections": serialize_connections(connections),
            "analysisRegions": serialize_regions(regions),
            "cropPath": save_face_crop(image, request_uid, index, candidate["box"], face_mask),
            "analysisInput": {
                "mode": analysis_mode,
                "detectionImage": "full_image",
                "featureImage": "cropped_face",
                "deepfakeImage": "cropped_face",
                "cropOnly": crop_only_mode,
                "cropSize": {"w": int(face_region["bgr"].shape[1]), "h": int(face_region["bgr"].shape[0])},
            },
            "trainingSample": None,
            "featureSummary": {
                "eyeEvidence": round(float(eye_quality), 4),
                "eyeReason": eye_reason,
                "noseEvidence": nose_tip["confidence"],
                "mouthEvidence": mouth_center["confidence"],
                "softMouthEvidence": 0.0 if soft_mouth_center is None else soft_mouth_center["score"],
                "softMouthCenter": None
                if soft_mouth_center is None
                else {
                    "x": int(soft_mouth_center["point"][0]),
                    "y": int(soft_mouth_center["point"][1]),
                    "score": soft_mouth_center["score"],
                    "reason": soft_mouth_center["reason"],
                },
            },
            "deepfakeFeatures": deepfake_features,
        }
        face_like_score = compute_face_like_score(face)
        face["faceLikeScore"] = face_like_score
        face["deepfakeFeatures"]["faceLikeScore"] = face_like_score
        keep_face, keep_reason = should_keep_face(face)
        face["candidateDecision"] = {"accepted": keep_face, "reason": keep_reason}
        if not keep_face:
            if CANDIDATE_BBOX_RETENTION == "precision_guarded" and should_retain_rejected_face_service(face, candidate, image.shape, keep_reason):
                retention_pool.append(
                    {
                        "face": face,
                        "candidate": candidate,
                        "originalKeepReason": keep_reason,
                        "eyeResponse": eye_response,
                        "noseResponse": nose_response,
                        "mouthResponse": mouth_response,
                    }
                )
            continue
        refine_face_label_after_quality(face)
        face["trainingSample"] = build_training_sample(face)
        faces.append(face)
        debug_maps["eye"].append({"bbox": face["bbox"], "response": eye_response})
        debug_maps["nose"].append({"bbox": face["bbox"], "response": nose_response})
        debug_maps["mouth"].append({"bbox": face["bbox"], "response": mouth_response})
    if CANDIDATE_BBOX_RETENTION == "precision_guarded" and retention_pool:
        retention_pool = sorted(
            retention_pool,
            key=lambda item: candidate_retention_priority_service(item["candidate"], image.shape),
            reverse=True,
        )
        for retained in retention_pool:
            face = retained["face"]
            if any(calculate_iou(bbox_dict_to_tuple(face["bbox"]), bbox_dict_to_tuple(existing["bbox"])) >= 0.35 for existing in faces):
                continue
            face["candidateRetention"] = {
                "strategy": CANDIDATE_BBOX_RETENTION,
                "originalRejectReason": retained["originalKeepReason"],
            }
            face["candidateDecision"] = {
                "accepted": True,
                "reason": "candidate-bbox-retention-precision-guarded",
            }
            refine_face_label_after_quality(face)
            face["trainingSample"] = build_training_sample(face)
            faces.append(face)
            debug_maps["eye"].append({"bbox": face["bbox"], "response": retained["eyeResponse"]})
            debug_maps["nose"].append({"bbox": face["bbox"], "response": retained["noseResponse"]})
            debug_maps["mouth"].append({"bbox": face["bbox"], "response": retained["mouthResponse"]})
            break
    if is_likely_mannequin_head_cluster(faces):
        return [], {"eye": [], "nose": [], "mouth": []}
    return faces, debug_maps


def draw_face_overlay(image, faces, output_path):
    canvas = image.copy()
    for index, face in enumerate(faces):
        bbox = face["bbox"]
        x = bbox["x"]
        y = bbox["y"]
        w = bbox["w"]
        h = bbox["h"]
        color = (0, 200, 0) if face["quality"]["label"] != "poor" else (0, 165, 255)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        contour = face.get("faceContour", [])
        if contour:
            pts = np.array([[point["x"], point["y"]] for point in contour], dtype=np.int32)
            cv2.polylines(canvas, [pts], True, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{index + 1}: {face['pose']['label']} {face['detectionConfidence']:.2f}", (x, max(y - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        for eye in face["eyes"]:
            cv2.rectangle(canvas, (eye["x"], eye["y"]), (eye["x"] + eye["w"], eye["y"] + eye["h"]), (255, 0, 255), 1)
    cv2.putText(canvas, "Detected Face Boxes", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(output_path, canvas)


def draw_response_map(image, debug_maps, output_path, title):
    canvas = image.copy()
    overlay = canvas.copy()
    for item in debug_maps:
        bbox = item["bbox"]
        x = bbox["x"]
        y = bbox["y"]
        w = bbox["w"]
        h = bbox["h"]
        response = item["response"]
        if response.size == 0:
            continue
        heat = np.uint8(np.clip(response * 255.0, 0, 255))
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        roi = overlay[y : y + h, x : x + w]
        if roi.shape[:2] != heat.shape[:2]:
            heat = cv2.resize(heat, (roi.shape[1], roi.shape[0]))
        overlay[y : y + h, x : x + w] = cv2.addWeighted(roi, 0.35, heat, 0.65, 0)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), 1)
    canvas = cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0)
    cv2.putText(canvas, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(output_path, canvas)


def draw_panel_text(canvas, lines, start_x, start_y, color=(235, 235, 235), line_gap=18):
    y = start_y
    for line in lines:
        cv2.putText(canvas, str(line), (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        y += line_gap
    return y


def draw_analysis_map(image, faces, output_path):
    image_h, image_w = image.shape[:2]
    panel_width = 430
    canvas = np.full((image_h, image_w + panel_width, 3), 18, dtype=np.uint8)
    canvas[:, :image_w] = image.copy()
    overlay = canvas[:, :image_w].copy()
    for face in faces:
        face_overlay = np.zeros_like(overlay)
        for region in face.get("analysisRegions", []):
            pts = np.array([[point["x"], point["y"]] for point in region["points"]], dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(face_overlay, [pts], tuple(region["color"]))
        contour = face.get("faceContour", [])
        if contour:
            contour_mask = np.zeros((image_h, image_w), dtype=np.uint8)
            contour_points = np.array([[point["x"], point["y"]] for point in contour], dtype=np.int32)
            cv2.fillPoly(contour_mask, [contour_points], 255)
            face_overlay = cv2.bitwise_and(face_overlay, face_overlay, mask=contour_mask)
        overlay = cv2.addWeighted(overlay, 1.0, face_overlay, 1.0, 0.0)
    canvas[:, :image_w] = cv2.addWeighted(overlay, 0.18, canvas[:, :image_w], 0.82, 0)
    for index, face in enumerate(faces):
        bbox = face["bbox"]
        x = bbox["x"]
        y = bbox["y"]
        w = bbox["w"]
        h = bbox["h"]
        cv2.rectangle(canvas[:, :image_w], (x, y), (x + w, y + h), (0, 255, 255), 2)
        contour = face.get("faceContour", [])
        if contour:
            pts = np.array([[point["x"], point["y"]] for point in contour], dtype=np.int32)
            cv2.polylines(canvas[:, :image_w], [pts], True, (255, 220, 120), 1, cv2.LINE_AA)
        keypoints = face["keypoints"]
        for connection in face["analysisConnections"]:
            start = keypoints.get(connection["from"])
            end = keypoints.get(connection["to"])
            if start and end:
                cv2.line(canvas[:, :image_w], point_xy(start), point_xy(end), (255, 180, 80), 1, cv2.LINE_AA)
        for keypoint in keypoints.values():
            color = (0, 255, 120) if keypoint["source"] == "detected" else (0, 180, 255)
            cv2.circle(canvas[:, :image_w], point_xy(keypoint), 4, color, -1, cv2.LINE_AA)
            cv2.circle(canvas[:, :image_w], point_xy(keypoint), 5, (255, 255, 255), 1, cv2.LINE_AA)
            label = POINT_LABELS.get(keypoint["name"], keypoint["name"][:2].upper())
            cv2.putText(
                canvas[:, :image_w],
                label,
                (keypoint["x"] + 6, max(keypoint["y"] - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(canvas[:, :image_w], f"Face {index + 1} | {face['pose']['label']}", (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    panel_x = image_w + 18
    y = draw_panel_text(canvas, ["VeritAI Original Anchor Graph", "Detected points = green", "Estimated points = orange", "", "Legend"], panel_x, 30, color=(255, 255, 255))
    for region_name, color in REGION_COLORS.items():
        cv2.rectangle(canvas, (panel_x, y - 10), (panel_x + 14, y + 4), color, -1)
        y = draw_panel_text(canvas, [region_name], panel_x + 24, y)
    if not faces:
        draw_panel_text(canvas, ["", "No faces detected.", "Try a larger, front-facing face image."], panel_x, y + 12)
        cv2.imwrite(output_path, canvas)
        return
    for index, face in enumerate(faces):
        quality = face["quality"]
        pose = face["pose"]
        y = draw_panel_text(canvas, ["", f"[Face {index + 1}]", f"detector={face['detector']}", f"pose={pose['label']} ({pose['confidence']:.4f})", f"quality={quality['label']} ({quality['score']:.4f})", f"detectedPointRatio={quality['detectedPointRatio']:.4f}", f"maskCoverage={face['deepfakeFeatures']['visibility'].get('faceMaskCoverage', 1.0):.4f}", f"eye={face['featureSummary']['eyeEvidence']:.4f}", f"nose={face['featureSummary']['noseEvidence']:.4f}", f"mouth={face['featureSummary']['mouthEvidence']:.4f}", f"edgeDensity={face['deepfakeFeatures']['texture']['edgeDensity']:.4f}", f"noiseScore={face['deepfakeFeatures']['texture']['noiseScore']:.4f}"], panel_x, y + 8)
        point_lines = [f"{point_name}: ({face['keypoints'][point_name]['x']},{face['keypoints'][point_name]['y']}) {face['keypoints'][point_name]['source']} {face['keypoints'][point_name]['confidence']:.2f}" for point_name in TRAINING_POINT_ORDER]
        y = draw_panel_text(canvas, point_lines, panel_x + 10, y, color=(210, 210, 210), line_gap=17)
    cv2.imwrite(output_path, canvas)


def save_debug_metadata(request_uid, faces, debug_paths):
    payload = {
        "requestUid": request_uid,
        "pipeline": {"name": "veritai-pose-aware-anchor-graph", "version": "v1", "description": "Original pose-aware facial anchor graph with detected/estimated keypoint separation."},
        "outputStructure": {
            "overlays": "images/analysis/overlays",
            "anchorMaps": "images/analysis/anchor_maps",
            "metadata": "images/analysis/metadata",
            "eyeResponses": "images/analysis/response_maps/eyes",
            "noseResponses": "images/analysis/response_maps/nose",
            "mouthResponses": "images/analysis/response_maps/mouth",
        },
        "faceCount": len(faces),
        "faces": faces,
        "generatedFiles": {key: normalize_path(path) for key, path in debug_paths.items()},
        "trainingDatasetCandidate": [face["trainingSample"] for face in faces],
    }
    with open(debug_paths["metadata"], "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def _resize_for_service(image_bgr):
    img = image_bgr
    max_width = MAX_IMAGE_WIDTH
    if img.shape[1] > max_width:
        ratio = max_width / float(img.shape[1])
        img = cv2.resize(img, None, fx=ratio, fy=ratio)
    return img


def _detect_service_face_candidates(img, preprocessed):
    candidates = detect_faces(preprocessed)
    if candidates:
        return candidates
    for rescue_detector in (
        detect_with_landmark_consensus,
        detect_with_mediapipe_landmarker,
        detect_with_yunet,
    ):
        rescue_candidates = rescue_detector(preprocessed)
        if rescue_candidates:
            return rescue_candidates
    return []


def prepare_service_cnn_inputs(image_bgr, analysis_mode="face_crop_only", crop_zip_mode=False, timings=None):
    """
    POST /predict 및 crop_all --pipeline service 에서 사용.

    crop_zip_mode=True (crop_all --service-depth bbox, 기본):
      detect_faces + extract_face_region 까지만 → CNN bbox 와 동일, 훨씬 빠름.
    crop_zip_mode=False (/predict):
      build_face_output 전체 (앵커·포즈·품질 분석 포함).
    """
    if image_bgr is None or image_bgr.size == 0:
        return None, [], {}

    resize_started = time.time()
    img = _resize_for_service(image_bgr)
    if timings is not None:
        timings["resizeTimeMs"] = int((time.time() - resize_started) * 1000)

    preprocess_started = time.time()
    preprocessed = preprocess_image(img)
    if timings is not None:
        timings["preprocessTimeMs"] = int((time.time() - preprocess_started) * 1000)

    candidate_started = time.time()
    candidates = _detect_service_face_candidates(img, preprocessed)
    if timings is not None:
        timings["candidateDetectionTimeMs"] = int((time.time() - candidate_started) * 1000)
        timings["candidateCount"] = len(candidates)

    if crop_zip_mode:
        bbox_started = time.time()
        faces = []
        for candidate in candidates:
            face_region = extract_face_region(
                img, preprocessed, candidate["box"], candidate["detector"]
            )
            if face_region is None:
                continue
            bbox = face_region["bbox"]
            if isinstance(bbox, dict):
                x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            else:
                x, y, w, h = bbox
            faces.append(
                {
                    "bbox": {
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h),
                    }
                }
            )
        if timings is not None:
            timings["bboxExtractionTimeMs"] = int((time.time() - bbox_started) * 1000)
        return img, faces, {}

    request_uid = uuid.uuid4().hex[:12]
    face_output_started = time.time()
    faces, debug_maps = build_face_output(img, preprocessed, candidates, request_uid, analysis_mode)
    if timings is not None:
        timings["faceOutputTimeMs"] = int((time.time() - face_output_started) * 1000)
    rescue_time_ms = 0
    if not faces and candidates:
        for rescue_detector in (
            detect_with_landmark_consensus,
            detect_with_mediapipe_landmarker,
            detect_with_yunet,
        ):
            rescue_started = time.time()
            rescue_candidates = rescue_detector(preprocessed)
            rescue_time_ms += int((time.time() - rescue_started) * 1000)
            if not rescue_candidates:
                continue
            rescue_build_started = time.time()
            faces, debug_maps = build_face_output(
                img, preprocessed, rescue_candidates, request_uid, analysis_mode
            )
            rescue_time_ms += int((time.time() - rescue_build_started) * 1000)
            if faces:
                break
    if timings is not None:
        timings["rescueTimeMs"] = rescue_time_ms
        timings["faceCountAfterAnalysis"] = len(faces)
    return img, faces, debug_maps


@app.post("/predict")
async def predict(file: UploadFile = File(...), analysisMode: str = Form("full_image")):
    start = time.time()
    normalized_mode = analysisMode if analysisMode in {"full_image", "face_crop_only"} else "full_image"
    mode_label = "face crop only" if normalized_mode == "face_crop_only" else "full image"
    read_started = time.time()
    contents = await file.read()
    read_time_ms = int((time.time() - read_started) * 1000)
    decode_started = time.time()
    img = decode_image(contents)
    decode_time_ms = int((time.time() - decode_started) * 1000)
    if img is None:
        return {"isDeepfake": False, "confidence": 0.0, "faceCount": 0, "watermarkDetected": False, "modelVersion": "veritai-pose-aware-anchor-graph-v1", "analysisMode": normalized_mode, "analysisInput": {"detectionImage": "full_image", "featureImage": "cropped_face", "deepfakeImage": "cropped_face"} if normalized_mode == "face_crop_only" else {"detectionImage": "full_image", "featureImage": "full_image", "deepfakeImage": "full_image"}, "timings": {"imageReadTimeMs": read_time_ms, "decodeTimeMs": decode_time_ms, "totalTimeMs": int((time.time() - start) * 1000)}, "processingTimeMs": 0, "message": f"{mode_label}: image decoding failed.", "faces": [], "debugImages": {}}
    detection_started = time.time()
    service_timings = {}
    img, faces, debug_maps = prepare_service_cnn_inputs(img, normalized_mode, timings=service_timings)
    detection_time_ms = int((time.time() - detection_started) * 1000)
    face_analysis_time_ms = service_timings.get("faceOutputTimeMs", 0) + service_timings.get("rescueTimeMs", 0)
    request_uid, debug_paths = make_debug_paths()
    debug_started = time.time()
    if DEBUG_ARTIFACTS:
        draw_face_overlay(img, faces, debug_paths["overlay"])
        draw_analysis_map(img, faces, debug_paths["analysisMap"])
        draw_response_map(img, debug_maps["eye"], debug_paths["eyeResponse"], "Eye Response Map")
        draw_response_map(img, debug_maps["nose"], debug_paths["noseResponse"], "Nose Response Map")
        draw_response_map(img, debug_maps["mouth"], debug_paths["mouthResponse"], "Mouth Response Map")
        save_debug_metadata(request_uid, faces, debug_paths)
    debug_time_ms = int((time.time() - debug_started) * 1000)
    cnn_started = time.time()
    cnn_result = run_cnn_prediction(img, faces)
    cnn_time_ms = int((time.time() - cnn_started) * 1000)
    processing_time_ms = int((time.time() - start) * 1000)
    ready_faces = [face for face in faces if face["quality"]["label"] != "poor"]
    anchor_confidence = round(sum(face["detectionConfidence"] for face in faces) / len(faces), 4) if faces else 0.0
    cnn_loaded = bool(cnn_result.get("modelLoaded"))
    if cnn_loaded:
        confidence = float(cnn_result.get("fakeProbability", 0.0))
        is_deepfake = bool(cnn_result.get("isDeepfake", False))
        model_version = "veritai-anchor-cnn-v1"
    else:
        # Do not treat face-detection score as deepfake confidence when CNN is unavailable.
        confidence = 0.0
        is_deepfake = False
        model_version = "veritai-pose-aware-anchor-graph-v1"
    cnn_error = cnn_result.get("error")
    message = (
        f"{mode_label}: detected {len(faces)} face(s); usable faces={len(ready_faces)}; "
        f"cnnLoaded={cnn_loaded}; detectionAnchorConfidence={anchor_confidence}."
    )
    if not cnn_loaded and cnn_error:
        message += f" cnnError={cnn_error}"
    debug_images = {key: normalize_path(path) for key, path in debug_paths.items()} if DEBUG_ARTIFACTS else {}
    return {
        "isDeepfake": is_deepfake,
        "confidence": confidence,
        "faceCount": len(faces),
        "watermarkDetected": False,
        "modelVersion": model_version,
        "analysisMode": normalized_mode,
        "analysisInput": {"detectionImage": "full_image", "featureImage": "cropped_face", "deepfakeImage": "cropped_face"} if normalized_mode == "face_crop_only" else {"detectionImage": "full_image", "featureImage": "full_image", "deepfakeImage": "full_image"},
        "timings": {"imageReadTimeMs": read_time_ms, "decodeTimeMs": decode_time_ms, **service_timings, "detectionTimeMs": detection_time_ms, "faceAnalysisTimeMs": face_analysis_time_ms, "debugArtifactTimeMs": debug_time_ms, "cnnTimeMs": cnn_time_ms, "totalTimeMs": processing_time_ms},
        "processingTimeMs": processing_time_ms,
        "message": message,
        "faces": faces,
        "cnn": cnn_result,
        "debugImages": debug_images,
    }


@app.get("/")
async def root():
    return {"message": "VeritAI original anchor graph server is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
