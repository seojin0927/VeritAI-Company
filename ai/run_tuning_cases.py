from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2

try:
    from ai import main as pipeline
except ImportError:
    import main as pipeline

pipeline.DEBUG_ARTIFACTS = True


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "ai" / "tuning_cases"
RUN_DIR = ROOT / "images" / "tuning_runs"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SKIP_NAME_TOKENS = ("_analysis", "_overlay", "_response", "case_summary")


def analyze_file(image_path: Path) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        return {
            "input": image_path.as_posix(),
            "status": "failed",
            "reason": "image-load-failed",
        }

    preprocessed = pipeline.preprocess_image(image)
    request_uid, debug_paths = pipeline.make_debug_paths()
    candidates = pipeline.detect_faces(preprocessed)
    faces, debug_maps = pipeline.build_face_output(image, preprocessed, candidates, request_uid)
    if not faces:
        for rescue_detector in (pipeline.detect_with_mediapipe_landmarker, pipeline.detect_with_yunet):
            rescue_candidates = rescue_detector(preprocessed)
            if not rescue_candidates:
                continue
            faces, debug_maps = pipeline.build_face_output(image, preprocessed, rescue_candidates, request_uid)
            if faces:
                break

    pipeline.draw_face_overlay(image, faces, debug_paths["overlay"])
    pipeline.draw_analysis_map(image, faces, debug_paths["analysisMap"])
    pipeline.draw_response_map(image, debug_maps["eye"], debug_paths["eyeResponse"], "Eye Response Map")
    pipeline.draw_response_map(image, debug_maps["nose"], debug_paths["noseResponse"], "Nose Response Map")
    pipeline.draw_response_map(image, debug_maps["mouth"], debug_paths["mouthResponse"], "Mouth Response Map")
    pipeline.save_debug_metadata(request_uid, faces, debug_paths)

    return {
        "input": image_path.as_posix(),
        "status": "ok",
        "uid": request_uid,
        "faceCount": len(faces),
        "poses": [face["pose"]["label"] for face in faces],
        "qualities": [face["quality"]["label"] for face in faces],
        "detectedPointRatios": [face["quality"]["detectedPointRatio"] for face in faces],
        "eyeClosureIndices": [
            face.get("deepfakeFeatures", {}).get("visibility", {}).get("eyeClosureIndex", 0.0)
            for face in faces
        ],
        "eyeClosureScores": [
            face.get("deepfakeFeatures", {}).get("visibility", {}).get("eyeClosureScore", 0.0)
            for face in faces
        ],
        "occlusionScores": [
            face.get("deepfakeFeatures", {}).get("visibility", {}).get("occlusionScore", 0.0)
            for face in faces
        ],
        "profileEyeClosureScores": [
            face.get("deepfakeFeatures", {}).get("visibility", {}).get("profileEyeClosureScore", 0.0)
            for face in faces
        ],
        "generatedFiles": {key: path.replace("\\", "/") for key, path in debug_paths.items()},
    }


def evaluate_result(category: str, result: dict) -> dict:
    if result.get("status") != "ok":
        return {"rating": "fail", "reason": result.get("reason", "analysis-failed")}

    face_count = result.get("faceCount", 0)
    poses = result.get("poses", [])
    qualities = result.get("qualities", [])
    detected_ratios = result.get("detectedPointRatios", [])
    eye_closure = result.get("eyeClosureIndices", [])
    eye_closure_scores = result.get("eyeClosureScores", eye_closure)
    occlusion_scores = result.get("occlusionScores", [])
    profile_eye_closure_scores = result.get("profileEyeClosureScores", eye_closure_scores)

    if category == "false_positive":
        if face_count == 0:
            return {"rating": "pass", "reason": "no-face-detected-as-expected"}
        return {"rating": "fail", "reason": f"false-positive-faceCount={face_count}"}

    if category == "false_negative":
        if face_count >= 1:
            return {"rating": "pass", "reason": "face-detected"}
        return {"rating": "fail", "reason": "expected-face-but-none-detected"}

    if face_count == 0:
        return {"rating": "fail", "reason": "no-face-detected"}

    if category == "frontal_good":
        if "frontal" in poses and any(ratio >= 0.4 for ratio in detected_ratios):
            return {"rating": "pass", "reason": "frontal-pose-and-sufficient-detected-points"}
        if "frontal" in poses:
            return {"rating": "warn", "reason": "frontal-pose-but-detected-points-are-still-low"}
        if any(quality == "good" for quality in qualities):
            return {"rating": "warn", "reason": f"good-quality-but-pose={poses}"}
        return {"rating": "fail", "reason": f"unexpected-pose={poses}"}

    if category == "profile_left":
        if "profile-left" in poses:
            return {"rating": "pass", "reason": "profile-left-detected"}
        if "profile-right" in poses:
            return {"rating": "fail", "reason": "profile-direction-flipped"}
        return {"rating": "warn", "reason": f"profile-left-not-detected poses={poses}"}

    if category == "profile_right":
        if "profile-right" in poses:
            return {"rating": "pass", "reason": "profile-right-detected"}
        if "profile-left" in poses:
            return {"rating": "fail", "reason": "profile-direction-flipped"}
        return {"rating": "warn", "reason": f"profile-right-not-detected poses={poses}"}

    if category in {"eyes_closed", "eyes_closed_frontal"}:
        if "eyes_closed" in poses:
            return {"rating": "pass", "reason": "eyes-closed-detected"}
        if any(value >= 0.65 for value in eye_closure_scores):
            return {"rating": "pass", "reason": "high-eye-closure-score"}
        if any(value >= 0.65 for value in eye_closure):
            return {"rating": "warn", "reason": "high-eye-closure-index-but-pose-not-eyes_closed"}
        if "frontal" in poses:
            return {"rating": "warn", "reason": "frontal-detected-but-eyes-closed-label-missed"}
        return {"rating": "fail", "reason": f"eyes-closed-missed poses={poses}"}

    if category == "eyes_closed_profile":
        if "eyes_closed" in poses:
            return {"rating": "pass", "reason": "eyes-closed-detected"}
        if any(value >= 0.55 for value in profile_eye_closure_scores):
            return {"rating": "pass", "reason": "high-profile-eye-closure-score"}
        if any(pose in {"profile-left", "profile-right"} for pose in poses) and any(value >= 0.12 for value in eye_closure_scores):
            return {"rating": "pass", "reason": "profile-detected-with-meaningful-eye-closure-signal"}
        if any(pose in {"profile-left", "profile-right"} for pose in poses):
            return {"rating": "warn", "reason": "profile-detected-but-eye-closure-signal-is-weak"}
        if any(value >= 0.40 for value in profile_eye_closure_scores):
            return {"rating": "warn", "reason": "profile-eye-closure-score-suggests-closed-profile"}
        if any(value >= 0.65 for value in eye_closure_scores):
            return {"rating": "warn", "reason": "high-eye-closure-index-but-profile-label-missed"}
        return {"rating": "fail", "reason": f"eyes-closed-profile-missed poses={poses}"}

    if category == "eyes_closed_occluded":
        if "eyes_closed" in poses or "occluded" in poses:
            return {"rating": "pass", "reason": "occluded-or-eyes-closed-detected"}
        if any(value >= 0.58 for value in eye_closure_scores) or any(value >= 0.50 for value in occlusion_scores):
            return {"rating": "pass", "reason": "occlusion-or-eye-closure-score-high"}
        if any(value >= 0.35 for value in occlusion_scores) or any(value >= 0.35 for value in eye_closure_scores) or any(ratio < 0.35 for ratio in detected_ratios):
            return {"rating": "warn", "reason": "occluded-eyes-closed-signal-present-but-label-missed"}
        return {"rating": "fail", "reason": f"eyes-closed-occluded-missed poses={poses}"}

    if category == "occluded":
        if "occluded" in poses:
            return {"rating": "pass", "reason": "occlusion-detected"}
        if any(value >= 0.50 for value in occlusion_scores):
            return {"rating": "pass", "reason": "high-occlusion-score"}
        if any(value >= 0.35 for value in occlusion_scores):
            return {"rating": "warn", "reason": "occlusion-score-suggests-occlusion"}
        if any(ratio < 0.35 for ratio in detected_ratios):
            return {"rating": "warn", "reason": "low-detected-point-ratio-suggests-occlusion"}
        return {"rating": "fail", "reason": f"occlusion-missed poses={poses}"}

    return {"rating": "warn", "reason": "no-category-rule"}


def detect_label_issue(category: str, result: dict) -> str | None:
    if result.get("status") != "ok":
        return None

    face_count = result.get("faceCount", 0)
    poses = result.get("poses", [])
    qualities = result.get("qualities", [])
    detected_ratios = result.get("detectedPointRatios", [])
    eye_closure = result.get("eyeClosureIndices", [])

    if category == "false_positive" and face_count > 0 and any(label == "good" for label in qualities):
        return "possible-mislabeled-false-positive"
    if category == "profile_left" and "profile-right" in poses:
        return "possible-direction-label-mismatch"
    if category == "profile_right" and "profile-left" in poses:
        return "possible-direction-label-mismatch"
    if category == "frontal_good" and any(pose in {"profile-left", "profile-right"} for pose in poses) and any(ratio >= 0.3 for ratio in detected_ratios):
        return "possible-semi-profile-mixed-into-frontal"
    if category in {"eyes_closed", "eyes_closed_frontal"} and "frontal" in poses and max(eye_closure or [0.0]) < 0.12:
        return "eyes-closed-label-but-closure-signal-weak"
    if category == "eyes_closed_profile" and "frontal" in poses:
        return "possible-profile-label-mismatch"
    if category == "eyes_closed_occluded" and "occluded" not in poses and max(eye_closure or [0.0]) < 0.12:
        return "possible-occluded-label-mismatch"
    return None


def sanitize_part(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)
    return sanitized.strip("-") or "unknown"


def classify_review_bucket(category: str, result: dict) -> str:
    if result.get("status") != "ok":
        return "analysis_failed"

    evaluation = result.get("evaluation", {})
    reason = evaluation.get("reason", "")

    if category == "false_positive":
        return "false_positive"
    if category == "false_negative":
        return "false_negative"
    if "profile" in reason or "pose" in reason:
        return "pose_mismatch"
    if "eyes-closed" in reason:
        return "eyes_closed_missed"
    if "occlusion" in reason:
        return "occlusion_missed"
    if "no-face-detected" in reason:
        return "missed_face"
    if "detected-point-ratio" in reason:
        return "low_detected_points"
    return "review_needed"


def export_review_artifacts(run_path: Path, result: dict) -> str | None:
    evaluation = result.get("evaluation", {})
    rating = evaluation.get("rating")
    if rating not in {"fail", "warn"}:
        return None

    category = sanitize_part(result.get("category", "uncategorized"))
    bucket = sanitize_part(classify_review_bucket(category, result))
    source_path = Path(result["input"])
    case_name = sanitize_part(source_path.stem)
    uid = sanitize_part(result.get("uid", "no-uid"))

    review_dir = run_path / "review" / bucket / rating / f"{case_name}-{uid}"
    review_dir.mkdir(parents=True, exist_ok=True)

    copied_files: dict[str, str] = {}

    input_copy = review_dir / f"input{source_path.suffix.lower()}"
    shutil.copy2(source_path, input_copy)
    copied_files["input"] = input_copy.relative_to(ROOT).as_posix()

    for key, file_path in result.get("generatedFiles", {}).items():
        artifact_path = Path(file_path)
        if not artifact_path.is_absolute():
            artifact_path = ROOT / artifact_path
        if artifact_path.exists():
            target = review_dir / artifact_path.name
            shutil.copy2(artifact_path, target)
            copied_files[key] = target.relative_to(ROOT).as_posix()

    case_summary = {
        "input": source_path.relative_to(ROOT).as_posix(),
        "category": result.get("category"),
        "reviewBucket": bucket,
        "evaluation": evaluation,
        "labelIssue": result.get("labelIssue"),
        "poses": result.get("poses", []),
        "qualities": result.get("qualities", []),
        "faceCount": result.get("faceCount", 0),
        "detectedPointRatios": result.get("detectedPointRatios", []),
        "eyeClosureIndices": result.get("eyeClosureIndices", []),
        "eyeClosureScores": result.get("eyeClosureScores", []),
        "occlusionScores": result.get("occlusionScores", []),
        "profileEyeClosureScores": result.get("profileEyeClosureScores", []),
        "copiedFiles": copied_files,
    }
    summary_path = review_dir / "case_summary.json"
    summary_path.write_text(json.dumps(case_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return review_dir.relative_to(ROOT).as_posix()


def collect_cases() -> list[Path]:
    cases = []
    for path in CASE_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        name_lower = path.stem.lower()
        if any(token in name_lower for token in SKIP_NAME_TOKENS):
            continue
        if path.name.startswith("."):
            continue
        cases.append(path)
    return sorted(cases)


def write_summary(run_path: Path, results: list[dict]) -> None:
    summary_json = run_path / "summary.json"
    summary_md = run_path / "summary.md"

    summary_payload = {
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "caseDir": CASE_DIR.as_posix(),
        "caseCount": len(results),
        "criteriaVersion": "v3",
        "reviewRoot": (run_path / "review").relative_to(ROOT).as_posix(),
        "results": results,
    }
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Tuning Case Summary",
        "",
        f"- Generated At: {summary_payload['generatedAt']}",
        f"- Case Count: {len(results)}",
        f"- Case Directory: `{CASE_DIR.as_posix()}`",
        f"- Criteria Version: `{summary_payload['criteriaVersion']}`",
        f"- Review Directory: `{summary_payload['reviewRoot']}`",
        "",
        "| Case | Category | Eval | Status | Face Count | Poses | Qualities | Review Dir |",
        "|---|---|---|---:|---:|---|---|---|",
    ]

    for result in results:
        case = Path(result["input"]).relative_to(ROOT).as_posix()
        poses = ", ".join(result.get("poses", [])) if result.get("poses") else "-"
        qualities = ", ".join(result.get("qualities", [])) if result.get("qualities") else "-"
        evaluation = result.get("evaluation", {})
        review_dir = result.get("reviewDir") or "-"
        lines.append(
            f"| `{case}` | {result.get('category', '-')} | {evaluation.get('rating', '-')} | "
            f"{result['status']} | {result.get('faceCount', 0)} | {poses} | {qualities} | `{review_dir}` |"
        )

    lines.extend(
        [
            "",
            "## Evaluation Rules",
            "",
            "- `frontal_good`: `frontal` pose and enough detected points means `pass`.",
            "- `profile_left`: `profile-left` pose means `pass`.",
            "- `profile_right`: `profile-right` pose means `pass`.",
            "- `eyes_closed_frontal`: `eyes_closed` pose means `pass`, and `frontal` with strong closure becomes `warn`.",
            "- `eyes_closed_profile`: `eyes_closed`, strong `profileEyeClosureScore`, or `profile-*` with meaningful closure signal means `pass`.",
            "- `eyes_closed_occluded`: `occluded` or `eyes_closed` pose means `pass`.",
            "- `occluded`: `occluded` pose means `pass`.",
            "- `false_positive`: zero detected faces means `pass`.",
            "- `false_negative`: at least one detected face means `pass`.",
            "",
            "## Review Export",
            "",
            "- `warn` and `fail` cases are copied to `images/tuning_runs/<timestamp>/review/...`.",
            "- Each review folder contains the original input image, generated analysis artifacts, and `case_summary.json`.",
        ]
    )

    label_issues = [result for result in results if result.get("labelIssue")]
    if label_issues:
        priority_issues = [
            result
            for result in label_issues
            if result["labelIssue"] in {
                "possible-direction-label-mismatch",
                "possible-mislabeled-false-positive",
                "possible-profile-label-mismatch",
            }
        ]
        if priority_issues:
            lines.extend(["", "## Priority Label Review", ""])
            for result in priority_issues:
                case = Path(result["input"]).relative_to(ROOT).as_posix()
                lines.append(f"- `{case}`: `{result['labelIssue']}`")
        lines.extend(["", "## Label Quality Notes", ""])
        for result in label_issues:
            case = Path(result["input"]).relative_to(ROOT).as_posix()
            lines.append(f"- `{case}`: `{result['labelIssue']}`")

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cases = collect_cases()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = RUN_DIR / timestamp
    run_path.mkdir(parents=True, exist_ok=True)

    results = []
    for case in cases:
        result = analyze_file(case)
        result["category"] = case.parent.name
        result["evaluation"] = evaluate_result(result["category"], result)
        result["labelIssue"] = detect_label_issue(result["category"], result)
        result["reviewDir"] = export_review_artifacts(run_path, result)
        results.append(result)

    write_summary(run_path, results)
    print(json.dumps({"runDir": run_path.as_posix(), "caseCount": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
