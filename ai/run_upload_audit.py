from __future__ import annotations

import json
import os
import shutil
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    from ai.run_tuning_cases import CASE_DIR, IMAGE_EXTENSIONS, ROOT, analyze_file, evaluate_result, sanitize_part
except ImportError:
    from run_tuning_cases import CASE_DIR, IMAGE_EXTENSIONS, ROOT, analyze_file, evaluate_result, sanitize_part


UPLOAD_DIR = ROOT / "backend" / "backend_spring" / "uploads"
RUN_DIR = ROOT / "images" / "upload_audit_runs"
SKIP_NAME_TOKENS = ("_analysis", "_overlay", "_response", "case_summary")


def collect_uploads() -> list[Path]:
    uploads = []
    for path in UPLOAD_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        name_lower = path.stem.lower()
        if any(token in name_lower for token in SKIP_NAME_TOKENS):
            continue
        if path.name.startswith("."):
            continue
        uploads.append(path)
    return sorted(uploads)


def build_label_index() -> dict[str, list[str]]:
    label_index: dict[str, list[str]] = defaultdict(list)
    for path in CASE_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        name_lower = path.stem.lower()
        if any(token in name_lower for token in SKIP_NAME_TOKENS):
            continue
        if path.name.startswith("."):
            continue
        category = path.parent.name
        label_index[path.name.lower()].append(category)
    return label_index


def expected_category_for_upload(path: Path, label_index: dict[str, list[str]]) -> str | None:
    categories = sorted(set(label_index.get(path.name.lower(), [])))
    if len(categories) == 1:
        return categories[0]
    if len(categories) > 1:
        return "ambiguous:" + ",".join(categories)
    return None


def infer_detected_type(result: dict) -> str:
    if result.get("status") != "ok":
        return "analysis_failed"
    face_count = int(result.get("faceCount", 0))
    if face_count <= 0:
        return "no_face"

    poses = result.get("poses", [])
    prefix = "multi_face_" if face_count > 1 else ""
    if "eyes_closed" in poses:
        return f"{prefix}eyes_closed"
    if "occluded" in poses:
        return f"{prefix}occluded"
    has_left = "profile-left" in poses
    has_right = "profile-right" in poses
    if has_left and has_right:
        return f"{prefix}profile_mixed"
    if has_left:
        return f"{prefix}profile_left"
    if has_right:
        return f"{prefix}profile_right"
    if "frontal" in poses:
        return f"{prefix}frontal"
    return f"{prefix}unknown_face"


def evaluate_upload_detection(result: dict) -> dict:
    if result.get("status") != "ok":
        return {"rating": "fail", "reason": result.get("reason", "analysis-failed")}

    face_count = int(result.get("faceCount", 0))
    if face_count <= 0:
        return {"rating": "warn", "reason": "unlabeled-no-face-needs-visual-review"}

    qualities = result.get("qualities", [])
    detected_ratios = [float(value) for value in result.get("detectedPointRatios", [])]
    occlusion_scores = [float(value) for value in result.get("occlusionScores", [])]
    eye_closure_scores = [float(value) for value in result.get("eyeClosureScores", [])]
    profile_eye_scores = [float(value) for value in result.get("profileEyeClosureScores", [])]

    has_usable_face = any(label in {"good", "usable"} for label in qualities)
    if not has_usable_face:
        return {"rating": "fail", "reason": "face-detected-but-quality-poor"}

    max_detected_ratio = max(detected_ratios or [0.0])
    strong_special_signal = (
        max(occlusion_scores or [0.0]) >= 0.50
        or max(eye_closure_scores or [0.0]) >= 0.55
        or max(profile_eye_scores or [0.0]) >= 0.55
    )
    if max_detected_ratio < 0.2222 and not strong_special_signal:
        return {"rating": "warn", "reason": "face-detected-but-anchor-support-very-low"}
    if max_detected_ratio < 0.3333 and not strong_special_signal:
        return {"rating": "warn", "reason": "face-detected-but-anchor-support-low"}

    return {"rating": "pass", "reason": "usable-face-detected"}


def review_bucket(result: dict) -> str:
    evaluation = result.get("evaluation", {})
    reason = evaluation.get("reason", "")
    detected_type = result.get("detectedType", "unknown")
    if result.get("status") != "ok":
        return "analysis_failed"
    if "no-face" in reason or "no-face" in detected_type or detected_type == "no_face":
        return "no_face"
    if "unlabeled-no-face" in reason:
        return "unlabeled_no_face"
    if "quality-poor" in reason:
        return "poor_quality"
    if "anchor-support" in reason:
        return "low_anchor_support"
    return sanitize_part(detected_type)


def export_review_artifacts(run_path: Path, result: dict) -> str | None:
    rating = result.get("evaluation", {}).get("rating")
    if rating not in {"warn", "fail"}:
        return None

    source_path = Path(result["input"])
    bucket = sanitize_part(review_bucket(result))
    case_name = sanitize_part(source_path.stem)
    uid = sanitize_part(result.get("uid", "no-uid"))
    review_dir = run_path / "review" / bucket / rating / f"{case_name}-{uid}"
    review_dir.mkdir(parents=True, exist_ok=True)

    copied_files = {}
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
        "detectedType": result.get("detectedType"),
        "evaluation": result.get("evaluation"),
        "faceCount": result.get("faceCount", 0),
        "poses": result.get("poses", []),
        "qualities": result.get("qualities", []),
        "detectedPointRatios": result.get("detectedPointRatios", []),
        "occlusionScores": result.get("occlusionScores", []),
        "eyeClosureScores": result.get("eyeClosureScores", []),
        "profileEyeClosureScores": result.get("profileEyeClosureScores", []),
        "copiedFiles": copied_files,
    }
    (review_dir / "case_summary.json").write_text(
        json.dumps(case_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return review_dir.relative_to(ROOT).as_posix()


def write_summary(run_path: Path, results: list[dict], *, total_uploads: int, offset: int, limit: int | None) -> None:
    by_rating = Counter(result["evaluation"]["rating"] for result in results)
    by_group_rating = defaultdict(Counter)
    by_detected_type_rating = defaultdict(Counter)
    for result in results:
        group_type = result.get("expectedCategory") or result["detectedType"]
        by_group_rating[group_type][result["evaluation"]["rating"]] += 1
        by_detected_type_rating[result["detectedType"]][result["evaluation"]["rating"]] += 1

    summary_payload = {
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "uploadDir": UPLOAD_DIR.relative_to(ROOT).as_posix(),
        "caseCount": len(results),
        "totalUploadCount": total_uploads,
        "offset": offset,
        "limit": limit,
        "retentionFeatureGuard": os.getenv("VERITAI_RETENTION_FEATURE_GUARD", "none"),
        "criteriaVersion": "upload-audit-v1",
        "ratingCounts": dict(sorted(by_rating.items())),
        "groupRatingCounts": {
            group_type: dict(sorted(counter.items()))
            for group_type, counter in sorted(by_group_rating.items())
        },
        "detectedTypeRatingCounts": {
            detected_type: dict(sorted(counter.items()))
            for detected_type, counter in sorted(by_detected_type_rating.items())
        },
        "reviewRoot": (run_path / "review").relative_to(ROOT).as_posix(),
        "results": results,
    }
    (run_path / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Upload Face Detection Audit",
        "",
        f"- Generated At: `{summary_payload['generatedAt']}`",
        f"- Upload Directory: `{summary_payload['uploadDir']}`",
        f"- Image Count: `{len(results)}`",
        f"- Total Upload Count: `{total_uploads}`",
        f"- Offset: `{offset}`",
        f"- Limit: `{limit if limit is not None else 'all'}`",
        f"- Retention Feature Guard: `{summary_payload['retentionFeatureGuard']}`",
        f"- Criteria Version: `{summary_payload['criteriaVersion']}`",
        f"- Review Directory: `{summary_payload['reviewRoot']}`",
        "",
        "## Overall",
        "",
        "| Rating | Count |",
        "|---|---:|",
    ]
    for rating in ("pass", "warn", "fail"):
        lines.append(f"| {rating} | {by_rating.get(rating, 0)} |")

    lines.extend(["", "## By Expected Type", "", "| Expected Type | Pass | Warn | Fail | Total |", "|---|---:|---:|---:|---:|"])
    for group_type, counter in sorted(by_group_rating.items()):
        total = sum(counter.values())
        lines.append(
            f"| `{group_type}` | {counter.get('pass', 0)} | {counter.get('warn', 0)} | "
            f"{counter.get('fail', 0)} | {total} |"
        )

    lines.extend(["", "## By Detected Type", "", "| Detected Type | Pass | Warn | Fail | Total |", "|---|---:|---:|---:|---:|"])
    for detected_type, counter in sorted(by_detected_type_rating.items()):
        total = sum(counter.values())
        lines.append(
            f"| `{detected_type}` | {counter.get('pass', 0)} | {counter.get('warn', 0)} | "
            f"{counter.get('fail', 0)} | {total} |"
        )

    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| File | Expected | Detected Type | Eval | Face Count | Poses | Qualities | Reason | Review Dir |",
            "|---|---|---|---|---:|---|---|---|---|",
        ]
    )
    for result in results:
        source_path = Path(result["input"])
        rel_path = source_path.relative_to(ROOT).as_posix()
        poses = ", ".join(result.get("poses", [])) if result.get("poses") else "-"
        qualities = ", ".join(result.get("qualities", [])) if result.get("qualities") else "-"
        evaluation = result["evaluation"]
        review_dir = result.get("reviewDir") or "-"
        expected = result.get("expectedCategory") or "-"
        lines.append(
            f"| `{rel_path}` | `{expected}` | `{result['detectedType']}` | {evaluation['rating']} | "
            f"{result.get('faceCount', 0)} | {poses} | {qualities} | {evaluation['reason']} | `{review_dir}` |"
        )

    lines.extend(
        [
            "",
            "## Evaluation Rules",
            "",
            "- If an upload filename matches `ai/tuning_cases`, the matching tuning category is used as the expected type.",
            "- Matched uploads use the same pass/warn/fail rules as `run_tuning_cases.py`.",
            "- Unmatched uploads fall back to detection-quality audit rules.",
            "- For unmatched uploads, `no_face` is `warn` rather than `fail` because unrelated/non-human photos may be valid no-face inputs.",
        ]
    )
    (run_path / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit face detection behavior on backend uploaded images.")
    parser.add_argument("--offset", type=int, default=0, help="Number of sorted upload images to skip.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of upload images to audit.")
    parser.add_argument("--label", default=None, help="Optional suffix for the output run directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uploads = collect_uploads()
    total_uploads = len(uploads)
    offset = max(args.offset, 0)
    if args.limit is not None and args.limit >= 0:
        uploads = uploads[offset : offset + args.limit]
    else:
        uploads = uploads[offset:]
    label_index = build_label_index()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = timestamp
    if args.label:
        run_name = f"{run_name}_{sanitize_part(args.label)}"
    run_path = RUN_DIR / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    results = []
    for upload in uploads:
        result = analyze_file(upload)
        expected_category = expected_category_for_upload(upload, label_index)
        result["detectedType"] = infer_detected_type(result)
        result["expectedCategory"] = expected_category
        if expected_category and not expected_category.startswith("ambiguous:"):
            result["evaluation"] = evaluate_result(expected_category, result)
        else:
            result["evaluation"] = evaluate_upload_detection(result)
        result["reviewDir"] = export_review_artifacts(run_path, result)
        results.append(result)

    write_summary(run_path, results, total_uploads=total_uploads, offset=offset, limit=args.limit)
    print(
        json.dumps(
            {
                "runDir": run_path.as_posix(),
                "imageCount": len(results),
                "totalUploadCount": total_uploads,
                "offset": offset,
                "limit": args.limit,
                "retentionFeatureGuard": os.getenv("VERITAI_RETENTION_FEATURE_GUARD", "none"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
