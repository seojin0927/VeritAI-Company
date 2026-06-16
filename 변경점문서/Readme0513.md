# VeritAI 변경사항 요약 - 2026-05-13

> 브랜치: `feature/#3얼굴외각선검출및정확도향상`  
> 기준 커밋: `f7a245b 얼굴 검출 정확도 개선`

## 한눈에 보기

이번 변경은 AI 얼굴 검출 정확도를 높이기 위한 개선입니다.

- Haar cascade 기반 검출에 더해 MediaPipe FaceLandmarker와 YuNet을 선택적 보조 검출기로 추가했습니다.
- 얼굴 후보를 무조건 받지 않고, `faceLikeScore`, 색상/피부/노이즈/윤곽/랜드마크 근거로 false positive를 걸러냅니다.
- 눈감음, 측면 얼굴, 가림 얼굴을 분리해서 평가하도록 feature와 tuning 기준을 보강했습니다.
- 튜닝 산출물과 임시 파일은 GitHub에 올리지 않도록 `.gitignore`에 제외 규칙을 추가했습니다.

## 최종 튜닝 결과

기준 run: `20260512_095253`  
최종 run: `20260512_102232`

| 구분 | pass | warn | fail |
|---|---:|---:|---:|
| 개선 전 | 115 | 28 | 26 |
| 개선 후 | 123 | 28 | 18 |

주요 변화:

| 항목 | 개선 전 | 개선 후 |
|---|---:|---:|
| `false_positive` fail | 9 | 8 |
| `eyes_closed_profile` fail | 6 | 2 |
| `frontal_good` fail | 1 | 0 |
| `occluded` fail | 6 | 5 |
| `profile_left` fail | 2 | 1 |

## 핵심 변경 파일

### `ai/main.py`

얼굴 검출 및 분석 파이프라인의 핵심 변경이 들어간 파일입니다.

- MediaPipe FaceLandmarker optional detector 추가
- OpenCV YuNet optional detector 추가
- 얼굴 윤곽 mask, contour, crop/overlay 보강
- `faceLikeScore` 추가
- `occlusionScore`, `eyeClosureScore`, `profileEyeClosureScore` 분리
- 색상 기반 non-photoreal 판별 feature 추가
- profile 방향 voting 보강
- no-face rescue 경로 추가
- false positive guard 강화

### `ai/run_tuning_cases.py`

튜닝 케이스 실행 및 평가 기준이 보강되었습니다.

- MediaPipe/YuNet rescue 결과도 평가에 반영
- `eyeClosureScores`, `occlusionScores`, `profileEyeClosureScores` 수집
- 카테고리별 pass/warn/fail 기준 정리
- label issue 후보를 summary에 표시

### `ai/models/`

선택적 보조 검출기에 필요한 모델 파일입니다.

- `face_landmarker.task`
- `face_detection_yunet_2023mar.onnx`

### `.gitignore`

다음 자료는 팀원이 볼 필요가 없거나 자동 생성물이므로 제외했습니다.

- `ai/tmp/`
- `images/tuning_runs/`
- 기존 `images/analysis/`, `images/faces/`, `uploads/`

## 단계별 개선 요약

### 1. Non-photoreal / False Positive 감소

얼굴처럼 보이는 그림, 조각상, 물체 패턴을 줄이기 위해 색상/chroma feature를 추가했습니다.

결과:

- `false_positive` fail `9 -> 8`
- 회귀 없음

### 2. No-face rescue 강화

Haar cascade가 놓친 얼굴을 MediaPipe/YuNet으로 보완하되, 강한 얼굴 근거가 있을 때만 통과시키도록 했습니다.

결과:

- `frontal_good` no-face 1건 복구
- `occluded` no-face 1건 복구
- false positive 증가 없음

### 3. Profile 방향 안정화

넓은 edge voting은 회귀가 있어 제외했고, 매우 좁은 조건의 nose 기반 보정만 유지했습니다.

결과:

- `profile_left` 방향 flip 1건 복구
- profile_right 회귀 없음

### 4. 눈감음 profile 전용 feature

정면처럼 잡히지만 실제로는 profile 눈감음에 가까운 케이스를 위해 `profileEyeClosureScore`를 보강했습니다.

결과:

- `eyes_closed_profile` fail `6 -> 2`
- 회귀 없음

### 5. 라벨 품질 분리

남은 non-pass 중 일부는 코드로 맞추기보다 라벨 재검토가 필요한 케이스로 분리했습니다.

남은 non-pass 46건 중 label issue 후보 14건:

- `possible-mislabeled-false-positive`: 7건
- `eyes-closed-label-but-closure-signal-weak`: 4건
- `possible-direction-label-mismatch`: 2건
- `possible-semi-profile-mixed-into-frontal`: 1건

## 검증 방법

문법 확인:

```bash
ai\venv\Scripts\python.exe -m py_compile ai\main.py ai\run_tuning_cases.py
```

전체 튜닝 실행:

```bash
ai\venv\Scripts\python.exe ai\run_tuning_cases.py
```

Windows에서 matplotlib cache 권한 경고가 나면 다음 환경변수를 먼저 지정합니다.

```bash
set MPLCONFIGDIR=C:\Users\Administrator\VeritAI-Project\VeritAI\ai\tmp\matplotlib
```

## 팀원이 보면 좋은 포인트

- 실제 서비스 로직은 `ai/main.py` 중심으로 보면 됩니다.
- 튜닝 기준과 평가 결과 해석은 `ai/run_tuning_cases.py`를 보면 됩니다.
- `images/tuning_runs/`는 로컬 검증 산출물이므로 GitHub에 올리지 않습니다.
- 남은 실패 케이스 중 일부는 모델 문제가 아니라 라벨 기준 문제일 수 있어, 추가 rule을 넣기 전에 label review가 필요합니다.
