# 2026-07-09 얼굴 검출 v2 적용 판단 요약

## 0616~0709 진행 요약

이 문서는 2026-06-16부터 2026-07-09까지 진행한 얼굴 검출 파이프라인 검증/튜닝 내용을 팀 공유용으로 축약한 것이다.

상세 실험 로그, 중간 리포트, 이미지 리뷰 artifact 전체는 커밋 대상에서 제외한다. 팀원이 확인해야 할 결론과 적용 방법만 남긴다.

### 1. Baseline 고정

WIDER FACE, Kaggle face-detection, Kaggle face-mask를 JSONL manifest로 변환해 동일 조건으로 평가했다.

초기 full baseline:

| Dataset | Candidate Precision | Candidate Recall | Analysis Precision | Analysis Recall |
|---|---:|---:|---:|---:|
| WIDER val | 0.3285 | 0.1937 | 0.3470 | 0.1652 |
| Kaggle face-detection val | 0.2686 | 0.5952 | 0.2832 | 0.4945 |
| Kaggle face-mask | 0.5460 | 0.4153 | 0.5737 | 0.3519 |

해석:

- WIDER는 crowd/small face가 많아 recall이 낮다.
- Kaggle face-detection은 recall은 상대적으로 높지만 FP가 많다.
- mask 세트는 precision이 가장 안정적이지만 mask/occlusion 구간에서 놓치는 얼굴이 남았다.

### 2. 실패 유형 분리

검출 bbox 성능과 analysis 통과율을 분리해서 봤다.

주요 실패 유형:

- false negative
- false positive
- profile 방향 오류
- occlusion 과대판정
- 비인간 객체 얼굴 오탐
- 분석 단계에서 품질/anchor 부족으로 탈락

이 단계에서 새 모델 학습보다, 반복 실패에 대한 후보 생성/retention 튜닝이 먼저라고 판단했다.

### 3. Retention 튜닝 방향

후보 bbox 단계에서 분석 탈락 후보 중 일부를 다시 살리는 `precision_guarded` retention 방향을 검증했다.

핵심 원칙:

- recall을 올리되 FP 폭증은 막는다.
- 반복 FP 유형은 feature guard로 다시 차단한다.
- 성능이 좋아지지 않는 튜닝은 적용하지 않는다.

### 4. `retention_precision_combo` 검증

기존 retention guard 조합은 FP를 줄이면서 명확한 TP 손실이 없어 적용 후보가 됐다.

대표 결과:

- Mask full: TP 유지, FP 감소
- WIDER paired: TP 손실 없음, FP 감소
- Kaggle FD paired: TP 손실 없음, FP 감소

이후 남은 FP 중 `frontal-alt-anthropometric-outlier` 비중이 커서 v2 조건을 추가 검토했다.

### 5. `retention_precision_combo_v2` 검증

v2는 기존 combo에 아래 guard를 추가한 것이다.

```text
keep_reason == "frontal-alt-anthropometric-outlier"
and colorSaturationMean < 0.1264
and qualityScore < 0.8252
```

목적:

- low saturation + 낮은 quality를 가진 anthropometric outlier FP를 줄인다.
- 정상 얼굴 TP 손실은 만들지 않는다.

외부 데이터셋 최종 합산:

| Guard | TP | FP | Precision |
|---|---:|---:|---:|
| `retention_precision_combo` | 291 | 136 | 0.6815 |
| `retention_precision_combo_v2` | 292 | 127 | 0.6969 |

Delta:

- TP +1
- FP -9
- precision +0.0154

보수적으로 TP +1은 재실행 변동으로 보고, 핵심 판단은 “TP 손실 없이 FP 감소”다.

### 6. 내부 업로드/서비스 smoke

내부 unlabeled smoke:

- retained 후보 수 증가 없음
- pipeline error 없음

업로드 감사 979장:

| Images | Pass | Warn | Fail |
|---:|---:|---:|---:|
| 979 | 898 | 80 | 1 |

Fail 1건은 기존 combo와 v2가 동일하게 실패했으므로 v2 신규 부작용으로 보지 않는다.

FastAPI `/predict` smoke:

- health check OK
- multipart 요청 4건 OK
- frontal/profile/no_face/poor_quality 대표 케이스 정상

Spring Backend -> FastAPI smoke:

| Metric | Value |
|---|---:|
| caseCount | 4 |
| doneCount | 4 |
| failedCount | 0 |
| totalAiCallCount | 4 |
| totalRetryCount | 0 |

Spring `/api/detections` -> queue worker -> FastAPI `/predict` -> polling 응답까지 정상 확인했다.

## 결론

`retention_precision_combo_v2`는 적용 가능 상태로 판단한다.

단, 적용 방식은 코드 기본값 변경이 아니라 실행 환경변수 적용을 권장한다.

```text
VERITAI_RETENTION_FEATURE_GUARD=retention_precision_combo_v2
```

즉:

- 환경변수 적용: GO
- `ai/main.py` 기본값 변경: 보류
- 새 모델 학습: 불필요

## 판단 근거

### 1. 외부 라벨 데이터셋 검증

결과 파일:

```text
images/eval_failure_reviews/service_retention/retention_precision_combo_v2_final_decision_summary_20260707.json
```

전체 합산:

| Guard | TP | FP | Precision |
|---|---:|---:|---:|
| `retention_precision_combo` | 291 | 136 | 0.6815 |
| `retention_precision_combo_v2` | 292 | 127 | 0.6969 |

Delta:

- TP +1
- FP -9
- precision +0.0154

보수적 해석:

- TP +1은 재실행 변동 가능성이 있으므로 강한 개선으로 주장하지 않는다.
- 중요한 점은 명확한 TP 손실이 없고 FP가 줄었다는 것이다.

### 2. 내부 unlabeled smoke

결과 파일:

```text
images/eval_failure_reviews/service_retention/retention_precision_combo_v2_internal_unlabeled_compare_20260707.json
```

요약:

- retained candidates 변화 없음
- retained images 변화 없음
- pipeline error 없음

해석:

내부 unlabeled 세트에서 v2가 후보를 과하게 차단하거나 실행 오류를 만들었다는 증거는 없다.

### 3. 업로드 감사 full chunk

결과 파일:

```text
images/upload_audit_runs/20260708_143842_retention_v2_smoke_0_100/summary.json
images/upload_audit_runs/20260708_144247_retention_v2_upload_100_300/summary.json
images/upload_audit_runs/20260708_145035_retention_v2_upload_300_500/summary.json
images/upload_audit_runs/20260708_145746_retention_v2_upload_500_700/summary.json
images/upload_audit_runs/20260708_150305_retention_v2_upload_700_end/summary.json
```

전체:

| Images | Pass | Warn | Fail |
|---:|---:|---:|---:|
| 979 | 898 | 80 | 1 |

Fail 1건:

- 기존 `retention_precision_combo`와 v2 결과가 동일했다.
- 따라서 v2 신규 부작용으로 보지 않는다.

### 4. FastAPI `/predict` smoke

결과:

- `/` health check 200 OK
- `/predict` multipart 요청 4건 200 OK
- `frontal`, `profile`, `no_face`, `poor_quality` 대표 케이스 정상 응답
- CNN model loaded: `veritai-anchor-cnn-v1`

### 5. Spring Backend -> FastAPI smoke

결과 파일:

```text
images/eval_failure_reviews/service_retention/spring_fastapi_v2_smoke_20260708_153624.json
```

요약:

| Metric | Value |
|---|---:|
| caseCount | 4 |
| doneCount | 4 |
| failedCount | 0 |
| totalEnqueuedCount | 4 |
| totalCompletedCount | 4 |
| totalFailedCount | 0 |
| totalAiCallCount | 4 |
| totalRetryCount | 0 |

해석:

Spring `/api/detections` 업로드, queue worker, FastAPI `/predict`, polling 결과 저장까지 정상이다.

### 6. 릴리즈 산출물 체크

체크 스크립트:

```text
scripts/check_retention_v2_release_artifacts.ps1
```

결과 파일:

```text
images/eval_failure_reviews/service_retention/retention_v2_release_artifacts_check_20260709_101928.json
```

결과:

```text
total=11
present=11
missing=0
ready=true
```

## 적용 방법

### 권장 적용

AI 서버 실행 환경에서 아래 환경변수를 설정한다.

```powershell
$env:VERITAI_RETENTION_FEATURE_GUARD='retention_precision_combo_v2'
.\ai\venv\Scripts\python.exe ai\main.py
```

또는 고정 스크립트를 사용한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_ai_server_retention_v2.ps1
```

Spring 설정은 변경하지 않는다.

```properties
app.ai-server-url=http://localhost:8000/predict
```

## 적용 후 확인

AI 서버와 Spring 서버를 띄운 뒤 아래 smoke를 실행한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\smoke_spring_fastapi_v2.ps1
```

성공 기준:

- `doneCount=4`
- `failedCount=0`
- `totalFailedCount=0`
- `totalRetryCount=0`
- 4개 대표 케이스가 모두 `DONE`

## 롤백

환경변수를 제거하거나 `none`으로 설정한다.

```powershell
Remove-Item Env:\VERITAI_RETENTION_FEATURE_GUARD -ErrorAction SilentlyContinue
.\ai\venv\Scripts\python.exe ai\main.py
```

또는:

```powershell
$env:VERITAI_RETENTION_FEATURE_GUARD='none'
.\ai\venv\Scripts\python.exe ai\main.py
```

롤백 후에도 같은 smoke를 실행해 Spring 연동이 정상인지 확인한다.

## 왜 코드 기본값 변경은 보류하는가

현재 v2는 성능상 유리하지만, 환경변수 적용만으로도 동일한 효과를 얻을 수 있다.

코드 기본값을 바꾸면:

- 모든 실행 환경에 즉시 영향을 준다.
- 롤백이 코드 수정/재배포가 된다.
- 실험 guard와 기본 정책의 경계가 흐려진다.

환경변수 적용은:

- 적용 범위를 staging/internal/production별로 나눌 수 있다.
- 문제가 생기면 즉시 `none`으로 되돌릴 수 있다.
- 현재 검증 단계에 더 적합하다.

## 최종 결정

`retention_precision_combo_v2`는 운영 적용 후보가 아니라, 환경변수 기반 적용 대상으로 승격한다.

최종 권장:

```text
GO: VERITAI_RETENTION_FEATURE_GUARD=retention_precision_combo_v2
NO-GO: ai/main.py default 변경
```
