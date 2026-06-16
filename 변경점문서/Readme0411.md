# VeritAI 변경사항 정리

> 기준 날짜: **2026-04-11**
>
> 이 문서는 기존 `ReadmePR0401.md` 이후 추가로 진행된 작업을 반영한 문서입니다.
> 오늘까지의 개선 사항과 앞으로 더 다듬어야 할 항목을 팀원이 빠르게 이해할 수 있도록 정리했습니다.

---

## 1. 이번 문서에서 새로 반영한 핵심 변화

- AI 얼굴 분석 튜닝을 여러 차례 반복 실행할 수 있는 **재실행 가능한 Python 환경 복구**
- 튜닝 데이터셋 구조를 세분화하고, 그 구조에 맞춘 **평가 규칙 개편**
- 얼굴 감김/가림/측면 분류를 더 잘 보기 위한 **pose-aware 튜닝 강화**
- 브라우저 검사 결과 alert와 오류 문구를 **한국어로 통일**
- 최신 튜닝 결과를 바탕으로 **남은 병목 구간을 구체화**

---

## 2. 개선된 점

### 2-1. AI 실행 환경 복구

- 가상환경을 다시 구성해 `run_tuning_cases.py`를 실제로 반복 실행할 수 있게 복구했습니다. **(2026-04-11 추가)**
- 이제 수집한 케이스 이미지를 넣고 바로 재실행해 결과를 비교할 수 있습니다. **(2026-04-11 추가)**

관련 경로:
- [ai/venv](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/venv)
- [ai/run_tuning_cases.py](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/run_tuning_cases.py)

---

### 2-2. 튜닝 데이터셋 구조 세분화

기존에는 `eyes_closed`, `profile_left`처럼 일부 카테고리가 넓게 묶여 있어서 오분류 원인을 정확히 보기 어려웠습니다.

아래 구조로 세분화했습니다. **(2026-04-11 추가)**

- [ai/tuning_cases/eyes_closed_frontal](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/tuning_cases/eyes_closed_frontal)
- [ai/tuning_cases/eyes_closed_profile](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/tuning_cases/eyes_closed_profile)
- [ai/tuning_cases/eyes_closed_occluded](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/tuning_cases/eyes_closed_occluded)
- [ai/tuning_cases/profile_left](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/tuning_cases/profile_left)
- [ai/tuning_cases/profile_right](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/tuning_cases/profile_right)
- [ai/tuning_cases/frontal_good](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/tuning_cases/frontal_good)
- [ai/tuning_cases/occluded](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/tuning_cases/occluded)
- [ai/tuning_cases/false_positive](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/tuning_cases/false_positive)
- [ai/tuning_cases/false_negative](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/tuning_cases/false_negative)

효과:
- `눈 감음` 실패인지 `측면 얼굴` 실패인지 분리해서 볼 수 있습니다. **(2026-04-11 추가)**
- `좌측 프로필 / 우측 프로필` 방향 뒤집힘 문제를 별도로 추적할 수 있습니다. **(2026-04-11 추가)**
- `false_positive`와 `false_negative`를 더 명확하게 튜닝할 수 있습니다. **(2026-04-11 추가)**

---

### 2-3. 평가 로직 개편

데이터 구조 변화에 맞춰 튜닝 평가 기준도 업데이트했습니다. **(2026-04-11 추가)**

개편 내용:
- `eyes_closed_frontal`, `eyes_closed_profile`, `eyes_closed_occluded`별로 pass/warn/fail 기준 분리
- `profile_right` 별도 평가 추가
- 라벨 품질이 의심되는 항목을 찾는 `Label Quality Notes` 섹션 추가

관련 파일:
- [ai/run_tuning_cases.py](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/run_tuning_cases.py)
- [ai/tuning_cases/README.md](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/tuning_cases/README.md)
- [ai/tuning_cases/TUNING_GUIDE.md](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/tuning_cases/TUNING_GUIDE.md)

---

### 2-4. AI 얼굴 분석 로직 추가 개선

다음 항목을 중심으로 `ai/main.py`를 여러 차례 튜닝했습니다. **(2026-04-11 추가)**

- 눈 후보가 눈썹이나 머리카락 그림자를 잡지 않도록 eye scoring 보정
- `no-eye` 상황에서 곧바로 잘못된 profile로 넘어가지 않도록 fallback 조정
- `false_positive`를 줄이기 위한 약한 후보 제거 강화
- `frontal`, `profile-left`, `profile-right`, `occluded`, `eyes_closed` 경계 조정
- 얼굴 중심축 편향값(`signedCenterAxisBias`)을 활용한 pose 재보정 추가
- detector score, edge density, closure signal을 함께 보도록 보정

관련 파일:
- [ai/main.py](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/main.py)

---

### 2-5. 최신 튜닝 결과 반영

최신 기준으로 확인한 튜닝 결과는 아래 리포트입니다. **(2026-04-11 추가)**

- [images/tuning_runs/20260411_150724/summary.md](C:/Users/Administrator/VeritAI-Project/VeritAI/images/tuning_runs/20260411_150724/summary.md)

핵심 결과 요약:
- 전체: `pass 35 / warn 33 / fail 21`
- `false_negative`: `10/10 pass`
- `false_positive`: `7/12 pass`
- `eyes_closed_frontal`: `5/7 pass`
- `eyes_closed_occluded`: `5/6 pass`
- `eyes_closed_profile`: `5/9 pass`

의미:
- 얼굴이 있는데 아예 못 찾는 문제는 크게 줄었습니다. **(2026-04-11 추가)**
- `eyes_closed` 계열 정확도는 이전보다 확실히 좋아졌습니다. **(2026-04-11 추가)**
- 반면 `frontal_good`, `profile_left`, `profile_right`, `occluded`는 아직 경계가 불안정합니다. **(2026-04-11 추가)**

---

### 2-6. 브라우저 안내 문구 한글화

검사 후 브라우저에 뜨는 문구를 한국어로 정리했습니다. **(2026-04-11 추가)**

반영 내용:
- 검사 버튼 라벨: `검사`, `분석중`
- 결과 안내: `요청 ID`, `판정 결과`, `신뢰도`, `감지된 얼굴 수`, `워터마크 감지`, `모델 버전`, `처리 시간`, `분석 메시지`
- 얼굴 요약: `위치`, `유형`, `검출 신뢰도`, `품질`
- 오류 문구: 한국어로 통일

관련 파일:
- [content.js](C:/Users/Administrator/VeritAI-Project/VeritAI/content.js)

---

## 3. 현재까지 정리된 개선 포인트

### 잘 개선된 부분

- 튜닝 실행 환경 복구
- 데이터셋 카테고리 세분화
- 평가 기준 정교화
- false negative 감소
- eyes closed 계열 분류 개선
- false positive 일부 억제
- 브라우저 안내 문구 한국어화

### 아직 흔들리는 부분

- 정면 눈뜸 얼굴이 `eyes_closed`로 잘못 가는 경우
- `profile_left / profile_right` 방향 뒤집힘
- `occluded`와 `eyes_closed` 경계 혼선
- 일부 `frontal_good`이 `warn`으로 많이 남는 문제

---

## 4. 앞으로 개선해야 할 사항

### AI 얼굴 분석

1. `frontal_good`와 `eyes_closed`를 더 안정적으로 분리해야 합니다.
2. `profile_left`와 `profile_right`의 방향 판정을 더 정확히 보정해야 합니다.
3. `occluded`와 `eyes_closed`의 경계를 분리하는 규칙이 더 필요합니다.
4. 눈 외에 입 영역 response map도 더 강화해야 합니다.
5. 현재 메타데이터를 딥페이크 판별용 전처리 피처로 쓰기 전에 pose 안정성을 더 높여야 합니다.

### 프론트엔드

1. 팝업 UI 추가
2. 플러그인 on/off 기능
3. `프레임 추출 중 > 데이터 전송 중 > 판별 완료` 상태 표시
4. alert 대신 오버레이/배지 형태의 결과 UI

### 백엔드

1. 메시지 큐 기반 비동기 처리
2. 작업 상태 조회 고도화
3. 트래픽 과부하 방지용 대기열 관리

### 딥페이크 판별

1. PyTorch/TensorFlow 기반 실제 분류 모델 연결
2. 현재 저장 중인 `deepfakeFeatures` 기반 데이터셋 설계
3. 프레임 단위 결과를 집계하는 판별 파이프라인 설계

---

## 5. 팀원이 보면 좋은 파일

- [content.js](C:/Users/Administrator/VeritAI-Project/VeritAI/content.js)
- [ai/main.py](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/main.py)
- [ai/run_tuning_cases.py](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/run_tuning_cases.py)
- [ai/tuning_cases/README.md](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/tuning_cases/README.md)
- [ai/tuning_cases/TUNING_GUIDE.md](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/tuning_cases/TUNING_GUIDE.md)
- [images/tuning_runs/20260411_150724/summary.md](C:/Users/Administrator/VeritAI-Project/VeritAI/images/tuning_runs/20260411_150724/summary.md)

---

## 6. 한 줄 요약

`0401 기준 문서 이후, 얼굴 분석 튜닝 환경과 데이터 구조가 한 단계 정리되었고, false negative와 eyes_closed 계열은 개선되었지만, frontal/profile/occluded 경계 안정화는 아직 추가 튜닝이 필요한 상태입니다.`
