# 📘 VeritAI PR 통합 문서

> 기준일: **2026-04-10**
>
> 이 문서는 현재 PR 기준으로 팀원이 빠르게 이해할 수 있도록
> 구현 현황, 얼굴 검사 방식, 저장 폴더 구조, 사용 방법, 미구현 기능, 개선 포인트를 통합 정리한 문서입니다.

---

## 🎯 현재 진척도

### 프론트엔드 / 브라우저 확장
**약 40%**

- 웹페이지의 `img`, `video` 요소 감지
- 미디어 위 검사 버튼 부착
- 이미지 / 비디오 프레임 캡처 후 백엔드 전송
- 분석 결과 `alert` 출력
- DOM 변경, 스크롤, 리사이즈 대응
- 버튼 재부착 로직 경량화 및 반복 스캔 최적화 **(2026-04-10 완료)**

### 백엔드 / Spring Boot
**약 50%**

- 파일 업로드 API 구현
- 요청 메타데이터 DB 저장
- AI 서버 호출
- AI 결과 저장 및 재조회 API 구현
- 상태값 `PROCESSING / DONE / FAILED` 관리

### AI / FastAPI
**약 55%**

- 얼굴 후보 검출
- pose 분류
- 눈 / 코 / 입 response map 생성
- `detected / estimated` 포인트 구분 저장
- anchor graph 시각화 저장
- 얼굴 crop 저장
- 딥페이크 확장용 feature 저장
- 튜닝 케이스 실행 스크립트 추가 **(2026-04-10 완료)**
- 오검출 리뷰용 자동 분류 저장 구조 추가 **(2026-04-10 완료)**
- 튜닝 가이드 문서 추가 **(2026-04-10 완료)**
- edge 연산 중복 제거 경량화 **(2026-04-10 완료)**

---

## ✅ 현재 구현된 기능

### 1. 브라우저 확장 기능

- 모든 페이지의 `img`, `video`를 탐지합니다.
- 미디어 위에 빨간 원형 `🔍` 검사 버튼을 붙입니다.
- 이미지 클릭 시 원본 이미지를 캡처해서 전송합니다.
- 비디오 클릭 시 현재 보이는 프레임을 캡처해서 전송합니다.
- 백엔드 응답을 받아 요청 ID, 신뢰도, 얼굴 수, 품질 정보를 표시합니다.
- 버튼 부착 로직을 `requestAnimationFrame` 기반으로 조정해 과도한 반복 탐색을 줄였습니다. **(2026-04-10 완료)**

### 2. 백엔드 기능

- `POST /api/detections` 로 파일 업로드를 받습니다.
- 업로드 파일을 `uploads/` 에 저장합니다.
- 요청 메타데이터를 H2 DB에 저장합니다.
- AI 서버 `http://localhost:8000/predict` 로 파일을 전달합니다.
- AI 응답 전체를 JSON 형태로 저장합니다.
- `GET /api/detections/{requestId}` 로 결과 재조회가 가능합니다.

### 3. AI 분석 기능

- OpenCV cascade 기반 얼굴 후보를 추출합니다.
- 눈 / 코 / 입 response map 을 만들어 주요 포인트를 찾습니다.
- 얼굴 상태를 `frontal`, `profile-left`, `profile-right`, `occluded`, `eyes_closed` 로 분류합니다.
- keypoint를 `detected` 와 `estimated` 로 나누어 저장합니다.
- 분석 결과를 anchor graph 이미지와 JSON으로 저장합니다.
- 얼굴 crop 을 별도 저장합니다.
- `trainingSample`, `deepfakeFeatures` 를 생성합니다.
- 튜닝용 케이스 폴더를 기반으로 일괄 분석할 수 있습니다. **(2026-04-10 완료)**
- `warn`, `fail` 케이스를 자동으로 리뷰 폴더로 복사 저장합니다. **(2026-04-10 완료)**

---

## 🧠 얼굴 검사 방식

### 1. 얼굴 후보 검출

- OpenCV Haar Cascade 를 사용해 정면 얼굴과 측면 얼굴 후보를 찾습니다.
- `frontal`, `frontal_alt`, `profile` 검출기를 함께 사용합니다.
- 좌우 반전 이미지를 같이 검사해 측면 얼굴도 보완합니다.
- 중복 박스는 `non_max_suppression` 으로 제거합니다.

### 2. 눈 검출

- 얼굴 상단 영역에서 `eye response map` 을 생성합니다.
- 어두운 수평 구조, gradient, 위치 prior 를 합쳐 눈 후보를 찾습니다.
- 후보 쌍의 간격, 높이 차이, 면적 비율을 이용해 가장 그럴듯한 눈 구성을 선택합니다.

### 3. 코 검출

- 얼굴 중앙 세로 영역에서 `nose response map` 을 생성합니다.
- 밝기, tophat, 세로 구조를 조합해 코 중심부와 코끝 후보를 찾습니다.
- 측면 얼굴일 때는 pose 방향에 맞는 prior 를 적용합니다.

### 4. 입 검출

- 얼굴 하단 영역에서 `mouth response map` 을 생성합니다.
- 검은 수평 구조와 bbox 비율을 이용해 입 후보를 찾습니다.
- pose 에 따라 입 양 끝점의 `detected / estimated` 처리 기준이 달라집니다.

### 5. 얼굴 상태 분류

- 눈 검출 개수
- 눈 좌우 정렬 상태
- 눈 aspect 비율
- 얼굴 가장자리 에너지 분포

이 정보를 이용해 아래 상태 중 하나로 분류합니다.

- `frontal`
- `profile-left`
- `profile-right`
- `occluded`
- `eyes_closed`

### 6. 얼굴 구역 분할

아래 주요 포인트를 기준으로 anchor graph 를 만듭니다.

- `forehead_center`
- `left_eye_center`
- `right_eye_center`
- `nose_bridge_top`
- `nose_tip`
- `mouth_left`
- `mouth_center`
- `mouth_right`
- `chin`

이 포인트를 연결해 아래 영역을 시각화합니다.

- `forehead`
- `left_eye_zone`
- `right_eye_zone`
- `nose`
- `mouth`
- `jaw`

### 7. 오늘 추가된 점 **(2026-04-10 추가)**

- 튜닝 케이스 기준 문서화
- `warn / fail` 케이스 자동 리뷰 폴더 저장
- profile / eyes_closed / occluded 케이스 검토용 배치 실행 구조 보강
- edge 연산 재사용으로 얼굴당 중복 계산 일부 제거

---

## 📁 결과 저장 폴더

### 기본 분석 결과

- [images/analysis/overlays](C:/Users/Administrator/VeritAI-Project/VeritAI/images/analysis/overlays)
  얼굴 bbox 와 눈 박스가 그려진 이미지 저장

- [images/analysis/anchor_maps](C:/Users/Administrator/VeritAI-Project/VeritAI/images/analysis/anchor_maps)
  얼굴 anchor graph, keypoint, 분석 영역이 그려진 이미지 저장

- [images/analysis/metadata](C:/Users/Administrator/VeritAI-Project/VeritAI/images/analysis/metadata)
  전체 분석 결과 JSON 저장

- [images/analysis/response_maps/eyes](C:/Users/Administrator/VeritAI-Project/VeritAI/images/analysis/response_maps/eyes)
  눈 response heatmap 저장

- [images/analysis/response_maps/nose](C:/Users/Administrator/VeritAI-Project/VeritAI/images/analysis/response_maps/nose)
  코 response heatmap 저장

- [images/analysis/response_maps/mouth](C:/Users/Administrator/VeritAI-Project/VeritAI/images/analysis/response_maps/mouth)
  입 response heatmap 저장

- [images/faces](C:/Users/Administrator/VeritAI-Project/VeritAI/images/faces)
  검출된 얼굴 crop 저장

### 튜닝 케이스 입력 폴더 **(2026-04-10 추가)**

- [ai/tuning_cases/frontal_good](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/tuning_cases/frontal_good)
- [ai/tuning_cases/profile_left](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/tuning_cases/profile_left)
- [ai/tuning_cases/eyes_closed](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/tuning_cases/eyes_closed)
- [ai/tuning_cases/occluded](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/tuning_cases/occluded)
- [ai/tuning_cases/false_positive](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/tuning_cases/false_positive)
- [ai/tuning_cases/false_negative](C:/Users/Administrator/VeritAI-Project/VeritAI/ai/tuning_cases/false_negative)

### 튜닝 실행 결과 폴더 **(2026-04-10 추가)**

- [images/tuning_runs](C:/Users/Administrator/VeritAI-Project/VeritAI/images/tuning_runs)
  배치 실행별 요약 결과 저장

- `summary.json`
  전체 실행 결과의 원본 데이터

- `summary.md`
  케이스별 `pass / warn / fail` 요약표

- `review/<bucket>/<rating>/<case-name>-<uid>/`
  `warn`, `fail` 케이스 자동 복사 저장 폴더

### 리뷰 폴더에 저장되는 파일 **(2026-04-10 추가)**

- 원본 입력 이미지
- `*_analysis.jpg`
- `*_eye_response.jpg`
- `*_nose_response.jpg`
- `*_mouth_response.jpg`
- `*_analysis.json`
- `case_summary.json`

---

## 🧾 JSON 에 저장되는 주요 데이터

### `bbox`
- 얼굴 박스 좌표와 크기

### `pose`
- 얼굴 방향 또는 상태 분류
- `label`, `confidence`, `reason`

### `quality`
- `blur`
- `brightness`
- `contrast`
- `detectedPointRatio`
- 최종 품질 label

### `keypoints`
- 각 포인트 좌표
- `source: detected | estimated`
- `confidence`
- `reason`

### `analysisConnections`
- 포인트 간 연결 정보

### `analysisRegions`
- forehead / eyes / nose / mouth / jaw 영역 정보

### `trainingSample`
- 이후 학습 데이터로 활용 가능한 정규화 포인트 정보

### `deepfakeFeatures`
- `geometry`
- `texture`
- `visibility`

### `generatedFiles`
- 생성된 overlay / analysis / response map / metadata 경로

---

## ▶️ 실행 방법

### 1. AI 서버 실행

```powershell
.\ai\venv\Scripts\python.exe .\ai\main.py
```

### 2. 백엔드 실행

```powershell
cd .\backend\backend_spring
.\gradlew.bat bootRun
```

### 3. 브라우저 확장 실행

1. 크롬에서 `chrome://extensions` 접속
2. 개발자 모드 활성화
3. `압축해제된 확장 프로그램 로드` 선택
4. 프로젝트 루트 폴더 로드

### 4. 튜닝 케이스 실행 **(2026-04-10 추가)**

```powershell
.\ai\venv\Scripts\python.exe .\ai\run_tuning_cases.py
```

---

## 🚧 아직 구현되지 않은 기능

### 프론트엔드

- 플러그인 `on / off` 토글
- 팝업 UI
- 실시간 진행 상태 표시
- `프레임 추출 중 > 데이터 전송 중 > 판별 완료` 단계 안내
- SNS 피드 상시 자동 검사
- 결과 배지 / 경고 오버레이 UI

### 백엔드

- 메시지 큐 기반 비동기 처리
- 대기열 제어
- 워커 분리 구조
- 실시간 상태 push
- 재시도 / 장애 복구 정책

### AI

- 실제 딥페이크 판별 모델
- PyTorch / TensorFlow 기반 경량 분류기
- 학습 파이프라인
- 비디오 시계열 기반 판별
- 모델 성능 평가 지표 자동 관리

---

## 🔄 미구현에서 구현으로 이동한 항목

- 튜닝 케이스 일괄 실행 스크립트 **(2026-04-10 완료)**
- 오검출 리뷰용 자동 분류 저장 구조 **(2026-04-10 완료)**
- 튜닝 가이드 문서화 **(2026-04-10 완료)**
- 분석 결과 저장 폴더 세분화 **(2026-04-10 완료)**
- 얼굴 분석 시각화 구조 고도화 **(2026-04-10 완료)**

---

## ⚠️ 현재 한계와 개선 포인트

### 1. 측면 얼굴 한계
- `profile-left`, `profile-right` 보정이 들어가 있지만 완전하지 않습니다.

### 2. 눈 감김 / 가림 케이스 한계
- `eyes_closed`, `occluded` 분류는 가능하지만 실제 landmark 수준 정확도는 아닙니다.

### 3. 비디오 분석 한계
- 현재는 전체 비디오 분석이 아니라 현재 보이는 프레임 캡처 기반입니다.

### 4. 딥페이크 판별 미구현
- 현재 AI는 얼굴 분석과 feature 추출 중심입니다.

### 5. 백엔드 비동기 미구현
- 지금은 요청이 들어오면 즉시 AI 서버를 호출하는 동기식 구조입니다.

### 6. 테스트 환경 이슈 **(2026-04-10 추가)**
- 백엔드 `compileJava` 는 통과했지만 `gradlew test` 는 현재 Gradle 캐시 접근 권한 문제로 환경에 따라 실패할 수 있습니다.
- AI Python 실행 환경은 일부 로컬 환경에서 `venv` 실행기 문제를 점검할 필요가 있습니다.

---

## 📌 핵심 요약

현재 프로젝트는 **브라우저 확장 → Spring Boot 백엔드 → FastAPI AI 서버** 까지 기본 연동이 가능한 상태입니다.

AI 쪽은 단순 bbox 검출을 넘어서
**pose-aware anchor graph**, **response map**, **detected / estimated keypoint 분리**, **시각화 저장**, **튜닝 케이스 자동 검토 구조**까지 들어와 있습니다.

오늘 2026-04-10 기준으로는 특히
**튜닝 자동화**, **오검출 리뷰 저장**, **문서화**, **프론트/AI 경량화**가 추가된 상태입니다. **(2026-04-10 추가)**
