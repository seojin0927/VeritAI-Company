# 2026-05-18 얼굴 crop 중심 검사 방식 변경

## 문제 상황

브라우저에서 확장 프로그램을 켜면 원래는 모든 이미지/영상에 검사 버튼이 붙고, 사용자가 검사 버튼을 누르면 해당 미디어가 분석되어야 한다.

10:51 변경에서는 팝업/확대 이미지 후보만 검사하도록 프론트 필터를 강하게 걸었고, 그 결과 일반 이미지에는 검사 버튼이 생기지 않거나 검사가 시작되지 않는 문제가 발생했다.

이번 변경에서는 프론트 UX는 원래대로 복구하고, 병목 해결은 AI 입력 구조에서 처리하도록 방향을 바꿨다.

## 최종 방향

유지하는 것:

- 브라우저의 모든 일반 이미지/영상에 검사 버튼 표시
- 사용자가 검사 버튼을 누르는 기존 흐름 유지
- 자동 검사 모드에서도 일반 이미지/영상 검사 가능

변경하는 것:

- 전체 이미지는 얼굴 후보를 찾는 데만 사용
- 얼굴 특징 추출과 딥페이크 판정용 입력은 crop된 얼굴 이미지를 기준으로 사용
- 같은 원본 이미지가 본문과 팝업에 동시에 등장하면 동일 URL 기준으로 중복 검사를 막음

## 적용 파일

### `content.js`

롤백/수정 내용:

- 팝업/라이트박스/모달 후보만 검사하던 제한을 제거
- `shouldInspectMedia(media)`가 이제 팝업 여부가 아니라 보이는 이미지/영상인지 여부만 확인
- 모든 보이는 이미지/영상에 검사 버튼이 붙도록 복구
- 전송 모드를 `face_crop_only`로 변경
- 같은 원본 URL 또는 같은 미디어 위치는 `scannedMediaKeys`로 중복 검사 방지

핵심 동작:

```text
브라우저 이미지 버튼 클릭
-> 전체 이미지 캡처
-> backend /api/detections 로 전송
-> analysisMode=face_crop_only 포함
```

### `DetectionController.java`

변경 내용:

- `analysisMode` 요청 파라미터를 AI 서버로 전달
- 허용 모드에 `face_crop_only` 추가
- 알 수 없는 값은 `full_image`로 정규화

### `AiPredictionDto.java`

변경 내용:

- `analysisMode` 유지
- `analysisInput` 추가
- `timings` 추가

이제 백엔드 raw result JSON에서 어떤 입력 구조로 분석했는지 확인할 수 있다.

### `ai/main.py`

변경 내용:

- `/predict`에서 `analysisMode=face_crop_only` 수신
- 전체 이미지는 `detect_faces()` 단계에서 얼굴 후보 탐색에 사용
- 후보별 얼굴 영역을 crop한 뒤 얼굴 특징/품질/포즈/딥페이크 feature 계산에 사용
- 응답에 다음 정보를 추가

```json
{
  "analysisMode": "face_crop_only",
  "analysisInput": {
    "detectionImage": "full_image",
    "featureImage": "cropped_face",
    "deepfakeImage": "cropped_face"
  }
}
```

각 얼굴 객체에도 crop 입력 정보가 들어간다.

```json
{
  "analysisInput": {
    "mode": "face_crop_only",
    "detectionImage": "full_image",
    "featureImage": "cropped_face",
    "deepfakeImage": "cropped_face",
    "cropOnly": true,
    "cropSize": { "w": 159, "h": 181 }
  }
}
```

## 시간 비교

벤치마크 방식:

- 샘플: `ai/tuning_cases`의 frontal/profile 계열 10장
- 기존 방식 가정: 같은 이미지를 전체 이미지 + 팝업 이미지처럼 2회 분석
- 변경 방식: 전체 이미지로 얼굴 후보 탐색 후 crop 얼굴 기준으로 1회 분석
- 워밍업 1회 후 측정

| 항목 | 총 시간 | 평균 시간 |
|---|---:|---:|
| 기존 중복 처리 가정 | 11,315.1ms | 1,131.5ms |
| 변경 후 face crop 중심 처리 | 5,713.4ms | 571.3ms |
| 절감 추정 | 5,601.6ms | 약 49.5% |

변경 후 내부 평균:

| 단계 | 평균 시간 |
|---|---:|
| 전체 이미지 얼굴 후보 탐색 | 443.0ms |
| crop 얼굴 특징/딥페이크 feature 분석 | 126.2ms |

얼굴 검출 개수 비교:

- 기존 중복 처리의 1회차/2회차 faceCount와 변경 후 faceCount가 10개 샘플에서 동일하게 유지됨
- 즉, 이번 변경은 검사 입력 구조와 중복 호출을 줄이는 변경이며 얼굴 검출 결과 자체를 훼손하지 않았다.

## 검증 결과

수행한 검증:

- `C:\Program Files\nodejs\node.exe --check content.js`
- `ai\venv\Scripts\python.exe -m py_compile ai\main.py ai\run_tuning_cases.py ai\run_upload_audit.py`
- `backend\backend_spring\gradlew.bat clean compileJava`
- AI 직접 호출로 `analysisMode=face_crop_only` 확인
- AI 직접 호출로 per-face `deepfakeImage=cropped_face` 확인
- 10장 샘플 기준 변경 전/후 시간 비교

참고:

- `gradlew.bat compileJava`만 단독 실행하면 Gradle 증분 캐시가 컨트롤러만 보다가 기존 패키지들을 못 찾는 오류가 재현될 수 있었다.
- `gradlew.bat clean compileJava` 기준으로는 성공했다.

## 판단

적용 유지.

이유:

- 사용자가 기대한 브라우저 UX, 즉 모든 이미지에 검사 버튼이 붙는 동작을 복구했다.
- 전체 이미지를 완전히 버리지 않고 얼굴 후보 탐색에만 사용하므로 검출 안정성을 유지한다.
- 무거운 특징/딥페이크 분석 입력은 crop 얼굴 기준으로 고정해 처리 범위를 줄였다.
- 동일 원본 중복 검사를 줄여 전체 체감 시간이 약 49.5% 감소했다.

## 다음 확인 포인트

1. 실제 크롬 확장에서 일반 이미지마다 검사 버튼이 붙는지 확인
2. 버튼 클릭 시 `analysisMode=face_crop_only`로 요청되는지 확인
3. 팝업 이미지를 열었을 때 같은 원본 URL이면 중복 검사가 막히는지 확인
4. 서로 다른 URL이지만 같은 이미지인 경우까지 막으려면 추후 perceptual hash 기반 중복 제거를 추가 검토

