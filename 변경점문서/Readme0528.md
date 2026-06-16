# Readme0528 - 대용량 이미지 처리 부하 개선

## 목적

브라우저에서 여러 이미지를 동시에 검사할 때 발생하는 부하를 줄이기 위해, AI에 보내는 이미지와 데이터는 변경하지 않고 요청 제어, 중복 재사용, polling, 큐 관측 기능만 개선했다.

유지한 사항:

- AI 서버로 전달되는 이미지 파일 자체는 변경하지 않음
- multipart 요청 구조 유지
- `analysisMode`, `file`, `sourceUrl`, `mediaType`, `clientType` 흐름 유지
- 얼굴 검출 및 딥페이크 분석 파이프라인의 판단 로직은 변경하지 않음

## 적용한 개선 사항

### 1. 브라우저 동시 검사 제한

파일: `content.js`

- 동시에 실행되는 검사 요청을 `MAX_CONCURRENT_INSPECTIONS = 3`으로 제한
- 초과 요청은 브라우저 내부 대기열에 넣고 순차 실행
- 화면 안에 있고 크기가 큰 미디어를 우선 검사하도록 priority queue 적용
- DOM에서 사라진 이미지나 보이지 않는 이미지는 대기열 실행 직전에 제외

효과:

- 브라우저가 한 번에 너무 많은 캡처와 API 요청을 보내는 문제 완화
- 대량 이미지 페이지에서 Spring/AI 서버로 몰리는 순간 피크 감소

### 2. 백엔드 fileHash 기반 중복 검사 재사용

파일:

- `DetectionController.java`
- `DetectionRequestRepository.java`

개선 내용:

- 업로드된 파일의 SHA-256 `fileHash`와 `analysisMode`를 기준으로 기존 요청 재사용
- 같은 이미지가 이미 `QUEUED`, `PROCESSING`, `DONE` 상태라면 새 AI 작업을 만들지 않음
- 완료된 요청은 기존 분석 결과를 즉시 반환
- 대기/처리 중 요청은 기존 `requestId`를 반환해 같은 작업을 polling

효과:

- 같은 이미지가 여러 번 검사되는 페이지에서 AI 호출, 파일 저장, 큐 적체가 크게 감소

### 3. polling backoff와 서버 권장 지연

파일:

- `content.js`
- `DetectionResponseDto.java`
- `DetectionProcessingService.java`

개선 내용:

- 백엔드 응답에 `retryAfterMs` 추가
- 큐 적체량에 따라 권장 polling 간격을 조정
- 브라우저는 결과가 아직 준비되지 않았을 때 polling 간격을 늘림

효과:

- 결과 대기 중인 요청이 많을 때 `/api/detections/{requestId}` 반복 호출 감소

### 4. batch status API 추가

파일:

- `DetectionController.java`
- `content.js`
- `scripts/benchmark_detections.py`

추가 API:

```http
GET /api/detections/status?ids=1,2,3
```

응답 구조:

```json
{
  "items": [
    {
      "requestId": 1,
      "status": "DONE",
      "message": "Analysis completed.",
      "result": {},
      "retryAfterMs": null
    }
  ],
  "queue": {}
}
```

개선 내용:

- 브라우저가 여러 requestId를 개별 polling하지 않고 한 번의 요청으로 묶어 조회
- 완료가 확인되면 다음 polling 간격을 다시 짧게 조정
- 처리 중일 때만 서버의 `retryAfterMs`를 반영

효과:

- polling API 호출 수를 크게 줄이면서 완료 인지가 과하게 늦어지지 않도록 조정

### 5. 큐 지표와 처리 시간 관측 강화

파일: `DetectionProcessingService.java`

추가 지표:

- `avgProcessingTimeMs`
- `avgAiCallTimeMs`
- `estimatedWaitMs`
- `totalAiCallCount`
- `totalRetryCount`
- `queuedCount`
- `activeProcessingCount`

확인 API:

```http
GET /api/detections/queue
```

효과:

- 큐 적체, worker 처리량, AI 호출 평균 시간을 운영 중 확인 가능
- 대량 요청에서 병목이 브라우저, Spring 큐, AI 서버 중 어디인지 분리해 볼 수 있음

## 벤치마크 결과

### A. 중복 이미지 40건

조건:

- Mock AI delay: 250ms/request
- 같은 이미지 40건 요청
- 동시 요청 수: 10
- 분석 모드: `face_crop_only`

결과 파일:

- `images/benchmark_runs/before_duplicate_40.json`
- `images/benchmark_runs/after_duplicate_40.json`
- `images/benchmark_runs/load_optimization_20260528_summary.md`

| 항목 | 개선 전 | 개선 후 | 개선율 |
|---|---:|---:|---:|
| 전체 완료 시간 | 6,560ms | 1,681ms | 74.4% 감소 |
| 요청 평균 완료 시간 | 1,433.85ms | 413.35ms | 71.2% 감소 |
| P50 완료 시간 | 1,066ms | 107ms | 90.0% 감소 |
| P95 완료 시간 | 2,407ms | 1,370ms | 43.1% 감소 |
| 생성된 requestId | 40개 | 1개 | 97.5% 감소 |
| polling 요청 수 | 52회 | 10회 | 80.8% 감소 |

해석:

- 같은 이미지가 반복 검사되는 경우에는 `fileHash + analysisMode` 재사용 효과가 가장 크다.
- 실제 AI 작업이 40개에서 1개로 줄어 AI 서버 부하가 크게 감소했다.

### B. 서로 다른 이미지 40건

조건:

- Mock AI delay: 250ms/request
- `ai/tuning_cases/frontal_good` 이미지 40건
- 동시 요청 수: 10
- 분석 모드: `face_crop_only`
- AI로 전달되는 이미지와 multipart 데이터는 동일

결과 파일:

- 개선 전: `images/benchmark_runs/before_batch_status_unique_40.json`
- 개선 후: `images/benchmark_runs/after_batch_status_unique_40_tuned.json`
- 큐 지표: `images/benchmark_runs/after_batch_status_queue_tuned.json`

| 항목 | 개선 전 | 개선 후 | 변화 |
|---|---:|---:|---:|
| 전체 완료 시간 | 8,915ms | 8,496ms | 4.7% 감소 |
| 요청 평균 완료 시간 | 2,024.22ms | 1,996.05ms | 1.4% 감소 |
| P50 완료 시간 | 1,563ms | 2,031ms | 29.9% 증가 |
| P95 완료 시간 | 4,070ms | 2,390ms | 41.3% 감소 |
| 평균 submit 시간 | 173.12ms | 113.10ms | 34.7% 감소 |
| polling API 호출 수 | 55회 | 8회 | 85.5% 감소 |
| 완료/실패 | 39/1 | 39/1 | 동일 |

해석:

- 서로 다른 이미지는 AI가 처리해야 하는 총 작업량이 줄지 않기 때문에 전체 시간 개선 폭은 작다.
- 대신 polling API 호출 수가 85.5% 줄어 Spring/DB 조회 부하가 크게 감소했다.
- P95가 41.3% 줄어 후반부 요청의 긴 대기 시간이 완화됐다.
- P50은 증가했는데, batch polling이 여러 요청을 묶어 확인하면서 일부 빠른 요청의 완료 인지가 1초 단위로 맞춰졌기 때문이다. 전체 완료 시간과 P95, API 부하는 개선됐다.

벤치마크 후 큐 지표:

```json
{
  "queueCapacity": 100,
  "queuedCount": 0,
  "remainingCapacity": 100,
  "workerCount": 2,
  "activeProcessingCount": 0,
  "totalEnqueuedCount": 40,
  "totalCompletedCount": 39,
  "totalFailedCount": 1,
  "totalAiCallCount": 41,
  "totalRetryCount": 1,
  "avgProcessingTimeMs": 295,
  "avgAiCallTimeMs": 247,
  "estimatedWaitMs": 0,
  "recommendedPollDelayMs": 1000
}
```

## 결론

이번 개선은 AI 입력 데이터를 바꾸지 않고, 대량 처리 시 부하를 줄이는 데 초점을 맞췄다.

적용 판단:

- 중복 이미지가 많은 상황: 적용 가치가 매우 큼
- 서로 다른 이미지가 많은 상황: 전체 처리 시간 자체는 AI 처리량에 묶이지만, polling/API/DB 부하는 크게 감소하므로 적용 가치가 있음
- batch polling은 초기 튜닝에서 대기 시간이 과하게 늘어났으나, 완료 시 polling 간격을 다시 줄이도록 조정해 최종 적용 가능하다고 판단

## 검증

실행한 검증:

```bash
python -m py_compile scripts/benchmark_detections.py
node --check content.js
./gradlew.bat clean compileJava --no-daemon --console=plain
```

벤치마크:

```bash
python scripts/benchmark_detections.py --image-dir ai/tuning_cases/frontal_good --count 40 --concurrency 10 --poll-backoff --batch-poll
```

## 남은 개선 후보

AI 입력과 기본 파이프라인을 유지한다는 조건에서 다음 개선을 추가로 검토할 수 있다.

1. 큐 worker 수를 환경별로 조정할 수 있는 운영 프로파일 분리
2. Redis/RabbitMQ 기반 외부 큐 전환
3. 오래된 업로드 원본과 결과 정리 정책 추가
4. batch status API의 최대 ids 개수 제한
5. 큐 지표를 관리자 화면 또는 health check에 연결
