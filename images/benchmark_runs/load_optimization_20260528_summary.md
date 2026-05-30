# Load Optimization Benchmark - 2026-05-28

## Scope

- 브라우저 동시 요청 제한
- 백엔드 `fileHash + analysisMode` 기반 중복 검사 재사용
- polling backoff
- 큐 지표/적체량 API
- AI 처리 시간 breakdown

AI로 전달하는 이미지 파일과 multipart 필드 구조는 변경하지 않았다.  
고유 이미지 요청은 기존과 같이 `file`, `analysisMode`를 Python AI 서버로 전달한다.  
중복 이미지 요청은 이미 처리 중이거나 완료된 동일 `fileHash + analysisMode` 요청을 재사용한다.

## Benchmark Setup

- Spring backend: `bootRun`
- AI server: `scripts/mock_ai_server.py`
- Mock AI delay: 250ms/request
- Test image: `ai/tuning_cases/frontal_good/photo-1542909168-82c3e7fdca5c.jpg`
- Scenario: 동일 이미지 40건 동시 요청
- Client concurrency: 10
- Analysis mode: `face_crop_only`

## Result

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Total elapsed | 6,560ms | 1,681ms | 74.4% faster |
| Avg request completion | 1,433.85ms | 413.35ms | 71.2% faster |
| P50 completion | 1,066ms | 107ms | 90.0% faster |
| P95 completion | 2,407ms | 1,370ms | 43.1% faster |
| Unique request IDs | 40 | 1 | 97.5% fewer queued jobs |
| Poll requests | 52 | 10 | 80.8% fewer polling calls |
| Done / Failed | 40 / 0 | 40 / 0 | same |

## After Queue Metrics

```json
{
  "queueCapacity": 100,
  "queuedCount": 0,
  "remainingCapacity": 100,
  "workerCount": 2,
  "activeProcessingCount": 0,
  "totalEnqueuedCount": 1,
  "totalCompletedCount": 1,
  "totalFailedCount": 0,
  "totalAiCallCount": 1,
  "totalRetryCount": 0,
  "recommendedPollDelayMs": 1000
}
```

## Notes

- 중복 이미지가 많은 대량 페이지에서는 AI 호출 자체가 줄어들기 때문에 처리 시간이 크게 줄어든다.
- 서로 다른 이미지가 많은 페이지에서는 총 AI 연산량은 유지된다. 이 경우 브라우저 동시 요청 제한과 polling backoff는 총 처리시간을 무리하게 줄이기보다, 서버 peak 부하와 DB polling 부하를 낮추는 역할을 한다.
- 운영 환경에서는 이 지표를 바탕으로 `worker-count`, queue capacity, external queue 전환 여부를 조정하면 된다.
