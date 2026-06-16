# VeritAI 변경사항 정리 - 2026-05-19

## 개요

이번 변경은 대량 이미지 검사 상황에서 Spring 백엔드가 AI 응답을 기다리느라 요청 처리가 막히는 문제를 줄이기 위해, 기존 검사 파이프라인에 비동기 큐를 적용한 작업이다.

중요한 원칙은 다음과 같다.

- AI로 보내는 이미지 파일은 변형하지 않는다.
- AI로 보내는 데이터 형식은 기존 multipart 구조를 유지한다.
- 얼굴 검출, 크롭, 딥페이크 판별 로직은 변경하지 않는다.
- API 응답 흐름은 비동기 처리에 맞게 바꾸되, 최종 검사 결과 payload는 유지한다.

## 변경 전 구조

기존 구조는 확장 프로그램 또는 클라이언트가 `/api/detections`로 이미지를 보내면 Spring 컨트롤러가 다음 작업을 한 요청 안에서 모두 처리했다.

1. 이미지 업로드 파일 저장
2. 요청 메타데이터 저장
3. AI 서버에 multipart 요청 전송
4. AI 응답 수신 대기
5. 결과 DB 저장
6. `DONE + result` 응답 반환

이 방식은 단건 처리에는 단순하지만, 이미지가 많아지면 Spring 요청 스레드가 AI 처리 시간만큼 계속 점유된다. 여러 사용자가 동시에 검사를 요청하는 상황에서는 백엔드 병목이 커질 수 있다.

## 변경 후 구조

변경 후에는 `/api/detections` 요청이 들어오면 Spring은 업로드 파일과 요청 정보를 저장한 뒤 작업을 내부 큐에 넣고 바로 `202 Accepted`를 반환한다.

이후 별도 워커가 큐에서 작업을 꺼내 AI 서버로 보내고, 완료된 결과를 DB에 저장한다. 확장 프로그램은 `requestId`로 `/api/detections/{requestId}`를 polling 하며 최종 결과를 기다린다.

처리 흐름은 다음과 같다.

1. 클라이언트가 이미지 검사 요청
2. Spring이 원본 업로드 파일 저장
3. 요청 상태를 `QUEUED`로 저장
4. 작업을 비동기 큐에 등록
5. 클라이언트에 `202 Accepted + requestId` 반환
6. 워커가 작업을 가져오면 상태를 `PROCESSING`으로 변경
7. 기존과 동일한 파일과 `analysisMode`로 AI 서버 호출
8. AI 결과 저장 후 상태를 `DONE`으로 변경
9. 클라이언트 polling에서 최종 `DONE + result` 확인

## 주요 변경 파일

### `backend/backend_spring/src/main/java/com/example/backend_spring/Service/DetectionProcessingService.java`

비동기 큐 처리 전담 서비스가 추가되었다.

주요 역할:

- `ArrayBlockingQueue` 기반 검사 작업 큐 관리
- 워커 스레드 실행
- `QUEUED`, `PROCESSING`, `DONE`, `FAILED` 상태 관리
- AI 서버 호출
- AI 결과 DB 저장
- 서버 재시작 시 `QUEUED/PROCESSING` 상태 요청 복구
- AI 서버 일시 오류에 대한 제한 재시도
- 큐 초과 시 `QueueFullException` 발생

AI 호출 부분은 기존 파이프라인을 유지한다.

```java
body.add("file", new FileSystemResource(filePath.toFile()));
body.add("analysisMode", normalizeAnalysisMode(analysisMode));
```

즉, AI로 보내는 이미지는 저장된 원본 파일이며, 추가 변형이나 압축, 리사이즈를 하지 않는다.

### `backend/backend_spring/src/main/java/com/example/backend_spring/Controller/DetectionController.java`

컨트롤러는 더 이상 AI 응답을 직접 기다리지 않는다.

변경 사항:

- 요청 저장 후 상태를 `QUEUED`로 설정
- `DetectionProcessingService.enqueue(...)`로 작업 등록
- 즉시 `202 Accepted` 반환
- 큐가 꽉 찬 경우 `429 Too Many Requests` 반환
- `Retry-After: 5` 헤더 추가
- `GET /api/detections/{requestId}`는 기존처럼 현재 상태와 결과를 조회

초기 응답 예시:

```json
{
  "requestId": 1,
  "status": "QUEUED",
  "message": "Analysis request queued.",
  "result": null
}
```

완료 응답은 기존과 같이 `DONE + result` 형태를 유지한다.

### `backend/backend_spring/src/main/java/com/example/backend_spring/Entity/DetectionRequestEntity.java`

요청별 `analysisMode`를 저장하도록 필드가 추가되었다.

이유:

- 재시작 복구 시에도 원래 요청이 `full_image`였는지 `face_crop_only`였는지 유지해야 한다.
- 복구 과정에서 AI로 보내는 데이터 조건이 바뀌지 않게 하기 위함이다.

### `backend/backend_spring/src/main/java/com/example/backend_spring/Repository/DetectionRequestRepository.java`

재시작 복구를 위해 상태 기반 조회 메서드가 추가되었다.

```java
List<DetectionRequestEntity> findByStatusIn(Collection<String> statuses);
```

### `backend/backend_spring/src/main/resources/application.properties`

큐와 재시도 관련 설정이 추가되었다.

```properties
app.detection.queue-capacity=100
app.detection.worker-count=2
app.detection.ai-retry-count=1
app.detection.ai-retry-delay-ms=500
```

설정 의미:

- `queue-capacity`: 동시에 대기 가능한 검사 작업 수
- `worker-count`: 큐를 소비하는 백엔드 워커 수
- `ai-retry-count`: AI 호출 실패 시 추가 재시도 횟수
- `ai-retry-delay-ms`: 재시도 전 대기 시간

### `content.js`

확장 프로그램은 이제 최초 응답에서 `DONE`이 오지 않아도 실패로 처리하지 않는다.

변경 사항:

- `PROCESSING` 또는 `QUEUED` 상태와 `requestId`가 오면 polling 시작
- `/api/detections/{requestId}`를 주기적으로 조회
- `DONE + result`가 오면 기존 배지 처리 로직으로 전달
- `FAILED` 또는 timeout 시 오류 처리

## 상태값 정의

| 상태 | 의미 |
|---|---|
| `QUEUED` | 요청이 저장되고 큐에 대기 중 |
| `PROCESSING` | 워커가 작업을 꺼내 AI 서버 처리 중 |
| `DONE` | AI 결과 저장 완료 |
| `FAILED` | 처리 실패 |

## 검증 결과

### 컴파일 및 문법 검사

```text
gradlew clean compileJava
결과: BUILD SUCCESSFUL
```

```text
node --check content.js
결과: 성공
```

### 80장 대량 샘플 처리 비교

업로드 원본 이미지 중 80장을 사용해 비교했다.

| 항목 | 결과 |
|---|---:|
| 기존 동기식 기준선, AI 순차 직접 호출 | 158.76초 |
| 기존 동기식 평균 | 1.98초/장 |
| 비동기 큐 접수 시간 | 1.80초 |
| 비동기 큐 접수 평균 | 0.022초/장 |
| 비동기 큐 전체 DONE 완료 | 182.35초 |
| 접수 건수 | 80 |
| 완료 건수 | 80 |
| 실패 | 0 |
| timeout | 0 |

해석:

- 큐 적용 후 사용자의 요청 접수는 매우 빨라졌다.
- 전체 AI 완료 시간은 AI 서버 처리 성능에 좌우된다.
- 현재 병목은 Spring 요청 처리보다 Python AI 처리 시간에 더 가깝다.
- 따라서 비동기 큐의 가장 큰 효과는 최종 AI 계산 속도 향상보다는, 백엔드 요청 스레드 점유 감소와 대량 요청 흡수 능력 개선이다.

### 실제 E2E 확인

AI 서버와 Spring 서버를 실제로 실행한 뒤 6장을 검사했다.

결과:

- 초기 응답: 6건 모두 `202 + QUEUED`
- 최종 상태: 6건 모두 `DONE`
- 실패: 0
- timeout: 0
- 결과 payload에 `analysisMode=face_crop_only` 유지
- 결과 payload에 `isDeepfake` 필드 유지

## 판단

이번 변경은 적용하는 것이 맞다.

이유:

- AI 입력 이미지와 요청 데이터는 유지됐다.
- 최종 검사 결과 payload 구조도 유지됐다.
- 대량 요청 시 백엔드가 AI 응답을 기다리며 막히는 문제가 줄었다.
- 큐 초과, 재시도, 재시작 복구 같은 운영 안정성이 개선됐다.
- 확장 프로그램의 검사 버튼 및 배지 흐름은 polling 방식으로 이어진다.

단, 이 구조는 아직 Redis나 RabbitMQ 같은 외부 메시지 큐는 아니다. 현재 프로젝트 파이프라인을 크게 흔들지 않기 위해 DB 복구가 가능한 메모리 큐 수준으로 적용했다. 실제 서비스 규모가 더 커지면 외부 큐, AI 서버 수평 확장, 해시 기반 중복 검사 캐시를 다음 단계로 검토하는 것이 좋다.
