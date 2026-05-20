1. 프론트엔드 (Chrome Extension) 개선 사항

주관식 신고 사유 입력 UI 추가

/api/feedback 엔드포인트로 전송하는 JSON 데이터에 reason(신고 사유) 필드를 추가하여 맵핑.

UI/UX 코드 개선

2. 백엔드 (Spring Boot) 개선 사항

피드백 수신 로직 확장 (POST /api/feedback)

FeedbackRequestDto에 reason 필드를 추가하여 프론트엔드의 확장된 데이터를 수신.

AI 모델 개선 및 데이터 분석을 위해, 접수된 피드백만 모아서 볼 수 있는 조회용 API 추가.




