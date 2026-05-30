package com.example.backend_spring.Dto;

public class DetectionResponseDto {

    private Long requestId;
    private String status;
    private String message;
    private AiPredictionDto result;
    private Integer retryAfterMs;

    public DetectionResponseDto() {
    }

    public DetectionResponseDto(Long requestId, String status, String message, AiPredictionDto result) {
        this.requestId = requestId;
        this.status = status;
        this.message = message;
        this.result = result;
    }

    public DetectionResponseDto(Long requestId, String status, String message, AiPredictionDto result, Integer retryAfterMs) {
        this.requestId = requestId;
        this.status = status;
        this.message = message;
        this.result = result;
        this.retryAfterMs = retryAfterMs;
    }

    public Long getRequestId() {
        return requestId;
    }

    public void setRequestId(Long requestId) {
        this.requestId = requestId;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public AiPredictionDto getResult() {
        return result;
    }

    public void setResult(AiPredictionDto result) {
        this.result = result;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public Integer getRetryAfterMs() {
        return retryAfterMs;
    }

    public void setRetryAfterMs(Integer retryAfterMs) {
        this.retryAfterMs = retryAfterMs;
    }
}
