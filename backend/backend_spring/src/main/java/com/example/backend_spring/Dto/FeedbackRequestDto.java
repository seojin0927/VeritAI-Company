package com.example.backend_spring.Dto;

public class FeedbackRequestDto {
    private Long requestId;
    private String reportedAt;
    private String reason;

    public Long getRequestId() {
        return requestId;
    }

    public void setRequestId(Long requestId) {
        this.requestId = requestId;
    }

    public String getReportedAt() {
        return reportedAt;
    }

    public void setReportedAt(String reportedAt) {
        this.reportedAt = reportedAt;
    }

    public String getReason() {
        return reason;
    }

    public void setReason(String reason) {
        this.reason = reason;
    }
}