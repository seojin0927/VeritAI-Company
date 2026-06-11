package com.example.backend_spring.Dto;

import lombok.Builder;
import lombok.Getter;
import java.util.List;
import java.util.Map;

@Getter
@Builder
public class ExtensionResponseDto {
    private String status;
    private String requestId;
    private ResultDetail result;
    private String message;

    @Getter
    @Builder
    public static class ResultDetail {
        private boolean isDeepfake;
        private double confidence;
        private int faceCount;
        private List<Map<String, Object>> faces;
        private String heatmapBase64;
    }
}