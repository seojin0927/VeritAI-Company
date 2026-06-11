package com.example.backend_spring.Dto;

import lombok.Getter;
import lombok.Setter;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@Getter
@Setter
@JsonIgnoreProperties(ignoreUnknown = true)
public class PythonApiResponse {
    private boolean isDeepfake;
    private double confidence;
    private int faceCount;
    private List<Map<String, Object>> faces;
    private String heatmapBase64;
    private String message;
}