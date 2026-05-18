package com.example.backend_spring.Dto;

import java.util.List;
import java.util.Map;

public class AiPredictionDto {

    private boolean isDeepfake;
    private double confidence;
    private int faceCount;
    private boolean watermarkDetected;
    private String modelVersion;
    private String analysisMode;
    private Map<String, Object> analysisInput;
    private Map<String, Object> timings;
    private long processingTimeMs;
    private String message;
    private List<Map<String, Object>> faces;
    private Map<String, String> debugImages;

    public boolean isDeepfake() {
        return isDeepfake;
    }

    public void setDeepfake(boolean deepfake) {
        isDeepfake = deepfake;
    }

    public double getConfidence() {
        return confidence;
    }

    public void setConfidence(double confidence) {
        this.confidence = confidence;
    }

    public int getFaceCount() {
        return faceCount;
    }

    public void setFaceCount(int faceCount) {
        this.faceCount = faceCount;
    }

    public boolean isWatermarkDetected() {
        return watermarkDetected;
    }

    public void setWatermarkDetected(boolean watermarkDetected) {
        this.watermarkDetected = watermarkDetected;
    }

    public String getModelVersion() {
        return modelVersion;
    }

    public void setModelVersion(String modelVersion) {
        this.modelVersion = modelVersion;
    }

    public String getAnalysisMode() {
        return analysisMode;
    }

    public void setAnalysisMode(String analysisMode) {
        this.analysisMode = analysisMode;
    }

    public Map<String, Object> getAnalysisInput() {
        return analysisInput;
    }

    public void setAnalysisInput(Map<String, Object> analysisInput) {
        this.analysisInput = analysisInput;
    }

    public Map<String, Object> getTimings() {
        return timings;
    }

    public void setTimings(Map<String, Object> timings) {
        this.timings = timings;
    }

    public long getProcessingTimeMs() {
        return processingTimeMs;
    }

    public void setProcessingTimeMs(long processingTimeMs) {
        this.processingTimeMs = processingTimeMs;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public List<Map<String, Object>> getFaces() {
        return faces;
    }

    public void setFaces(List<Map<String, Object>> faces) {
        this.faces = faces;
    }

    public Map<String, String> getDebugImages() {
        return debugImages;
    }

    public void setDebugImages(Map<String, String> debugImages) {
        this.debugImages = debugImages;
    }
}
