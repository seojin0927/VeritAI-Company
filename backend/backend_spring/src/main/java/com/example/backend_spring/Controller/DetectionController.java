package com.example.backend_spring.Controller;

import com.example.backend_spring.Dto.AiPredictionDto;
import com.example.backend_spring.Dto.DetectionResponseDto;
import com.example.backend_spring.Dto.FeedbackRequestDto;
import com.example.backend_spring.Entity.DetectionRequestEntity;
import com.example.backend_spring.Entity.DetectionResultEntity;
import com.example.backend_spring.Repository.DetectionRequestRepository;
import com.example.backend_spring.Repository.DetectionResultRepository;
import com.example.backend_spring.Service.DetectionProcessingService;
import com.example.backend_spring.Service.DetectionProcessingService.QueueFullException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

@RestController
@RequestMapping("/api")
public class DetectionController {

    private static final Logger log = LoggerFactory.getLogger(DetectionController.class);
    private static final String RETRY_AFTER_SECONDS = "5";

    private final DetectionRequestRepository detectionRequestRepository;
    private final DetectionResultRepository detectionResultRepository;
    private final DetectionProcessingService detectionProcessingService;
    private final ObjectMapper objectMapper;

    @Value("${app.upload-dir}")
    private String uploadDir;

    public DetectionController(DetectionRequestRepository detectionRequestRepository,
                               DetectionResultRepository detectionResultRepository,
                               DetectionProcessingService detectionProcessingService,
                               ObjectMapper objectMapper) {
        this.detectionRequestRepository = detectionRequestRepository;
        this.detectionResultRepository = detectionResultRepository;
        this.detectionProcessingService = detectionProcessingService;
        this.objectMapper = objectMapper;
    }

    @PostMapping("/detections")
    public ResponseEntity<?> createDetection(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "sourceUrl", required = false) String sourceUrl,
            @RequestParam(value = "mediaType", defaultValue = "image") String mediaType,
            @RequestParam(value = "clientType", defaultValue = "chrome-extension") String clientType,
            @RequestParam(value = "analysisMode", defaultValue = "full_image") String analysisMode
    ) {
        DetectionRequestEntity requestEntity = new DetectionRequestEntity();

        try {
            if (file.isEmpty()) {
                return ResponseEntity.badRequest().body("Uploaded file is empty.");
            }

            byte[] bytes = file.getBytes();
            String fileHash = sha256(bytes);

            Path uploadPath = Paths.get(uploadDir);
            Files.createDirectories(uploadPath);

            String originalFileName = sanitizeFileName(file.getOriginalFilename());
            String savedFileName = UUID.randomUUID() + "_" + originalFileName;
            Path savedPath = uploadPath.resolve(savedFileName);
            Files.write(savedPath, bytes, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);

            String normalizedAnalysisMode = detectionProcessingService.normalizeAnalysisMode(analysisMode);
            requestEntity.setSourceUrl(truncate(sourceUrl, 2000));
            requestEntity.setMediaType(mediaType);
            requestEntity.setClientType(clientType);
            requestEntity.setFileName(originalFileName);
            requestEntity.setFilePath(savedPath.toString());
            requestEntity.setFileHash(fileHash);
            requestEntity.setMimeType(file.getContentType());
            requestEntity.setFileSize(file.getSize());
            requestEntity.setAnalysisMode(normalizedAnalysisMode);
            requestEntity.setStatus(DetectionProcessingService.STATUS_QUEUED);
            detectionRequestRepository.save(requestEntity);

            detectionProcessingService.enqueue(requestEntity.getId(), savedPath, normalizedAnalysisMode);

            DetectionResponseDto responseDto = new DetectionResponseDto(
                    requestEntity.getId(),
                    requestEntity.getStatus(),
                    "Analysis request queued.",
                    null
            );

            return ResponseEntity.accepted().body(responseDto);

        } catch (QueueFullException e) {
            requestEntity.setStatus(DetectionProcessingService.STATUS_FAILED);
            if (requestEntity.getId() != null) {
                detectionRequestRepository.save(requestEntity);
            }

            DetectionResponseDto errorDto = new DetectionResponseDto(
                    requestEntity.getId(),
                    DetectionProcessingService.STATUS_FAILED,
                    "Detection queue is full. Please retry later.",
                    null
            );

            return ResponseEntity.status(HttpStatus.TOO_MANY_REQUESTS)
                    .header("Retry-After", RETRY_AFTER_SECONDS)
                    .body(errorDto);

        } catch (Exception e) {
            requestEntity.setStatus(DetectionProcessingService.STATUS_FAILED);
            if (requestEntity.getId() != null) {
                detectionRequestRepository.save(requestEntity);
            }

            DetectionResponseDto errorDto = new DetectionResponseDto(
                    requestEntity.getId(),
                    DetectionProcessingService.STATUS_FAILED,
                    "Analysis failed: " + e.getMessage(),
                    null
            );

            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorDto);
        }
    }

    @GetMapping("/detections/{requestId}")
    public ResponseEntity<?> getDetection(@PathVariable Long requestId) {
        Optional<DetectionRequestEntity> requestOpt = detectionRequestRepository.findById(requestId);
        if (requestOpt.isEmpty()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Request not found.");
        }

        DetectionRequestEntity requestEntity = requestOpt.get();
        Optional<DetectionResultEntity> resultOpt = detectionResultRepository.findByRequestId(requestId);

        AiPredictionDto resultDto = null;
        if (resultOpt.isPresent()) {
            DetectionResultEntity resultEntity = resultOpt.get();
            resultDto = deserializeResult(resultEntity);
        }

        DetectionResponseDto responseDto = new DetectionResponseDto(
                requestEntity.getId(),
                requestEntity.getStatus(),
                getStatusMessage(requestEntity.getStatus()),
                resultDto
        );

        return ResponseEntity.ok(responseDto);
    }

    @PostMapping("/feedback")
    public ResponseEntity<?> receiveFeedback(@RequestBody FeedbackRequestDto feedbackDto) {
        log.info("Feedback received - ID: {}, reason: {}", feedbackDto.getRequestId(), feedbackDto.getReason());

        Optional<DetectionRequestEntity> requestOpt = detectionRequestRepository.findById(feedbackDto.getRequestId());
        if (requestOpt.isEmpty()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body(java.util.Map.of("status", "FAILED", "message", "Request not found."));
        }

        DetectionRequestEntity entity = requestOpt.get();
        entity.setReported(true);
        entity.setReportedAt(feedbackDto.getReportedAt());
        entity.setReportReason(feedbackDto.getReason());
        detectionRequestRepository.save(entity);

        return ResponseEntity.ok(java.util.Map.of("status", "SUCCESS"));
    }

    @GetMapping("/feedback/list")
    public ResponseEntity<?> getFeedbackList() {
        List<DetectionRequestEntity> reportedList = detectionRequestRepository.findByIsReportedTrue();
        return ResponseEntity.ok(reportedList);
    }

    private AiPredictionDto deserializeResult(DetectionResultEntity resultEntity) {
        if (resultEntity.getRawResultJson() != null && !resultEntity.getRawResultJson().isBlank()) {
            try {
                return objectMapper.readValue(resultEntity.getRawResultJson(), AiPredictionDto.class);
            } catch (Exception ignored) {
            }
        }

        AiPredictionDto resultDto = new AiPredictionDto();
        resultDto.setDeepfake(resultEntity.isDeepfake());
        resultDto.setConfidence(resultEntity.getConfidence());
        resultDto.setFaceCount(resultEntity.getFaceCount());
        resultDto.setWatermarkDetected(resultEntity.isWatermarkDetected());
        resultDto.setModelVersion(resultEntity.getModelVersion());
        resultDto.setProcessingTimeMs(resultEntity.getProcessingTimeMs());
        resultDto.setMessage(resultEntity.getMessage());
        return resultDto;
    }

    private String truncate(String value, int maxLength) {
        if (value == null) return null;
        return value.length() <= maxLength ? value : value.substring(0, maxLength);
    }

    private String sanitizeFileName(String fileName) {
        String fallback = "capture.jpg";
        if (fileName == null || fileName.isBlank()) {
            return fallback;
        }

        String normalized = Paths.get(fileName).getFileName().toString();
        String sanitized = normalized.replaceAll("[^A-Za-z0-9._-]", "_");
        return sanitized.isBlank() ? fallback : sanitized;
    }

    private String getStatusMessage(String status) {
        if (DetectionProcessingService.STATUS_QUEUED.equals(status)) {
            return "Analysis request is queued.";
        }
        if (DetectionProcessingService.STATUS_PROCESSING.equals(status)) {
            return "Analysis is processing.";
        }
        if (DetectionProcessingService.STATUS_DONE.equals(status)) {
            return "Analysis completed.";
        }
        if (DetectionProcessingService.STATUS_FAILED.equals(status)) {
            return "Analysis failed.";
        }
        return "Analysis status loaded.";
    }

    private String sha256(byte[] bytes) throws Exception {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        byte[] digest = md.digest(bytes);
        return HexFormat.of().formatHex(digest);
    }
}
