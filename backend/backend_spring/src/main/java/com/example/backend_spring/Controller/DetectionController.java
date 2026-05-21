package com.example.backend_spring.Controller;

import com.example.backend_spring.Dto.AiPredictionDto;
import com.example.backend_spring.Dto.DetectionResponseDto;
import com.example.backend_spring.Dto.FeedbackRequestDto;
import com.example.backend_spring.Entity.DetectionRequestEntity;
import com.example.backend_spring.Entity.DetectionResultEntity;
import com.example.backend_spring.Repository.DetectionRequestRepository;
import com.example.backend_spring.Repository.DetectionResultRepository;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.Optional;
import java.util.UUID;
import java.util.List;

@RestController
@RequestMapping("/api")

public class DetectionController {

    private static final org.slf4j.Logger log = org.slf4j.LoggerFactory.getLogger(DetectionController.class);
    private final RestTemplate restTemplate;
    private final DetectionRequestRepository detectionRequestRepository;
    private final DetectionResultRepository detectionResultRepository;
    private final ObjectMapper objectMapper;

    @Value("${app.upload-dir}")
    private String uploadDir;

    @Value("${app.ai-server-url}")
    private String aiServerUrl;

    public DetectionController(RestTemplate restTemplate,
            DetectionRequestRepository detectionRequestRepository,
            DetectionResultRepository detectionResultRepository,
            ObjectMapper objectMapper) {
        this.restTemplate = restTemplate;
        this.detectionRequestRepository = detectionRequestRepository;
        this.detectionResultRepository = detectionResultRepository;
        this.objectMapper = objectMapper;
    }

    @PostMapping("/detections")
    public ResponseEntity<?> createDetection(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "sourceUrl", required = false) String sourceUrl,
            @RequestParam(value = "mediaType", defaultValue = "image") String mediaType,
            @RequestParam(value = "clientType", defaultValue = "chrome-extension") String clientType,
            @RequestParam(value = "analysisMode", defaultValue = "full_image") String analysisMode) {
        DetectionRequestEntity requestEntity = new DetectionRequestEntity();

        try {
            if (file.isEmpty()) {
                return ResponseEntity.badRequest().body("업로드 파일이 비어 있습니다.");
            }

            byte[] bytes = file.getBytes();
            String fileHash = sha256(bytes);

            Path uploadPath = Paths.get(uploadDir);
            Files.createDirectories(uploadPath);

            String originalFileName = sanitizeFileName(file.getOriginalFilename());
            String savedFileName = UUID.randomUUID() + "_" + originalFileName;
            Path savedPath = uploadPath.resolve(savedFileName);
            Files.write(savedPath, bytes, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);

            requestEntity.setSourceUrl(truncate(sourceUrl, 2000));
            requestEntity.setMediaType(mediaType);
            requestEntity.setClientType(clientType);
            requestEntity.setFileName(originalFileName);
            requestEntity.setFilePath(savedPath.toString());
            requestEntity.setFileHash(fileHash);
            requestEntity.setMimeType(file.getContentType());
            requestEntity.setFileSize(file.getSize());
            requestEntity.setStatus("PROCESSING");
            detectionRequestRepository.save(requestEntity);

            AiPredictionDto aiResult = callAiServer(savedPath, analysisMode);

            DetectionResultEntity resultEntity = new DetectionResultEntity();
            resultEntity.setRequestId(requestEntity.getId());
            resultEntity.setDeepfake(aiResult.isDeepfake());
            resultEntity.setConfidence(aiResult.getConfidence());
            resultEntity.setFaceCount(aiResult.getFaceCount());
            resultEntity.setWatermarkDetected(aiResult.isWatermarkDetected());
            resultEntity.setModelVersion(aiResult.getModelVersion());
            resultEntity.setProcessingTimeMs(aiResult.getProcessingTimeMs());
            resultEntity.setMessage(aiResult.getMessage());
            resultEntity.setRawResultJson(objectMapper.writeValueAsString(aiResult));
            detectionResultRepository.save(resultEntity);

            requestEntity.setStatus("DONE");
            detectionRequestRepository.save(requestEntity);

            DetectionResponseDto responseDto = new DetectionResponseDto(
                    requestEntity.getId(),
                    "DONE",
                    "분석 완료",
                    aiResult);

            return ResponseEntity.ok(responseDto);

        } catch (Exception e) {
            requestEntity.setStatus("FAILED");
            if (requestEntity.getId() != null) {
                detectionRequestRepository.save(requestEntity);
            }

            DetectionResponseDto errorDto = new DetectionResponseDto(
                    requestEntity.getId(),
                    "FAILED",
                    "분석 실패: " + e.getMessage(),
                    null);

            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorDto);
        }
    }

    @GetMapping("/detections/{requestId}")
    public ResponseEntity<?> getDetection(@PathVariable Long requestId) {
        Optional<DetectionRequestEntity> requestOpt = detectionRequestRepository.findById(requestId);
        if (requestOpt.isEmpty()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body("요청을 찾을 수 없습니다.");
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
                "조회 성공",
                resultDto);

        return ResponseEntity.ok(responseDto);
    }

    private AiPredictionDto callAiServer(Path filePath, String analysisMode) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new FileSystemResource(filePath.toFile()));
        body.add("analysisMode", normalizeAnalysisMode(analysisMode));

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        ResponseEntity<AiPredictionDto> response = restTemplate.exchange(
                aiServerUrl,
                HttpMethod.POST,
                requestEntity,
                AiPredictionDto.class);

        if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null) {
            throw new RuntimeException("AI 서버 응답이 비정상입니다.");
        }

        return response.getBody();
    }

    private String normalizeAnalysisMode(String analysisMode) {
        if ("face_crop_only".equals(analysisMode)) {
            return "face_crop_only";
        }
        return "full_image";
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
        if (value == null)
            return null;
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

    @PostMapping("/feedback")
    public ResponseEntity<?> receiveFeedback(@RequestBody FeedbackRequestDto feedbackDto) {
        log.info("🚨 오답 신고 접수 - ID: {}, 사유: {}", feedbackDto.getRequestId(), feedbackDto.getReason());

        Optional<DetectionRequestEntity> requestOpt = detectionRequestRepository.findById(feedbackDto.getRequestId());

        if (requestOpt.isPresent()) {
            DetectionRequestEntity entity = requestOpt.get();

            entity.setReported(true);
            entity.setReportedAt(feedbackDto.getReportedAt());
            entity.setReportReason(feedbackDto.getReason());

            detectionRequestRepository.save(entity);

            return ResponseEntity.ok(java.util.Map.of("status", "SUCCESS"));
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body(java.util.Map.of("status", "FAILED", "message", "해당 ID를 찾을 수 없습니다."));
        }
    }

    private String sha256(byte[] bytes) throws Exception {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        byte[] digest = md.digest(bytes);
        return HexFormat.of().formatHex(digest);
    }

    @GetMapping("/feedback/list")
    public ResponseEntity<?> getFeedbackList() {
        List<DetectionRequestEntity> reportedList = detectionRequestRepository.findByIsReportedTrue();

        return ResponseEntity.ok(reportedList);
    }
}
