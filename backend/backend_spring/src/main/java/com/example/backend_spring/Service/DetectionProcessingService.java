package com.example.backend_spring.Service;

import com.example.backend_spring.Dto.AiPredictionDto;
import com.example.backend_spring.Entity.DetectionRequestEntity;
import com.example.backend_spring.Entity.DetectionResultEntity;
import com.example.backend_spring.Repository.DetectionRequestRepository;
import com.example.backend_spring.Repository.DetectionResultRepository;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

@Service
public class DetectionProcessingService {

    public static final String STATUS_QUEUED = "QUEUED";
    public static final String STATUS_PROCESSING = "PROCESSING";
    public static final String STATUS_DONE = "DONE";
    public static final String STATUS_FAILED = "FAILED";

    private final RestTemplate restTemplate;
    private final DetectionRequestRepository detectionRequestRepository;
    private final DetectionResultRepository detectionResultRepository;
    private final ObjectMapper objectMapper;
    private final BlockingQueue<DetectionJob> queue;
    private final int queueCapacity;
    private final List<Thread> workers = new ArrayList<>();
    private final AtomicInteger activeProcessingCount = new AtomicInteger(0);
    private final AtomicLong totalEnqueuedCount = new AtomicLong(0);
    private final AtomicLong totalCompletedCount = new AtomicLong(0);
    private final AtomicLong totalFailedCount = new AtomicLong(0);
    private final AtomicLong totalAiCallCount = new AtomicLong(0);
    private final AtomicLong totalRetryCount = new AtomicLong(0);
    private final AtomicLong totalProcessingTimeMs = new AtomicLong(0);
    private final AtomicLong totalAiCallTimeMs = new AtomicLong(0);

    @Value("${app.ai-server-url}")
    private String aiServerUrl;

    @Value("${app.detection.worker-count:2}")
    private int workerCount;

    @Value("${app.detection.ai-retry-count:1}")
    private int aiRetryCount;

    @Value("${app.detection.ai-retry-delay-ms:500}")
    private long aiRetryDelayMs;

    private volatile boolean running = true;

    public DetectionProcessingService(RestTemplate restTemplate,
                                      DetectionRequestRepository detectionRequestRepository,
                                      DetectionResultRepository detectionResultRepository,
                                      ObjectMapper objectMapper,
                                      @Value("${app.detection.queue-capacity:100}") int queueCapacity) {
        this.restTemplate = restTemplate;
        this.detectionRequestRepository = detectionRequestRepository;
        this.detectionResultRepository = detectionResultRepository;
        this.objectMapper = objectMapper;
        this.queueCapacity = queueCapacity;
        this.queue = new ArrayBlockingQueue<>(queueCapacity);
    }

    @PostConstruct
    public void startWorkers() {
        int count = Math.max(1, workerCount);
        for (int i = 0; i < count; i += 1) {
            Thread worker = new Thread(this::runWorker, "veritai-detection-worker-" + (i + 1));
            worker.setDaemon(true);
            worker.start();
            workers.add(worker);
        }
        recoverPendingRequests();
    }

    @PreDestroy
    public void stopWorkers() {
        running = false;
        for (Thread worker : workers) {
            worker.interrupt();
        }
    }

    public void enqueue(Long requestId, Path filePath, String analysisMode) {
        DetectionJob job = new DetectionJob(requestId, filePath.toString(), normalizeAnalysisMode(analysisMode));
        if (!queue.offer(job)) {
            throw new QueueFullException("Detection queue is full.");
        }
        totalEnqueuedCount.incrementAndGet();
    }

    private void runWorker() {
        while (running) {
            try {
                DetectionJob job = queue.poll(1, TimeUnit.SECONDS);
                if (job != null) {
                    processJob(job);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private void processJob(DetectionJob job) {
        long processingStarted = System.currentTimeMillis();
        activeProcessingCount.incrementAndGet();
        try {
            DetectionRequestEntity requestEntity = detectionRequestRepository.findById(job.requestId())
                    .orElseThrow(() -> new IllegalStateException("Detection request not found: " + job.requestId()));

            if (detectionResultRepository.findByRequestId(job.requestId()).isPresent()) {
                requestEntity.setStatus(STATUS_DONE);
                detectionRequestRepository.save(requestEntity);
                totalCompletedCount.incrementAndGet();
                return;
            }

            requestEntity.setStatus(STATUS_PROCESSING);
            detectionRequestRepository.save(requestEntity);

            AiPredictionDto aiResult = callAiServerWithRetry(Paths.get(job.filePath()), job.analysisMode());

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

            requestEntity.setStatus(STATUS_DONE);
            detectionRequestRepository.save(requestEntity);
            totalCompletedCount.incrementAndGet();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            detectionRequestRepository.findById(job.requestId()).ifPresent(requestEntity -> {
                requestEntity.setStatus(STATUS_QUEUED);
                detectionRequestRepository.save(requestEntity);
            });
        } catch (Exception e) {
            detectionRequestRepository.findById(job.requestId()).ifPresent(requestEntity -> {
                requestEntity.setStatus(STATUS_FAILED);
                detectionRequestRepository.save(requestEntity);
            });
            totalFailedCount.incrementAndGet();
        } finally {
            activeProcessingCount.decrementAndGet();
            totalProcessingTimeMs.addAndGet(Math.max(0, System.currentTimeMillis() - processingStarted));
        }
    }

    private void recoverPendingRequests() {
        Set<String> pendingStatuses = Set.of(STATUS_QUEUED, STATUS_PROCESSING);
        List<DetectionRequestEntity> pendingRequests = detectionRequestRepository.findByStatusIn(pendingStatuses);
        for (DetectionRequestEntity requestEntity : pendingRequests) {
            if (detectionResultRepository.findByRequestId(requestEntity.getId()).isPresent()) {
                requestEntity.setStatus(STATUS_DONE);
                detectionRequestRepository.save(requestEntity);
                continue;
            }

            Path filePath = Paths.get(requestEntity.getFilePath());
            if (!Files.exists(filePath)) {
                requestEntity.setStatus(STATUS_FAILED);
                detectionRequestRepository.save(requestEntity);
                continue;
            }

            requestEntity.setStatus(STATUS_QUEUED);
            detectionRequestRepository.save(requestEntity);
            if (!queue.offer(new DetectionJob(
                    requestEntity.getId(),
                    requestEntity.getFilePath(),
                    normalizeAnalysisMode(requestEntity.getAnalysisMode())
            ))) {
                break;
            }
            totalEnqueuedCount.incrementAndGet();
        }
    }

    private AiPredictionDto callAiServerWithRetry(Path filePath, String analysisMode) throws InterruptedException {
        int attempts = Math.max(1, aiRetryCount + 1);
        RuntimeException lastError = null;
        for (int attempt = 1; attempt <= attempts; attempt += 1) {
            try {
                totalAiCallCount.incrementAndGet();
                long aiStarted = System.currentTimeMillis();
                AiPredictionDto result = callAiServer(filePath, analysisMode);
                totalAiCallTimeMs.addAndGet(Math.max(0, System.currentTimeMillis() - aiStarted));
                return result;
            } catch (RuntimeException e) {
                lastError = e;
                if (attempt < attempts) {
                    totalRetryCount.incrementAndGet();
                    Thread.sleep(Math.max(0, aiRetryDelayMs));
                }
            }
        }
        throw lastError == null ? new RuntimeException("AI server request failed.") : lastError;
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
                AiPredictionDto.class
        );

        if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null) {
            throw new RuntimeException("AI server response is invalid.");
        }

        return response.getBody();
    }

    public String normalizeAnalysisMode(String analysisMode) {
        if ("face_crop_only".equals(analysisMode)) {
            return "face_crop_only";
        }
        return "full_image";
    }

    public int recommendedPollDelayMs() {
        int queued = queue.size();
        if (queued >= Math.max(1, queueCapacity * 0.8)) {
            return 5000;
        }
        if (queued >= Math.max(1, queueCapacity * 0.5)) {
            return 3000;
        }
        if (queued > workerCount) {
            return 2000;
        }
        return 1000;
    }

    public Map<String, Object> getQueueMetrics() {
        long completed = totalCompletedCount.get();
        long aiCalls = totalAiCallCount.get();
        long avgProcessingMs = completed == 0 ? 0 : totalProcessingTimeMs.get() / completed;
        long avgAiCallMs = aiCalls == 0 ? 0 : totalAiCallTimeMs.get() / aiCalls;
        int workers = Math.max(1, workerCount);
        long estimatedWaitMs = avgProcessingMs == 0 ? 0 : ((long) Math.ceil(queue.size() / (double) workers)) * avgProcessingMs;
        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("queueCapacity", queueCapacity);
        metrics.put("queuedCount", queue.size());
        metrics.put("remainingCapacity", queue.remainingCapacity());
        metrics.put("workerCount", workers);
        metrics.put("activeProcessingCount", activeProcessingCount.get());
        metrics.put("running", running);
        metrics.put("totalEnqueuedCount", totalEnqueuedCount.get());
        metrics.put("totalCompletedCount", totalCompletedCount.get());
        metrics.put("totalFailedCount", totalFailedCount.get());
        metrics.put("totalAiCallCount", totalAiCallCount.get());
        metrics.put("totalRetryCount", totalRetryCount.get());
        metrics.put("avgProcessingTimeMs", avgProcessingMs);
        metrics.put("avgAiCallTimeMs", avgAiCallMs);
        metrics.put("estimatedWaitMs", estimatedWaitMs);
        metrics.put("recommendedPollDelayMs", recommendedPollDelayMs());
        return metrics;
    }

    private record DetectionJob(Long requestId, String filePath, String analysisMode) {
    }

    public static class QueueFullException extends RuntimeException {
        public QueueFullException(String message) {
            super(message);
        }
    }
}
