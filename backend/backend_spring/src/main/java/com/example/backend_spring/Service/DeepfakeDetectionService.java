package com.example.backend_spring.Service;

import com.example.backend_spring.Dto.ExtensionResponseDto;
import com.example.backend_spring.Dto.PythonApiResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.util.MultiValueMap;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.client.WebClient;

import java.io.IOException;
import java.util.UUID;

@Slf4j
@Service
@RequiredArgsConstructor
public class DeepfakeDetectionService {

    private final WebClient pythonWebClient;

    public ExtensionResponseDto analyzeMedia(MultipartFile mediaFile) {
        String requestId = UUID.randomUUID().toString();
        log.info("[{}] 스캔 요청 시작. 파일명: {}", requestId, mediaFile.getOriginalFilename());

        try {
            MultiValueMap<String, HttpEntity<?>> multipartBody = createMultipartBody(mediaFile);

            PythonApiResponse pythonResponse = pythonWebClient.post()
                    .uri("/predict")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .bodyValue(multipartBody)
                    .retrieve()
                    .bodyToMono(PythonApiResponse.class)
                    .block();

            if (pythonResponse == null) {
                return handleFallback(requestId, "FAIL", "AI 모델 서버로부터 응답을 받지 못했습니다.");
            }

            ExtensionResponseDto.ResultDetail resultDetail = ExtensionResponseDto.ResultDetail.builder()
                    .isDeepfake(pythonResponse.isDeepfake())
                    .confidence(pythonResponse.getConfidence())
                    .faceCount(pythonResponse.getFaceCount())
                    .faces(pythonResponse.getFaces())
                    .heatmapBase64(pythonResponse.getHeatmapBase64())
                    .build();

            return ExtensionResponseDto.builder()
                    .status("DONE")
                    .requestId(requestId)
                    .message(pythonResponse.getMessage())
                    .result(resultDetail)
                    .build();

        } catch (Exception e) {
            log.error("[{}] 내부 파이프라인 처리 에러", requestId, e);
            return handleFallback(requestId, "FAIL", "서버 오류로 스캔 실패");
        }
    }

    private MultiValueMap<String, HttpEntity<?>> createMultipartBody(MultipartFile file) throws IOException {
        MultipartBodyBuilder builder = new MultipartBodyBuilder();
        ByteArrayResource resource = new ByteArrayResource(file.getBytes()) {
            @Override
            public String getFilename() {
                return file.getOriginalFilename();
            }
        };
        builder.part("file", resource, MediaType.parseMediaType(file.getContentType()));
        return builder.build();
    }

    private ExtensionResponseDto handleFallback(String requestId, String status, String message) {
        return ExtensionResponseDto.builder()
                .status(status)
                .requestId(requestId)
                .message(message)
                .result(null)
                .build();
    }
}