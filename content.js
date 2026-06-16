console.log("VeritAI content script loaded");
const API_URL = "http://localhost:8080/api/detections";
const FEEDBACK_URL = "http://localhost:8080/api/feedback";
const scanCache = new Map();
const POLL_INITIAL_INTERVAL_MS = 300;
const POLL_MAX_INTERVAL_MS = 2000;
const POLL_TIMEOUT_MS = 180000;
const MAX_CONCURRENT_INSPECTIONS = (navigator.hardwareConcurrency || 4) <= 4 ? 2 : 3;

let isSystemOn = true;
let isAutoScanMode = false;
const FACE_CROP_ANALYSIS_MODE = "face_crop_only";
const scannedMediaKeys = new Set();
let activeInspectionCount = 0;
const pendingInspectionQueue = [];
const pendingDetectionPolls = new Map();
let batchPollingActive = false;

const MAX_CACHE_SIZE = 500;
function manageMemoryCache() {
    if (scannedMediaKeys.size > MAX_CACHE_SIZE) {
        scannedMediaKeys.delete(scannedMediaKeys.keys().next().value);
    }
    if (scanCache.size > MAX_CACHE_SIZE) {
        scanCache.delete(scanCache.keys().next().value);
    }
}

function getMediaSource(media) {
    if (!media) return "";
    return media.currentSrc || media.src || media.poster || "";
}

function getMediaKey(media) {
    const source = getMediaSource(media);
    if (source) return `${media.tagName}:${source}`;
    const rect = media.getBoundingClientRect();
    return `${media.tagName}:${Math.round(rect.left)}:${Math.round(rect.top)}:${Math.round(rect.width)}:${Math.round(rect.height)}`;
}

function isVisibleMedia(media) {
    if (!media || !media.isConnected) return false;
    const rect = media.getBoundingClientRect();
    if (rect.width < 80 || rect.height < 80) return false;
    const style = getComputedStyle(media);
    return style.display !== "none" && style.visibility !== "hidden" && style.opacity !== "0";
}

function shouldInspectMedia(media) {
    if (media.closest('.veritai-details-box') || media.closest('.veritai-ui-container')) {
        return false;
    }
    return isVisibleMedia(media);
}

function readDeepfakeFlag(result) {
    if (!result) return false;
    return Boolean(result.isDeepfake ?? result.deepfake);
}

function getInspectionPriority(media) {
    if (!media || !media.isConnected) return -1;
    const rect = media.getBoundingClientRect();
    const viewportOverlap =
        rect.bottom > 0 && rect.top < window.innerHeight &&
        rect.right > 0 && rect.left < window.innerWidth;
    const area = Math.max(0, rect.width) * Math.max(0, rect.height);
    return (viewportOverlap ? 1_000_000 : 0) + Math.min(area, 999_999);
}

function runWithInspectionLimit(task, media = null) {
    return new Promise((resolve, reject) => {
        pendingInspectionQueue.push({ task, media, resolve, reject });
        pendingInspectionQueue.sort((a, b) => getInspectionPriority(b.media) - getInspectionPriority(a.media));
        drainInspectionQueue();
    });
}

function drainInspectionQueue() {
    while (activeInspectionCount < MAX_CONCURRENT_INSPECTIONS && pendingInspectionQueue.length > 0) {
        const next = pendingInspectionQueue.shift();
        if (next.media && (!next.media.isConnected || !shouldInspectMedia(next.media))) {
            next.reject(new Error("검사 대상이 화면에서 사라졌습니다."));
            continue;
        }
        activeInspectionCount += 1;
        Promise.resolve()
            .then(next.task)
            .then(next.resolve, next.reject)
            .finally(() => {
                activeInspectionCount -= 1;
                drainInspectionQueue();
            });
    }
}

async function captureImageBlob(imageUrl) {
    if (!imageUrl) throw new Error("이미지 주소가 없습니다.");
    
    return new Promise((resolve, reject) => {
        chrome.runtime.sendMessage({
            action: "resize_image",
            url: imageUrl
        }, async (response) => {
            if (chrome.runtime.lastError) {
                return reject(new Error(chrome.runtime.lastError.message));
            }
            if (response && response.success && response.base64) {
                try {
                    const res = await fetch(response.base64);
                    const blob = await res.blob();
                    resolve(blob);
                } catch (e) {
                    reject(new Error("이미지 변환 실패"));
                }
            } else {
                reject(new Error(response?.error || "리사이징 실패"));
            }
        });
    });
}

async function captureVideoBlob(video) {
    if (!video) throw new Error("영상 요소를 찾을 수 없습니다.");
    let width = video.videoWidth || video.clientWidth;
    let height = video.videoHeight || video.clientHeight;
    if (width === 0 || height === 0) throw new Error("영상 크기를 인식할 수 없습니다.");

    return new Promise((resolve, reject) => {
        try {
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");
            if (!ctx) return reject(new Error("캔버스 컨텍스트를 생성하지 못했습니다."));

            const MAX_SIZE = 1280;
            if (width > MAX_SIZE || height > MAX_SIZE) {
                const ratio = Math.min(MAX_SIZE / width, MAX_SIZE / height);
                width = Math.round(width * ratio);
                height = Math.round(height * ratio);
            }
            canvas.width = width;
            canvas.height = height;

            if (!video.crossOrigin) { video.crossOrigin = "anonymous"; }
            
            ctx.drawImage(video, 0, 0, width, height);
            canvas.toBlob((blob) => {
                if (!blob) return reject(new Error("영상 프레임 데이터를 생성하지 못했습니다."));
                resolve(blob);
            }, "image/webp", 0.7);
        } catch (error) {
            reject(new Error("비디오 프레임에 접근할 수 없습니다 (CORS 보안)."));
        }
    });
}

async function sendToBackend(blob, mediaType, analysisMode = FACE_CROP_ANALYSIS_MODE) {
    const formData = new FormData();
    formData.append("file", blob, "capture.webp"); 
    formData.append("sourceUrl", window.location.href);
    formData.append("mediaType", mediaType);
    formData.append("clientType", "chrome-extension");
    formData.append("analysisMode", analysisMode);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000);

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData,
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const error = new Error(`Server Error`);
            error.status = response.status;
            throw error;
        }

        const data = await response.json();
        if (!data) throw new Error("분석이 정상적으로 완료되지 않았습니다.");

        if (data.status === "DONE" && data.result) return data;

        if ((data.status === "PROCESSING" || data.status === "QUEUED") && data.requestId) {
            return await pollDetectionResult(data.requestId);
        }

        if (data.status === "FAILED") throw new Error(data?.message || "Analysis failed");

        throw new Error(data?.message || "분석이 정상적으로 완료되지 않았습니다.");

    } catch (err) {
        clearTimeout(timeoutId);
        if (err.name === 'AbortError') {
            const timeoutErr = new Error("Timeout");
            timeoutErr.status = 408;
            throw timeoutErr;
        }
        throw err;
    }
}

function updateStatusBadge(media, status, data = null) {
    const wrapper = ensureWrapper(media);
    if (!wrapper) return;

    if (!media.dataset.veritaiScanned && status !== "loading") return;

    const existingContainers = wrapper.querySelectorAll('.veritai-ui-container');
    if (existingContainers.length > 1) {
        existingContainers.forEach(c => c.remove());
    }

    let uiContainer = wrapper.querySelector('.veritai-ui-container');
    if (!uiContainer) {
        uiContainer = document.createElement('div');
        uiContainer.className = 'veritai-ui-container';

        uiContainer.style.cssText = `
            position: absolute; 
            top: 6px; 
            left: 6px; 
            z-index: 2147483647;
            display: flex; flex-direction: column; align-items: flex-start;
            pointer-events: none; 
        `;
        wrapper.appendChild(uiContainer);
    }

    let badge = uiContainer.querySelector('.veritai-status-badge');
    if (!badge) {
        badge = document.createElement('div');
        badge.className = 'veritai-status-badge';
        uiContainer.appendChild(badge);
    }

    badge.onclick = null;
    badge.onmouseenter = null;
    badge.onmouseleave = null;
    badge.dataset.pinned = "false";

    badge.style.cssText = `
        padding: 4px 8px; border-radius: 4px; color: white; font-size: 11px; 
        font-weight: bold; font-family: sans-serif; box-shadow: 0 2px 4px rgba(0,0,0,0.5);
        transition: all 0.2s ease; user-select: none; cursor: default;
        pointer-events: auto !important;
        box-sizing: border-box !important;
        line-height: normal !important;
    `;
    media.style.border = "none";

    if (status === "loading") {
        badge.innerHTML = `
            <div style="display: flex; align-items: center; gap: 5px;">
                <div style="width: 10px; height: 10px; border: 2px solid white; border-top-color: transparent; border-radius: 50%; animation: veritai-spin 1s linear infinite;"></div>
                분석 중...
            </div>
            <style>
                @keyframes veritai-spin { to { transform: rotate(360deg); } }
            </style>
        `;
        badge.style.background = "rgba(59, 130, 246, 0.9)";
    }
    else if (status === "error") {
        const errorMsg = data?.message || "분석 실패";
        badge.innerText = errorMsg;
        badge.style.background = "rgba(100, 116, 139, 0.9)";

        setTimeout(() => {
            if (uiContainer && uiContainer.parentNode) {
                uiContainer.remove();
            }
        }, 3000);
    }
    else if (status === "fake" || status === "real") {
        badge.style.cursor = "pointer";

        if (status === "fake") {
            const conf = ((data.result.confidence || 0) * 100).toFixed(1);
            badge.innerText = `조작 의심 (${conf}%)`;
            badge.style.background = "rgba(239, 68, 68, 0.95)";
            badge.style.borderRadius = "4px";
            badge.style.padding = "4px 8px";
            media.style.border = "2px solid rgba(239, 68, 68, 0.8)";
        } else {
            badge.innerText = "✓";
            badge.style.background = "rgba(16, 185, 129, 0.8)";
            badge.style.color = "white";
            badge.style.width = "18px";
            badge.style.height = "18px";
            badge.style.borderRadius = "50%";
            badge.style.display = "flex";
            badge.style.justifyContent = "center";
            badge.style.alignItems = "center";
            badge.style.fontSize = "12px";
            badge.style.padding = "0";
            badge.style.fontWeight = "bold";
            media.style.border = "none";
            media.style.boxShadow = "inset 4px 0 0 rgba(0, 200, 0, 0.8)";
            badge.style.opacity = "0.4";
        }

        const showReportBox = (e) => {
            if (e) {
                e.preventDefault();
                e.stopPropagation();
            }

            const mediaSrc = media.currentSrc || media.src || "unknown_media";
            const existingBoxes = document.querySelectorAll('.veritai-details-box');

            if (e && e.type === "click") {
                if (badge.dataset.pinned === "true") {
                    badge.dataset.pinned = "false";
                    existingBoxes.forEach(box => { if (box.cleanupListeners) box.cleanupListeners(); box.remove(); });
                    return;
                } else {
                    badge.dataset.pinned = "true";
                    existingBoxes.forEach(box => { if (box.cleanupListeners) box.cleanupListeners(); box.remove(); });
                }
            } else {
                if (badge.dataset.pinned === "true") return;
                existingBoxes.forEach(box => { if (box.cleanupListeners) box.cleanupListeners(); box.remove(); });
            }

            const result = data.result;
            const faces = result.faces || [];

            const faceText = faces.length === 0 ?
                "<div style='text-align:center; color:#94a3b8; padding: 10px 0;'>검출된 얼굴 없음</div>" :
                faces.slice(0, 3).map((f, i) => {
                    const bbox = f.bbox || {};
                    const quality = f.quality || {};
                    const detConf = ((f.detectionConfidence || f.score || 0) * 100).toFixed(1);
                    const qualScore = ((quality.score || 0) * 100).toFixed(1);

                    return `
                    <div style="background: rgba(0, 0, 0, 0.2); padding: 8px 10px; border-radius: 6px; margin-bottom: 8px; border: 1px solid rgba(255,255,255,0.05);">
                        <div style="color:#fbbf24; font-weight:bold; margin-bottom: 6px; font-size: 11.5px;">[얼굴 ${i + 1}]</div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; color: #cbd5e1; font-size: 11px; line-height: 1.3;">
                            <div>• 유형: <span style="color:#fff">${f.faceMode || '?'}</span></div>
                            <div>• 검출률: <span style="color:#fff">${detConf}%</span></div>
                            <div>• 크기: <span style="color:#fff">${bbox.w ?? '?'}x${bbox.h ?? '?'}</span></div>
                            <div>• 품질: <span style="color:#fff">${quality.label || '?'}</span></div>
                        </div>
                    </div>`;
                }).join("");

            const detailsBox = document.createElement('div');
            detailsBox.className = 'veritai-details-box';
            detailsBox.dataset.targetMedia = mediaSrc;

            Object.assign(detailsBox.style, {
                position: "absolute",
                top: "0px",
                left: "0px",
                willChange: "transform",
                zIndex: "2147483647",
                background: "rgba(30, 41, 59, 0.95)",
                backdropFilter: "blur(12px)",
                color: "#F8FAFC",
                padding: "16px",
                borderRadius: "12px",
                border: `1px solid ${status === "fake" ? "rgba(239, 68, 68, 0.5)" : "rgba(16, 185, 129, 0.5)"}`,
                fontSize: "12px",
                whiteSpace: "normal",
                lineHeight: "1.6",
                boxShadow: badge.dataset.pinned === "true"
                    ? `0 0 15px ${status === "fake" ? "rgba(239, 68, 68, 0.4)" : "rgba(16, 185, 129, 0.4)"}`
                    : "0 10px 25px -5px rgba(0, 0, 0, 0.5)",
                fontFamily: "monospace",
                width: "280px",
                maxHeight: "400px",
                overflowY: "auto",
                textAlign: "left",
                cursor: "default",
                pointerEvents: "auto",
                transition: "box-shadow 0.3s ease",
                boxSizing: "border-box",
                fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                margin: "0",
                letterSpacing: "normal"
            });

            detailsBox.innerHTML = `
<div class="veritai-drag-handle" style="color:lightskyblue; font-weight:bold; margin-bottom:12px; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:8px; font-size:14px; display:flex; justify-content:space-between; align-items: center; cursor: grab; user-select: none;">
    <span>🔍 분석 리포트 ${badge.dataset.pinned === "true" ? "📌" : ""}</span>
    <span class="veritai-close-btn" style="cursor:pointer; color:#94a3b8; padding: 0 5px; font-size: 16px;">✕</span>
</div>
<div style="display: flex; flex-direction: column; gap: 6px;">
    <div><b>ID:</b> <span style="color:#e2e8f0;">${data.requestId || 'N/A'}</span></div>
    <div><b>판정:</b> ${readDeepfakeFlag(result) ? "<span style='color:#ef4444; font-weight:bold;'>조작 의심</span>" : "<span style='color:#10b981; font-weight:bold;'>정상</span>"}</div>
</div>

<div style="margin:12px 0; border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; padding: 8px; background: rgba(0,0,0,0.2);">
    
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <div style="font-weight: bold; font-size: 11px; color: #94a3b8;" id="veritai-visual-title">감지 영역 (기본)</div>
        <button id="veritai-toggle-xai-btn" style="font-size: 10px; padding: 2px 6px; background: #3b82f6; color: white; border: none; border-radius: 3px; cursor: pointer;" ${!result.heatmapBase64 ? 'disabled' : ''}>
            ${result.heatmapBase64 ? 'AI 분석 근거 보기' : '히트맵 데이터 없음'}
        </button>
    </div>

    <div style="position: relative; width: 100%; height: 160px; background: #0f172a; border-radius: 4px; overflow: hidden; border: 1px solid #333; display: flex; align-items: center; justify-content: center;">
        <canvas id="veritai-bbox-canvas" style="position: absolute; max-width: 100%; max-height: 100%; object-fit: contain; z-index: 2;"></canvas>
        <canvas id="veritai-heatmap-canvas" style="display: none; position: absolute; max-width: 100%; max-height: 100%; object-fit: contain; z-index: 1;"></canvas>
    </div>

    <div id="veritai-slider-container" style="display: none; margin-top: 8px; align-items: center; gap: 8px;">
        <span style="font-size: 10px; color: #64748b;">원본</span>
        <input type="range" id="veritai-heatmap-slider" min="0" max="100" value="70" style="flex: 1; accent-color: #ef4444; cursor: pointer;">
        <span style="font-size: 10px; color: #ef4444;">히트맵</span>
    </div>
</div>

<div style="margin:12px 0; border-top:1px dashed rgba(255,255,255,0.2);"></div>
${faceText}
<div style="margin-top: 15px; display: flex; justify-content: flex-end;">
    <button class="veritai-feedback-btn" style="font-size: 11px; padding: 4px 8px; cursor: pointer; background: rgba(255, 60, 60, 0.1); color: #ff6b6b; border: 1px solid rgba(255, 60, 60, 0.3); border-radius: 4px; transition: all 0.2s;">🚨 오답 신고</button>
</div>
            `.trim();

            detailsBox.onclick = (evt) => evt.stopPropagation();
            detailsBox.onmouseenter = () => { detailsBox.dataset.isHovered = "true"; };
            detailsBox.onmouseleave = () => {
                detailsBox.dataset.isHovered = "false";
                if (status === "real" && badge.dataset.pinned !== "true") {
                    setTimeout(() => {
                        if (detailsBox.dataset.isHovered !== "true" && badge.dataset.isHovered !== "true") {
                            badge.dataset.pinned = "false";
                            detailsBox.remove();
                        }
                    }, 400);
                }
            };

            document.body.appendChild(detailsBox);

            const bboxCanvas = detailsBox.querySelector('#veritai-bbox-canvas');
            const heatmapCanvas = detailsBox.querySelector('#veritai-heatmap-canvas');
            const slider = detailsBox.querySelector('#veritai-heatmap-slider');

            if (bboxCanvas && heatmapCanvas) {
                const bCtx = bboxCanvas.getContext('2d');
                const hCtx = heatmapCanvas.getContext('2d');
                const imgObj = new Image();
                imgObj.crossOrigin = "anonymous";
                
                imgObj.onload = () => {
                    bboxCanvas.width = imgObj.width;
                    bboxCanvas.height = imgObj.height;
                    heatmapCanvas.width = imgObj.width;
                    heatmapCanvas.height = imgObj.height;

                    bCtx.drawImage(imgObj, 0, 0);
                    if (faces && faces.length > 0) {
                        faces.forEach((f, i) => {
                            if (f.bbox && f.bbox.w > 0) {
                                bCtx.lineWidth = Math.max(3, imgObj.width / 150);
                                const isFake = (f.fakeProbability || f.confidence || 0) >= 0.5;
                                bCtx.strokeStyle = isFake ? "#ef4444" : "#10b981"; 
                                bCtx.fillStyle = isFake ? "rgba(239, 68, 68, 0.2)" : "rgba(16, 185, 129, 0.2)";
                                bCtx.strokeRect(f.bbox.x, f.bbox.y, f.bbox.w, f.bbox.h);
                                bCtx.fillRect(f.bbox.x, f.bbox.y, f.bbox.w, f.bbox.h);
                                bCtx.font = `${Math.max(16, imgObj.width / 25)}px sans-serif`;
                                bCtx.fillStyle = bCtx.strokeStyle;
                                bCtx.fillText(`얼굴 ${i + 1}`, f.bbox.x, f.bbox.y - 5);
                            }
                        });
                    }

                    // 히트맵
                    if (result.heatmapBase64) {
                        let bestFace = faces && faces.length > 0 ? faces[0] : null;
                        let maxScore = -1;
                        if (faces) {
                            faces.forEach(f => {
                                const score = f.fakeProbability || f.confidence || (f.detectionConfidence || 0);
                                if (score > maxScore) { maxScore = score; bestFace = f; }
                            });
                        }

                        const hmImg = new Image();
                        hmImg.onload = () => {
                            const drawHeatmap = (opacity) => {
                                hCtx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
                                hCtx.drawImage(imgObj, 0, 0); 
                                
                                hCtx.globalAlpha = opacity;
                                hCtx.globalCompositeOperation = "screen"; 
                                
                                if (bestFace && bestFace.bbox) {
                                    hCtx.drawImage(hmImg, bestFace.bbox.x, bestFace.bbox.y, bestFace.bbox.w, bestFace.bbox.h);
                                } else {
                                    hCtx.drawImage(hmImg, 0, 0, imgObj.width, imgObj.height);
                                }
                                
                                hCtx.globalAlpha = 1.0;
                                hCtx.globalCompositeOperation = "source-over";
                            };

                            drawHeatmap(slider.value / 100);

                            if (slider) {
                                slider.addEventListener('input', (e) => {
                                    drawHeatmap(e.target.value / 100);
                                });
                            }
                        };
                        hmImg.src = "data:image/jpeg;base64," + result.heatmapBase64;
                    }
                };
                imgObj.src = mediaSrc;
            }

            const toggleBtn = detailsBox.querySelector('#veritai-toggle-xai-btn');
            const visualTitle = detailsBox.querySelector('#veritai-visual-title');
            const sliderContainer = detailsBox.querySelector('#veritai-slider-container');
            let isHeatmapMode = false;

            if (toggleBtn && result.heatmapBase64) {
                toggleBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    isHeatmapMode = !isHeatmapMode;
                    
                    if (isHeatmapMode) {
                        bboxCanvas.style.display = 'none';
                        heatmapCanvas.style.display = 'block';
                        sliderContainer.style.display = 'flex';
                        toggleBtn.innerText = '감지 박스로 돌아가기';
                        toggleBtn.style.background = '#64748b';
                        visualTitle.innerText = 'XAI 분석 근거 (히트맵)';
                    } else {
                        bboxCanvas.style.display = 'block';
                        heatmapCanvas.style.display = 'none';
                        sliderContainer.style.display = 'none';
                        toggleBtn.innerText = 'AI 분석 근거 보기';
                        toggleBtn.style.background = '#3b82f6';
                        visualTitle.innerText = '감지 영역 (기본)';
                    }
                });
            }

            const dragHandle = detailsBox.querySelector('.veritai-drag-handle');
            let isDragging = false;
            let startX, startY, initialLeft, initialTop;

            dragHandle.addEventListener('mousedown', (e) => {
                if (e.target.classList.contains('veritai-close-btn')) return;
                isDragging = true;
                detailsBox.dataset.isDragged = "true";
                dragHandle.style.cursor = 'grabbing';

                const rect = detailsBox.getBoundingClientRect();
                detailsBox.style.transform = 'none';
                detailsBox.style.left = (rect.left + window.scrollX) + 'px';
                detailsBox.style.top = (rect.top + window.scrollY) + 'px';

                startX = e.clientX;
                startY = e.clientY;
                initialLeft = parseFloat(detailsBox.style.left) || 0;
                initialTop = parseFloat(detailsBox.style.top) || 0;

                e.preventDefault();
            });

            const onMouseMove = (e) => {
                if (!isDragging) return;
                const dx = e.clientX - startX;
                const dy = e.clientY - startY;
                detailsBox.style.left = (initialLeft + dx) + 'px';
                detailsBox.style.top = (initialTop + dy) + 'px';
            };

            const onMouseUp = () => {
                if (isDragging) {
                    isDragging = false;
                    dragHandle.style.cursor = 'grab';
                }
            };

            window.addEventListener('mousemove', onMouseMove);
            window.addEventListener('mouseup', onMouseUp);

            const updatePosition = () => {
                if (!document.body.contains(detailsBox)) return;
                if (detailsBox.dataset.isDragged === "true") return;

                const badgeRect = badge.getBoundingClientRect();
                const boxWidth = 280;
                const boxMaxHeight = 400;

                let leftPos = badgeRect.left + window.scrollX;
                if (leftPos + boxWidth > window.innerWidth + window.scrollX) {
                    leftPos = window.innerWidth + window.scrollX - boxWidth - 10;
                }

                let topPos = badgeRect.bottom + window.scrollY + 5;
                if (badgeRect.bottom + boxMaxHeight > window.innerHeight) {
                    topPos = badgeRect.top + window.scrollY - detailsBox.offsetHeight - 5;
                    if (topPos < window.scrollY) {
                        topPos = window.scrollY + 50;
                    }
                }

                detailsBox.style.transform = `translate3d(${leftPos}px, ${topPos}px, 0)`;
            };

            updatePosition();
            window.addEventListener('resize', updatePosition);

            let closeDetails;
            detailsBox.cleanupListeners = () => {
                window.removeEventListener('resize', updatePosition);
                window.removeEventListener('mousemove', onMouseMove);
                window.removeEventListener('mouseup', onMouseUp);
                if (closeDetails) document.removeEventListener('click', closeDetails);
            };

            const closeBtn = detailsBox.querySelector('.veritai-close-btn');
            if (closeBtn) {
                closeBtn.addEventListener('click', (evt) => {
                    evt.preventDefault();
                    evt.stopImmediatePropagation();
                    badge.dataset.pinned = "false";
                    detailsBox.cleanupListeners();
                    detailsBox.remove();
                });
            }

            const feedbackBtn = detailsBox.querySelector('.veritai-feedback-btn');
            if (feedbackBtn) {
                feedbackBtn.addEventListener('mouseenter', () => {
                    if (!feedbackBtn.disabled) feedbackBtn.style.background = 'rgba(255, 60, 60, 0.2)';
                });
                feedbackBtn.addEventListener('mouseleave', () => {
                    if (!feedbackBtn.disabled) feedbackBtn.style.background = 'rgba(255, 60, 60, 0.1)';
                });

                feedbackBtn.onclick = (e) => {
                    e.stopPropagation();
                    if (feedbackBtn.disabled) return;
                    feedbackBtn.style.display = 'none';

                    const reasonContainer = document.createElement('div');
                    reasonContainer.style.cssText = 'display: flex; flex-direction: column; gap: 5px; margin-top: 5px; width: 100%;';
                    const reasonInput = document.createElement('textarea');
                    reasonInput.placeholder = "어떤 부분이 잘못되었나요?";
                    reasonInput.style.cssText = `font-size: 11px; padding: 5px; border-radius: 4px; border: 1px solid #555; background: #222; color: white; resize: none; height: 40px; font-family: sans-serif;`;

                    const actionContainer = document.createElement('div');
                    actionContainer.style.cssText = 'display: flex; justify-content: flex-end; gap: 5px;';
                    const cancelBtn = document.createElement('button');
                    cancelBtn.innerText = "취소";
                    cancelBtn.style.cssText = 'font-size: 11px; padding: 2px 8px; cursor: pointer; background: #444; color: white; border: none; border-radius: 3px;';
                    const submitBtn = document.createElement('button');
                    submitBtn.innerText = "제출";
                    submitBtn.style.cssText = 'font-size: 11px; padding: 2px 8px; cursor: pointer; background: #ff6b6b; color: white; border: none; border-radius: 3px; font-weight: bold;';

                    actionContainer.appendChild(cancelBtn);
                    actionContainer.appendChild(submitBtn);
                    reasonContainer.appendChild(reasonInput);
                    reasonContainer.appendChild(actionContainer);
                    feedbackBtn.parentNode.appendChild(reasonContainer);

                    cancelBtn.onclick = (cancelEvent) => {
                        cancelEvent.stopPropagation();
                        reasonContainer.remove();
                        feedbackBtn.style.display = 'flex';
                    };

                    submitBtn.onclick = async (submitEvent) => {
                        submitEvent.stopPropagation();
                        const textReason = reasonInput.value.trim();
                        if (!textReason) {
                            reasonInput.style.border = "1px solid red";
                            reasonInput.placeholder = "신고 이유를 적어주세요.";
                            return;
                        }
                        submitBtn.innerText = "전송 중...";
                        submitBtn.disabled = true;
                        cancelBtn.disabled = true;

                        try {
                            const response = await fetch(FEEDBACK_URL, {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({
                                    requestId: data.requestId,
                                    reportedAt: new Date().toISOString(),
                                    reason: textReason
                                })
                            });
                            if (!response.ok) throw new Error("전송 실패");
                            reasonContainer.innerHTML = "<span style='color: lightgreen; font-size: 11px; text-align: right;'>피드백이 접수되었습니다!</span>";
                        } catch (err) {
                            submitBtn.innerText = "실패(재시도)";
                            submitBtn.disabled = false;
                            cancelBtn.disabled = false;
                        }
                    };
                };
            };

            setTimeout(() => {
                closeDetails = (evt) => {
                    if (!detailsBox.contains(evt.target) && !badge.contains(evt.target)) {
                        badge.dataset.pinned = "false";
                        detailsBox.cleanupListeners();
                        detailsBox.remove();
                    }
                };
                document.addEventListener('click', closeDetails);
            }, 10);
        };

        badge.onclick = (e) => showReportBox(e);

        if (status === "real") {
            badge.onmouseenter = (e) => {
                badge.style.opacity = "1";
                badge.dataset.isHovered = "true";
                showReportBox(e);
            };

            badge.onmouseleave = () => {
                badge.dataset.isHovered = "false";
                if (badge.dataset.pinned !== "true") {
                    badge.style.opacity = "0.4";
                    setTimeout(() => {
                        const existingBox = document.querySelector('.veritai-details-box');
                        if (existingBox && existingBox.dataset.isHovered !== "true" && badge.dataset.isHovered !== "true" && badge.dataset.pinned !== "true") {
                            if (existingBox.cleanupListeners) existingBox.cleanupListeners();
                            existingBox.remove();
                        }
                    }, 400);
                }
            };
        }
    }
}

async function startInspection(media) {
    if (!isSystemOn || !shouldInspectMedia(media)) return;

    if (media.dataset.veritaiScanned === "true") return;

    const mediaUrl = media.currentSrc || media.src;
    const scanKey = getMediaKey(media);

    if (mediaUrl && scanCache.has(mediaUrl)) {
        media.dataset.veritaiScanned = "true";
        media.dataset.veritaiScanKey = scanKey;
        scannedMediaKeys.add(scanKey); 
        const cachedData = scanCache.get(mediaUrl);
        updateStatusBadge(media, readDeepfakeFlag(cachedData.result) ? "fake" : "real", cachedData);
        return;
    }

    if (scannedMediaKeys.has(scanKey)) return;
    scannedMediaKeys.add(scanKey);
    manageMemoryCache();
    media.dataset.veritaiScanned = "true";
    media.dataset.veritaiScanKey = scanKey;

    const wrapper = ensureWrapper(media);
    if (wrapper) {
        const btn = wrapper.querySelector('.veritai-check-btn');
        if (btn) btn.remove();
    }

    return runWithInspectionLimit(async () => {
        updateStatusBadge(media, "loading");

        if (mediaUrl && scanCache.has(mediaUrl)) {
            console.log("캐시된 결과를 재활용합니다:", mediaUrl);
            const cachedData = scanCache.get(mediaUrl);

            if (readDeepfakeFlag(cachedData.result)) {
                updateStatusBadge(media, "fake", cachedData);
            } else {
                updateStatusBadge(media, "real", cachedData);
            }
            return;
        }

        let blob;
        let mediaType = "image";
        if (media.tagName === "VIDEO") {
            blob = await captureVideoBlob(media);
            mediaType = "video_frame";
        } else {
            blob = await captureImageBlob(mediaUrl);
        }

        const data = await sendToBackend(blob, mediaType);

        if (!media.dataset.veritaiScanned) {
            console.log("검사 중지됨: 로딩 중 모드가 해제되었습니다.");
            return;
        }

        if (mediaUrl) {
            scanCache.set(mediaUrl, data);
        }

        if (readDeepfakeFlag(data.result)) {
            updateStatusBadge(media, "fake", data);
        } else {
            updateStatusBadge(media, "real", data);
        }

    }, media).catch((err) => {
        console.error("Analysis Error:", err);
        let friendlyMessage = "분석 오류";

        if (err.status === 429) {
            friendlyMessage = "요청 과다 (잠시 후 시도)";
        } else if (err.status === 408) {
            friendlyMessage = "응답 지연 (서버 혼잡)";
        } else if (err.status >= 500) {
            friendlyMessage = "서버 내부 오류";
        } else if (err.name === 'TypeError' && err.message === 'Failed to fetch') {
            friendlyMessage = "서버 연결 실패 (서버 꺼짐)";
        } else if (err.message.includes("CORS") || err.message.includes("보안 차단됨")) {
            friendlyMessage = "보안 정책 차단";
        } else if (err.status === 400 || err.status === 415) {
            friendlyMessage = "지원하지 않는 이미지";
        } else {
            friendlyMessage = err.message || "분석 실패";
        }
        
        updateStatusBadge(media, "error", { message: friendlyMessage });
        delete media.dataset.veritaiScanned;
        if (media.dataset.veritaiScanKey) {
            scannedMediaKeys.delete(media.dataset.veritaiScanKey);
            delete media.dataset.veritaiScanKey;
        }

        setTimeout(() => {
            delete media.dataset.veritaiAttached;
            attachUI(media);
        }, 3000);
    });
}

const autoScanObserver = new IntersectionObserver((entries) => {
    if (!isSystemOn || !isAutoScanMode) return;
    entries.forEach(entry => {
        if (entry.isIntersecting && entry.target.clientWidth > 80) {
            if (entry.target.dataset.scanTimer) clearTimeout(entry.target.dataset.scanTimer);

            entry.target.dataset.scanTimer = setTimeout(() => {
                const rect = entry.target.getBoundingClientRect();
                if (rect.top < window.innerHeight && rect.bottom > 0) {
                    startInspection(entry.target);
                    autoScanObserver.unobserve(entry.target);
                }
            }, 300);
        }
    });
}, { threshold: 0.3 });

let debounceTimer;

const domObserver = new MutationObserver((mutations) => {
    if (!isSystemOn) return;

    mutations.forEach(mutation => {
        if (mutation.addedNodes) {
            mutation.addedNodes.forEach(node => {
                if (node.nodeType === 1 && (node.tagName === 'IMG' || node.tagName === 'VIDEO')) {
                    attachUI(node); 
                }
                else if (node.nodeType === 1 && node.querySelectorAll) {
                    node.querySelectorAll('img, video').forEach(media => attachUI(media));
                }
            });
        }
        
        if (mutation.type === 'attributes' && (mutation.attributeName === 'src' || mutation.attributeName === 'srcset')) {
            const target = mutation.target;
            if (target.tagName === 'IMG' || target.tagName === 'VIDEO') {
                delete target.dataset.veritaiAttached;
                delete target.dataset.veritaiScanned;
                delete target.dataset.veritaiScanKey;
                const wrapper = ensureWrapper(target);
                if (wrapper) {
                    const oldBadge = wrapper.querySelector('.veritai-ui-container');
                    if (oldBadge) oldBadge.remove();
                    const oldBtn = wrapper.querySelector('.veritai-check-btn');
                    if (oldBtn) oldBtn.remove();
                }
                
                if (target.tagName === 'IMG' && !target.complete) {
                    target.addEventListener('load', () => attachUI(target), { once: true });
                } else {
                    attachUI(target); 
                }
            }
        }
    });
});

function ensureWrapper(media) {
    let parent = media.parentElement;
    if (!parent) return null;

    if (parent.tagName === 'PICTURE' || parent.tagName === 'YT-IMAGE' || parent.tagName === 'YT-IMG-SHADOW') {
        parent = parent.parentElement;
        if (!parent) return null;
    }

    if (getComputedStyle(parent).position === "static") {
        parent.style.position = "relative";
    }
    return parent;
}

function attachUI(media, retryCount = 0) {
    if (media.tagName === 'IMG' && !media.complete) {
        media.addEventListener('load', () => attachUI(media, retryCount), { once: true });
        return;
    }

    if (!shouldInspectMedia(media)) {
        if (retryCount < 5) {
            setTimeout(() => attachUI(media, retryCount + 1), 300);
        }
        return;
    }

    const wrapper = ensureWrapper(media);
    const hasUI = wrapper && (wrapper.querySelector('.veritai-check-btn') || wrapper.querySelector('.veritai-status-badge'));

    const mediaUrl = media.currentSrc || media.src;

    if (mediaUrl && scanCache.has(mediaUrl) && !hasUI) {
        media.dataset.veritaiAttached = "true";
        media.dataset.veritaiScanned = "true";
        media.dataset.veritaiScanKey = getMediaKey(media);
        const data = scanCache.get(mediaUrl);
        updateStatusBadge(media, readDeepfakeFlag(data.result) ? "fake" : "real", data);
        return;
    }

    if (media.dataset.veritaiAttached === "true" && hasUI) return;

    media.dataset.veritaiAttached = "true";
    delete media.dataset.veritaiScanned;

    if (isAutoScanMode && media.tagName !== 'VIDEO') {
        autoScanObserver.observe(media); 
    } else {
        if (wrapper && !wrapper.querySelector('.veritai-check-btn')) {
            const btn = document.createElement("button");
            btn.innerText = "🔍 검사";
            btn.className = "veritai-check-btn";
            
            btn.style.cssText = `
                position: absolute; 
                top: 8px; 
                left: 8px; 
                z-index: 2147483647;
                padding: 4px 10px; 
                background-color: rgba(59, 130, 246, 0.9);
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 0.2); 
                border-radius: 6px;
                cursor: pointer;
                font-weight: 600; font-size: 11px; backdrop-filter: blur(4px);
                transition: all 0.2s ease;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                pointer-events: auto !important;

                box-sizing: border-box !important;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
                line-height: normal !important;
            `;
            btn.onmouseenter = () => btn.style.backgroundColor = "rgba(37, 99, 235, 1)";
            btn.onmouseleave = () => btn.style.backgroundColor = "rgba(59, 130, 246, 0.9)";

            btn.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                startInspection(media);
            });
            wrapper.appendChild(btn);
        }
    }
}

chrome.runtime.onMessage.addListener((msg) => {
    if (msg.action === "TOGGLE_SYSTEM") {
        isSystemOn = msg.isSystemOn;
        isAutoScanMode = msg.isAutoScanOn;
        autoScanObserver.disconnect();
        domObserver.disconnect();
        clearAllUI();
        if (isSystemOn) {
            document.querySelectorAll('img, video').forEach(media => attachUI(media));
            domObserver.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["src", "class", "style", "open", "aria-hidden", "aria-modal"] });
        }
    }
});

function clearAllUI() {
    document.querySelectorAll('img, video').forEach(media => {
        media.style.border = "none";
        delete media.dataset.veritaiScanned;
        delete media.dataset.veritaiAttached;
        delete media.dataset.veritaiScanKey;
        const wrapper = ensureWrapper(media);
        if (wrapper) {
            const container = wrapper.querySelector('.veritai-ui-container');
            if (container) container.remove();
            const btn = wrapper.querySelector('.veritai-check-btn');
            if (btn) btn.remove();
        }
    });
    scannedMediaKeys.clear();

    document.querySelectorAll('.veritai-details-box').forEach(box => {
        if (box.cleanupListeners) box.cleanupListeners();
        box.remove();
    });
}

chrome.storage.local.get(['isSystemOn', 'isAutoScanOn'], (result) => {
    isSystemOn = result.isSystemOn !== false;
    isAutoScanMode = result.isAutoScanOn || false;
    setTimeout(() => {
        if (isSystemOn) {
            document.querySelectorAll('img, video').forEach(media => attachUI(media));
            domObserver.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: [ "src","class", "style", "open", "aria-hidden", "aria-modal"] });
        }
    }, 500);
});

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function pollDetectionResult(requestId) {
    return new Promise((resolve, reject) => {
        pendingDetectionPolls.set(String(requestId), {
            requestId,
            resolve,
            reject,
            startedAt: Date.now(),
        });
        ensureBatchPolling();
    });
}

function ensureBatchPolling() {
    if (batchPollingActive) return;
    batchPollingActive = true;
    runBatchPollingLoop().finally(() => {
        batchPollingActive = false;
        if (pendingDetectionPolls.size > 0) {
            ensureBatchPolling();
        }
    });
}

async function runBatchPollingLoop() {
    let delayMs = POLL_INITIAL_INTERVAL_MS;
    while (pendingDetectionPolls.size > 0) {
        await delay(delayMs);
        const now = Date.now();
        const timedOut = [];
        pendingDetectionPolls.forEach((entry, key) => {
            if (now - entry.startedAt >= POLL_TIMEOUT_MS) {
                timedOut.push(key);
            }
        });
        timedOut.forEach(key => {
            const entry = pendingDetectionPolls.get(key);
            if (entry) entry.reject(new Error("Analysis timed out."));
            pendingDetectionPolls.delete(key);
        });
        if (pendingDetectionPolls.size === 0) break;

        const ids = Array.from(pendingDetectionPolls.keys()).join(",");
        let response;
        try {
            response = await fetch(`${API_URL}/status?ids=${encodeURIComponent(ids)}`);
        } catch (error) {
            pendingDetectionPolls.forEach(entry => entry.reject(error));
            pendingDetectionPolls.clear();
            break;
        }
        if (!response.ok) {
            const error = new Error(`Server response error: ${response.status}`);
            pendingDetectionPolls.forEach(entry => entry.reject(error));
            pendingDetectionPolls.clear();
            break;
        }

        const data = await response.json();
        const items = Array.isArray(data?.items) ? data.items : [];
        let maxRetryAfterMs = 0;
        let completedCount = 0;
        items.forEach(item => {
            const key = String(item.requestId);
            const entry = pendingDetectionPolls.get(key);
            if (!entry) return;
            const retryAfterMs = Number(item.retryAfterMs);
            if (Number.isFinite(retryAfterMs) && retryAfterMs > 0) {
                maxRetryAfterMs = Math.max(maxRetryAfterMs, retryAfterMs);
            }
            if (item.status === "DONE" && item.result) {
                entry.resolve(item);
                pendingDetectionPolls.delete(key);
                completedCount += 1;
            } else if (item.status === "FAILED") {
                entry.reject(new Error(item?.message || "Analysis failed"));
                pendingDetectionPolls.delete(key);
                completedCount += 1;
            }
        });
        if (completedCount > 0) {
            delayMs = POLL_INITIAL_INTERVAL_MS;
        } else if (maxRetryAfterMs > 0) {
            delayMs = Math.min(POLL_MAX_INTERVAL_MS, Math.max(POLL_INITIAL_INTERVAL_MS, maxRetryAfterMs));
        } else {
            delayMs = Math.min(POLL_MAX_INTERVAL_MS, Math.round(delayMs * 1.25));
        }
    }
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        const detailsBox = document.querySelector('.veritai-details-box');
        if (detailsBox) {
            if (detailsBox.cleanupListeners) detailsBox.cleanupListeners();
            detailsBox.remove();
        }
    }
});

document.addEventListener('click', (e) => {
    if (!e.isTrusted) return; 

    const customUIs = document.querySelectorAll('.veritai-check-btn, .veritai-status-badge');
    if (customUIs.length === 0) return; 

    for (let ui of customUIs) {
        const rect = ui.getBoundingClientRect();
        if (e.clientX >= rect.left && e.clientX <= rect.right &&
            e.clientY >= rect.top && e.clientY <= rect.bottom) {
            
            e.preventDefault();
            e.stopPropagation();
            
            if (ui.classList.contains('veritai-check-btn')) {
                ui.click();
            } else if (ui.classList.contains('veritai-status-badge') && ui.onclick) {
                ui.onclick(e);
            }
            return; 
        }
    }
}, true);