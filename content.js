console.log("VeritAI content script loaded");
const API_URL = "http://localhost:8080/api/detections";

let isSystemOn = true;
let isAutoScanMode = false;
const FACE_CROP_ANALYSIS_MODE = "face_crop_only";
const scannedMediaKeys = new Set();

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
    return isVisibleMedia(media);
}

function readDeepfakeFlag(result) {
    if (!result) return false;
    return Boolean(result.isDeepfake ?? result.deepfake);
}

function updateStatusBadge(media, status, data = null) {
    const wrapper = ensureWrapper(media);
    if (!wrapper) return;

    let uiContainer = wrapper.querySelector('.veritai-ui-container');
    if (!uiContainer) {
        uiContainer = document.createElement('div');
        uiContainer.className = 'veritai-ui-container';
        uiContainer.style.cssText = `
            position: absolute; top: 6px; left: 6px; z-index: 2147483647;
            display: flex; flex-direction: column; align-items: flex-start;
        `;
        wrapper.appendChild(uiContainer);
    }

    let badge = uiContainer.querySelector('.veritai-status-badge');
    if (!badge) {
        badge = document.createElement('div');
        badge.className = 'veritai-status-badge';
        badge.style.cssText = `
            padding: 4px 8px; border-radius: 4px; color: white; font-size: 11px; 
            font-weight: bold; font-family: sans-serif; box-shadow: 0 2px 4px rgba(0,0,0,0.5);
            transition: all 0.2s ease; user-select: none;
        `;
        uiContainer.appendChild(badge);
    }

    badge.onclick = null;
    badge.style.cursor = "default";
    media.style.border = "none";

    if (status === "loading") {
        badge.innerText = "분석 중...";
        badge.style.background = "blue";
    }
    else if (status === "error") {
        const errorMsg = data?.message || "분석 실패";
        badge.innerText = errorMsg;
        badge.style.background = "dimgray";
        
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
            badge.style.background = "red";
            media.style.border = "2px solid red";
        } else {
            badge.innerText = "정상 이미지";
            badge.style.background = "green";
            media.style.border = "2px solid green";
        }

        badge.onclick = (e) => {
            e.preventDefault();
            e.stopPropagation();

            const existingBox = document.querySelector('.veritai-details-box');
            if (existingBox) {
                existingBox.remove();
                if (existingBox.dataset.targetMedia === (media.currentSrc || media.src)) return;
            }

            const result = data.result;
            const faces = result.faces || [];
            const faceText = faces.length === 0 ? "검출된 얼굴 없음" :
                faces.slice(0, 3).map((f, i) => {
                    const bbox = f.bbox || {};
                    const quality = f.quality || {};
                    const detConf = ((f.detectionConfidence || f.score || 0) * 100).toFixed(1);
                    const qualScore = ((quality.score || 0) * 100).toFixed(1);
                    return `<span style="color:yellow; font-weight:bold;">[얼굴 ${i + 1}]</span>
 - 유형: ${f.faceMode || '?'}
 - 검출 신뢰도: ${detConf}%
 - 위치: (${bbox.x || '?'}, ${bbox.y || '?'}, ${bbox.w || '?'}x${bbox.h || '?'})
 - 품질: ${quality.label || '?'} (${qualScore}%)`;
                }).join("\n\n");

            const detailsBox = document.createElement('div');
            detailsBox.className = 'veritai-details-box';
            detailsBox.dataset.targetMedia = media.currentSrc || media.src; 

            const badgeRect = badge.getBoundingClientRect();

            Object.assign(detailsBox.style, {
                position: "fixed", 
                top: `${badgeRect.bottom + 5}px`,
                left: `${badgeRect.left}px`,
                zIndex: "2147483647",
                background: "black",
                backdropFilter: "blur(10px)",
                color: "white",
                padding: "15px",
                borderRadius: "12px",
                border: `1px solid ${status === "fake" ? "red" : "green"}`,
                fontSize: "12px",
                whiteSpace: "pre-wrap",
                lineHeight: "1.6",
                boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.7)",
                fontFamily: "monospace",
                width: "280px",
                maxHeight: "400px",
                overflowY: "auto",
                textAlign: "left",
                cursor: "default",
                pointerEvents: "auto"
            });

            detailsBox.innerHTML = `
                <div style="color:lightskyblue; font-weight:bold; margin-bottom:10px; border-bottom:1px solid grey; padding-bottom:6px; font-size:14px; display:flex; justify-content:space-between;">
                    <span>🔍 분석 리포트</span>
                    <span style="cursor:pointer; color:gray;" onclick="this.closest('.veritai-details-box').remove()">✕</span>
</div>
<b>ID:</b> ${data.requestId}
<b>판정:</b> ${readDeepfakeFlag(result) ? "<span style='color:crimson; font-weight:bold;'>조작 의심</span>" : "<span style='color:green; font-weight:bold;'>정상</span>"} (${((result.confidence || 0) * 100).toFixed(1)}%)
<b>시간:</b> ${result.processingTimeMs}ms
<b>얼굴 수:</b> ${result.faceCount || faces.length}명
<div style="margin:10px 0; border-top:1px dashed grey;"></div>
${faceText}
            `.trim();

            detailsBox.onclick = (evt) => evt.stopPropagation();
            document.body.appendChild(detailsBox);

            setTimeout(() => {
                const closeDetails = (evt) => {
                    if (!detailsBox.contains(evt.target) && !badge.contains(evt.target)) {
                        detailsBox.remove();
                        document.removeEventListener('click', closeDetails);
                    }
                };
                document.addEventListener('click', closeDetails);
            }, 10);
        };
    }
}

async function startInspection(media) {
    if (!isSystemOn || !shouldInspectMedia(media)) return;

    const scanKey = getMediaKey(media);
    if (media.dataset.veritaiScanned === "true" || scannedMediaKeys.has(scanKey)) return;
    scannedMediaKeys.add(scanKey);
    media.dataset.veritaiScanned = "true";
    media.dataset.veritaiScanKey = scanKey;

    const wrapper = ensureWrapper(media);
    if (wrapper) {
        const btn = wrapper.querySelector('.veritai-check-btn');
        if (btn) btn.remove();
    }

    try {
        updateStatusBadge(media, "loading");

        let blob;
        let mediaType = "image";
        if (media.tagName === "VIDEO") {
            blob = await captureVideoBlob(media);
            mediaType = "video_frame";
        } else {
            blob = await captureImageBlob(media.currentSrc || media.src);
        }

        const data = await sendToBackend(blob, mediaType, FACE_CROP_ANALYSIS_MODE);
        
        if (readDeepfakeFlag(data.result)) {
            updateStatusBadge(media, "fake", data);
        } else {
            updateStatusBadge(media, "real", data);
        }

    } catch (err) {
        console.error("Analysis Error:", err);
        let friendlyMessage = "Analysis error";
        if (err.name === 'TypeError' && err.message === 'Failed to fetch') {
            friendlyMessage = "Server connection failed";
        }
        else if (err.message && err.message.includes("Server response error")) {
            friendlyMessage = "Server response error";
        }
        else if (err.message && err.message.includes("CORS")) {
            friendlyMessage = "Blocked by browser security";
        }
        updateStatusBadge(media, "error", { message: friendlyMessage });
        delete media.dataset.veritaiScanned;
        scannedMediaKeys.delete(scanKey);
    }
}

const autoScanObserver = new IntersectionObserver((entries) => {
    if (!isSystemOn || !isAutoScanMode) return;
    entries.forEach(entry => {
        if (entry.isIntersecting && shouldInspectMedia(entry.target)) {
            startInspection(entry.target);
        }
    });
}, { threshold: 0.1 });

const domObserver = new MutationObserver((mutations) => {
    if (!isSystemOn) return;
    mutations.forEach(mutation => {
        if (mutation.type === "attributes") {
            const node = mutation.target;
            if (node.nodeType !== 1) return;
            if (node.tagName === 'IMG' || node.tagName === 'VIDEO') {
                attachUI(node);
            } else {
                node.querySelectorAll?.('img, video').forEach(media => attachUI(media));
            }
            return;
        }

        mutation.addedNodes.forEach(node => {
            if (node.nodeType !== 1) return; 
            if (node.tagName === 'IMG' || node.tagName === 'VIDEO') {
                attachUI(node);
            } else {
                node.querySelectorAll('img, video').forEach(media => attachUI(media));
            }
        });
    });
});

function ensureWrapper(media) {
    let parent = media.parentElement;
    if (!parent) return null;

    if (parent.tagName === 'PICTURE') {
        parent = parent.parentElement;
    }

    if (getComputedStyle(parent).position === "static") {
        parent.style.position = "relative";
    }
    return parent;
}

function attachUI(media) {
    if (media.dataset.veritaiAttached || media.clientWidth < 80 || media.clientHeight < 80) return;
    if (!shouldInspectMedia(media)) {
        delete media.dataset.veritaiAttached;
        return;
    }
    media.dataset.veritaiAttached = "true";

    if (isAutoScanMode) {
        autoScanObserver.observe(media);
    } else {
        const wrapper = ensureWrapper(media);
        if (wrapper && !wrapper.querySelector('.veritai-check-btn')) {
            const btn = document.createElement("button");
            btn.innerText = "🔍 검사";
            btn.className = "veritai-check-btn";
            btn.style.cssText = `
                position: absolute; top: 10px; left: 10px; z-index: 2147483647;
                padding: 4px 8px; background-color: midnightblue; color: aqua;
                border: 1px solid aqua; border-radius: 999px; cursor: pointer;
                font-weight: bold; font-size: 11px; backdrop-filter: blur(2px);
            `;

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
        clearAllUI();
        if (isSystemOn) {
            document.querySelectorAll('img, video').forEach(media => attachUI(media));
            domObserver.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["class", "style", "open", "aria-hidden", "aria-modal"] });
        } else {
            autoScanObserver.disconnect();
            domObserver.disconnect();
        }
    }
});

function clearAllUI() {
    scannedMediaKeys.clear();
    document.querySelectorAll('img, video').forEach(media => {
        media.style.border = "none";
        delete media.dataset.veritaiScanned;
        delete media.dataset.veritaiAttached;
        delete media.dataset.veritaiScanKey;
        const wrapper = media.parentElement;
        if (wrapper) {
            const container = wrapper.querySelector('.veritai-ui-container');
            if (container) container.remove();
            const btn = wrapper.querySelector('.veritai-check-btn');
            if (btn) btn.remove();
        }
    });
    document.querySelectorAll('.veritai-details-box').forEach(box => box.remove());
}

chrome.storage.local.get(['isSystemOn', 'isAutoScanOn'], (result) => {
    isSystemOn = result.isSystemOn !== false;
    isAutoScanMode = result.isAutoScanOn || false;
    setTimeout(() => {
        if (isSystemOn) {
            document.querySelectorAll('img, video').forEach(media => attachUI(media));
            domObserver.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["class", "style", "open", "aria-hidden", "aria-modal"] });
        }
    }, 500);
});

async function captureVideoBlob(video) {
    if (!video) throw new Error("영상 요소를 찾을 수 없습니다.");
    const width = video.videoWidth || video.clientWidth;
    const height = video.videoHeight || video.clientHeight;
    if (width === 0 || height === 0) throw new Error("영상 크기를 인식할 수 없습니다.");

    return new Promise((resolve, reject) => {
        try {
            const canvas = document.createElement("canvas");
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext("2d");
            if (!ctx) return reject(new Error("캔버스 컨텍스트를 생성하지 못했습니다."));
            ctx.drawImage(video, 0, 0, width, height);
            canvas.toBlob((blob) => {
                if (!blob) return reject(new Error("영상 프레임 데이터를 생성하지 못했습니다."));
                resolve(blob);
            }, "image/jpeg", 0.9);
        } catch (error) {
            reject(new Error("비디오 프레임에 접근할 수 없습니다 (CORS 보안)."));
        }
    });
}

async function captureImageBlob(url) {
    if (!url) throw new Error("이미지 주소가 없습니다.");
    return new Promise((resolve, reject) => {
        chrome.runtime.sendMessage({ action: "fetch_image", url: url }, (response) => {
            if (chrome.runtime.lastError || !response || response.error) {
                return reject(new Error("이미지를 불러오지 못했습니다 (보안 차단됨)."));
            }
            const img = new Image();
            img.onload = () => {
                try {
                    const canvas = document.createElement("canvas");
                    const ctx = canvas.getContext("2d");
                    if (!ctx) return reject(new Error("캔버스 컨텍스트를 생성하지 못했습니다."));
                    canvas.width = img.naturalWidth || img.width;
                    canvas.height = img.naturalHeight || img.height;
                    if (canvas.width === 0 || canvas.height === 0)
                        return reject(new Error("이미지 크기가 0입니다."));
                    ctx.drawImage(img, 0, 0);
                    canvas.toBlob((blob) => {
                        if (!blob) return reject(new Error("이미지 데이터를 생성하지 못했습니다."));
                        resolve(blob);
                    }, "image/jpeg", 0.9);
                } catch (error) { reject(error); }
            };
            img.onerror = () => reject(new Error("가져온 이미지를 렌더링하지 못했습니다."));
            img.src = response.dataUrl;
        });
    });
}

async function sendToBackend(blob, mediaType, analysisMode = FACE_CROP_ANALYSIS_MODE) {
    const formData = new FormData();
    formData.append("file", blob, "capture.jpg");
    formData.append("sourceUrl", window.location.href);
    formData.append("mediaType", mediaType);
    formData.append("clientType", "chrome-extension");
    formData.append("analysisMode", analysisMode);

    const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) throw new Error(`Server response error: ${response.status}`);
    const data = await response.json();
    if (!data || data.status !== "DONE" || !data.result) {
        throw new Error(data?.message || "분석이 정상적으로 완료되지 않았습니다.");
    }
    return data;
}
