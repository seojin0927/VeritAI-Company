async function setupOffscreen() {
    if (await chrome.offscreen.hasDocument()) return;
    await chrome.offscreen.createDocument({
        url: 'offscreen.html',
        reasons: ['WORKERS'], 
        justification: '이미지 리사이징 및 처리 연산'
    });
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    
    if (request.action === "fetch_image") {
        fetch(request.url, { mode: 'cors' })
            .then(res => res.blob())
            .then(blob => {
                const reader = new FileReader();
                reader.onloadend = () => sendResponse({ dataUrl: reader.result });
                reader.readAsDataURL(blob);
            })
            .catch(err => {
                sendResponse({ error: "CORS fetch 실패: " + err.message });
            });
        return true; 
    } 
    
    else if (request.action === "resize_image") {
        setupOffscreen().then(() => {
            chrome.runtime.sendMessage(request, (response) => {
                sendResponse(response);
            });
        });
        return true; 
    }
    else {
        sendResponse({ error: "알 수 없는 요청입니다." });
    }
});