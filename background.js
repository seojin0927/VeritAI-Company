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
    sendResponse({ error: "알 수 없는 요청입니다." });
});