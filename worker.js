chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "resize_image") {
        
        (async () => {
            try {
                const response = await fetch(request.url);
                const blob = await response.blob();
                const imageBitmap = await createImageBitmap(blob);

                const width = 500;
                const height = (imageBitmap.height / imageBitmap.width) * width;

                const offscreen = new OffscreenCanvas(width, height);
                const ctx = offscreen.getContext('2d');
                ctx.drawImage(imageBitmap, 0, 0, width, height);

                const resizedBlob = await offscreen.convertToBlob({ type: 'image/webp', quality: 0.8 });
                imageBitmap.close();

                const reader = new FileReader();
                reader.readAsDataURL(resizedBlob);
                reader.onloadend = () => {
                    sendResponse({ success: true, base64: reader.result });
                };

            } catch (error) {
                sendResponse({ success: false, error: error.message });
            }
        })();
        return true; 
    }
});