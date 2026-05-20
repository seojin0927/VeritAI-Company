const systemToggle = document.getElementById('system-toggle');
const autoToggle = document.getElementById('auto-toggle');
const statusText = document.getElementById('status-text');

chrome.storage.local.get(['isSystemOn', 'isAutoScanOn'], (res) => {
    const sysOn = res.isSystemOn !== false;
    const autoOn = !!res.isAutoScanOn;

    systemToggle.checked = sysOn;
    autoToggle.checked = autoOn;
    
    updateUI();
});

function updateUI() {
    const sysOn = systemToggle.checked;
    const autoOn = autoToggle.checked;

    autoToggle.disabled = !sysOn;
    statusText.innerText = !sysOn ? "시스템 전원 OFF" : (autoOn ? "자동 검사 모드" : "수동 검사 모드");
    statusText.style.color = !sysOn ? "dimgrey" : "lightskyblue";
}

function syncState() {
    updateUI();
    const state = { 
        isSystemOn: systemToggle.checked, 
        isAutoScanOn: autoToggle.checked 
    };
    
    chrome.storage.local.set(state);
    
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs[0]?.id) {
            chrome.tabs.sendMessage(tabs[0].id, { action: "TOGGLE_SYSTEM", ...state });
        }
    });
}

systemToggle.addEventListener('change', syncState);
autoToggle.addEventListener('change', syncState);

document.getElementById('feedback-link').addEventListener('click', () => {
    const feedbackUrl = "https://forms.gle/실제_주소"; 
    
    chrome.tabs.create({ url: feedbackUrl });
});