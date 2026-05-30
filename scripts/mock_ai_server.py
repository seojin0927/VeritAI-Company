from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import time


DELAY_MS = int(os.environ.get("VERITAI_MOCK_AI_DELAY_MS", "250"))


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length:
            self.rfile.read(length)

        started = time.time()
        if DELAY_MS > 0:
            time.sleep(DELAY_MS / 1000.0)
        elapsed_ms = int((time.time() - started) * 1000)
        payload = {
            "isDeepfake": False,
            "confidence": 0.0,
            "faceCount": 1,
            "watermarkDetected": False,
            "modelVersion": "mock-ai",
            "analysisMode": "face_crop_only",
            "analysisInput": {
                "detectionImage": "full_image",
                "featureImage": "cropped_face",
                "deepfakeImage": "cropped_face",
            },
            "timings": {
                "mockDelayMs": DELAY_MS,
                "totalTimeMs": elapsed_ms,
            },
            "processingTimeMs": elapsed_ms,
            "message": "mock ai response",
            "faces": [],
            "cnn": {"modelLoaded": False},
            "debugImages": {},
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


if __name__ == "__main__":
    port = int(os.environ.get("VERITAI_MOCK_AI_PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"mock ai server listening on {port}, delay={DELAY_MS}ms", flush=True)
    server.serve_forever()
