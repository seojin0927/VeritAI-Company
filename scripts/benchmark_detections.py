import argparse
import http.client
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
import uuid


BOUNDARY = "----VeritAIBenchmarkBoundary"


def build_multipart(image_path: Path, media_type: str, analysis_mode: str, source_url: str) -> tuple[bytes, str]:
    image = image_path.read_bytes()
    fields = [
        ("sourceUrl", source_url.encode("utf-8"), None),
        ("mediaType", media_type.encode("utf-8"), None),
        ("clientType", b"benchmark", None),
        ("analysisMode", analysis_mode.encode("utf-8"), None),
        ("file", image, "capture.jpg"),
    ]
    chunks: list[bytes] = []
    for name, value, filename in fields:
        chunks.append(f"--{BOUNDARY}\r\n".encode("ascii"))
        if filename:
            chunks.append(
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                "Content-Type: image/jpeg\r\n\r\n".encode("ascii")
            )
        else:
            chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii"))
        chunks.append(value)
        chunks.append(b"\r\n")
    chunks.append(f"--{BOUNDARY}--\r\n".encode("ascii"))
    return b"".join(chunks), f"multipart/form-data; boundary={BOUNDARY}"


def request_json(method: str, url: str, body: bytes | None = None, content_type: str | None = None) -> dict:
    parsed = urlparse(url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=30)
    headers = {}
    if content_type:
        headers["Content-Type"] = content_type
    if body is not None:
        headers["Content-Length"] = str(len(body))
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    try:
        parsed_body = json.loads(data.decode("utf-8"))
    except Exception:
        parsed_body = {"raw": data.decode("utf-8", errors="replace")}
    parsed_body["_httpStatus"] = response.status
    return parsed_body


def wait_result(base_url: str, request_id: int, initial_delay_ms: int = 1000, max_delay_ms: int = 5000) -> tuple[dict, int]:
    polls = 0
    delay_ms = initial_delay_ms
    started = time.time()
    while time.time() - started < 180:
        time.sleep(delay_ms / 1000.0)
        polls += 1
        data = request_json("GET", f"{base_url}/api/detections/{request_id}")
        if data.get("status") == "DONE" or data.get("status") == "FAILED":
            return data, polls
        server_delay = data.get("retryAfterMs")
        if isinstance(server_delay, (int, float)) and server_delay > 0:
            delay_ms = max(delay_ms, int(server_delay))
        delay_ms = min(max_delay_ms, int(delay_ms * 1.5))
    raise TimeoutError(f"request {request_id} timed out")


def wait_results_batch(base_url: str, request_ids: list[int], initial_delay_ms: int = 1000, max_delay_ms: int = 5000) -> tuple[dict[int, dict], int]:
    remaining = set(request_ids)
    completed: dict[int, dict] = {}
    polls = 0
    delay_ms = initial_delay_ms
    started = time.time()
    while remaining and time.time() - started < 180:
        time.sleep(delay_ms / 1000.0)
        polls += 1
        ids = ",".join(str(request_id) for request_id in sorted(remaining))
        data = request_json("GET", f"{base_url}/api/detections/status?ids={ids}")
        items = data.get("items", data if isinstance(data, list) else [])
        max_retry_after = 0
        for item in items:
            request_id = item.get("requestId")
            if request_id is None:
                continue
            retry_after = item.get("retryAfterMs")
            if isinstance(retry_after, (int, float)):
                max_retry_after = max(max_retry_after, int(retry_after))
            if item.get("status") in {"DONE", "FAILED"}:
                completed[int(request_id)] = item
                remaining.discard(int(request_id))
        if max_retry_after > 0:
            delay_ms = max(delay_ms, max_retry_after)
        delay_ms = min(max_delay_ms, int(delay_ms * 1.5))
    if remaining:
        raise TimeoutError(f"requests timed out: {sorted(remaining)[:10]}")
    return completed, polls


class BatchPoller:
    def __init__(self, base_url: str, initial_delay_ms: int = 1000, max_delay_ms: int = 5000):
        self.base_url = base_url
        self.initial_delay_ms = initial_delay_ms
        self.max_delay_ms = max_delay_ms
        self.lock = threading.Lock()
        self.pending: dict[int, dict] = {}
        self.running = False
        self.total_polls = 0

    def wait(self, request_id: int) -> dict:
        event = threading.Event()
        entry = {"event": event, "result": None}
        with self.lock:
            self.pending[request_id] = entry
            if not self.running:
                self.running = True
                thread = threading.Thread(target=self._run, daemon=True)
                thread.start()

        if not event.wait(180):
            with self.lock:
                self.pending.pop(request_id, None)
            raise TimeoutError(f"request {request_id} timed out")
        return entry["result"]

    def _run(self) -> None:
        delay_ms = self.initial_delay_ms
        while True:
            with self.lock:
                request_ids = sorted(self.pending.keys())
                if not request_ids:
                    self.running = False
                    return

            time.sleep(delay_ms / 1000.0)
            ids = ",".join(str(request_id) for request_id in request_ids)
            data = request_json("GET", f"{self.base_url}/api/detections/status?ids={ids}")
            with self.lock:
                self.total_polls += 1

            items = data.get("items", data if isinstance(data, list) else [])
            max_retry_after = 0
            completed_count = 0
            for item in items:
                request_id = item.get("requestId")
                if request_id is None:
                    continue
                retry_after = item.get("retryAfterMs")
                if isinstance(retry_after, (int, float)):
                    max_retry_after = max(max_retry_after, int(retry_after))
                if item.get("status") in {"DONE", "FAILED"}:
                    with self.lock:
                        entry = self.pending.pop(int(request_id), None)
                    if entry is not None:
                        entry["result"] = item
                        entry["event"].set()
                        completed_count += 1

            if completed_count > 0:
                delay_ms = self.initial_delay_ms
            elif max_retry_after > 0:
                delay_ms = min(self.max_delay_ms, max(self.initial_delay_ms, max_retry_after))
            else:
                delay_ms = min(self.max_delay_ms, int(delay_ms * 1.25))


def submit_one(base_url: str, payload: bytes, content_type: str, poll_backoff: bool) -> dict:
    started = time.time()
    data = request_json("POST", f"{base_url}/api/detections", payload, content_type)
    submit_ms = int((time.time() - started) * 1000)
    request_id = data.get("requestId")
    polls = 0
    final = data
    if data.get("status") in {"QUEUED", "PROCESSING"} and request_id is not None:
        if poll_backoff:
            final, polls = wait_result(base_url, int(request_id))
        else:
            # Baseline-like polling: 1s fixed interval.
            wait_started = time.time()
            while time.time() - wait_started < 180:
                time.sleep(1.0)
                polls += 1
                final = request_json("GET", f"{base_url}/api/detections/{request_id}")
                if final.get("status") in {"DONE", "FAILED"}:
                    break
    total_ms = int((time.time() - started) * 1000)
    return {
        "requestId": request_id,
        "status": final.get("status"),
        "submitMs": submit_ms,
        "totalMs": total_ms,
        "polls": polls,
        "httpStatus": data.get("_httpStatus"),
    }


def submit_one_with_batch_poll(base_url: str, payload: bytes, content_type: str, poller: BatchPoller) -> dict:
    started = time.time()
    data = request_json("POST", f"{base_url}/api/detections", payload, content_type)
    submit_ms = int((time.time() - started) * 1000)
    request_id = data.get("requestId")
    final = data
    if data.get("status") in {"QUEUED", "PROCESSING"} and request_id is not None:
        final = poller.wait(int(request_id))
    total_ms = int((time.time() - started) * 1000)
    return {
        "requestId": request_id,
        "status": final.get("status"),
        "submitMs": submit_ms,
        "totalMs": total_ms,
        "polls": 0,
        "httpStatus": data.get("_httpStatus"),
    }


def submit_only(base_url: str, payload: bytes, content_type: str) -> dict:
    started = time.time()
    data = request_json("POST", f"{base_url}/api/detections", payload, content_type)
    submit_ms = int((time.time() - started) * 1000)
    return {
        "requestId": data.get("requestId"),
        "status": data.get("status"),
        "submitMs": submit_ms,
        "totalMs": submit_ms if data.get("status") == "DONE" else 0,
        "polls": 0,
        "httpStatus": data.get("_httpStatus"),
        "submittedAt": started,
    }


def percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return float(ordered[index])


def summarize(results: list[dict], elapsed_ms: int) -> dict:
    totals = [r["totalMs"] for r in results]
    submits = [r["submitMs"] for r in results]
    request_ids = [r["requestId"] for r in results if r.get("requestId") is not None]
    return {
        "count": len(results),
        "elapsedMs": elapsed_ms,
        "done": sum(1 for r in results if r.get("status") == "DONE"),
        "failed": sum(1 for r in results if r.get("status") == "FAILED"),
        "uniqueRequestIds": len(set(request_ids)),
        "totalMsAvg": round(statistics.mean(totals), 2) if totals else 0,
        "totalMsP50": percentile(totals, 0.5),
        "totalMsP95": percentile(totals, 0.95),
        "submitMsAvg": round(statistics.mean(submits), 2) if submits else 0,
        "pollsTotal": sum(r.get("polls", 0) for r in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--image")
    parser.add_argument("--image-dir")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--media-type", default="image")
    parser.add_argument("--analysis-mode", default="face_crop_only")
    parser.add_argument("--poll-backoff", action="store_true")
    parser.add_argument("--batch-poll", action="store_true")
    args = parser.parse_args()

    if args.image_dir:
        image_paths = [
            path for path in sorted(Path(args.image_dir).rglob("*"))
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
        if not image_paths:
            raise SystemExit(f"no images found in {args.image_dir}")
        selected_paths = [image_paths[index % len(image_paths)] for index in range(args.count)]
    elif args.image:
        selected_paths = [Path(args.image) for _ in range(args.count)]
    else:
        raise SystemExit("--image or --image-dir is required")

    payloads = [
        build_multipart(path, args.media_type, args.analysis_mode, f"benchmark://{uuid.uuid4().hex}/{index}")
        for index, path in enumerate(selected_paths)
    ]

    started = time.time()
    results: list[dict] = []
    if args.batch_poll:
        poller = BatchPoller(args.base_url)
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(submit_one_with_batch_poll, args.base_url, payload, content_type, poller)
                for payload, content_type in payloads
            ]
            for future in as_completed(futures):
                results.append(future.result())
        if results:
            per_result_polls = poller.total_polls / len(results)
            for result in results:
                result["polls"] = per_result_polls
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(submit_one, args.base_url, payload, content_type, args.poll_backoff)
                for payload, content_type in payloads
            ]
            for future in as_completed(futures):
                results.append(future.result())
    elapsed_ms = int((time.time() - started) * 1000)
    print(json.dumps({"summary": summarize(results, elapsed_ms), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
