import base64
import io
import json
import logging
import os
import sqlite3
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import numpy as np
import cv2
import cloudinary
import cloudinary.uploader
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT / "backend" / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = ROOT / "backend" / "accidents.db"
CAMERA_CONFIG_PATH = ROOT / "backend" / "camera_sources.json"
load_dotenv(ROOT / ".env")

DETECTION_MODE = os.getenv("DETECTION_MODE", "roboflow_cloud").strip().lower()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "road-accident-2vton/4")
ROBOFLOW_CONFIDENCE = os.getenv("ROBOFLOW_CONFIDENCE", "40")

LOCAL_INFERENCE_URL = os.getenv(
    "LOCAL_INFERENCE_URL",
    "http://localhost:9001"
)

LOCAL_CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key=ROBOFLOW_API_KEY,
)

OFFLINE_MODEL_PATH = os.getenv("OFFLINE_MODEL_PATH", "models/road-accident.onnx")
OFFLINE_INPUT_SIZE = int(os.getenv("OFFLINE_INPUT_SIZE", "640"))
OFFLINE_CONFIDENCE = float(os.getenv("OFFLINE_CONFIDENCE", "0.35"))
OFFLINE_IOU = float(os.getenv("OFFLINE_IOU", "0.45"))
MODEL_CLASS_NAMES = [
    name.strip() for name in os.getenv("MODEL_CLASS_NAMES", "accident").split(",") if name.strip()
]

ACCIDENT_CLASSES = {
    c.strip().lower() for c in os.getenv("ACCIDENT_CLASSES", "accident,crash,collision").split(",") if c.strip()
}

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
)

app = FastAPI(title="AI Accident Detection Dashboard API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ONNX_SESSION = None
_ROBOFLOW_SESSION = requests.Session()
_ROBOFLOW_SESSION.headers.update({"Connection": "keep-alive"})
LOGGER = logging.getLogger(__name__)
ANALYSIS_SERIAL_LOCK = threading.Lock()


class CameraConfigUpdate(BaseModel):
    source_url: str = Field(default="", description="RTSP, HTTP, or other OpenCV-compatible source URL")
    label: str | None = Field(default=None, description="Optional display label")


def normalize_source_url(source_url: str) -> str:
    cleaned = source_url.strip()
    if not cleaned:
        return ""

    if re.match(r"^(https?|rtsp|rtsps):[^/]", cleaned):
        scheme, remainder = cleaned.split(":", 1)
        return f"{scheme}://{remainder.lstrip('/')}"

    return cleaned


class CameraStream:
    def __init__(self, camera_id: str, source_url: str = "", label: str = "") -> None:
        self.camera_id = camera_id
        self.source_url = source_url.strip()
        self.label = label.strip()
        self.last_error = ""
        self.last_frame_jpeg: bytes | None = None
        self.last_frame_at: float | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, source_url: str, label: str | None = None) -> None:
        with self._lock:
            self.source_url = normalize_source_url(source_url)
            if label is not None:
                self.label = label.strip()

    def snapshot(self) -> bytes | None:
        with self._lock:
            return self.last_frame_jpeg

    def _open_capture(self, source_url: str):
        import cv2

        if not hasattr(cv2, "VideoCapture"):
            raise RuntimeError(
                "OpenCV is installed, but this Python environment does not expose cv2.VideoCapture. "
                "Reinstall opencv-python-headless inside the active venv."
            )

        normalized_source = normalize_source_url(source_url)
        backend_candidates: list[tuple[Any, Any | None]] = []

        if normalized_source.isdigit():
            backend_candidates.append((int(normalized_source), None))
            if hasattr(cv2, "CAP_V4L2"):
                backend_candidates.append((int(normalized_source), cv2.CAP_V4L2))
        else:
            backend_candidates.append((normalized_source, cv2.CAP_FFMPEG))
            if hasattr(cv2, "CAP_GSTREAMER"):
                backend_candidates.append((normalized_source, cv2.CAP_GSTREAMER))
            backend_candidates.append((normalized_source, None))

        last_capture = None
        for candidate_source, backend in backend_candidates:
            capture = cv2.VideoCapture(candidate_source) if backend is None else cv2.VideoCapture(candidate_source, backend)
            last_capture = capture
            if capture and capture.isOpened():
                try:
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                return capture

            try:
                capture.release()
            except Exception:
                pass

        return last_capture

    def _run(self) -> None:
        backoff_seconds = 1.0
        while not self._stop_event.is_set():
            with self._lock:
                source_url = self.source_url

            if not source_url:
                with self._lock:
                    self.last_error = "No source configured"
                time.sleep(0.5)
                continue

            try:
                capture = self._open_capture(source_url)
            except Exception as exc:  # pragma: no cover - hardware/backend dependent
                with self._lock:
                    self.last_error = f"Open error: {exc}"
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 1.5, 5.0)
                continue

            if not capture or not capture.isOpened():
                with self._lock:
                    self.last_error = f"Could not open source: {source_url}"
                try:
                    capture.release()
                except Exception:
                    pass
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 1.5, 5.0)
                continue

            backoff_seconds = 1.0
            with self._lock:
                self.last_error = ""

            try:
                import cv2

                while not self._stop_event.is_set():
                    with self._lock:
                        current_source = self.source_url

                    if current_source != source_url:
                        break

                    ok, frame = capture.read()
                    if not ok or frame is None:
                        with self._lock:
                            self.last_error = f"Failed to read frame from {source_url}"
                        break

                    success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if success:
                        jpeg_bytes = buffer.tobytes()
                        with self._lock:
                            self.last_frame_jpeg = jpeg_bytes
                            self.last_frame_at = time.time()
                            self.last_error = ""
                    else:
                        with self._lock:
                            self.last_error = f"Failed to encode frame from {source_url}"

                    time.sleep(0.02)
            finally:
                try:
                    capture.release()
                except Exception:
                    pass

            time.sleep(0.25)


class CameraManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._configs = self._load_configs()
        self._streams: dict[str, CameraStream] = {}

    def _default_configs(self) -> dict[str, dict[str, str]]:
        return {
            "int-1": {"source_url": "", "label": "Main St & 1st Ave"},
            "int-2": {"source_url": "", "label": "Central Blvd & Lake Rd"},
            "int-3": {"source_url": "", "label": "Airport Road Junction"},
            "int-4": {"source_url": "", "label": "Market Square Cross"},
            "int-5": {"source_url": "", "label": "River Bridge Exit"},
            "int-6": {"source_url": "", "label": "Highway 4 Flyover"},
            "int-7": {"source_url": "", "label": "Industrial Gate Loop"},
            "int-8": {"source_url": "", "label": "University Circle"},
            "int-9": {"source_url": "", "label": "Bus Terminal North"},
        }

    def _load_configs(self) -> dict[str, dict[str, str]]:
        if not CAMERA_CONFIG_PATH.exists():
            return self._default_configs()

        try:
            payload = json.loads(CAMERA_CONFIG_PATH.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return self._default_configs()
        except Exception:
            return self._default_configs()

        defaults = self._default_configs()
        for camera_id, config in payload.items():
            if camera_id not in defaults or not isinstance(config, dict):
                continue
            defaults[camera_id]["source_url"] = str(config.get("source_url", "")).strip()
            defaults[camera_id]["label"] = str(config.get("label", defaults[camera_id]["label"])).strip() or defaults[camera_id]["label"]
        return defaults

    def _save_configs(self) -> None:
        CAMERA_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CAMERA_CONFIG_PATH.write_text(json.dumps(self._configs, indent=2), encoding="utf-8")

    def list_cameras(self) -> list[dict[str, Any]]:
        with self._lock:
            items = []
            for camera_id, config in self._configs.items():
                stream = self._streams.get(camera_id)
                items.append(
                    {
                        "id": camera_id,
                        "label": config.get("label", camera_id),
                        "source_url": config.get("source_url", ""),
                        "active": bool(config.get("source_url")),
                        "has_frame": bool(stream and stream.snapshot()),
                        "last_error": stream.last_error if stream else "",
                    }
                )
            return items

    def get_camera(self, camera_id: str) -> dict[str, str]:
        with self._lock:
            if camera_id not in self._configs:
                raise HTTPException(status_code=404, detail=f"Unknown camera id: {camera_id}")
            return dict(self._configs[camera_id])

    def set_camera(self, camera_id: str, source_url: str, label: str | None = None) -> dict[str, str]:
        with self._lock:
            if camera_id not in self._configs:
                raise HTTPException(status_code=404, detail=f"Unknown camera id: {camera_id}")

            current = self._configs[camera_id]
            current["source_url"] = normalize_source_url(source_url)
            if label is not None and label.strip():
                current["label"] = label.strip()

            stream = self._streams.get(camera_id)
            if stream is None:
                stream = CameraStream(camera_id, current["source_url"], current["label"])
                self._streams[camera_id] = stream
            else:
                stream.update(current["source_url"], current["label"])

            self._save_configs()
            return dict(current)

    def clear_camera(self, camera_id: str) -> None:
        self.set_camera(camera_id, "")

    def ensure_stream(self, camera_id: str) -> CameraStream:
        with self._lock:
            config = self._configs.get(camera_id)
            if config is None:
                raise HTTPException(status_code=404, detail=f"Unknown camera id: {camera_id}")

            stream = self._streams.get(camera_id)
            if stream is None:
                stream = CameraStream(camera_id, config.get("source_url", ""), config.get("label", camera_id))
                self._streams[camera_id] = stream
            return stream

    def latest_frame(self, camera_id: str) -> bytes | None:
        stream = self.ensure_stream(camera_id)
        return stream.snapshot()

    def stream_response(self, camera_id: str) -> StreamingResponse:
        stream = self.ensure_stream(camera_id)

        def frame_iterator():
            boundary = b"--frame\r\n"
            while True:
                frame = stream.snapshot()
                if frame is None:
                    time.sleep(0.25)
                    continue
                yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                time.sleep(0.08)

        return StreamingResponse(frame_iterator(), media_type="multipart/x-mixed-replace; boundary=frame")


CAMERA_MANAGER = CameraManager()


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS accident_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source_name TEXT,
                confidence REAL,
                class_name TEXT,
                bbox_json TEXT,
                camera_id TEXT,
                image_path TEXT,
                image_url TEXT,
                ai_summary TEXT,
                severity TEXT,
                camera_location TEXT,
                emergency_status TEXT,
                raw_prediction_json TEXT
            )
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(accident_events)").fetchall()}
        for column_sql, column_name in [
            ("ALTER TABLE accident_events ADD COLUMN image_url TEXT", "image_url"),
            ("ALTER TABLE accident_events ADD COLUMN camera_id TEXT", "camera_id"),
            ("ALTER TABLE accident_events ADD COLUMN ai_summary TEXT", "ai_summary"),
            ("ALTER TABLE accident_events ADD COLUMN severity TEXT", "severity"),
            ("ALTER TABLE accident_events ADD COLUMN camera_location TEXT", "camera_location"),
            ("ALTER TABLE accident_events ADD COLUMN emergency_status TEXT", "emergency_status"),
        ]:
            if column_name not in columns:
                conn.execute(column_sql)
        conn.commit()


def roboflow_cloud_predict(image_bytes: bytes) -> dict[str, Any]:
    try:

        # Convert bytes -> numpy array
        np_arr = np.frombuffer(image_bytes, np.uint8)

        # Decode image
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to decode image"
            )

        # Run local Jetson inference
        result = LOCAL_CLIENT.infer(
            image,
            model_id=ROBOFLOW_MODEL_ID
        )

        print("LOCAL INFERENCE RESULT:", result)

        if not isinstance(result, dict):
            return {"predictions": []}

        if "predictions" not in result:
            result["predictions"] = []

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Local Jetson inference failed: {exc}"
        ) from exc


def get_onnx_session() -> Any:
    global _ONNX_SESSION
    if _ONNX_SESSION is not None:
        return _ONNX_SESSION

    model_path = ROOT / OFFLINE_MODEL_PATH
    if not model_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Offline model file not found at {model_path}. Export ONNX and place it there.",
        )

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="onnxruntime is not installed. Run: pip install onnxruntime numpy pillow",
        ) from exc

    _ONNX_SESSION = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return _ONNX_SESSION


def _nms(boxes: list[list[float]], scores: list[float], iou_thresh: float) -> list[int]:
    if not boxes:
        return []

    x1 = [b[0] for b in boxes]
    y1 = [b[1] for b in boxes]
    x2 = [b[2] for b in boxes]
    y2 = [b[3] for b in boxes]
    areas = [(x2[i] - x1[i]) * (y2[i] - y1[i]) for i in range(len(boxes))]
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        remaining = []
        for j in order:
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[j] - inter
            iou = inter / union if union > 0 else 0
            if iou < iou_thresh:
                remaining.append(j)
        order = remaining
    return keep


def onnx_offline_predict(image_bytes: bytes) -> dict[str, Any]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Offline mode requires numpy and pillow. Run: pip install numpy pillow",
        ) from exc

    session = get_onnx_session()
    input_info = session.get_inputs()[0]
    input_name = input_info.name

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = image.size

    target_w = OFFLINE_INPUT_SIZE
    target_h = OFFLINE_INPUT_SIZE

    scale = min(target_w / orig_w, target_h / orig_h)
    resized_w = int(orig_w * scale)
    resized_h = int(orig_h * scale)

    resized = image.resize((resized_w, resized_h))
    canvas = Image.new("RGB", (target_w, target_h), (114, 114, 114))
    pad_x = (target_w - resized_w) // 2
    pad_y = (target_h - resized_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    input_tensor = np.asarray(canvas, dtype=np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))[None, :, :, :]

    outputs = session.run(None, {input_name: input_tensor})
    if not outputs:
        return {"predictions": []}

    out = outputs[0]
    if out.ndim != 3:
        return {"predictions": []}

    pred = out[0]
    if pred.shape[0] <= 128 and pred.shape[1] > pred.shape[0]:
        pred = pred.T

    if pred.shape[1] < 6:
        return {"predictions": []}

    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]

    keep_mask = confidences >= OFFLINE_CONFIDENCE
    boxes_xywh = boxes_xywh[keep_mask]
    class_ids = class_ids[keep_mask]
    confidences = confidences[keep_mask]

    if boxes_xywh.shape[0] == 0:
        return {"predictions": []}

    boxes_xyxy: list[list[float]] = []
    for cx, cy, bw, bh in boxes_xywh.tolist():
        x1 = (cx - bw / 2.0 - pad_x) / scale
        y1 = (cy - bh / 2.0 - pad_y) / scale
        x2 = (cx + bw / 2.0 - pad_x) / scale
        y2 = (cy + bh / 2.0 - pad_y) / scale

        x1 = max(0.0, min(float(orig_w), x1))
        y1 = max(0.0, min(float(orig_h), y1))
        x2 = max(0.0, min(float(orig_w), x2))
        y2 = max(0.0, min(float(orig_h), y2))
        boxes_xyxy.append([x1, y1, x2, y2])

    keep_indices = _nms(boxes_xyxy, confidences.tolist(), OFFLINE_IOU)

    predictions = []
    for i in keep_indices:
        x1, y1, x2, y2 = boxes_xyxy[i]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        cx = x1 + width / 2.0
        cy = y1 + height / 2.0
        class_id = int(class_ids[i])
        class_name = MODEL_CLASS_NAMES[class_id] if 0 <= class_id < len(MODEL_CLASS_NAMES) else f"class_{class_id}"

        predictions.append(
            {
                "x": float(cx),
                "y": float(cy),
                "width": float(width),
                "height": float(height),
                "confidence": float(confidences[i]),
                "class": class_name,
            }
        )

    return {"predictions": predictions}


def predict(image_bytes: bytes) -> dict[str, Any]:
    if DETECTION_MODE == "onnx_offline":
        return onnx_offline_predict(image_bytes)
    return roboflow_cloud_predict(image_bytes)


def is_accident_prediction(pred: dict[str, Any]) -> bool:
    class_name = str(pred.get("class", "")).strip().lower()
    return class_name in ACCIDENT_CLASSES


def normalize_image_url(image_url: str | None) -> str | None:
    if not image_url:
        return None

    cleaned = str(image_url).strip()
    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        return cleaned

    return None


def normalize_ai_severity(severity: str | None) -> str:
    normalized = str(severity or "").strip().upper()
    if normalized in {"HIGH", "FATAL", "SEVERE", "CRITICAL"}:
        return "HIGH"
    if normalized in {"MEDIUM", "MAJOR"}:
        return "MEDIUM"
    if normalized in {"LOW", "MINOR"}:
        return "LOW"
    return "UNKNOWN"


def fetch_image_bytes(image_url: str) -> bytes:
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    return response.content


def parse_ai_report_payload(raw_response: str) -> dict[str, str]:
    parsed = parse_ollama_json_report(raw_response)
    if parsed:
        return parsed

    return {
        "severity": infer_severity_from_text(raw_response),
        "accident_type": "Unknown accident type",
        "emergency": "Emergency response recommended.",
        "summary": raw_response.strip() or "AI analysis unavailable.",
    }


def generate_accident_report_from_bytes(
    image_bytes: bytes,
    camera_id: str,
    location: str,
) -> dict[str, str]:
    start_time = time.perf_counter()
    try:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        prompt = f"""
You are an AI traffic accident analysis system.

Analyze this accident image carefully.

Determine:

1. Severity Level (LOW, MEDIUM, HIGH)
2. Accident Type
3. Visible hazards
4. Emergency urgency
5. Short professional incident summary

Rules:

* Fire or explosion = HIGH
* Vehicle rollover = MEDIUM/HIGH
* Heavy frontal collision = HIGH
* Minor impact with pole/wall = LOW

Return ONLY valid JSON:

{{
  "severity": "HIGH",
  "accident_type": "Vehicle rollover collision",
  "emergency": "Immediate ambulance dispatch recommended",
  "summary": "Major rollover accident detected with severe roadway obstruction."
}}
        """

        LOGGER.info("Starting Ollama accident analysis for camera_id=%s location=%s", camera_id, location)
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "moondream:latest",
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "format": "json",
            },
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()
        raw_response = str(data.get("response", "")).strip()
        LOGGER.info("Ollama raw response: %s", raw_response)
        print("OLLAMA RAW RESPONSE:", raw_response)

        parsed = parse_ai_report_payload(raw_response)
        summary_text = parsed.get("summary") or raw_response or "AI analysis unavailable."
        severity = normalize_ai_severity(parsed.get("severity"))
        if severity == "UNKNOWN":
            severity = infer_severity_from_text(summary_text)

        emergency = parsed.get("emergency") or "Emergency response recommended."
        accident_type = parsed.get("accident_type") or "Unknown accident type"

        return {
            "summary": summary_text,
            "severity": severity,
            "emergency": emergency,
            "accident_type": accident_type,
            "raw_response": raw_response,
        }
    except Exception:
        LOGGER.exception("Failed to generate AI report")
        return {
            "summary": "AI analysis unavailable.",
            "severity": "UNKNOWN",
            "emergency": "Emergency response recommended.",
            "accident_type": "Unknown accident type",
            "raw_response": "",
        }
    finally:
        duration_seconds = time.perf_counter() - start_time
        LOGGER.info("Ollama inference duration: %.3fs", duration_seconds)


def generate_accident_report(
    image_path: Path,
    confidence: float,
    camera_id: str,
    location: str,
) -> dict[str, str]:
    return generate_accident_report_from_bytes(image_path.read_bytes(), camera_id, location)


def classify_severity(confidence: float) -> str:
    if confidence >= 0.85:
        return "FATAL"

    if confidence >= 0.65:
        return "MAJOR"

    return "MINOR"


def infer_severity_from_text(text: str) -> str:
    lowered = text.lower()
    if any(keyword in lowered for keyword in ["fire", "explosion", "smoke", "pile-up", "destroyed", "major roadway obstruction", "severe"]):
        return "HIGH"
    if any(keyword in lowered for keyword in ["rollover", "overturned", "multi-vehicle", "collision", "impact", "obstruction", "blocked"]):
        return "MEDIUM"
    if any(keyword in lowered for keyword in ["minor", "bumped", "fender", "light", "small"]):
        return "LOW"
    return "UNKNOWN"


def parse_ollama_json_report(raw_response: str) -> dict[str, str]:
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    candidates = [cleaned]
    if not cleaned.startswith("{") or not cleaned.endswith("}"):
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            candidates.insert(0, match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue

        if isinstance(parsed, dict):
            return {
                "severity": str(parsed.get("severity", "")).strip(),
                "accident_type": str(parsed.get("accident_type", "")).strip(),
                "emergency": str(parsed.get("emergency", "")).strip(),
                "summary": str(parsed.get("summary", "")).strip(),
            }

    return {}


def upload_accident_image(image_path: Path) -> str:
    result = cloudinary.uploader.upload(
        str(image_path),
        folder="accidents",
    )
    return result["secure_url"]


def generate_accident_report(
    image_path: Path,
    confidence: float,
    camera_id: str,
    location: str,
) -> dict[str, str]:
    start_time = time.perf_counter()
    try:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

        prompt = f"""
You are an AI traffic accident analysis system.

Analyze this accident image carefully.

Determine:

1. Severity Level (LOW, MEDIUM, HIGH)
2. Accident Type
3. Visible hazards
4. Emergency urgency
5. Short professional incident summary

Rules:

* Fire or explosion = HIGH
* Vehicle rollover = MEDIUM/HIGH
* Heavy frontal collision = HIGH
* Minor impact with pole/wall = LOW

Return ONLY valid JSON:

{{
    "severity": "HIGH",
    "accident_type": "Vehicle rollover collision",
    "emergency": "Immediate ambulance dispatch recommended",
    "summary": "Major rollover accident detected with severe roadway obstruction."
}}
        """

        LOGGER.info("Starting Ollama accident analysis for camera_id=%s location=%s", camera_id, location)
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "moondream:latest",
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "format": "json",
            },
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()
        raw_response = str(data.get("response", "")).strip()
        LOGGER.info("Ollama raw response: %s", raw_response)
        print("OLLAMA RAW RESPONSE:", raw_response)

        parsed = parse_ollama_json_report(raw_response)
        summary_text = parsed.get("summary") or raw_response or "AI analysis unavailable."
        severity = normalize_ai_severity(parsed.get("severity"))
        if severity == "UNKNOWN":
            severity = infer_severity_from_text(summary_text)

        emergency = parsed.get("emergency") or "Emergency response recommended."
        accident_type = parsed.get("accident_type") or "Unknown accident type"

        return {
            "summary": summary_text,
            "severity": severity,
            "emergency": emergency,
            "accident_type": accident_type,
            "raw_response": raw_response,
        }
    except Exception:
        LOGGER.exception("Failed to generate AI report")
        return {
            "summary": "AI analysis unavailable.",
            "severity": "UNKNOWN",
            "emergency": "Emergency response recommended.",
            "accident_type": "Unknown accident type",
            "raw_response": "",
        }
    finally:
        duration_seconds = time.perf_counter() - start_time
        LOGGER.info("Ollama inference duration: %.3fs", duration_seconds)


def save_accident_event(
    camera_id: str,
    location: str,
    pred: dict[str, Any],
    raw_prediction: dict[str, Any],
    image_bytes: bytes,
) -> None:
    ts = datetime.now(timezone.utc)
    file_name = f"accident_{ts.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    file_path = SNAPSHOT_DIR / file_name
    file_path.write_bytes(image_bytes)

    confidence = float(pred.get("confidence", 0.0))
    report = generate_accident_report(file_path, confidence, camera_id, location)
    ai_summary = report.get("summary", "AI analysis unavailable.")
    severity = normalize_ai_severity(report.get("severity"))
    if severity == "UNKNOWN":
        severity = infer_severity_from_text(ai_summary)
    emergency_status = report.get("emergency", "Emergency response recommended.")

    image_url: str | None = None
    try:
        image_url = upload_accident_image(file_path)
    except Exception:
        LOGGER.exception("Failed to upload accident image to Cloudinary")
    finally:
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO accident_events (
                created_at, source_name, confidence, class_name, bbox_json, camera_id, image_path, image_url, ai_summary, severity, camera_location, emergency_status, raw_prediction_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts.isoformat(),
                location,
                confidence,
                str(pred.get("class", "unknown")),
                json.dumps(
                    {
                        "x": pred.get("x"),
                        "y": pred.get("y"),
                        "width": pred.get("width"),
                        "height": pred.get("height"),
                    }
                ),
                camera_id,
                image_url,
                image_url,
                ai_summary,
                severity,
                location,
                emergency_status,
                json.dumps(raw_prediction),
            ),
        )
        conn.commit()


@app.on_event("startup")
def startup_event() -> None:
    init_db()


@app.get("/api/cameras")
def list_cameras() -> dict[str, Any]:
    return {"cameras": CAMERA_MANAGER.list_cameras()}


@app.get("/api/cameras/{camera_id}")
def get_camera(camera_id: str) -> dict[str, Any]:
    return {"camera": CAMERA_MANAGER.get_camera(camera_id)}


@app.post("/api/cameras/{camera_id}")
def update_camera(camera_id: str, payload: CameraConfigUpdate) -> dict[str, Any]:
    camera = CAMERA_MANAGER.set_camera(camera_id, payload.source_url, payload.label)
    return {"camera": camera}


@app.delete("/api/cameras/{camera_id}")
def clear_camera(camera_id: str) -> dict[str, str]:
    CAMERA_MANAGER.clear_camera(camera_id)
    return {"status": "cleared"}


@app.get("/api/cameras/{camera_id}/stream")
def camera_stream(camera_id: str) -> StreamingResponse:
    return CAMERA_MANAGER.stream_response(camera_id)


@app.get("/api/cameras/{camera_id}/snapshot")
def camera_snapshot(camera_id: str):
    """Return the latest JPEG frame for a camera as a single image response.

    This is useful for focus previews where a single image (with proper
    content-length and natural dimensions) is preferred over an MJPEG stream.
    """
    image_bytes = CAMERA_MANAGER.latest_frame(camera_id)
    if image_bytes is None:
        raise HTTPException(status_code=503, detail=f"Camera {camera_id} has no available frame yet")
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/jpeg", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": DETECTION_MODE}


@app.post("/api/detect")
def detect(
    frame: UploadFile | None = File(default=None),
    source_name: str = Form("ipcam"),
    camera_id: str | None = Form(default=None),
) -> dict[str, Any]:
    image_bytes: bytes | None = None
    input_source = "upload"

    if camera_id:
        image_bytes = CAMERA_MANAGER.latest_frame(camera_id)
        if image_bytes is None:
            raise HTTPException(status_code=503, detail=f"Camera {camera_id} has no available frame yet")
        camera = CAMERA_MANAGER.get_camera(camera_id)
        source_name = camera.get("label") or source_name
        input_source = "camera_id"
        LOGGER.info("Running live inference on camera %s (%s)", camera_id, source_name)
    elif frame is not None:
        image_bytes = frame.file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Provide either a frame upload or a camera_id")

    prediction = predict(image_bytes)
    print(prediction)
    predictions = prediction.get("predictions", [])

    accidents = [p for p in predictions if is_accident_prediction(p)]
    for accident in accidents:
        resolved_camera_id = camera_id or source_name
        save_accident_event(resolved_camera_id, source_name, accident, prediction, image_bytes)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_name": source_name,
        "input_source": input_source,
        "mode": DETECTION_MODE,
        "accident_detected": len(accidents) > 0,
        "accident_count": len(accidents),
        "predictions": predictions,
    }


@app.get("/api/events")
def events(limit: int = 25) -> dict[str, Any]:
    safe_limit = max(1, min(limit, 200))
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, source_name, confidence, class_name, bbox_json, camera_id, image_path, image_url, ai_summary, severity, camera_location, emergency_status
            FROM accident_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()

    items = []
    for row in rows:
        items.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "source_name": row["source_name"],
                "confidence": row["confidence"],
                "class_name": row["class_name"],
                "bbox": json.loads(row["bbox_json"]),
                "camera_id": row["camera_id"],
                "image_path": row["image_path"],
                "image_url": normalize_image_url(row["image_url"]),
                "ai_summary": row["ai_summary"],
                "severity": row["severity"],
                "camera_location": row["camera_location"],
                "emergency_status": row["emergency_status"],
            }
        )
    return {"events": items}


@app.post("/api/events/{event_id}/analyze")
def analyze_event(event_id: int) -> dict[str, Any]:
    print(f"\n{'='*80}")
    print(f"[BACKEND] POST /api/events/{event_id}/analyze - ENDPOINT CALLED")
    print(f"{'='*80}")
    LOGGER.info(f"[ANALYZE {event_id}] Endpoint called")
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT id, created_at, source_name, confidence, class_name, bbox_json, camera_id, image_path, image_url, ai_summary, severity, camera_location, emergency_status
            FROM accident_events
            WHERE id = ?
            """,
            (event_id,),
        ).fetchone()

        if row is None:
            print(f"[BACKEND] Event {event_id} NOT FOUND in database")
            LOGGER.error(f"[ANALYZE {event_id}] Event not found")
            raise HTTPException(status_code=404, detail=f"Unknown accident event id: {event_id}")

        existing_summary = str(row["ai_summary"] or "").strip()
        existing_severity = normalize_ai_severity(row["severity"])
        existing_emergency = str(row["emergency_status"] or "").strip()
        
        print(f"[BACKEND] Event {event_id} - has_summary: {bool(existing_summary)}, severity: {existing_severity}, emergency: {bool(existing_emergency)}")
        
        if existing_summary and existing_severity != "UNKNOWN" and existing_emergency:
            print(f"[BACKEND] Event {event_id} - Already fully analyzed, returning cached result")
            LOGGER.info(f"[ANALYZE {event_id}] Returning cached analysis")
            return {
                "id": row["id"],
                "ai_summary": existing_summary,
                "severity": existing_severity,
                "emergency_status": existing_emergency,
                "accident_type": "",
            }

        image_url = normalize_image_url(row["image_url"])
        if not image_url:
            print(f"[BACKEND] Event {event_id} - NO VALID IMAGE URL")
            LOGGER.error(f"[ANALYZE {event_id}] No valid Cloudinary image URL")
            raise HTTPException(status_code=400, detail=f"Event {event_id} does not have a downloadable Cloudinary image URL")

        print(f"[BACKEND] Event {event_id} - Queued for serial analysis")
        LOGGER.info(f"[ANALYZE {event_id}] Waiting for serial analysis lock")

        with ANALYSIS_SERIAL_LOCK:
            print(f"[BACKEND] Event {event_id} - Acquired analysis lock")
            LOGGER.info(f"[ANALYZE {event_id}] Analysis lock acquired")

            conn.row_factory = sqlite3.Row
            latest_row = conn.execute(
                """
                SELECT ai_summary, severity, emergency_status, camera_id, source_name, camera_location, confidence, image_url
                FROM accident_events
                WHERE id = ?
                """,
                (event_id,),
            ).fetchone()
            if latest_row is None:
                raise HTTPException(status_code=404, detail=f"Unknown accident event id: {event_id}")

            latest_summary = str(latest_row["ai_summary"] or "").strip()
            latest_severity = normalize_ai_severity(latest_row["severity"])
            latest_emergency = str(latest_row["emergency_status"] or "").strip()
            if latest_summary and latest_severity != "UNKNOWN" and latest_emergency:
                print(f"[BACKEND] Event {event_id} - Analysis completed by another request, returning cached result")
                LOGGER.info(f"[ANALYZE {event_id}] Returning cached analysis after lock")
                return {
                    "id": event_id,
                    "camera_id": str(latest_row["camera_id"] or latest_row["source_name"] or f"event-{event_id}"),
                    "camera_location": str(latest_row["camera_location"] or latest_row["source_name"] or "unknown location"),
                    "confidence": float(latest_row["confidence"] or 0.0),
                    "image_url": normalize_image_url(str(latest_row["image_url"] or "")),
                    "ai_summary": latest_summary,
                    "severity": latest_severity,
                    "emergency_status": latest_emergency,
                }

            image_url = normalize_image_url(str(latest_row["image_url"] or ""))
            if not image_url:
                print(f"[BACKEND] Event {event_id} - NO VALID IMAGE URL after lock")
                LOGGER.error(f"[ANALYZE {event_id}] No valid Cloudinary image URL after lock")
                raise HTTPException(status_code=400, detail=f"Event {event_id} does not have a downloadable Cloudinary image URL")

            print(f"[BACKEND] Event {event_id} - Fetching image from: {image_url[:80]}...")
            camera_id = str(latest_row["camera_id"] or latest_row["source_name"] or f"event-{event_id}")
            location = str(latest_row["camera_location"] or latest_row["source_name"] or "unknown location")
            confidence = float(latest_row["confidence"] or 0.0)
            LOGGER.info(f"[ANALYZE {event_id}] Starting Moondream LLM analysis with image_url")
            print(f"[BACKEND] Event {event_id} - Downloading image bytes...")
            image_bytes = fetch_image_bytes(image_url)
            print(f"[BACKEND] Event {event_id} - Downloaded {len(image_bytes)} bytes, calling Moondream LLM...")
            report = generate_accident_report_from_bytes(image_bytes, camera_id, location)
            print(f"[BACKEND] Event {event_id} - Moondream LLM response: {report}")

            ai_summary = report.get("summary", "AI analysis unavailable.")
            severity = normalize_ai_severity(report.get("severity"))
            if severity == "UNKNOWN":
                severity = infer_severity_from_text(ai_summary)
            emergency_status = report.get("emergency", "Emergency response recommended.")
            
            print(f"[BACKEND] Event {event_id} - Parsed report: severity={severity}, emergency={emergency_status}, summary_len={len(ai_summary)}")
            print(f"[BACKEND] Event {event_id} - Updating SQLite database...")

            conn.execute(
                """
                UPDATE accident_events
                SET ai_summary = ?, severity = ?, emergency_status = ?
                WHERE id = ?
                """,
                (ai_summary, severity, emergency_status, event_id),
            )
            conn.commit()
            print(f"[BACKEND] Event {event_id} - Database updated successfully!")
            LOGGER.info(f"[ANALYZE {event_id}] Analysis complete, database updated")

    print(f"[BACKEND] Event {event_id} - Returning result to client")
    return {
        "id": event_id,
        "camera_id": camera_id,
        "camera_location": location,
        "confidence": confidence,
        "image_url": image_url,
        "ai_summary": ai_summary,
        "severity": severity,
        "emergency_status": emergency_status,
        "accident_type": report.get("accident_type", "Unknown accident type"),
    }


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    with sqlite3.connect(DB_PATH) as conn:
        total = conn.execute("SELECT COUNT(*) FROM accident_events").fetchone()[0]
        last_24h = conn.execute(
            """
            SELECT COUNT(*) FROM accident_events
            WHERE julianday(created_at) >= julianday('now', '-1 day')
            """
        ).fetchone()[0]
    return {"total_events": total, "events_last_24h": last_24h}


app.mount("/backend/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")
app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/accidents")
def accident_gallery() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "accidents.html")


