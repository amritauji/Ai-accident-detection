import base64
import io
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT / "backend" / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = ROOT / "backend" / "accidents.db"
load_dotenv(ROOT / ".env")

DETECTION_MODE = os.getenv("DETECTION_MODE", "roboflow_cloud").strip().lower()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "road-accident-2vton/4")
ROBOFLOW_CONFIDENCE = os.getenv("ROBOFLOW_CONFIDENCE", "40")

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
                image_path TEXT,
                raw_prediction_json TEXT
            )
            """
        )
        conn.commit()


def roboflow_cloud_predict(image_bytes: bytes) -> dict[str, Any]:
    if not ROBOFLOW_API_KEY:
        raise HTTPException(status_code=500, detail="Set ROBOFLOW_API_KEY in .env")
    if not ROBOFLOW_MODEL_ID:
        raise HTTPException(status_code=500, detail="Set ROBOFLOW_MODEL_ID in .env")

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    url = (
        f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}"
        f"?api_key={ROBOFLOW_API_KEY}&confidence={ROBOFLOW_CONFIDENCE}"
    )

    try:
        response = _ROBOFLOW_SESSION.post(
            url,
            data=encoded,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return {"predictions": []}
        if not isinstance(payload.get("predictions"), list):
            payload["predictions"] = []
        return payload
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Roboflow cloud inference failed: {exc}") from exc


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


def save_accident_event(
    source_name: str,
    pred: dict[str, Any],
    raw_prediction: dict[str, Any],
    image_bytes: bytes,
) -> None:
    ts = datetime.now(timezone.utc)
    file_name = f"accident_{ts.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    file_path = SNAPSHOT_DIR / file_name
    file_path.write_bytes(image_bytes)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO accident_events (
                created_at, source_name, confidence, class_name, bbox_json, image_path, raw_prediction_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts.isoformat(),
                source_name,
                float(pred.get("confidence", 0.0)),
                str(pred.get("class", "unknown")),
                json.dumps(
                    {
                        "x": pred.get("x"),
                        "y": pred.get("y"),
                        "width": pred.get("width"),
                        "height": pred.get("height"),
                    }
                ),
                str(file_path.relative_to(ROOT)).replace("\\", "/"),
                json.dumps(raw_prediction),
            ),
        )
        conn.commit()


@app.on_event("startup")
def startup_event() -> None:
    init_db()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": DETECTION_MODE}


@app.post("/api/detect")
def detect(
    frame: UploadFile = File(...),
    source_name: str = Form("webcam"),
) -> dict[str, Any]:
    image_bytes = frame.file.read()
    prediction = predict(image_bytes)
    predictions = prediction.get("predictions", [])

    accidents = [p for p in predictions if is_accident_prediction(p)]
    for accident in accidents:
        save_accident_event(source_name, accident, prediction, image_bytes)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_name": source_name,
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
            SELECT id, created_at, source_name, confidence, class_name, bbox_json, image_path
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
                "image_path": row["image_path"],
                "image_url": f"/{row['image_path']}" if row["image_path"] else None,
            }
        )
    return {"events": items}


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


