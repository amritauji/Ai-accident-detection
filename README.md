# AI Car Accident Detection Dashboard

Live dashboard for IP camera and RTSP accident detection using Roboflow cloud or local ONNX offline inference.

## Features
- Live IP camera or RTSP feed in browser
- Real-time inference from the focused camera stream
- Two inference modes:
  - `roboflow_cloud` (online)
  - `onnx_offline` (no internet required at runtime)
- Bounding-box overlay with class and confidence
- Accident event logging into SQLite
- Snapshot storage for detected incidents

## How It Works
1. Add an IP camera or RTSP URL to a tile in the camera matrix.
2. Click `Focus` to open the live preview for that camera.
3. Click `Start Detection` to send the focused camera frames to Roboflow for live inferencing.
4. Accident detections are drawn on the focus view and saved to SQLite with snapshots.

## Project Structure
- `backend/app.py`: FastAPI API + inference pipeline + SQLite event store
- `frontend/index.html`: main dashboard UI
- `frontend/styles.css`: responsive UI styles
- `frontend/app.js`: camera control, polling, rendering, API integration
- `models/`: local offline model folder (`.onnx`)
- `backend/snapshots/`: stored accident images
- `backend/accidents.db`: SQLite DB (auto-created)

## Setup
1. Create virtual environment and install dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Configure environment:
   ```powershell
   Copy-Item .env.example .env
   ```
3. Start server:
   ```powershell
   python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
   ```
4. Open `http://localhost:8000`

## Live Camera Flow
- Enter the IP camera URL in the tile or camera dialog.
- The focus view uses the selected camera snapshot feed.
- Detection runs on the selected `camera_id`, so the frames going to Roboflow are the IP camera frames, not the laptop webcam.

## Offline Inference Setup
1. Export your Roboflow model as `YOLOv8 ONNX`.
2. Put the file at:
   - `models/road-accident.onnx`
3. Edit `.env`:
   ```env
   DETECTION_MODE=onnx_offline
   OFFLINE_MODEL_PATH=models/road-accident.onnx
   MODEL_CLASS_NAMES=accident
   ACCIDENT_CLASSES=accident,crash,collision
   ```
4. Restart server.

## API Endpoints
- `GET /api/health` (returns current `mode`)
- `POST /api/detect` (multipart: `frame` or `camera_id`, `source_name`)
- `GET /api/cameras/{camera_id}/snapshot`
- `GET /api/events?limit=20`
- `GET /api/stats`

## Notes
- If detections show class IDs, set `MODEL_CLASS_NAMES` in `.env` as comma-separated names in the exact model class order.
- For production, add authentication and HTTPS.
