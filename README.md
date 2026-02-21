# AI Car Accident Detection Dashboard

Live UI + backend service for camera-based accident detection using Roboflow cloud or local ONNX offline inference.

## Features
- Live webcam or USB camera feed in browser
- Real-time frame inference
- Two inference modes:
  - `roboflow_cloud` (online)
  - `onnx_offline` (no internet required at runtime)
- Bounding-box overlay with class and confidence
- Accident event logging into SQLite
- Snapshot storage for detected incidents

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
   uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
   ```
4. Open `http://localhost:8000`

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
- `POST /api/detect` (multipart: `frame`, `source_name`)
- `GET /api/events?limit=20`
- `GET /api/stats`

## Notes
- If detections show class IDs, set `MODEL_CLASS_NAMES` in `.env` as comma-separated names in the exact model class order.
- For production, add authentication and HTTPS.
