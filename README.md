# AI Car Accident Detection Dashboard

Live dashboard for IP camera and RTSP accident detection using Roboflow cloud, local Jetson GPU inference, or local ONNX offline inference.

## Features
- Live IP camera or RTSP feed in browser
- Real-time inference from the focused camera stream
- Three inference modes:
  - `roboflow_cloud` (online, cloud-based)
  - `local_jetson` (local Jetson GPU with TensorRT/CUDA acceleration)
  - `onnx_offline` (local ONNX, CPU-based, no internet required)
- Bounding-box overlay with class and confidence
- Accident event logging into SQLite
- Cloudinary-backed accident image storage for detected incidents

## How It Works
1. Add an IP camera or RTSP URL to a tile in the camera matrix.
2. Click `Focus` to open the live preview for that camera.
3. Click `Start Detection` to send the focused camera frames to the inference backend (Roboflow cloud, local Jetson, or local ONNX).
4. Accident detections are drawn on the focus view and saved to SQLite with Cloudinary image URLs.

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
   Add your Cloudinary credentials to `.env` so accident snapshots can be uploaded to the CDN.
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

## Local Jetson Inference Setup
1. Deploy the Roboflow inference server on your Jetson device:
   ```bash
   docker run -it -d -p 9001:9001 roboflow/roboflow-inference-server-jetson:latest
   ```
   Or use a local inference server pointed to by `LOCAL_INFERENCE_URL`.

2. Configure your `.env` on the backend machine:
   ```env
   DETECTION_MODE=roboflow_cloud
   ROBOFLOW_API_KEY=your_api_key_here
   ROBOFLOW_MODEL_ID=your-model-name/version
   LOCAL_INFERENCE_URL=http://jetson-ip-or-hostname:9001
   ```

3. Restart the backend server.

The backend will now use `InferenceHTTPClient` to send images to the local Jetson inference server, leveraging GPU acceleration (CUDA/TensorRT) without requiring internet connectivity.

## API Endpoints
- `GET /api/health` (returns current `mode`)
- `POST /api/detect` (multipart: `frame` or `camera_id`, `source_name`)
- `GET /api/cameras/{camera_id}/snapshot`
- `GET /api/events?limit=20`
- `GET /api/stats`

## Notes
- If detections show class IDs, set `MODEL_CLASS_NAMES` in `.env` as comma-separated names in the exact model class order.
- For production, add authentication and HTTPS.
