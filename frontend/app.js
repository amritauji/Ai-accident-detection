const focusVideo = document.getElementById("focusVideo");
const focusOverlay = document.getElementById("focusOverlay");
const intersectionGrid = document.getElementById("intersectionGrid");
const focusNameEl = document.getElementById("focusName");
const wallModeBtn = document.getElementById("wallModeBtn");
const focusModeBtn = document.getElementById("focusModeBtn");
const connectedCountEl = document.getElementById("connectedCount");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const intervalInput = document.getElementById("intervalInput");
const statusEl = document.getElementById("status");
const totalEventsEl = document.getElementById("totalEvents");
const last24hEl = document.getElementById("last24h");
const currentDetectionsEl = document.getElementById("currentDetections");
const eventsList = document.getElementById("eventsList");
const checklistList = document.getElementById("checklistList");
const missingList = document.getElementById("missingList");

const intersections = [
  { id: "int-1", name: "Main St & 1st Ave" },
  { id: "int-2", name: "Central Blvd & Lake Rd" },
  { id: "int-3", name: "Airport Road Junction" },
  { id: "int-4", name: "Market Square Cross" },
  { id: "int-5", name: "River Bridge Exit" },
  { id: "int-6", name: "Highway 4 Flyover" },
  { id: "int-7", name: "Industrial Gate Loop" },
  { id: "int-8", name: "University Circle" },
  { id: "int-9", name: "Bus Terminal North" },
];

const state = {
  devices: [],
  items: new Map(),
  focusedId: null,
  eventsTimer: null,
  detectionInFlight: false,
  detectionActive: false,
  lastPredictions: [],
  lastCaptureWidth: 0,
  lastCaptureHeight: 0,
  lastDetectionLatencyMs: null,
  lastHealth: null,
  lastEventsRefreshAt: 0,
  viewMode: "wall",
};

const DETECTION_MAX_DIM = 640;
const DETECTION_JPEG_QUALITY = 0.65;
const EVENTS_REFRESH_MS = 3000;
const HEALTH_REFRESH_MS = 5000;

function makeInitialIntersectionState() {
  return {
    stream: null,
    deviceId: "",
    elements: null,
  };
}

function setStatus(text, type = "") {
  statusEl.textContent = text;
  statusEl.className = type ? `status ${type}` : "status";
}

function renderAuthorityChecklist() {
  const connectedCount = [...state.items.values()].filter((item) => item.stream).length;
  const hasFocusedLive = Boolean(state.focusedId && state.items.get(state.focusedId)?.stream);
  const hasApiHealth = state.lastHealth?.status === "ok";
  const hasDetectionMode = Boolean(state.lastHealth?.mode);
  const hasRecentEventsSync = state.lastEventsRefreshAt > 0 && Date.now() - state.lastEventsRefreshAt <= EVENTS_REFRESH_MS * 2;
  const latencyOk = state.lastDetectionLatencyMs !== null && state.lastDetectionLatencyMs <= 1200;
  const noActiveAccident = Number(currentDetectionsEl.textContent || "0") === 0;

  const checks = [
    { label: "Backend API reachable", ok: hasApiHealth, bad: "System down risk" },
    { label: "Detection mode available", ok: hasDetectionMode, bad: "Detector not configured" },
    { label: "At least one camera connected", ok: connectedCount > 0, bad: "Blind monitoring" },
    { label: "Focused feed is live", ok: hasFocusedLive, bad: "No active target feed" },
    { label: "Detection latency <= 1.2s", ok: latencyOk, bad: "Slow alert response" },
    { label: "Events feed fresh", ok: hasRecentEventsSync, bad: "Stale dashboard data" },
    { label: "No active accident", ok: noActiveAccident, bad: "Incident ongoing" },
  ];

  checklistList.innerHTML = "";
  missingList.innerHTML = "";

  checks.forEach((check) => {
    const row = document.createElement("div");
    row.className = "check-item";
    row.innerHTML = `
      <span class="check-label">${check.label}</span>
      <span class="check-pill ${check.ok ? "ok" : "bad"}">${check.ok ? "Present" : "Missing"}</span>
    `;
    checklistList.appendChild(row);

    if (!check.ok) {
      const li = document.createElement("li");
      li.textContent = `${check.label}: ${check.bad}`;
      missingList.appendChild(li);
    }
  });

  if (!missingList.children.length) {
    const li = document.createElement("li");
    li.textContent = "No critical gaps right now.";
    missingList.appendChild(li);
  }
}

function setViewMode(mode) {
  state.viewMode = mode;
  document.body.classList.toggle("wall-mode", mode === "wall");
  document.body.classList.toggle("focus-mode", mode === "focus");
  wallModeBtn.classList.toggle("active", mode === "wall");
  focusModeBtn.classList.toggle("active", mode === "focus");
}

async function loadDevices() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  state.devices = devices.filter((device) => device.kind === "videoinput");
}

function getDeviceLabel(deviceId) {
  const device = state.devices.find((item) => item.deviceId === deviceId);
  return device ? device.label || "Camera" : "Unassigned";
}

function updateConnectedCount() {
  const liveCount = [...state.items.values()].filter((item) => item.stream).length;
  connectedCountEl.textContent = `${liveCount} camera${liveCount === 1 ? "" : "s"} live`;
}

function updateFocusedCardStyles() {
  state.items.forEach((itemState, id) => {
    if (!itemState.elements?.card) return;
    itemState.elements.card.classList.toggle("active", id === state.focusedId);
  });
}

function renderIntersectionCards() {
  intersectionGrid.innerHTML = "";

  intersections.forEach((intersection) => {
    if (!state.items.has(intersection.id)) {
      state.items.set(intersection.id, makeInitialIntersectionState());
    }

    const itemState = state.items.get(intersection.id);

    const card = document.createElement("article");
    card.className = "intersection-card";
    card.dataset.intersectionId = intersection.id;
    card.innerHTML = `
      <h3>${intersection.name}</h3>
      <p class="subtle">${intersection.id.toUpperCase()}</p>
      <video class="feed-preview" autoplay muted playsinline></video>
      <div class="intersection-controls">
        <select></select>
        <div class="control-row">
          <button data-action="connect">Connect</button>
          <button data-action="focus" class="secondary">Focus</button>
          <button data-action="disconnect" class="secondary">Off</button>
        </div>
      </div>
      <span class="feed-badge">Offline</span>
    `;

    const video = card.querySelector("video");
    const select = card.querySelector("select");
    const badge = card.querySelector(".feed-badge");
    const connectBtn = card.querySelector("button[data-action='connect']");
    const focusBtn = card.querySelector("button[data-action='focus']");
    const disconnectBtn = card.querySelector("button[data-action='disconnect']");

    select.innerHTML = "<option value=\"\">Select camera</option>";
    state.devices.forEach((camera, index) => {
      const option = document.createElement("option");
      option.value = camera.deviceId;
      option.textContent = camera.label || `Camera ${index + 1}`;
      select.appendChild(option);
    });

    if (itemState.deviceId) {
      select.value = itemState.deviceId;
    }

    connectBtn.addEventListener("click", async () => {
      const chosenId = select.value;
      if (!chosenId) {
        setStatus(`Select a camera for ${intersection.name}`, "alert");
        return;
      }
      try {
        await connectIntersection(intersection.id, chosenId);
        badge.textContent = `Live: ${getDeviceLabel(chosenId)}`;
        badge.classList.add("live");
      } catch (error) {
        setStatus(`Camera error at ${intersection.name}: ${error.message}`, "alert");
      }
    });

    focusBtn.addEventListener("click", () => setFocusedIntersection(intersection.id));
    video.addEventListener("click", () => setFocusedIntersection(intersection.id));

    disconnectBtn.addEventListener("click", () => {
      disconnectIntersection(intersection.id);
      badge.textContent = "Offline";
      badge.classList.remove("live");
    });

    itemState.elements = { card, video, select, badge };

    if (itemState.stream) {
      video.srcObject = itemState.stream;
      badge.textContent = `Live: ${getDeviceLabel(itemState.deviceId)}`;
      badge.classList.add("live");
    }

    intersectionGrid.appendChild(card);
  });

  updateFocusedCardStyles();
  updateConnectedCount();
  renderAuthorityChecklist();
}

async function connectIntersection(intersectionId, deviceId) {
  const itemState = state.items.get(intersectionId);
  if (!itemState) return;

  disconnectIntersection(intersectionId);

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { deviceId: { exact: deviceId } },
    audio: false,
  });

  itemState.stream = stream;
  itemState.deviceId = deviceId;
  if (itemState.elements?.video) {
    itemState.elements.video.srcObject = stream;
  }

  if (state.focusedId === intersectionId) {
    focusVideo.srcObject = stream;
  }

  updateConnectedCount();
  setStatus(`${intersections.find((entry) => entry.id === intersectionId)?.name} connected`, "ok");
  renderAuthorityChecklist();
}

function disconnectIntersection(intersectionId) {
  const itemState = state.items.get(intersectionId);
  if (!itemState) return;

  if (itemState.stream) {
    itemState.stream.getTracks().forEach((track) => track.stop());
  }

  itemState.stream = null;

  if (state.focusedId === intersectionId) {
    state.lastPredictions = [];
    clearOverlay();
    focusVideo.srcObject = null;
    currentDetectionsEl.textContent = "0";
    setStatus("Focused feed is offline", "alert");
  }

  updateConnectedCount();
  renderAuthorityChecklist();
}

function setFocusedIntersection(intersectionId) {
  const intersection = intersections.find((entry) => entry.id === intersectionId);
  if (!intersection) return;

  state.focusedId = intersectionId;
  focusNameEl.textContent = intersection.name;

  const itemState = state.items.get(intersectionId);
  if (itemState?.stream) {
    focusVideo.srcObject = itemState.stream;
    setStatus(`Focused on ${intersection.name}`, "ok");
  } else {
    focusVideo.srcObject = null;
    clearOverlay();
    currentDetectionsEl.textContent = "0";
    setStatus(`${intersection.name} selected but currently offline`, "alert");
  }

  updateFocusedCardStyles();
  renderAuthorityChecklist();
}

function clearOverlay() {
  const context = focusOverlay.getContext("2d");
  context.clearRect(0, 0, focusOverlay.width, focusOverlay.height);
}

function drawPredictions(predictions) {
  if (!focusVideo.videoWidth || !focusVideo.videoHeight) return;

  focusOverlay.width = focusVideo.clientWidth;
  focusOverlay.height = focusVideo.clientHeight;

  const frameWidth = state.lastCaptureWidth || focusVideo.videoWidth;
  const frameHeight = state.lastCaptureHeight || focusVideo.videoHeight;
  const scaleX = focusOverlay.width / frameWidth;
  const scaleY = focusOverlay.height / frameHeight;
  const context = focusOverlay.getContext("2d");
  context.clearRect(0, 0, focusOverlay.width, focusOverlay.height);
  context.lineWidth = 2;
  context.font = "13px Manrope";

  predictions.forEach((prediction) => {
    const x = (prediction.x - prediction.width / 2) * scaleX;
    const y = (prediction.y - prediction.height / 2) * scaleY;
    const width = prediction.width * scaleX;
    const height = prediction.height * scaleY;
    const className = (prediction.class || "unknown").toLowerCase();
    const isAccident = ["accident", "crash", "collision"].includes(className);

    context.strokeStyle = isAccident ? "#ff4f4f" : "#0fa176";
    context.fillStyle = isAccident ? "#ff4f4f" : "#0fa176";
    context.strokeRect(x, y, width, height);

    const label = `${prediction.class} ${(prediction.confidence * 100).toFixed(1)}%`;
    const labelWidth = context.measureText(label).width;
    context.fillRect(x, Math.max(0, y - 20), labelWidth + 8, 18);
    context.fillStyle = "#fff";
    context.fillText(label, x + 4, Math.max(12, y - 6));
  });
}

async function detectFocusFrame() {
  const focused = state.items.get(state.focusedId);
  if (!focused?.stream || state.detectionInFlight || focusVideo.readyState < 2) return;

  state.detectionInFlight = true;
  try {
    const detectStartedAt = performance.now();
    const canvas = document.createElement("canvas");
    const sourceWidth = focusVideo.videoWidth;
    const sourceHeight = focusVideo.videoHeight;
    const scale = Math.min(1, DETECTION_MAX_DIM / Math.max(sourceWidth, sourceHeight));
    canvas.width = Math.max(1, Math.round(sourceWidth * scale));
    canvas.height = Math.max(1, Math.round(sourceHeight * scale));
    const context = canvas.getContext("2d");
    context.drawImage(focusVideo, 0, 0, canvas.width, canvas.height);

    state.lastCaptureWidth = canvas.width;
    state.lastCaptureHeight = canvas.height;
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", DETECTION_JPEG_QUALITY));
    if (!blob) return;

    const focusedIntersection = intersections.find((entry) => entry.id === state.focusedId);
    const formData = new FormData();
    formData.append("frame", blob, "frame.jpg");
    formData.append("source_name", focusedIntersection?.name || "unknown intersection");

    const response = await fetch("/api/detect", { method: "POST", body: formData });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || "Detection request failed");
    }

    const data = await response.json();
    state.lastDetectionLatencyMs = performance.now() - detectStartedAt;
    state.lastPredictions = data.predictions || [];
    drawPredictions(state.lastPredictions);

    const accidents = data.accident_count || 0;
    currentDetectionsEl.textContent = String(accidents);
    if (data.accident_detected) {
      setStatus(`ALERT on ${focusedIntersection?.name}: ${accidents} accident detection(s)`, "alert");
    } else {
      setStatus(`Monitoring ${focusedIntersection?.name || "focus feed"}`, "ok");
    }
    renderAuthorityChecklist();
  } finally {
    state.detectionInFlight = false;
  }
}

async function refreshEvents() {
  const [eventsResp, statsResp] = await Promise.all([
    fetch("/api/events?limit=20"),
    fetch("/api/stats"),
  ]);

  const eventsData = await eventsResp.json();
  const statsData = await statsResp.json();
  state.lastEventsRefreshAt = Date.now();

  totalEventsEl.textContent = String(statsData.total_events ?? 0);
  last24hEl.textContent = String(statsData.events_last_24h ?? 0);

  eventsList.innerHTML = "";
  (eventsData.events || []).forEach((event) => {
    const item = document.createElement("article");
    item.className = "event";
    item.innerHTML = `
      <h4>${event.class_name} (${(event.confidence * 100).toFixed(1)}%)</h4>
      <p><strong>Time:</strong> ${new Date(event.created_at).toLocaleString()}</p>
      <p><strong>Source:</strong> ${event.source_name || "unknown"}</p>
      <p><strong>BBox:</strong> x=${Math.round(event.bbox.x || 0)}, y=${Math.round(event.bbox.y || 0)}, w=${Math.round(event.bbox.width || 0)}, h=${Math.round(event.bbox.height || 0)}</p>
      ${event.image_url ? `<img class="thumb" src="${event.image_url}" alt="Accident snapshot ${event.id}"/>` : ""}
    `;
    eventsList.appendChild(item);
  });
  renderAuthorityChecklist();
}

async function refreshHealth() {
  try {
    const resp = await fetch("/api/health");
    if (!resp.ok) throw new Error("health endpoint failed");
    state.lastHealth = await resp.json();
  } catch (_error) {
    state.lastHealth = null;
  }
  renderAuthorityChecklist();
}

function startDetection() {
  if (!state.focusedId) {
    setStatus("Choose a focused intersection first", "alert");
    return;
  }

  const focused = state.items.get(state.focusedId);
  if (!focused?.stream) {
    setStatus("Focused intersection is offline. Connect camera first", "alert");
    return;
  }

  const intervalMs = Math.max(300, Number(intervalInput.value) || 1000);
  if (state.detectionActive) return;
  state.detectionActive = true;

  const runLoop = async () => {
    while (state.detectionActive) {
      const startedAt = performance.now();
      try {
        await detectFocusFrame();
      } catch (error) {
        setStatus(`Detection error: ${error.message}`, "alert");
      }
      const elapsed = performance.now() - startedAt;
      const delay = Math.max(0, intervalMs - elapsed);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  };
  runLoop();

  if (state.eventsTimer) clearInterval(state.eventsTimer);
  refreshEvents().catch(() => {});
  state.eventsTimer = setInterval(async () => {
    try {
      await refreshEvents();
    } catch (_error) {
      // Keep detection loop independent from dashboard refresh failures.
    }
  }, EVENTS_REFRESH_MS);

  setStatus("Detection started", "ok");
  renderAuthorityChecklist();
}

function stopDetection() {
  state.detectionActive = false;
  if (state.eventsTimer) clearInterval(state.eventsTimer);
  state.eventsTimer = null;
  state.lastPredictions = [];
  state.lastCaptureWidth = 0;
  state.lastCaptureHeight = 0;
  clearOverlay();
  currentDetectionsEl.textContent = "0";
  setStatus("Detection stopped");
  renderAuthorityChecklist();
}

function cleanupAllStreams() {
  state.items.forEach((itemState) => {
    if (itemState.stream) {
      itemState.stream.getTracks().forEach((track) => track.stop());
    }
  });
}

startBtn.addEventListener("click", startDetection);
stopBtn.addEventListener("click", stopDetection);
wallModeBtn.addEventListener("click", () => setViewMode("wall"));
focusModeBtn.addEventListener("click", () => setViewMode("focus"));
window.addEventListener("beforeunload", cleanupAllStreams);
window.addEventListener("resize", () => drawPredictions(state.lastPredictions));

(async function init() {
  try {
    const permissionStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    permissionStream.getTracks().forEach((track) => track.stop());
  } catch (_error) {
    setStatus("Camera permission is required to load and switch CCTV feeds", "alert");
  }

  try {
    await loadDevices();
  } catch (_error) {
    setStatus("Could not enumerate camera devices", "alert");
  }

  renderIntersectionCards();
  setFocusedIntersection(intersections[0].id);
  setViewMode("wall");
  await refreshHealth();
  setInterval(() => {
    refreshHealth().catch(() => {});
  }, HEALTH_REFRESH_MS);
  await refreshEvents();
  renderAuthorityChecklist();
})();
