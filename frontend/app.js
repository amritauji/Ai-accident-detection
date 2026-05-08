const focusFeed = document.getElementById("focusVideo");
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
const cameraDialog = document.getElementById("cameraDialog");
const cameraDialogTitle = document.getElementById("cameraDialogTitle");
const cameraDialogInput = document.getElementById("cameraDialogInput");
const cameraDialogSaveBtn = document.getElementById("cameraDialogSaveBtn");
const cameraDialogCancelBtn = document.getElementById("cameraDialogCancelBtn");

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
  cameras: new Map(),
  items: new Map(),
  focusedId: null,
  dialogCameraId: null,
  eventsTimer: null,
  detectionInFlight: false,
  detectionActive: false,
  lastPredictions: [],
  lastDetectionLatencyMs: null,
  lastHealth: null,
  lastEventsRefreshAt: 0,
  viewMode: "wall",
  focusRefreshTimer: null,
};

const EVENTS_REFRESH_MS = 3000;
const HEALTH_REFRESH_MS = 5000;

function makeInitialIntersectionState() {
  return {
    sourceDraft: "",
    elements: null,
  };
}

function cameraStreamUrl(intersectionId) {
  return `/api/cameras/${intersectionId}/stream?ts=${Date.now()}`;
}

function placeholderFeed(title, subtitle) {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540">
      <defs>
        <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#0b1320" />
          <stop offset="100%" stop-color="#162633" />
        </linearGradient>
      </defs>
      <rect width="960" height="540" rx="24" fill="url(#g)" />
      <circle cx="120" cy="120" r="70" fill="rgba(41,164,255,0.14)" />
      <circle cx="820" cy="430" r="100" fill="rgba(15,161,118,0.14)" />
      <text x="60" y="220" fill="#e8f0f8" font-size="40" font-family="Manrope,Segoe UI,sans-serif" font-weight="700">${title}</text>
      <text x="60" y="278" fill="#9cb2c7" font-size="22" font-family="Manrope,Segoe UI,sans-serif">${subtitle}</text>
    </svg>`;
  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
}

function getCameraConfig(intersectionId) {
  return state.cameras.get(intersectionId) || { source_url: "", label: "" };
}

function setStatus(text, type = "") {
  statusEl.textContent = text;
  statusEl.className = type ? `status ${type}` : "status";
}

function renderAuthorityChecklist() {
  const configuredCount = [...state.cameras.values()].filter((camera) => camera.source_url).length;
  const hasFocusedConfigured = Boolean(state.focusedId && getCameraConfig(state.focusedId)?.source_url);
  const hasApiHealth = state.lastHealth?.status === "ok";
  const hasDetectionMode = Boolean(state.lastHealth?.mode);
  const hasRecentEventsSync = state.lastEventsRefreshAt > 0 && Date.now() - state.lastEventsRefreshAt <= EVENTS_REFRESH_MS * 2;
  const latencyOk = state.lastDetectionLatencyMs !== null && state.lastDetectionLatencyMs <= 1200;
  const noActiveAccident = Number(currentDetectionsEl.textContent || "0") === 0;

  const checks = [
    { label: "Backend API reachable", ok: hasApiHealth, bad: "API offline" },
    { label: "Detection mode available", ok: hasDetectionMode, bad: "Detector not configured" },
    { label: "At least one camera configured", ok: configuredCount > 0, bad: "No RTSP/IP source set" },
    { label: "Focused feed has a source", ok: hasFocusedConfigured, bad: "Pick an IP camera URL for focus" },
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

function updateConnectedCount() {
  const configuredCount = [...state.cameras.values()].filter((camera) => camera.source_url).length;
  connectedCountEl.textContent = `${configuredCount} camera${configuredCount === 1 ? "" : "s"} configured`;
}

function updateFocusedCardStyles() {
  state.items.forEach((itemState, id) => {
    if (!itemState.elements?.card) return;
    itemState.elements.card.classList.toggle("active", id === state.focusedId);
  });
}

async function loadCameraConfigs() {
  try {
    const response = await fetch("/api/cameras");
    if (!response.ok) throw new Error("Failed to load cameras");
    const data = await response.json();
    state.cameras.clear();
    (data.cameras || []).forEach((camera) => {
      state.cameras.set(camera.id, camera);
    });
  } catch (_error) {
    state.cameras.clear();
    intersections.forEach((intersection) => {
      state.cameras.set(intersection.id, { id: intersection.id, label: intersection.name, source_url: "", last_error: "" });
    });
  }
}

function syncTileFromCamera(itemState, camera, intersection) {
  if (!itemState.elements) return;

  const { preview, badge, sourceInput } = itemState.elements;
  if (sourceInput && document.activeElement !== sourceInput) {
    sourceInput.value = itemState.sourceDraft || camera.source_url || "";
  }

  if (camera.source_url) {
    preview.src = cameraStreamUrl(intersection.id);
    badge.textContent = camera.last_error ? `Error: ${camera.last_error}` : `Live: ${camera.label || intersection.name}`;
    badge.classList.toggle("live", !camera.last_error);
  } else {
    preview.src = placeholderFeed(intersection.name, "No source configured");
    badge.textContent = "Offline";
    badge.classList.remove("live");
  }
}

function renderIntersectionCards() {
  intersectionGrid.innerHTML = "";

  intersections.forEach((intersection) => {
    if (!state.items.has(intersection.id)) {
      state.items.set(intersection.id, makeInitialIntersectionState());
    }

    const itemState = state.items.get(intersection.id);
    const camera = getCameraConfig(intersection.id);

    const card = document.createElement("article");
    card.className = "intersection-card";
    card.dataset.intersectionId = intersection.id;
    card.innerHTML = `
      <h3>${intersection.name}</h3>
      <p class="subtle">${intersection.id.toUpperCase()}</p>
      <img class="feed-preview" alt="${intersection.name} camera preview" />
      <div class="intersection-controls">
        <label class="source-label" for="source-${intersection.id}">IP / RTSP camera URL</label>
        <input id="source-${intersection.id}" class="source-input" type="text" placeholder="rtsp://user:pass@192.168.1.20:554/stream or http://192.168.1.20:8080/video" />
        <p class="source-hint">Use an RTSP, HTTP, or IP camera URL. Save it to start the backend stream.</p>
        <div class="control-row">
          <button data-action="save">Apply Feed</button>
          <button data-action="edit" class="secondary">Edit URL</button>
          <button data-action="focus" class="secondary">Focus</button>
          <button data-action="clear" class="secondary">Clear</button>
        </div>
      </div>
      <span class="feed-badge">Offline</span>
    `;

    const preview = card.querySelector("img.feed-preview");
    const sourceInput = card.querySelector("input.source-input");
    const badge = card.querySelector(".feed-badge");
    const saveBtn = card.querySelector("button[data-action='save']");
    const editBtn = card.querySelector("button[data-action='edit']");
    const focusBtn = card.querySelector("button[data-action='focus']");
    const clearBtn = card.querySelector("button[data-action='clear']");

    sourceInput.value = itemState.sourceDraft || camera.source_url || "";

    sourceInput.addEventListener("input", () => {
      itemState.sourceDraft = sourceInput.value;
    });

    sourceInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        saveBtn.click();
      }
    });

    saveBtn.addEventListener("click", async () => {
      await saveCameraSource(intersection.id, sourceInput.value.trim(), intersection.name);
    });

    editBtn.addEventListener("click", () => openCameraDialog(intersection.id));

    clearBtn.addEventListener("click", async () => {
      try {
        const response = await fetch(`/api/cameras/${intersection.id}`, { method: "DELETE" });
        if (!response.ok) {
          const detail = await response.text();
          throw new Error(detail || "Failed to clear camera source");
        }

        itemState.sourceDraft = "";
        await loadCameraConfigs();
        renderIntersectionCards();
        if (state.focusedId === intersection.id) {
          setFocusedIntersection(intersection.id);
        }
        setStatus(`${intersection.name} cleared`, "ok");
      } catch (error) {
        setStatus(`Clear failed for ${intersection.name}: ${error.message}`, "alert");
      }
    });

    focusBtn.addEventListener("click", () => setFocusedIntersection(intersection.id));
    preview.addEventListener("click", () => setFocusedIntersection(intersection.id));

    itemState.elements = { card, preview, sourceInput, badge };
    syncTileFromCamera(itemState, camera, intersection);

    intersectionGrid.appendChild(card);
  });

  updateFocusedCardStyles();
  updateConnectedCount();
  renderAuthorityChecklist();
}

function openCameraDialog(cameraId) {
  const intersection = intersections.find((item) => item.id === cameraId);
  if (!intersection) return;

  const itemState = state.items.get(cameraId);
  const camera = getCameraConfig(cameraId);

  state.dialogCameraId = cameraId;
  cameraDialogTitle.textContent = `Set source for ${intersection.name}`;
  cameraDialogInput.value = itemState?.sourceDraft || camera.source_url || "";
  cameraDialog.classList.remove("hidden");
  cameraDialog.setAttribute("aria-hidden", "false");

  window.setTimeout(() => {
    cameraDialogInput.focus();
    cameraDialogInput.select();
  }, 0);
}

function closeCameraDialog() {
  state.dialogCameraId = null;
  cameraDialog.classList.add("hidden");
  cameraDialog.setAttribute("aria-hidden", "true");
}

async function saveCameraSource(cameraId, sourceUrl, label) {
  try {
    const response = await fetch(`/api/cameras/${cameraId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_url: sourceUrl, label }),
    });

    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || "Failed to save camera source");
    }

    const itemState = state.items.get(cameraId);
    if (itemState) {
      itemState.sourceDraft = sourceUrl;
    }

    await loadCameraConfigs();
    renderIntersectionCards();
    if (state.focusedId === cameraId) {
      setFocusedIntersection(cameraId);
    }
    setStatus(`${label} saved`, "ok");
    closeCameraDialog();
  } catch (error) {
    setStatus(`Save failed for ${label}: ${error.message}`, "alert");
  }
}

function setFocusedIntersection(intersectionId) {
  const intersection = intersections.find((entry) => entry.id === intersectionId);
  if (!intersection) return;

  state.focusedId = intersectionId;
  focusNameEl.textContent = intersection.name;

  const camera = getCameraConfig(intersectionId);
  if (camera.source_url) {
    // Clear any previous focus refresh timer
    if (state.focusRefreshTimer) {
      clearInterval(state.focusRefreshTimer);
      state.focusRefreshTimer = null;
    }

    // Use snapshot endpoint for focus view (single-image) and refresh periodically.
    const setSnapshot = () => {
      focusFeed.src = `/api/cameras/${intersectionId}/snapshot?ts=${Date.now()}`;
    };

    setSnapshot();
    state.focusRefreshTimer = setInterval(setSnapshot, 800);
    setStatus(`Focused on ${intersection.name}`, "ok");
  } else {
    // Clear any previous focus refresh timer
    if (state.focusRefreshTimer) {
      clearInterval(state.focusRefreshTimer);
      state.focusRefreshTimer = null;
    }

    focusFeed.src = placeholderFeed(intersection.name, "No source configured for this tile");
    clearOverlay();
    currentDetectionsEl.textContent = "0";
    setStatus(`${intersection.name} selected but no source is configured`, "alert");
  }

  updateFocusedCardStyles();
  renderAuthorityChecklist();
}

function clearOverlay() {
  const context = focusOverlay.getContext("2d");
  context.clearRect(0, 0, focusOverlay.width, focusOverlay.height);
}

function drawPredictions(predictions) {
  const frameWidth = focusFeed.naturalWidth || 0;
  const frameHeight = focusFeed.naturalHeight || 0;
  if (!frameWidth || !frameHeight) return;

  focusOverlay.width = focusFeed.clientWidth;
  focusOverlay.height = focusFeed.clientHeight;

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
  const camera = getCameraConfig(state.focusedId);
  if (!state.focusedId || !camera.source_url || state.detectionInFlight) return;

  state.detectionInFlight = true;
  try {
    const detectStartedAt = performance.now();
    const formData = new FormData();
    formData.append("camera_id", state.focusedId);
    formData.append("source_name", intersections.find((entry) => entry.id === state.focusedId)?.name || "unknown intersection");

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
      setStatus(`ALERT on ${camera.label || state.focusedId}: ${accidents} accident detection(s)`, "alert");
    } else {
      setStatus(`Monitoring ${camera.label || "focus feed"}`, "ok");
    }
    renderAuthorityChecklist();
  } finally {
    state.detectionInFlight = false;
  }
}

async function refreshEvents() {
  const [eventsResp, statsResp] = await Promise.all([fetch("/api/events?limit=20"), fetch("/api/stats")]);

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

  const camera = getCameraConfig(state.focusedId);
  if (!camera.source_url) {
    setStatus("Focused intersection has no RTSP/IP source configured", "alert");
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
  clearOverlay();
  currentDetectionsEl.textContent = "0";
  setStatus("Detection stopped");
  renderAuthorityChecklist();
}

startBtn.addEventListener("click", startDetection);
stopBtn.addEventListener("click", stopDetection);
wallModeBtn.addEventListener("click", () => setViewMode("wall"));
focusModeBtn.addEventListener("click", () => setViewMode("focus"));
window.addEventListener("resize", () => drawPredictions(state.lastPredictions));
focusFeed.addEventListener("load", () => drawPredictions(state.lastPredictions));
cameraDialogInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    cameraDialogSaveBtn.click();
  }
  if (event.key === "Escape") {
    event.preventDefault();
    closeCameraDialog();
  }
});
cameraDialogSaveBtn.addEventListener("click", () => {
  if (!state.dialogCameraId) return;
  const intersection = intersections.find((item) => item.id === state.dialogCameraId);
  if (!intersection) return;
  saveCameraSource(state.dialogCameraId, cameraDialogInput.value.trim(), intersection.name);
});
cameraDialogCancelBtn.addEventListener("click", closeCameraDialog);
cameraDialog.addEventListener("click", (event) => {
  if (event.target?.dataset?.action === "close") {
    closeCameraDialog();
  }
});

(async function init() {
  await loadCameraConfigs();
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
