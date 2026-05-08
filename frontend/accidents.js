const galleryGrid = document.getElementById("galleryGrid");
const emptyState = document.getElementById("emptyState");
const latestIncident = document.getElementById("latestIncident");
const latestConfidence = document.getElementById("latestConfidence");
const galleryStatus = document.getElementById("galleryStatus");
const refreshBtn = document.getElementById("refreshBtn");
const analysisQueue = [];
let analysisQueueRunning = false;

function formatConfidence(value) {
  const numeric = Number(value || 0);
  return `${Math.round(numeric * 100)}%`;
}

function normalizeSeverity(value) {
  const text = String(value || "UNKNOWN").trim().toUpperCase();
  if (["HIGH", "FATAL", "SEVERE", "CRITICAL"].includes(text)) return "HIGH";
  if (["MEDIUM", "MAJOR"].includes(text)) return "MEDIUM";
  if (["LOW", "MINOR"].includes(text)) return "LOW";
  return "UNKNOWN";
}

function severityClass(value) {
  return `severity-${normalizeSeverity(value).toLowerCase()}`;
}

function buildDescription(event) {
  if (event.ai_summary) {
    return String(event.ai_summary)
      .split(/\n+/)
      .map((line) => line.trim())
      .filter(Boolean);
  }

  return [
    "Sending accident snapshot to local Moondream analysis.",
    "Generating incident summary and emergency guidance.",
  ];
}

function buildLoadingDescription() {
  return [
    "Passing this image to the local LLM.",
    "Waiting for an AI-generated report.",
  ];
}

function applyEventDetails(card, event, isLoading = false) {
  const titleEl = card.querySelector("[data-role='title']");
  const descriptionEl = card.querySelector("[data-role='description']");
  const tagsEl = card.querySelector("[data-role='tags']");
  const statusEl = card.querySelector("[data-role='status']");
  const overlayEl = card.querySelector("[data-role='overlay']");
  const severity = normalizeSeverity(event.severity);
  const cameraLabel = event.camera_id || event.source_name || "Unknown camera";
  const locationLabel = event.camera_location || cameraLabel;
  const emergencyLabel = event.emergency_status || (isLoading ? "Processing with local LLM" : "Pending");
  const confidence = formatConfidence(event.confidence);
  const timestamp = event.created_at ? new Date(event.created_at).toLocaleString() : "Unknown time";
  const descriptionLines = isLoading ? buildLoadingDescription() : buildDescription(event);

  titleEl.textContent = isLoading ? "Processing snapshot..." : "AI Incident Report";
  descriptionEl.innerHTML = descriptionLines.map((line) => `<p>${line}</p>`).join("");
  tagsEl.innerHTML = `
    <span class="tag severity ${severityClass(event.severity)}">Severity: ${severity}</span>
    <span class="tag accent">Confidence: ${confidence}</span>
    <span class="tag">Camera: ${cameraLabel}</span>
    <span class="tag alert">Location: ${locationLabel}</span>
    <span class="tag">Status: ${emergencyLabel}</span>
    <span class="tag">Time: ${timestamp}</span>
  `;
  statusEl.textContent = isLoading ? "Processing in local LLM" : "Ready";
  card.classList.toggle("is-loading", isLoading);
  overlayEl.hidden = !isLoading;
}

function renderEventCard(event) {
  const card = document.createElement("article");
  card.className = "card event-card";
  card.dataset.eventId = String(event.id);
  card.innerHTML = `
    <div class="card-image-wrap">
      <img src="${event.image_url}" alt="Accident snapshot ${event.id}" />
      <div class="processing-overlay" data-role="overlay">
        <div class="processing-spinner"></div>
        <div>
          <div class="processing-title">Local LLM processing</div>
          <div class="processing-subtitle">Analyzing accident snapshot with Moondream</div>
        </div>
      </div>
    </div>
    <div class="card-body">
      <div class="card-title" data-role="title"></div>
      <div class="description" data-role="description"></div>
      <div class="tags" data-role="tags"></div>
      <div class="card-status" data-role="status"></div>
    </div>
  `;

  const needsAnalysis = !String(event.ai_summary || "").trim() || normalizeSeverity(event.severity) === "UNKNOWN" || !String(event.emergency_status || "").trim();
  applyEventDetails(card, event, needsAnalysis);

  if (needsAnalysis) {
    enqueueAnalysis(event, card);
  }

  return card;
}

async function analyzeEvent(event, card) {
  if (card.dataset.analysisStarted === "true") {
    console.log(`[LLM ANALYSIS] Event ${event.id} already analyzed, skipping`);
    return;
  }
  card.dataset.analysisStarted = "true";
  console.log(`[LLM ANALYSIS] Starting LLM analysis for Event ${event.id}`);

  try {
    console.log(`[LLM ANALYSIS] POSTing to /api/events/${event.id}/analyze`);
    const response = await fetch(`/api/events/${event.id}/analyze`, { method: "POST" });
    
    if (!response.ok) {
      console.error(`[LLM ANALYSIS] Backend error for Event ${event.id}: HTTP ${response.status}`);
      throw new Error(`Failed to analyze event ${event.id}`);
    }

    const data = await response.json();
    console.log(`[LLM ANALYSIS] Event ${event.id} analysis complete:`, data);
    console.log(`[LLM ANALYSIS] AI Summary received: ${data.ai_summary?.substring(0, 100) || "(empty)"}...`);
    console.log(`[LLM ANALYSIS] Severity: ${data.severity}, Emergency Status: ${data.emergency_status}`);
    
    const updatedEvent = {
      ...event,
      ai_summary: data.ai_summary,
      severity: data.severity,
      emergency_status: data.emergency_status,
      camera_id: data.camera_id || event.camera_id,
      camera_location: data.camera_location || event.camera_location,
    };
    console.log(`[LLM ANALYSIS] Updating card DOM for Event ${event.id}`);
    applyEventDetails(card, updatedEvent, false);
  } catch (error) {
    console.error(`[LLM ANALYSIS] FAILED for Event ${event.id}:`, error.message);
    applyEventDetails(
      card,
      {
        ...event,
        ai_summary: event.ai_summary || "AI analysis unavailable.",
        severity: event.severity || "UNKNOWN",
        emergency_status: event.emergency_status || "Emergency response recommended.",
      },
      false,
    );
  }
}

function enqueueAnalysis(event, card) {
  if (card.dataset.analysisComplete === "true") {
    console.log(`[LLM ANALYSIS] Event ${event.id} already completed, skipping queue`);
    return;
  }

  console.log(`[LLM ANALYSIS] Queuing Event ${event.id} for sequential analysis`);
  analysisQueue.push({ event, card });
  processAnalysisQueue();
}

async function processAnalysisQueue() {
  if (analysisQueueRunning) {
    return;
  }

  analysisQueueRunning = true;

  try {
    while (analysisQueue.length > 0) {
      const nextItem = analysisQueue.shift();
      if (!nextItem) {
        continue;
      }

      const { event, card } = nextItem;
      if (!card.isConnected || card.dataset.analysisComplete === "true") {
        console.log(`[LLM ANALYSIS] Event ${event.id} is no longer eligible, skipping`);
        continue;
      }

      console.log(`[LLM ANALYSIS] Processing next queued event ${event.id}`);
      await analyzeEvent(event, card);
      card.dataset.analysisComplete = "true";
      console.log(`[LLM ANALYSIS] Event ${event.id} finished and marked complete`);
    }
  } finally {
    analysisQueueRunning = false;
  }
}

async function loadGallery() {
  console.log("[LLM ANALYSIS] Gallery load started");
  galleryStatus.textContent = "Fetching events";
  galleryGrid.innerHTML = "";

  try {
    const response = await fetch("/api/events?limit=24");
    if (!response.ok) throw new Error("Failed to load events");

    const data = await response.json();
    console.log(`[LLM ANALYSIS] Loaded ${data.events?.length || 0} events from backend`);
    
    const events = (data.events || []).filter((event) => typeof event.image_url === "string" && event.image_url.startsWith("http"));
    console.log(`[LLM ANALYSIS] Filtered to ${events.length} events with valid images`);

    if (!events.length) {
      emptyState.hidden = false;
      latestIncident.textContent = "No incidents yet";
      latestConfidence.textContent = "--";
      galleryStatus.textContent = "No images available";
      return;
    }

    emptyState.hidden = true;
    galleryStatus.textContent = `${events.length} images loaded`;

    const latest = events[0];
    latestIncident.textContent = `${normalizeSeverity(latest.severity)} · ${latest.source_name || latest.class_name || "Unknown"}`;
    latestConfidence.textContent = formatConfidence(latest.confidence);

    events.forEach((event) => {
      console.log(`[LLM ANALYSIS] Rendering Event ${event.id} - has ai_summary: ${Boolean(event.ai_summary)}`);
      galleryGrid.appendChild(renderEventCard(event));
    });
  } catch (error) {
    emptyState.hidden = false;
    emptyState.textContent = "Unable to load accident images right now.";
    galleryStatus.textContent = "Load failed";
    latestIncident.textContent = "Unavailable";
    latestConfidence.textContent = "--";
  }
}

refreshBtn.addEventListener("click", loadGallery);
loadGallery();
