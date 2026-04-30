const STORAGE_KEY = "academic-rag.sessions.v3";

const state = {
  sessions: [],
  activeSessionId: null,
  activeSources: [],
  jobs: new Map(),
  documents: [],
};

const els = {
  healthText: document.querySelector("#healthText"),
  refreshDocumentsBtn: document.querySelector("#refreshDocumentsBtn"),
  pdfInput: document.querySelector("#pdfInput"),
  jobList: document.querySelector("#jobList"),
  documentList: document.querySelector("#documentList"),
  sessionList: document.querySelector("#sessionList"),
  newSessionBtn: document.querySelector("#newSessionBtn"),
  clearCurrentChatBtn: document.querySelector("#clearCurrentChatBtn"),
  clearAllChatsBtn: document.querySelector("#clearAllChatsBtn"),
  activeSessionTitle: document.querySelector("#activeSessionTitle"),
  activeSessionMeta: document.querySelector("#activeSessionMeta"),
  agentToggle: document.querySelector("#agentToggle"),
  messages: document.querySelector("#messages"),
  chatForm: document.querySelector("#chatForm"),
  questionInput: document.querySelector("#questionInput"),
  memoryToggle: document.querySelector("#memoryToggle"),
  sendBtn: document.querySelector("#sendBtn"),
  searchForm: document.querySelector("#searchForm"),
  searchInput: document.querySelector("#searchInput"),
  sourceList: document.querySelector("#sourceList"),
  sourceCount: document.querySelector("#sourceCount"),
  pdfPanel: document.querySelector("#pdfPanel"),
  pdfTitle: document.querySelector("#pdfTitle"),
  pdfPreviewContent: document.querySelector("#pdfPreviewContent"),
  pdfOpenLink: document.querySelector("#pdfOpenLink"),
  closePdfBtn: document.querySelector("#closePdfBtn"),
};

function uid(prefix) {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function loadSessions() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    state.sessions = Array.isArray(parsed) ? parsed : [];
  } catch {
    state.sessions = [];
  }

  if (state.sessions.length === 0) {
    createSession(false);
    return;
  }

  state.activeSessionId = state.sessions[0].id;
  const lastAssistant = [...state.sessions[0].messages].reverse().find((item) => item.role === "assistant");
  state.activeSources = lastAssistant?.sources || [];
}

function saveSessions() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.sessions));
}

function createSession(render = true) {
  const now = new Date().toISOString();
  const session = {
    id: uid("chat"),
    title: "New chat",
    messages: [],
    createdAt: now,
    updatedAt: now,
  };
  state.sessions.unshift(session);
  state.activeSessionId = session.id;
  state.activeSources = [];
  saveSessions();
  if (render) renderAll();
  return session;
}

function activeSession() {
  return state.sessions.find((session) => session.id === state.activeSessionId);
}

function updateSessionTitle(session, question) {
  if (!session || session.title !== "New chat") return;
  session.title = question.trim().slice(0, 24) || "New chat";
}

function formatDate(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function renderAll() {
  renderSessions();
  renderMessages();
  renderSources(state.activeSources);
}

function renderSessions() {
  els.sessionList.innerHTML = "";
  for (const session of state.sessions) {
    const button = document.createElement("button");
    button.className = `session-item ${session.id === state.activeSessionId ? "active" : ""}`;
    button.type = "button";
    button.innerHTML = `
      <strong>${escapeHtml(session.title)}</strong>
      <div class="meta-line">${session.messages.length} messages - ${formatDate(session.updatedAt)}</div>
    `;
    button.addEventListener("click", () => {
      state.activeSessionId = session.id;
      const lastAssistant = [...session.messages].reverse().find((item) => item.role === "assistant");
      state.activeSources = lastAssistant?.sources || [];
      renderAll();
    });
    els.sessionList.appendChild(button);
  }
}

function renderMessages() {
  const session = activeSession();
  els.activeSessionTitle.textContent = session?.title || "New chat";
  els.activeSessionMeta.textContent = session
    ? `${session.id} - ${session.messages.length} messages`
    : "History is saved locally; backend memory uses session_id.";

  els.messages.innerHTML = "";
  if (!session || session.messages.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "Start asking questions about indexed papers. Sources will appear on the right.";
    els.messages.appendChild(empty);
    return;
  }

  for (const message of session.messages) {
    const item = document.createElement("article");
    item.className = `message ${message.role}`;
    const bubble = document.createElement("div");
    bubble.className = `bubble ${message.pending ? "pending" : ""}`;
    bubble.textContent = message.content || (message.pending ? "Thinking" : "");
    item.appendChild(bubble);
    els.messages.appendChild(item);
  }
  els.messages.scrollTop = els.messages.scrollHeight;
}

function renderDocuments(documents = []) {
  state.documents = documents;
  els.documentList.innerHTML = "";
  if (documents.length === 0) {
    els.documentList.innerHTML = `<div class="document-card"><strong>No papers</strong><div class="meta-line">Upload or index PDFs to show them here.</div></div>`;
    return;
  }

  for (const doc of documents) {
    const button = document.createElement("button");
    button.className = "document-card document-button";
    button.type = "button";
    const pages = doc.first_page && doc.last_page ? ` - pages ${doc.first_page}-${doc.last_page}` : "";
    const pdfText = doc.has_pdf ? "Click for preview" : "Original PDF not found";
    button.innerHTML = `
      <strong>${escapeHtml(doc.source_name || "unknown")}</strong>
      <div class="meta-line">${doc.chunk_count || 0} chunks${pages}</div>
      <div class="meta-line">${pdfText} - ${formatDate(doc.created_at)}</div>
    `;
    button.addEventListener("click", () => openDocumentPreview(doc));
    els.documentList.appendChild(button);
  }
}

async function openDocumentPreview(doc) {
  document.querySelector(".sources-panel")?.classList.add("pdf-open");
  els.pdfPanel.classList.add("visible");
  els.pdfTitle.textContent = doc.source_name || "Paper preview";
  els.pdfPreviewContent.textContent = "Loading preview...";
  els.pdfOpenLink.href = doc.pdf_url || "#";
  els.pdfOpenLink.style.display = doc.pdf_url ? "inline" : "none";

  try {
    const response = await fetch(`/documents/${doc.id}/preview`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Preview failed");
    els.pdfOpenLink.href = payload.pdf_url || doc.pdf_url || "#";
    els.pdfOpenLink.style.display = payload.pdf_url || doc.pdf_url ? "inline" : "none";
    els.pdfTitle.textContent =
      payload.preview_type === "abstract"
        ? `Abstract - ${payload.source_name || doc.source_name || "Paper"}`
        : `Preview - ${payload.source_name || doc.source_name || "Paper"}`;
    els.pdfPreviewContent.textContent =
      payload.preview_text ||
      "No text preview is available for this paper. Use the browser link to open the PDF.";
  } catch (error) {
    els.pdfPreviewContent.textContent = `Preview failed: ${error.message}`;
  }
}

function closePdf() {
  els.pdfPreviewContent.textContent = "";
  els.pdfOpenLink.href = "#";
  els.pdfOpenLink.style.display = "none";
  document.querySelector(".sources-panel")?.classList.remove("pdf-open");
  els.pdfPanel.classList.remove("visible");
  els.pdfTitle.textContent = "PDF preview";
}

function renderJobs() {
  els.jobList.innerHTML = "";
  for (const job of state.jobs.values()) {
    const card = document.createElement("article");
    card.className = "job-card";
    card.innerHTML = `
      <strong>${escapeHtml(job.filename || "Index job")}</strong>
      <div class="status ${job.status || ""}">${escapeHtml(job.status || "queued")}</div>
      <div class="meta-line">${job.chunks_added ? `${job.chunks_added} chunks` : "Waiting for worker"}</div>
    `;
    els.jobList.appendChild(card);
  }
}

function normalizeSources(sources = []) {
  return sources.map((source) => ({
    source: source.source || source.metadata?.source || "unknown",
    page: source.page ?? source.metadata?.page ?? null,
    score: source.score,
    chunk_id: source.chunk_id || source.id,
    content_preview: source.content_preview || source.text || source.content || "",
  }));
}

function renderSources(sources = []) {
  state.activeSources = normalizeSources(sources);
  els.sourceCount.textContent = String(state.activeSources.length);
  els.sourceList.innerHTML = "";
  if (state.activeSources.length === 0) {
    els.sourceList.innerHTML = `<div class="source-card"><strong>No sources</strong><p>Ask a question or search papers to show matching chunks here.</p></div>`;
    return;
  }

  for (const source of state.activeSources) {
    const card = document.createElement("article");
    card.className = "source-card";
    const page = source.page ? `page ${source.page}` : "page unknown";
    const score = typeof source.score === "number" ? source.score.toFixed(4) : source.score;
    card.innerHTML = `
      <strong>${escapeHtml(source.source)}</strong>
      <div class="meta-line">${page} - score ${escapeHtml(String(score || "-"))}</div>
      <p>${escapeHtml(source.content_preview)}</p>
    `;
    els.sourceList.appendChild(card);
  }
}

async function refreshHealth() {
  try {
    const response = await fetch("/health");
    const payload = await response.json();
    els.healthText.textContent = `Service ok - ${payload.index_size || 0} vectors`;
  } catch {
    els.healthText.textContent = "Service disconnected";
  }
}

async function refreshDocuments() {
  try {
    const response = await fetch("/documents");
    if (!response.ok) throw new Error("documents failed");
    const payload = await response.json();
    renderDocuments(payload.documents || []);
  } catch {
    renderDocuments([]);
  }
}

async function uploadPdf(file) {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch("/upload", { method: "POST", body: form });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Upload failed");
  }
  state.jobs.set(payload.job_id, payload);
  renderJobs();
  pollJob(payload.job_id);
}

async function pollJob(jobId) {
  const timer = setInterval(async () => {
    try {
      const response = await fetch(`/jobs/${jobId}`);
      const payload = await response.json();
      state.jobs.set(jobId, payload);
      renderJobs();
      if (["finished", "failed", "stopped", "canceled"].includes(payload.status)) {
        clearInterval(timer);
        await refreshHealth();
        await refreshDocuments();
      }
    } catch {
      clearInterval(timer);
    }
  }, 1800);
}

async function sendQuestion(question) {
  let session = activeSession();
  if (!session) session = createSession(false);

  updateSessionTitle(session, question);
  session.messages.push({ role: "user", content: question });
  const assistantMessage = {
    role: "assistant",
    content: "",
    sources: [],
    pending: true,
  };
  session.messages.push(assistantMessage);
  session.updatedAt = new Date().toISOString();
  saveSessions();
  renderAll();

  els.sendBtn.disabled = true;
  try {
    await streamQuery(question, session.id, assistantMessage);
    if (assistantMessage.sources.length === 0) {
      assistantMessage.sources = await fetchSourcesFallback(question);
      renderSources(assistantMessage.sources);
    }
  } catch (error) {
    assistantMessage.content = `Request failed: ${error.message}`;
  } finally {
    assistantMessage.pending = false;
    session.updatedAt = new Date().toISOString();
    saveSessions();
    els.sendBtn.disabled = false;
    renderAll();
  }
}

async function streamQuery(question, sessionId, assistantMessage) {
  const response = await fetch("/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      session_id: sessionId,
      use_agent: els.agentToggle.checked,
      use_memory: els.memoryToggle.checked,
    }),
  });

  if (!response.ok || !response.body) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || "Query API unavailable");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split(/\r?\n\r?\n/);
    buffer = events.pop() || "";
    for (const event of events) {
      handleSseEvent(event, assistantMessage);
    }
  }

  if (buffer.trim()) {
    handleSseEvent(buffer, assistantMessage);
  }
}

function handleSseEvent(rawEvent, assistantMessage) {
  const dataLines = rawEvent
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data: "))
    .map((line) => line.slice(6));
  if (dataLines.length === 0) return;

  const payload = JSON.parse(dataLines.join("\n"));
  if (payload.type === "sources") {
    assistantMessage.sources = normalizeSources(payload.data || []);
    renderSources(assistantMessage.sources);
  }
  if (payload.type === "token") {
    assistantMessage.content += payload.data || "";
    renderMessages();
  }
}

async function fetchSourcesFallback(query) {
  const response = await fetch("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: 8, score_threshold: 0 }),
  });
  const payload = await response.json();
  if (!response.ok) return [];
  return normalizeSources(payload.retrieved_chunks || []);
}

async function searchOnly(query) {
  const sources = await fetchSourcesFallback(query);
  renderSources(sources);
}

async function clearBackendMemory(sessionId) {
  const response = await fetch("/memory/clear", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "Failed to clear backend memory");
  }
}

async function clearCurrentChat() {
  const session = activeSession();
  if (!session) return;
  await clearBackendMemory(session.id);
  session.messages = [];
  session.updatedAt = new Date().toISOString();
  state.activeSources = [];
  saveSessions();
  renderAll();
}

async function clearAllChats() {
  await clearBackendMemory(null);
  state.sessions = [];
  createSession(false);
  saveSessions();
  renderAll();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

els.refreshDocumentsBtn.addEventListener("click", () => {
  refreshHealth();
  refreshDocuments();
});

els.pdfInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    await uploadPdf(file);
  } catch (error) {
    alert(error.message);
  } finally {
    els.pdfInput.value = "";
  }
});

els.newSessionBtn.addEventListener("click", () => createSession(true));
els.closePdfBtn.addEventListener("click", closePdf);
els.clearCurrentChatBtn.addEventListener("click", async () => {
  try {
    await clearCurrentChat();
  } catch (error) {
    alert(error.message);
  }
});
els.clearAllChatsBtn.addEventListener("click", async () => {
  if (!confirm("Clear all local chats and backend memory?")) return;
  try {
    await clearAllChats();
  } catch (error) {
    alert(error.message);
  }
});

els.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = els.questionInput.value.trim();
  if (!question) return;
  els.questionInput.value = "";
  await sendQuestion(question);
});

els.questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    els.chatForm.requestSubmit();
  }
});

els.searchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = els.searchInput.value.trim();
  if (!query) return;
  try {
    await searchOnly(query);
  } catch (error) {
    alert(error.message);
  }
});

loadSessions();
renderAll();
refreshHealth();
refreshDocuments();
