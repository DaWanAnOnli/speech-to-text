// --- State ---
const MAX_RECONNECT = 5;

// Map of jobId -> JobState
// { ws, resultId, reconnectCount, status, data, cardEl }
const jobs = new Map();

// --- DOM helper ---
const $ = id => document.getElementById(id);

// --- Stage config ---
const STAGES = [
    'loading_model', 'converting', 'noise_reducing', 'enhancing',
    'transcribing', 'saving', 'complete'
];
const STAGE_IDX = {
    loading_model: 0, converting: 1, noise_reducing: 2, enhancing: 3,
    transcribing: 4, saving: 5, complete: 6, queued: -1, queued_starting: -1
};

// === Drag and Drop ===
const dropZone = $('dropZone');
const fileInput = $('fileInput');

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
});
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const files = Array.from(e.dataTransfer.files);
  if (files.length > 0) files.forEach(f => uploadFile(f));
});
dropZone.addEventListener('click', e => {
  if (e.target !== fileInput) fileInput.click();
});
dropZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') fileInput.click();
});
fileInput.addEventListener('change', () => {
  const files = Array.from(fileInput.files);
  if (files.length > 0) {
    files.forEach(f => uploadFile(f));
    fileInput.value = '';
  }
});

// === Upload ===
async function uploadFile(file) {
  const tempId = 'temp_' + Math.random().toString(36).slice(2);
  const cardEl = createJobCard(tempId, file.name);

  // Show jobs list
  const jobsList = $('jobsList');
  jobsList.hidden = false;

  // Initialize job state
  jobs.set(tempId, {
    ws: null,
    resultId: null,
    reconnectCount: 0,
    status: 'uploading',
    data: {},
    cardEl: cardEl
  });

  const formData = new FormData();
  formData.append('file', file);
  formData.append('model', $('modelSelect').value);
  formData.append('enable_noise_reduction', $('toggleNoiseReduction').checked ? 'true' : 'false');
  formData.append('enable_audio_enhancement', $('toggleAudioEnhancement').checked ? 'true' : 'false');

  try {
    const res = await fetch('/upload', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      showErrorForJob(tempId, err.detail || 'Upload failed');
      return;
    }
    const data = await res.json();
    const jobId = data.job_id;

    // Migrate from temp to real jobId
    const job = jobs.get(tempId);
    jobs.delete(tempId);
    jobs.set(jobId, job);

    // Update card DOM with real jobId
    cardEl.id = 'job-card-' + jobId;
    cardEl.dataset.jobId = jobId;
    cardEl.querySelector('[data-action="remove"]').onclick = () => removeJob(jobId);

    // Update button handlers to use jobId
    const dlBtn = cardEl.querySelector('[data-action="download"]');
    if (dlBtn) dlBtn.onclick = () => downloadResult(jobId);

    connectWebSocket(jobId);
  } catch (e) {
    showErrorForJob(tempId, 'Upload failed: ' + e.message);
  }
}

// === Job Card ===
function createStepperHTML() {
  const steps = [
    { key: 'loading_model', label: 'Model' },
    { key: 'converting', label: 'Convert' },
    { key: 'noise_reducing', label: 'Denoise' },
    { key: 'enhancing', label: 'Enhance' },
    { key: 'transcribing', label: 'Transcribe' },
    { key: 'saving', label: 'Save' },
    { key: 'complete', label: 'Done' }
  ];

  return steps.map((s, i) => {
    const hasLine = i < steps.length - 1;
    return `<div class="step" data-step="${s.key}">
      <div class="step-indicator">
        <div class="step-dot"></div>
        ${hasLine ? '<div class="step-line"></div>' : ''}
      </div>
      <span class="step-label">${s.label}</span>
    </div>`;
  }).join('');
}

function createJobCard(jobId, filename) {
  const jobsList = $('jobsList');
  const card = document.createElement('div');
  card.className = 'card panel panel--job';
  card.id = 'job-card-' + jobId;
  card.dataset.jobId = jobId;

  card.innerHTML = `
    <div class="panel-top">
      <div class="panel-info">
        <div class="panel-filename" data-field="filename">${filename}</div>
        <div class="panel-status-row">
          <span class="status-dot status-dot--active" data-field="dot"></span>
          <span class="panel-status" data-field="status">Uploading</span>
        </div>
      </div>
      <div class="panel-actions">
        <span class="panel-message" data-field="message">Uploading to server...</span>
        <button class="btn btn--icon btn--remove" data-action="remove" title="Remove">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
            <line x1="18" y1="6" x2="6" y2="18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            <line x1="6" y1="6" x2="18" y2="18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </button>
      </div>
    </div>
    <div class="stepper" data-field="stepper">
      ${createStepperHTML()}
    </div>
    <div class="progress-wrap">
      <div class="progress-bar">
        <div class="progress-fill" data-field="fill"></div>
      </div>
      <span class="progress-pct" data-field="pct">0%</span>
    </div>
    <div class="card-actions" data-field="actions" style="display:none">
      <button class="btn btn--primary" data-action="download">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
          <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"
            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <polyline points="7 10 12 15 17 10" stroke="currentColor" stroke-width="2"
            stroke-linecap="round" stroke-linejoin="round"/>
          <line x1="12" y1="15" x2="12" y2="3" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        </svg>
        Download
      </button>
    </div>
  `;

  jobsList.appendChild(card);
  return card;
}

// === WebSocket ===
function connectWebSocket(jobId) {
  const job = jobs.get(jobId);
  if (!job) return;

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  job.ws = new WebSocket(proto + '//' + location.host + '/ws/' + jobId);

  job.ws.onopen = () => { job.reconnectCount = 0; };
  job.ws.onmessage = evt => {
    try { handleProgress(jobId, JSON.parse(evt.data)); }
    catch(e) { console.error(e); }
  };
  job.ws.onclose = () => {
    const j = jobs.get(jobId);
    if (j && j.status === 'processing' && j.reconnectCount < MAX_RECONNECT) {
      j.reconnectCount++;
      setTimeout(() => connectWebSocket(jobId), 1000 * j.reconnectCount);
    }
  };
  job.ws.onerror = () => { job.ws?.close(); };
}

// === Progress Handling ===
function handleProgress(jobId, data) {
  const job = jobs.get(jobId);
  if (!job) return;
  const card = job.cardEl;
  if (!card) return;

  const fill = card.querySelector('[data-field="fill"]');
  const pct = card.querySelector('[data-field="pct"]');
  const msg = card.querySelector('[data-field="message"]');
  const dot = card.querySelector('[data-field="dot"]');
  const statusEl = card.querySelector('[data-field="status"]');
  const actions = card.querySelector('[data-field="actions"]');

  const progress = data.progress || 0;
  if (fill) fill.style.width = progress + '%';
  if (pct) pct.textContent = progress + '%';
  if (msg) msg.textContent = data.message || '';

  // Handle queued stage
  if (data.stage === 'queued') {
    job.status = 'queued';
    if (dot) dot.className = 'status-dot status-dot--queued';
    if (statusEl) statusEl.textContent = 'Queued';
    if (msg) msg.textContent = 'Waiting for a GPU slot...';
    return;
  }
  if (data.stage === 'queued_starting') {
    job.status = 'processing';
    if (dot) dot.className = 'status-dot status-dot--active';
    if (statusEl) statusEl.textContent = 'Processing';
    return;
  }

  if (data.stage === 'error') {
    job.status = 'error';
    if (dot) dot.className = 'status-dot status-dot--error';
    if (statusEl) statusEl.textContent = 'Error';
    if (fill) fill.classList.add('error');
    if (actions) actions.style.display = 'none';
    job.ws?.close();
    return;
  }

  updateCardStages(card, data.stage);

  if (data.stage === 'complete') {
    job.status = 'complete';
    job.resultId = data.result_id;
    job.data = data;
    if (dot) dot.className = 'status-dot status-dot--done';
    if (statusEl) statusEl.textContent = 'Complete';
    if (fill) fill.style.width = '100%';
    if (fill) fill.classList.remove('error');
    if (pct) pct.textContent = '100%';
    if (msg) msg.textContent = 'Transcription complete.';
    if (actions) actions.style.display = '';
    // Update navbar badge and footer with model name from first completed job
    if (data.model_name) {
      $('modelBadge').textContent = data.model_name;
      $('footerCredit').textContent = `Transcribed with ${data.model_name} · All processing done locally`;
    }
    job.ws?.close();
  }
}

function updateCardStages(card, currentStage) {
  const idx = STAGE_IDX[currentStage];
  if (idx === undefined || idx < 0) return;
  card.querySelectorAll('.step').forEach(el => el.classList.remove('active', 'done'));
  const steps = card.querySelectorAll('.step');
  for (let i = 0; i < idx; i++) {
    if (steps[i]) steps[i].classList.add('done');
  }
  if (idx < STAGES.length - 1 && steps[idx]) {
    steps[idx].classList.add('active');
  } else if (steps[idx]) {
    steps[idx].classList.add('done');
  }
}

function showErrorForJob(jobId, msg) {
  const job = jobs.get(jobId);
  if (!job) return;
  const card = job.cardEl;
  if (!card) return;
  job.status = 'error';
  const dot = card.querySelector('[data-field="dot"]');
  const statusEl = card.querySelector('[data-field="status"]');
  const msgEl = card.querySelector('[data-field="message"]');
  const fill = card.querySelector('[data-field="fill"]');
  const actions = card.querySelector('[data-field="actions"]');
  if (dot) dot.className = 'status-dot status-dot--error';
  if (statusEl) statusEl.textContent = 'Error';
  if (msgEl) msgEl.textContent = msg;
  if (fill) { fill.style.width = '100%'; fill.classList.add('error'); }
  if (actions) actions.style.display = 'none';
}

function removeJob(jobId) {
  const job = jobs.get(jobId);
  if (!job) return;
  job.ws?.close();
  jobs.delete(jobId);
  job.cardEl?.remove();
  if (jobs.size === 0) {
    $('jobsList').hidden = true;
  }
}

function downloadResult(jobId) {
  const job = jobs.get(jobId);
  if (!job || !job.resultId) return;
  window.location.href = '/download/' + job.resultId;
}
