// --- State ---
let currentJobId = null;
let ws = null;
let currentResultId = null;
let reconnectCount = 0;
const MAX_RECONNECT = 5;

// --- DOM helper ---
const $ = id => document.getElementById(id);

// --- Stage config ---
const STAGES = [
    'loading_model', 'converting', 'noise_reducing', 'enhancing',
    'transcribing', 'saving', 'complete'
];
const STAGE_IDX = {
    loading_model: 0, converting: 1, noise_reducing: 2, enhancing: 3,
    transcribing: 4, saving: 5, complete: 6
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
  if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
});
dropZone.addEventListener('click', e => {
  if (e.target !== fileInput) fileInput.click();
});
dropZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') fileInput.click();
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) uploadFile(fileInput.files[0]);
});

// === Upload ===
async function uploadFile(file) {
  if (currentJobId) return;

  $('resultsPanel').hidden = true;
  $('jobPanel').hidden = false;
  $('activeFilename').textContent = file.name;
  setStatus('Processing', 'active');
  $('progressFill').style.width = '0%';
  $('progressFill').classList.remove('error');
  $('progressPct').textContent = '0%';
  $('jobMessage').textContent = 'Uploading file to server...';
  resetStages();

  const formData = new FormData();
  formData.append('file', file);
  formData.append('model', $('modelSelect').value);
  formData.append('enable_noise_reduction', $('toggleNoiseReduction').checked ? 'true' : 'false');
  formData.append('enable_audio_enhancement', $('toggleAudioEnhancement').checked ? 'true' : 'false');

  try {
    const res = await fetch('/upload', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      showError(err.detail || 'Upload failed');
      return;
    }
    const data = await res.json();
    currentJobId = data.job_id;
    setStatus('Processing', 'active');
    $('jobMessage').textContent = 'Connecting to progress stream...';
    connectWebSocket(currentJobId);
  } catch (e) {
    showError('Upload failed: ' + e.message);
  }
}

// === WebSocket ===
function connectWebSocket(jobId) {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/ws/' + jobId);
  ws.onopen = () => { reconnectCount = 0; };
  ws.onmessage = evt => {
    try { handleProgress(JSON.parse(evt.data)); }
    catch(e) { console.error(e); }
  };
  ws.onclose = () => {
    if (currentJobId && reconnectCount < MAX_RECONNECT) {
      reconnectCount++;
      setTimeout(() => connectWebSocket(jobId), 1000 * reconnectCount);
    }
  };
  ws.onerror = () => { ws.close(); };
}

// === Progress Handling ===
function handleProgress(data) {
  const stage = data.stage;
  const pct = data.progress || 0;
  $('progressFill').style.width = pct + '%';
  $('progressPct').textContent = pct + '%';
  $('jobMessage').textContent = data.message || '';

  if (stage === 'error') { showError(data.message); return; }

  updateStages(stage);

  if (stage === 'complete') {
    setStatus('Complete', 'done');
    $('jobMessage').textContent = 'Transcription complete.';
    currentResultId = data.result_id;
    currentJobId = null;
    if (ws) ws.close();
    showResults(data);
  }
}

function updateStages(currentStage) {
  const idx = STAGE_IDX[currentStage];
  if (idx === undefined) return;
  resetStages();
  for (let i = 0; i < idx; i++) markDone(STAGES[i]);
  if (idx < STAGES.length - 1) markActive(currentStage);
  else markDone(currentStage);
}

function resetStages() {
  document.querySelectorAll('.step').forEach(el => el.classList.remove('active', 'done'));
}
function markActive(name) {
  const el = document.querySelector('[data-step="' + name + '"]');
  if (el) el.classList.add('active');
}
function markDone(name) {
  const el = document.querySelector('[data-step="' + name + '"]');
  if (el) { el.classList.remove('active'); el.classList.add('done'); }
}
function setStatus(text, state) {
  const badge = $('statusBadge');
  const dot = $('statusDot');
  badge.textContent = text;
  dot.className = 'status-dot status-dot--' + state;
}

// === Results ===
function showResults(data) {
  $('resultFilename').textContent = data.filename || data.result_id;
  $('resultLang').textContent = data.language || 'unknown';
  $('resultModel').textContent = data.model_name || '';
  $('resultTime').textContent = new Date().toLocaleString();
  $('resultText').textContent = data.text || '(no transcription)';
  $('modelBadge').textContent = data.model_name || '';
  $('footerCredit').textContent = data.model_name
    ? `Transcribed with ${data.model_name} · All processing done locally`
    : 'All processing done locally';
  $('resultsPanel').hidden = false;
  $('jobPanel').hidden = true;
  $('resultsPanel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// === Download SRT ===
function downloadResult() {
  if (!currentResultId) return;
  window.location.href = '/download/' + currentResultId;
}

// === Utilities ===
function showError(msg) {
  setStatus('Error', 'error');
  $('jobMessage').textContent = msg;
  $('progressFill').style.width = '100%';
  $('progressFill').classList.add('error');
  currentJobId = null;
  resetStages();
}
