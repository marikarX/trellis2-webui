/* ============================================================
   TRELLIS.2 WebUI — Main Application Logic
   ============================================================ */

// -------- State --------
const state = {
  activeTab: 'single',
  singleImage: null,       // File
  multiviewImages: [null, null, null, null, null, null],
  multiviewLabels: ['Front', 'Front Left', 'Left', 'Back', 'Back Right', 'Right'],
  multiviewPreviewUrls: [null, null, null, null, null, null],
  pendingViewIndex: null,
  meshFile: null,
  texImage: null,
  sessionId: null,
  previews: null,           // { render_key: [base64...] }
  currentRenderMode: 'shaded_forest',
  currentAngle: 3,
  glbUrl: null,
  generating: false,
  extracting: false,
  currentPreset: '16gb-fast',
  presetTouched: false,
  runtimeInfo: null,
  currentSurface: 'hard',
};

// -------- Preset Definitions --------
// Each preset defines base values; surface type adds guidance offsets.
const PRESETS = {
  '16gb-fast': {
    ss_gs: 7.5, ss_gr: 0.7, ss_steps: 12, ss_rt: 5.0,
    shape_gs: 7.5, shape_gr: 0.5, shape_steps: 12, shape_rt: 3.0,
    tex_gs: 1.0, tex_gr: 0.0, tex_steps: 12, tex_rt: 3.0,
    decimation: 500000, texSize: 2048, resolution: '512',
  },
  '16gb-quality': {
    ss_gs: 8.0, ss_gr: 0.7, ss_steps: 20, ss_rt: 5.0,
    shape_gs: 8.0, shape_gr: 0.5, shape_steps: 20, shape_rt: 3.0,
    tex_gs: 2.0, tex_gr: 0.1, tex_steps: 20, tex_rt: 3.0,
    decimation: 800000, texSize: 2048, resolution: '512',
  },
  '24gb-high': {
    ss_gs: 8.0, ss_gr: 0.7, ss_steps: 30, ss_rt: 5.0,
    shape_gs: 8.5, shape_gr: 0.5, shape_steps: 30, shape_rt: 4.0,
    tex_gs: 2.5, tex_gr: 0.15, tex_steps: 30, tex_rt: 4.0,
    decimation: 800000, texSize: 4096, resolution: '1024',
  },
  maximum: {
    ss_gs: 8.0, ss_gr: 0.7, ss_steps: 50, ss_rt: 6.0,
    shape_gs: 8.5, shape_gr: 0.5, shape_steps: 50, shape_rt: 6.0,
    tex_gs: 2.5, tex_gr: 0.2, tex_steps: 50, tex_rt: 6.0,
    decimation: 1000000, texSize: 4096, resolution: '1536',
  },
};

const VIEW_LABELS = [
  'Front', 'Front Left', 'Left', 'Back Left', 'Back', 'Back Right',
  'Right', 'Front Right', 'Top', 'Bottom', 'Custom',
];

// Surface type guidance offsets (added to preset base)
const SURFACE_OFFSETS = {
  hard:    { ss_gs: +0.5, shape_gs: +0.5, tex_gs: 0 },   // higher guidance for crisp edges
  organic: { ss_gs: -0.5, shape_gs: -1.0, tex_gs: +0.5 }, // softer shape, richer textures
};

// -------- DOM Refs --------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// -------- Init --------
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initSingleUpload();
  initMultiViewUpload();
  initTexturingUpload();
  initRadioGroups();
  initSliders();
  initPresets();
  initSurfaceType();
  initGenerateButton();
  initViewerModes();
  initRenderModes();
  initAngleSlider();
  initExtractButton();
  loadExamples();
  applyPreset(); // Set initial slider values from default preset
  loadRuntimeInfo();
});

async function loadRuntimeInfo() {
  try {
    const response = await fetch('/api/runtime');
    if (!response.ok) throw new Error('Runtime details unavailable');
    const info = await response.json();
    state.runtimeInfo = info;
    const gpuSummary = $('#gpuSummary');
    const recommendation = $('#runtimeRecommendation');
    if (!info.cuda_available) {
      gpuSummary.textContent = 'CUDA GPU unavailable';
      recommendation.textContent = 'The UI remains available; GPU generation needs a CUDA runtime.';
    } else {
      const vram = Number(info.total_vram_gb).toFixed(1);
      gpuSummary.textContent = `GPU: ${info.gpu_name} · ${vram} GB VRAM`;
      const recommended = prettyPreset(info.recommended_preset);
      const selected = info.default_preset === info.recommended_preset
        ? ''
        : ` · selected: ${prettyPreset(info.default_preset)}`;
      const runtimeWarnings = (info.warnings || []).join(' ');
      recommendation.textContent = `Recommended: ${recommended}${selected} · CPU offload ${info.cpu_offload ? 'on' : 'off'} · ${info.attention_backend}${runtimeWarnings ? ` · ${runtimeWarnings}` : ''}`;
      recommendation.classList.toggle('warning-text', Boolean(runtimeWarnings));
    }
    $('#runtimeStatus').title = [
      info.precision,
      `BF16: ${info.bf16_supported ? 'supported' : 'not reported'}`,
      ...(info.warnings || []).filter(Boolean),
    ].join('\n');
    if (!state.presetTouched && PRESETS[info.default_preset]) {
      state.currentPreset = info.default_preset;
      $$('[data-preset]').forEach((button) => button.classList.toggle('active', button.dataset.preset === state.currentPreset));
      applyPreset();
    }
  } catch (error) {
    $('#gpuSummary').textContent = 'GPU status unavailable';
    $('#runtimeRecommendation').textContent = 'Check the server runtime log for CUDA details.';
  }
}

function prettyPreset(preset) {
  return ({
    '16gb-fast': '16GB Fast',
    '16gb-quality': '16GB Quality',
    '24gb-high': '24GB+ High',
    maximum: 'Maximum',
  })[preset] || preset || '16GB Fast';
}

// ============================================================
// TABS
// ============================================================
function initTabs() {
  $$('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      state.activeTab = tab;
      $$('.tab-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      $$('.tab-content').forEach((c) => {
        c.classList.toggle('active', c.dataset.tab === tab);
      });
    });
  });
}

// ============================================================
// SINGLE IMAGE UPLOAD
// ============================================================
function initSingleUpload() {
  const zone = $('#singleUpload');
  const input = $('#singleFileInput');
  const preview = $('#singlePreview');
  const placeholder = $('#singlePlaceholder');

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleSingleFile(e.dataTransfer.files[0]);
  });
  input.addEventListener('change', () => {
    if (input.files.length) handleSingleFile(input.files[0]);
  });

  function handleSingleFile(file) {
    state.singleImage = file;
    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.hidden = false;
    zone.classList.add('has-image');
  }
}

// ============================================================
// MULTI-VIEW UPLOAD
// ============================================================
function initMultiViewUpload() {
  const grid = $('#mvGrid');
  const input = $('#mvFileInput');
  const dropZone = $('#mvDropZone');
  const addButton = $('#mvAddButton');
  const labelOptions = VIEW_LABELS.map((label) => `<option value="${label}">${label}</option>`).join('');

  for (let idx = 0; idx < 6; idx += 1) {
    const slot = document.createElement('div');
    slot.className = 'mv-slot';
    slot.dataset.index = String(idx);
    slot.innerHTML = `
      <div class="mv-placeholder"><span>Drop or select View ${idx + 1}</span></div>
      <img class="mv-preview" alt="View ${idx + 1}" />
      <select class="mv-label" aria-label="View ${idx + 1} label">${labelOptions}</select>
      <button type="button" class="mv-remove" aria-label="Remove view ${idx + 1}" hidden>×</button>
    `;
    const selector = slot.querySelector('.mv-label');
    const remove = slot.querySelector('.mv-remove');
    selector.value = state.multiviewLabels[idx];
    selector.addEventListener('click', (event) => event.stopPropagation());
    selector.addEventListener('change', () => { state.multiviewLabels[idx] = selector.value; });
    remove.addEventListener('click', (event) => {
      event.stopPropagation();
      clearMvSlot(idx, slot);
    });

    slot.addEventListener('click', () => {
      state.pendingViewIndex = idx;
      input.click();
    });
    slot.addEventListener('dragover', (event) => { event.preventDefault(); slot.classList.add('dragover'); });
    slot.addEventListener('dragleave', () => slot.classList.remove('dragover'));
    slot.addEventListener('drop', (event) => {
      event.preventDefault();
      event.stopPropagation();
      slot.classList.remove('dragover');
      assignMvFiles(event.dataTransfer.files, idx);
    });
    grid.appendChild(slot);
  }

  addButton.addEventListener('click', () => {
    state.pendingViewIndex = null;
    input.click();
  });
  input.addEventListener('change', () => {
    assignMvFiles(input.files, state.pendingViewIndex);
    state.pendingViewIndex = null;
    input.value = '';
  });
  dropZone.addEventListener('dragover', (event) => { event.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropZone.classList.remove('dragover');
    state.pendingViewIndex = null;
    assignMvFiles(event.dataTransfer.files, null);
  });
}

function updateMvCount() {
  const count = state.multiviewImages.filter((f) => f !== null).length;
  $('#mvCount').textContent = `${count} / 6 views loaded`;
  $('#mvDropZone').textContent = count >= 6
    ? 'All 6 view slots are filled. Remove a view to replace it.'
    : `Drop ${count ? 'more ' : ''}images here (${count} / 6 loaded), or choose files above`;
}

function assignMvFiles(fileList, startIndex = null) {
  const files = Array.from(fileList || []).filter((file) => file.type.startsWith('image/'));
  if (!files.length) return;
  let target = Number.isInteger(startIndex) ? startIndex : state.multiviewImages.findIndex((file) => file === null);
  let accepted = 0;
  files.forEach((file) => {
    while (target < 6 && state.multiviewImages[target] !== null && target !== startIndex) target += 1;
    if (target < 0) target = state.multiviewImages.findIndex((item) => item === null);
    if (target < 0 || target >= 6) return;
    const slot = $(`.mv-slot[data-index="${target}"]`);
    if (state.multiviewPreviewUrls[target]) URL.revokeObjectURL(state.multiviewPreviewUrls[target]);
    state.multiviewImages[target] = file;
    state.multiviewPreviewUrls[target] = URL.createObjectURL(file);
    slot.querySelector('.mv-preview').src = state.multiviewPreviewUrls[target];
    slot.querySelector('.mv-remove').hidden = false;
    slot.classList.add('has-image');
    accepted += 1;
    target += 1;
  });
  updateMvCount();
  if (accepted < files.length) alert('The multi-view workflow supports up to 6 images.');
}

function clearMvSlot(index, slot) {
  if (state.multiviewPreviewUrls[index]) URL.revokeObjectURL(state.multiviewPreviewUrls[index]);
  state.multiviewPreviewUrls[index] = null;
  state.multiviewImages[index] = null;
  slot.querySelector('.mv-preview').removeAttribute('src');
  slot.querySelector('.mv-remove').hidden = true;
  slot.classList.remove('has-image');
  updateMvCount();
}

// ============================================================
// TEXTURING UPLOAD
// ============================================================
function initTexturingUpload() {
  // Mesh upload
  const meshZone = $('#meshUpload');
  const meshInput = $('#meshFileInput');
  meshZone.addEventListener('click', () => meshInput.click());
  meshZone.addEventListener('dragover', (e) => { e.preventDefault(); meshZone.classList.add('dragover'); });
  meshZone.addEventListener('dragleave', () => meshZone.classList.remove('dragover'));
  meshZone.addEventListener('drop', (e) => {
    e.preventDefault();
    meshZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleMeshFile(e.dataTransfer.files[0]);
  });
  meshInput.addEventListener('change', () => {
    if (meshInput.files.length) handleMeshFile(meshInput.files[0]);
  });

  function handleMeshFile(file) {
    state.meshFile = file;
    meshZone.classList.add('has-image');
    $('#meshPlaceholder').hidden = true;
    $('#meshInfo').hidden = false;
    $('#meshFileName').textContent = file.name;
  }

  // Texture reference image upload
  const texZone = $('#texImageUpload');
  const texInput = $('#texImageInput');
  const texPreview = $('#texImagePreview');
  texZone.addEventListener('click', () => texInput.click());
  texZone.addEventListener('dragover', (e) => { e.preventDefault(); texZone.classList.add('dragover'); });
  texZone.addEventListener('dragleave', () => texZone.classList.remove('dragover'));
  texZone.addEventListener('drop', (e) => {
    e.preventDefault();
    texZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleTexImage(e.dataTransfer.files[0]);
  });
  texInput.addEventListener('change', () => {
    if (texInput.files.length) handleTexImage(texInput.files[0]);
  });

  function handleTexImage(file) {
    state.texImage = file;
    texPreview.src = URL.createObjectURL(file);
    texPreview.hidden = false;
    texZone.classList.add('has-image');
  }
}

// ============================================================
// RADIO GROUPS
// ============================================================
function initRadioGroups() {
  $$('.radio-group').forEach((group) => {
    group.querySelectorAll('.radio-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        group.querySelectorAll('.radio-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        if (group.id === 'resolutionGroup') updateResolutionWarning();
      });
    });
  });
}

// ============================================================
// SLIDERS
// ============================================================
function initSliders() {
  const sliderMap = {
    ss_gs: 'ss_gs_val',
    ss_gr: 'ss_gr_val',
    ss_steps: 'ss_steps_val',
    ss_rt: 'ss_rt_val',
    shape_gs: 'shape_gs_val',
    shape_gr: 'shape_gr_val',
    shape_steps: 'shape_steps_val',
    shape_rt: 'shape_rt_val',
    tex_gs: 'tex_gs_val',
    tex_gr: 'tex_gr_val',
    tex_steps: 'tex_steps_val',
    tex_rt: 'tex_rt_val',
  };

  Object.entries(sliderMap).forEach(([sliderId, valId]) => {
    const slider = $(`#${sliderId}`);
    const val = $(`#${valId}`);
    if (slider && val) {
      slider.addEventListener('input', () => {
        val.textContent = slider.value;
      });
    }
  });

  // Decimation slider
  const decSlider = $('#decimationTarget');
  const decVal = $('#dec_val');
  if (decSlider && decVal) {
    decSlider.addEventListener('input', () => {
      const v = parseInt(decSlider.value);
      decVal.textContent = v >= 1000000 ? `${(v / 1000000).toFixed(1)}M` : `${Math.round(v / 1000)}K`;
    });
  }

  // Texture size slider
  const texSizeSlider = $('#textureSize');
  const texSizeVal = $('#texsize_val');
  if (texSizeSlider && texSizeVal) {
    texSizeSlider.addEventListener('input', () => {
      texSizeVal.textContent = texSizeSlider.value;
    });
  }
}

// ============================================================
// PRESETS & SURFACE TYPE
// ============================================================
function initPresets() {
  $$('.preset-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        $$('.preset-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        state.currentPreset = btn.dataset.preset;
        state.presetTouched = true;
        applyPreset();
    });
  });
}

function initSurfaceType() {
  $$('.surface-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      $$('.surface-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      state.currentSurface = btn.dataset.surface;
      applyPreset();
    });
  });
}

function applyPreset() {
  const preset = PRESETS[state.currentPreset];
  const offset = SURFACE_OFFSETS[state.currentSurface];
  if (!preset) return;

  // Helper: set slider value and update displayed label
  function setSlider(id, valId, value, format) {
    const slider = $(`#${id}`);
    const label = $(`#${valId}`);
    if (slider) {
      // Clamp to slider min/max
      const clamped = Math.max(parseFloat(slider.min), Math.min(parseFloat(slider.max), value));
      slider.value = clamped;
      if (label) label.textContent = format ? format(clamped) : clamped;
    }
  }

  // Stage 1: Sparse Structure
  setSlider('ss_gs', 'ss_gs_val', preset.ss_gs + (offset.ss_gs || 0));
  setSlider('ss_gr', 'ss_gr_val', preset.ss_gr);
  setSlider('ss_steps', 'ss_steps_val', preset.ss_steps);
  setSlider('ss_rt', 'ss_rt_val', preset.ss_rt);

  // Stage 2: Shape
  setSlider('shape_gs', 'shape_gs_val', preset.shape_gs + (offset.shape_gs || 0));
  setSlider('shape_gr', 'shape_gr_val', preset.shape_gr);
  setSlider('shape_steps', 'shape_steps_val', preset.shape_steps);
  setSlider('shape_rt', 'shape_rt_val', preset.shape_rt);

  // Stage 3: Texture
  setSlider('tex_gs', 'tex_gs_val', preset.tex_gs + (offset.tex_gs || 0));
  setSlider('tex_gr', 'tex_gr_val', preset.tex_gr);
  setSlider('tex_steps', 'tex_steps_val', preset.tex_steps);
  setSlider('tex_rt', 'tex_rt_val', preset.tex_rt);

  // Export
  setSlider('decimationTarget', 'dec_val', preset.decimation, (v) => {
    return v >= 1000000 ? `${(v / 1000000).toFixed(1)}M` : `${Math.round(v / 1000)}K`;
  });
  setSlider('textureSize', 'texsize_val', preset.texSize);
  $$('#resolutionGroup .radio-btn').forEach((button) => {
    button.classList.toggle('active', button.dataset.value === preset.resolution);
  });
  updateResolutionWarning();
}

function updateResolutionWarning() {
  const resolution = $('#resolutionGroup .radio-btn.active')?.dataset.value || '512';
  const lowVram = (state.runtimeInfo?.total_vram_gb ?? 16) < 24;
  const warning = $('#resolutionWarning');
  if (lowVram && resolution !== '512') {
    warning.textContent = `${resolution} geometry is experimental below 24 GB VRAM and may OOM. Use a 16GB preset for the 512 mode.`;
    warning.classList.add('warning-text');
  } else {
    warning.textContent = 'The 16GB presets use 512 geometry. Higher resolutions are experimental below 24GB.';
    warning.classList.remove('warning-text');
  }
}

// ============================================================
// GENERATE BUTTON
// ============================================================
function initGenerateButton() {
  const btn = $('#generateBtn');
  btn.addEventListener('click', () => {
    if (state.generating) return;
    if (state.activeTab === 'single') generateSingle();
    else if (state.activeTab === 'multiview') generateMultiView();
    else if (state.activeTab === 'texturing') generateTexturing();
  });
}

function getSettings() {
  return {
    resolution: $('#resolutionGroup .radio-btn.active')?.dataset.value || '512',
    execution_preset: state.currentPreset,
    seed: parseInt($('#seedInput').value) || 0,
    randomize_seed: $('#randomizeSeed').checked,
    ss_guidance_strength: parseFloat($('#ss_gs').value),
    ss_guidance_rescale: parseFloat($('#ss_gr').value),
    ss_sampling_steps: parseInt($('#ss_steps').value),
    ss_rescale_t: parseFloat($('#ss_rt').value),
    shape_guidance_strength: parseFloat($('#shape_gs').value),
    shape_guidance_rescale: parseFloat($('#shape_gr').value),
    shape_sampling_steps: parseInt($('#shape_steps').value),
    shape_rescale_t: parseFloat($('#shape_rt').value),
    tex_guidance_strength: parseFloat($('#tex_gs').value),
    tex_guidance_rescale: parseFloat($('#tex_gr').value),
    tex_sampling_steps: parseInt($('#tex_steps').value),
    tex_rescale_t: parseFloat($('#tex_rt').value),
  };
}

function setGenerating(isGenerating) {
  state.generating = isGenerating;
  const btn = $('#generateBtn');
  btn.disabled = isGenerating;
  btn.classList.toggle('loading', isGenerating);

  const overlay = $('#progressOverlay');
  overlay.hidden = !isGenerating;

  if (!isGenerating) {
    // Reset progress
    updateProgress(0, 'Ready');
  }
}

function showGenerationStats(data) {
  const stats = $('#generationStats');
  const peak = data.peak_vram_gb != null && Number.isFinite(Number(data.peak_vram_gb))
    ? `${Number(data.peak_vram_gb).toFixed(2)} GB peak allocated`
    : 'Peak VRAM unavailable';
  const views = Number(data.num_views || 1);
  const resolutionLabel = data.resolution_label || `${data.resolution} geometry`;
  stats.textContent = `${prettyPreset(data.preset)} · ${resolutionLabel} · ${views} view${views === 1 ? '' : 's'} · ${peak}`;
  if (data.warning) stats.textContent += ` — ${data.warning}`;
  stats.hidden = false;
}

function updateProgress(percent, stage) {
  const circumference = 2 * Math.PI * 52; // r=52
  const offset = circumference - (percent / 100) * circumference;
  $('#progressRingFill').style.strokeDashoffset = offset;
  $('#progressPercent').textContent = `${Math.round(percent)}%`;
  $('#progressStage').textContent = stage;
}

// SSE progress poller — connects to /api/progress/{taskId} and updates the
// progress ring in real time.  Returns a stop() function to close the stream.
function pollProgress(taskId) {
  const es = new EventSource(`/api/progress/${taskId}`);
  es.onmessage = (evt) => {
    try {
      const info = JSON.parse(evt.data);
      const pct = info.step != null && info.total ? Math.round((info.step / info.total) * 100) : 0;
      updateProgress(pct, info.stage || 'Processing…');
      if (info.done) es.close();
    } catch (_) { /* ignore parse errors */ }
  };
  es.onerror = () => { es.close(); };
  return () => es.close();
}

async function generateSingle() {
  if (!state.singleImage) {
    alert('Please upload an image first.');
    return;
  }
  setGenerating(true);
  updateProgress(5, 'Uploading image…');

  const taskId = crypto.randomUUID();
  const settings = getSettings();
  const formData = new FormData();
  formData.append('image', state.singleImage);
  formData.append('task_id', taskId);
  Object.entries(settings).forEach(([k, v]) => formData.append(k, String(v)));

  // Start SSE progress polling immediately
  const stopPolling = pollProgress(taskId);

  try {
    const res = await fetch('/api/generate', { method: 'POST', body: formData });
    stopPolling();
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Generation failed');
    }
    const data = await res.json();
    updateProgress(100, 'Done!');
    state.sessionId = data.session_id;
    state.previews = data.previews;
    state.glbUrl = null;
    showGenerationStats(data);
    showPreviews();
  } catch (err) {
    stopPolling();
    alert(`Error: ${err.message}`);
  } finally {
    setGenerating(false);
  }
}

async function generateMultiView() {
  const images = state.multiviewImages
    .map((file, index) => ({ file, index, label: state.multiviewLabels[index] }))
    .filter((entry) => entry.file !== null);
  if (images.length < 2) {
    alert('Please upload at least 2 images for multi-view.');
    return;
  }
  setGenerating(true);
  updateProgress(5, 'Uploading images…');

  const taskId = crypto.randomUUID();
  const settings = getSettings();
  const formData = new FormData();
  formData.append('task_id', taskId);
  images.forEach(({ file, index, label }) => {
    formData.append(`image_${index}`, file);
    formData.append(`view_label_${index}`, label);
  });
  Object.entries(settings).forEach(([k, v]) => formData.append(k, String(v)));

  // Start SSE progress polling immediately
  const stopPolling = pollProgress(taskId);

  try {
    const res = await fetch('/api/generate-multiview', { method: 'POST', body: formData });
    stopPolling();
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Generation failed');
    }
    const data = await res.json();
    updateProgress(100, 'Done!');
    state.sessionId = data.session_id;
    state.previews = data.previews;
    state.glbUrl = null;
    showGenerationStats(data);
    showPreviews();
  } catch (err) {
    stopPolling();
    alert(`Error: ${err.message}`);
  } finally {
    setGenerating(false);
  }
}

async function generateTexturing() {
  if (!state.meshFile || !state.texImage) {
    alert('Please upload both a mesh file and a reference image.');
    return;
  }
  setGenerating(true);
  updateProgress(5, 'Uploading files…');

  const settings = getSettings();
  const formData = new FormData();
  formData.append('mesh_file', state.meshFile);
  formData.append('image', state.texImage);
  formData.append('seed', String(settings.seed || 11456));
  formData.append('randomize_seed', 'false');  // Stage 2: fixed seed by default
  formData.append('resolution', settings.resolution === '512' ? '512' : '1024');
  formData.append('execution_preset', settings.execution_preset);
  formData.append('texture_size', $('#textureSize').value);
  formData.append('tex_guidance_strength', String(settings.tex_guidance_strength));
  formData.append('tex_guidance_rescale', String(settings.tex_guidance_rescale));
  formData.append('tex_sampling_steps', String(settings.tex_sampling_steps));
  formData.append('tex_rescale_t', String(settings.tex_rescale_t));

  try {
    updateProgress(10, 'Re-texturing mesh…');
    const res = await fetch('/api/texturing', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Texturing failed');
    }
    updateProgress(100, 'Done!');
    // Texturing returns GLB directly — show in 3D viewer, download via buttons
    const blob = await res.blob();
    state.glbUrl = URL.createObjectURL(blob);
    state.previews = null;
    state.sessionId = null;
    const texturePeak = res.headers.get('X-Trellis-Peak-VRAM-GB');
    showGenerationStats({
      preset: settings.execution_preset,
      resolution_label: `${settings.resolution === '512' ? 512 : 1024} texture resolution`,
      num_views: 1,
      peak_vram_gb: texturePeak,
    });
    showGlbViewer();
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    setGenerating(false);
  }
}

// ============================================================
// PREVIEW VIEWER
// ============================================================
function showPreviews() {
  if (!state.previews) return;

  $('#emptyState').hidden = true;
  $('#viewerModeBar').hidden = false;
  $('#renderModes').hidden = false;
  $('#angleSliderRow').hidden = false;
  $('#actionBar').hidden = false;

  // Show preview mode
  setViewerMode('preview');
  updatePreviewImage();
}

function updatePreviewImage() {
  if (!state.previews) return;
  const imgs = state.previews[state.currentRenderMode];
  if (!imgs || !imgs[state.currentAngle]) return;
  const img = $('#previewImage');
  img.src = imgs[state.currentAngle];
  img.hidden = false;
  img.classList.add('fade-in');
  setTimeout(() => img.classList.remove('fade-in'), 300);
}

// ============================================================
// VIEWER MODES (Preview / 3D)
// ============================================================
function initViewerModes() {
  $$('.mode-pill').forEach((pill) => {
    pill.addEventListener('click', () => {
      setViewerMode(pill.dataset.mode);
    });
  });
}

function setViewerMode(mode) {
  $$('.mode-pill').forEach((p) => p.classList.toggle('active', p.dataset.mode === mode));

  const previewDiv = $('#previewViewer');
  const threeDiv = $('#threeViewer');

  if (mode === 'preview') {
    previewDiv.style.display = 'flex';
    threeDiv.hidden = true;
    $('#renderModes').hidden = false;
    $('#angleSliderRow').hidden = false;
  } else {
    previewDiv.style.display = 'none';
    threeDiv.hidden = false;
    $('#renderModes').hidden = true;
    $('#angleSliderRow').hidden = true;
    if (state.glbUrl) {
      loadGlbInViewer(state.glbUrl);
    } else if (state.sessionId) {
      // Auto-extract GLB for the 3D viewer
      extractGlbForViewer();
    }
  }
}

// Extract GLB silently (no download) and load into 3D viewer
async function extractGlbForViewer() {
  if (state.extracting) return; // prevent double-extraction
  state.extracting = true;

  // Show loading state in the viewer
  showViewerLoading(true);

  const formData = new FormData();
  formData.append('session_id', state.sessionId);
  formData.append('decimation_target', $('#decimationTarget').value);
  formData.append('texture_size', $('#textureSize').value);

  try {
    const res = await fetch('/api/extract-glb', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Extraction failed');
    }
    const blob = await res.blob();
    state.glbUrl = URL.createObjectURL(blob);
    showViewerLoading(false);
    loadGlbInViewer(state.glbUrl);
  } catch (err) {
    showViewerLoading(false);
    showViewerError(err.message);
  } finally {
    state.extracting = false;
  }
}

function showViewerLoading(show) {
  let overlay = $('#viewerLoadingOverlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'viewerLoadingOverlay';
    overlay.className = 'viewer-loading-overlay';
    overlay.innerHTML = `
      <div class="viewer-loading-spinner"></div>
      <p class="viewer-loading-text">Extracting 3D mesh…</p>
    `;
    $('#threeViewer').appendChild(overlay);
  }
  overlay.hidden = !show;
}

function showViewerError(message) {
  let overlay = $('#viewerLoadingOverlay');
  if (overlay) {
    overlay.innerHTML = `<p class="viewer-loading-text" style="color: var(--danger)">⚠ ${message}</p>`;
    overlay.hidden = false;
    setTimeout(() => { overlay.hidden = true; }, 4000);
  }
}

// ============================================================
// RENDER MODES
// ============================================================
function initRenderModes() {
  $$('.render-mode-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      $$('.render-mode-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      state.currentRenderMode = btn.dataset.render;
      updatePreviewImage();
    });
  });
}

// ============================================================
// ANGLE SLIDER
// ============================================================
function initAngleSlider() {
  const slider = $('#angleSlider');
  slider.addEventListener('input', () => {
    state.currentAngle = parseInt(slider.value);
    updatePreviewImage();
  });
}

// ============================================================
// EXTRACT GLB
// ============================================================
function initExtractButton() {
  $('#extractBtn').addEventListener('click', extractGlb);
  $('#extractObjBtn').addEventListener('click', extractObj);
}

function getModelName() {
  // Read from input, sanitize for filesystem use
  let raw = ($('#modelNameInput')?.value || '').trim();
  if (!raw) return '';
  // Replace spaces/special chars with underscores, lowercase, truncate
  return raw.replace(/[^a-zA-Z0-9_\-]/g, '_').replace(/_+/g, '_').replace(/^_|_$/g, '').substring(0, 64).toLowerCase();
}

async function extractGlb() {
  // If we already have a GLB (e.g. from texturing), download it directly
  if (state.glbUrl && !state.sessionId) {
    const name = getModelName() || 'model';
    triggerDownload(state.glbUrl, `${name}.glb`);
    return;
  }

  if (!state.sessionId) {
    alert('No model to extract. Generate first.');
    return;
  }

  const btn = $('#extractBtn');
  btn.disabled = true;
  btn.textContent = 'Extracting…';

  const formData = new FormData();
  formData.append('session_id', state.sessionId);
  formData.append('decimation_target', $('#decimationTarget').value);
  formData.append('texture_size', $('#textureSize').value);

  try {
    const res = await fetch('/api/extract-glb', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Extraction failed');
    }
    const blob = await res.blob();
    state.glbUrl = URL.createObjectURL(blob);
    const name = getModelName() || 'model';
    triggerDownload(state.glbUrl, `${name}.glb`);

    // Show 3D view option
    setViewerMode('3d');
    loadGlbInViewer(state.glbUrl);
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
      Download GLB
    `;
  }
}

async function extractObj() {
  if (!state.sessionId && !state.glbUrl) {
    alert('No model to extract. Generate first.');
    return;
  }

  // Require a model name for OBJ export
  let modelName = getModelName();
  if (!modelName) {
    // Auto-open export settings and focus the input
    const details = $('#exportSettings');
    if (details && !details.open) details.open = true;
    const input = $('#modelNameInput');
    if (input) {
      input.focus();
      input.style.outline = '2px solid var(--accent)';
      input.style.outlineOffset = '2px';
      setTimeout(() => { input.style.outline = ''; input.style.outlineOffset = ''; }, 2000);
    }
    alert('Please enter a Model Name in Export Settings before downloading OBJ.\n\nThis name will be used for all filenames and material references.');
    return;
  }

  const btn = $('#extractObjBtn');
  btn.disabled = true;
  btn.textContent = 'Extracting OBJ…';

  try {
    let blob;

    if (!state.sessionId && state.glbUrl) {
      // Texturing path: convert the GLB blob to OBJ via server
      const glbBlob = await fetch(state.glbUrl).then(r => r.blob());
      const formData = new FormData();
      formData.append('glb_file', glbBlob, 'model.glb');
      formData.append('model_name', modelName);
      const res = await fetch('/api/convert-to-obj', { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'OBJ conversion failed');
      }
      blob = await res.blob();
    } else {
      // Generation path: extract OBJ from latents
      const formData = new FormData();
      formData.append('session_id', state.sessionId);
      formData.append('model_name', modelName);
      formData.append('decimation_target', $('#decimationTarget').value);
      formData.append('texture_size', $('#textureSize').value);
      const res = await fetch('/api/extract-obj', { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'OBJ extraction failed');
      }
      blob = await res.blob();
    }

    const url = URL.createObjectURL(blob);
    triggerDownload(url, `${modelName}.zip`);
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
      Download OBJ
    `;
  }
}

function triggerDownload(url, filename) {
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// ============================================================
// THREE.JS 3D VIEWER
// ============================================================
let threeScene = null;

async function loadGlbInViewer(url) {
  const canvas = $('#threeCanvas');
  const container = $('#threeViewer');

  // Dynamically import three.js
  const THREE = await import('three');
  const { OrbitControls } = await import('three/examples/jsm/controls/OrbitControls.js');
  const { GLTFLoader } = await import('three/examples/jsm/loaders/GLTFLoader.js');

  // Clean up previous scene
  if (threeScene) {
    threeScene.renderer.dispose();
    if (threeScene.resizeObserver) threeScene.resizeObserver.disconnect();
  }

  // Wait one frame for layout reflow (container may have just become visible)
  await new Promise(resolve => requestAnimationFrame(resolve));

  const width = container.clientWidth || 800;
  const height = container.clientHeight || 600;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d0d18);

  const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
  camera.position.set(0, 0.5, 2);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(width, height, false);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0, 0);

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
  scene.add(ambientLight);

  const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
  dirLight.position.set(3, 5, 4);
  scene.add(dirLight);

  const dirLight2 = new THREE.DirectionalLight(0x8888ff, 0.5);
  dirLight2.position.set(-3, -2, -4);
  scene.add(dirLight2);

  // Load GLB
  const loader = new GLTFLoader();
  loader.load(
    url,
    (gltf) => {
      const model = gltf.scene;
      console.log('[3D Viewer] GLB loaded, meshes:', model.children.length);

      // Center and scale
      const box = new THREE.Box3().setFromObject(model);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      console.log('[3D Viewer] Bounding box size:', size, 'maxDim:', maxDim);

      if (maxDim === 0 || !isFinite(maxDim)) {
        console.warn('[3D Viewer] Model has zero or invalid dimensions');
        return;
      }

      const scale = 1.5 / maxDim;
      model.scale.setScalar(scale);
      model.position.sub(center.multiplyScalar(scale));

      scene.add(model);
      controls.target.set(0, 0, 0);
      controls.update();
    },
    undefined,
    (error) => {
      console.error('[3D Viewer] Failed to load GLB:', error);
    }
  );

  // Animate
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // Handle resize — guard against infinite loops
  let lastW = width, lastH = height;
  const resizeObserver = new ResizeObserver(() => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w > 0 && h > 0 && (w !== lastW || h !== lastH)) {
      lastW = w;
      lastH = h;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
    }
  });
  resizeObserver.observe(container);

  threeScene = { scene, camera, renderer, controls, resizeObserver };
}

function showGlbViewer() {
  $('#emptyState').hidden = true;
  $('#viewerModeBar').hidden = false;
  $('#actionBar').hidden = false;
  setViewerMode('3d');
}

// ============================================================
// EXAMPLES
// ============================================================
async function loadExamples() {
  try {
    const res = await fetch('/api/examples');
    if (!res.ok) return;
    const data = await res.json();
    const grid = $('#examplesGrid');
    grid.innerHTML = '';

    data.examples.forEach((name) => {
      const img = document.createElement('img');
      img.src = `/api/examples/${name}`;
      img.alt = name;
      img.className = 'example-thumb';
      img.addEventListener('click', async () => {
        // Fetch the example image and set it as the single image
        const imgRes = await fetch(`/api/examples/${name}`);
        const blob = await imgRes.blob();
        const file = new File([blob], name, { type: blob.type });
        state.singleImage = file;

        const preview = $('#singlePreview');
        preview.src = URL.createObjectURL(file);
        preview.hidden = false;
        $('#singleUpload').classList.add('has-image');

        // Switch to single tab if not there
        if (state.activeTab !== 'single') {
          document.querySelector('.tab-btn[data-tab="single"]').click();
        }
      });
      grid.appendChild(img);
    });
  } catch {
    // Examples not available during dev
  }
}
