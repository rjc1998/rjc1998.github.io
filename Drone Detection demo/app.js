'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const video = $('source-video');
  const canvases = ['prediction-view', 'truth-view', 'prediction-detail', 'truth-detail'].map($);
  const contexts = canvases.map(canvas => canvas.getContext('2d'));
  const colors = { prediction: '#ffc17c', truth: '#66f1d0' };
  const controls = ['play', 'previous', 'next', 'timeline', 'speed'].map($);
  let data, clip, model = 'map50', frameIndex = 0, sheets = [], loadVersion = 0, ready = false;
  let fallbackClock = 0, movieURL = null, loadController = null;

  function fail(message) { $('error').textContent = message; $('error').hidden = false; }
  function image(url) {
    return new Promise((resolve, reject) => {
      const result = new Image();
      result.onload = () => resolve(result);
      result.onerror = () => reject(new Error(`Could not load ${url}`));
      result.src = url;
    });
  }
  function pause() { video.pause(); updatePlay(); }
  function updatePlay() {
    $('play').innerHTML = video.paused ? 'Play <span aria-hidden="true">▶</span>' : 'Pause <span aria-hidden="true">Ⅱ</span>';
    $('play').setAttribute('aria-label', video.paused ? 'Play selected clip' : 'Pause selected clip');
  }
  function markButtons(container, key) {
    for (const button of container.querySelectorAll('button')) button.setAttribute('aria-pressed', String(button.dataset.key === key));
  }
  function setModel(key) {
    model = key;
    markButtons($('model-tabs'), key);
    // No src or currentTime changes: preserve the exact decoded frame and timeline.
    if (ready) render(frameIndex);
  }
  function label(ctx, text, x, y, color, maxWidth) {
    ctx.font = '600 18px Arial, sans-serif';
    const width = ctx.measureText(text).width + 14;
    const left = Math.max(0, Math.min(maxWidth - width, x));
    const top = Math.max(0, y - 28);
    ctx.fillStyle = '#102732ee'; ctx.fillRect(left, top, width, 26);
    ctx.fillStyle = color; ctx.fillText(text, left + 7, top + 19);
  }
  function boxes(ctx, rows, color, scale, origin, detail) {
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = detail ? 3 : 3;
    for (const box of rows) {
      const [x1, y1, x2, y2, score] = box;
      const x = (x1 - origin[0]) * scale, y = (y1 - origin[1]) * scale;
      const width = (x2 - x1) * scale, height = (y2 - y1) * scale;
      if (x + width < 0 || y + height < 0 || x > ctx.canvas.width || y > ctx.canvas.height) continue;
      ctx.strokeRect(x, y, width, height);
      label(ctx, score === undefined ? 'drone · GT' : `drone ${(score * 100).toFixed(0)}%`, x, y, color, ctx.canvas.width);
    }
    ctx.restore();
  }
  function render(index) {
    if (!ready || video.readyState < 2) return;
    frameIndex = Math.max(0, Math.min(clip.frameCount - 1, index));
    const frame = clip.frames[frameIndex];
    const predictions = frame.predictions[model];
    const group = data.sheetGrid * data.sheetGrid;
    const slot = frameIndex % group;
    const sprite = sheets[Math.floor(frameIndex / group)];
    if (!sprite) return;
    const sx = (slot % data.sheetGrid) * data.cropSize;
    const sy = Math.floor(slot / data.sheetGrid) * data.cropSize;
    for (let i = 0; i < 2; i++) {
      const ctx = contexts[i];
      ctx.drawImage(video, 0, 0, ctx.canvas.width, ctx.canvas.height);
      if ($('boxes').checked) boxes(ctx, i === 0 ? predictions : frame.gt, i === 0 ? colors.prediction : colors.truth, ctx.canvas.width / data.nativeWidth, [0, 0], false);
      const detail = contexts[i + 2];
      detail.imageSmoothingEnabled = false;
      detail.drawImage(sprite, sx, sy, data.cropSize, data.cropSize, 0, 0, 512, 512);
      if ($('boxes').checked) boxes(detail, i === 0 ? predictions : frame.gt, i === 0 ? colors.prediction : colors.truth, 512/data.cropSize, frame.crop, true);
    }
    const [x1, y1, x2, y2] = frame.gt[0];
    $('target-size').textContent = `${Math.round(x2 - x1)} × ${Math.round(y2 - y1)} px target`;
    $('timeline').value = String(frameIndex);
    $('clock').textContent = `${(frameIndex / clip.fps).toFixed(2)} / ${clip.duration.toFixed(2)} s`;
    $('frame-number').textContent = `Frame ${frameIndex + 1} / ${clip.frameCount}`;
    $('frame-status').textContent = `${clip.id} · source frame ${frame.sourceFrame} · ${data.models.find(item => item.id === model).name}`;
  }
  function decodedFrame(_now, metadata) {
    if (ready && !video.seeking) render(Math.round(metadata.mediaTime * clip.fps));
    video.requestVideoFrameCallback(decodedFrame);
  }
  // requestVideoFrameCallback binds boxes to decoded frames, including slow playback.
  if ('requestVideoFrameCallback' in video) video.requestVideoFrameCallback(decodedFrame);
  else {
    // Fallback for older browsers; seeked still provides exact paused-frame inspection.
    const tick = () => {
      if (ready && !video.paused && !video.seeking) render(Math.floor(video.currentTime * clip.fps + 0.001));
      fallbackClock = requestAnimationFrame(tick);
    };
    fallbackClock = requestAnimationFrame(tick);
  }
  function seek(index) {
    if (!ready) return;
    pause();
    const target = Math.max(0, Math.min(clip.frameCount - 1, index));
    video.currentTime = (target + .1) / clip.fps;
  }
  async function selectClip(key) {
    const version = ++loadVersion;
    pause(); ready = false; controls.forEach(control => control.disabled = true);
    if (loadController) loadController.abort();
    loadController = new AbortController();
    const controller = loadController;
    video.removeAttribute('src'); video.load();
    if (movieURL) { URL.revokeObjectURL(movieURL); movieURL = null; }
    $('error').hidden = true; $('loading').hidden = false; $('comparison').hidden = true;
    $('loading').textContent = 'Loading the selected sequence…';
    $('viewer').setAttribute('aria-busy', 'true');
    clip = data.clips.find(item => item.id === key);
    const selected = clip;
    markButtons($('clip-tabs'), key);
    $('clip-id').textContent = `${key} · ${selected.duration.toFixed(1)} seconds`;
    $('clip-title').textContent = selected.title;
    $('clip-description').textContent = selected.description;
    $('clip-description').hidden = !selected.description;
    $('timeline').max = String(selected.frameCount - 1);
    $('timeline').value = '0';
    $('clock').textContent = `0.00 / ${selected.duration.toFixed(2)} s`;
    $('frame-number').textContent = `Frame 1 / ${selected.frameCount}`;
    $('frame-status').textContent = '';
    try {
      // Buffer only the selected clip. A Blob gives reliable seeking even on
      // simple local servers without HTTP Range support, and avoids partial seeks.
      const movie = fetch(selected.video, { signal: controller.signal }).then(response => {
        if (!response.ok) throw new Error(`Video returned HTTP ${response.status}.`);
        return response.blob();
      });
      const [loadedSheets, blob] = await Promise.all([Promise.all(selected.sheets.map(image)), movie]);
      if (version !== loadVersion) return;
      const movieReady = new Promise((resolve, reject) => {
        const cleanup = () => { video.removeEventListener('loadeddata', loaded); video.removeEventListener('error', error); controller.signal.removeEventListener('abort', aborted); };
        const loaded = () => { cleanup(); resolve(); };
        const error = () => { cleanup(); reject(new Error('This browser could not decode the WebM video. Try a current Chrome, Edge, Firefox or Safari.')); };
        const aborted = () => { cleanup(); reject(new DOMException('Loading canceled', 'AbortError')); };
        video.addEventListener('loadeddata', loaded);
        video.addEventListener('error', error);
        controller.signal.addEventListener('abort', aborted, { once: true });
      });
      movieURL = URL.createObjectURL(blob);
      video.poster = selected.poster; video.src = movieURL; video.load();
      await movieReady;
      if (version !== loadVersion) return;
      sheets = loadedSheets; frameIndex = 0; ready = true;
      video.playbackRate = Number($('speed').value);
      controls.forEach(control => control.disabled = false);
      $('loading').hidden = true; $('comparison').hidden = false;
      $('viewer').setAttribute('aria-busy', 'false');
      render(0);
    } catch (error) {
      if (version !== loadVersion || error.name === 'AbortError') return;
      $('loading').textContent = 'The selected sequence could not be loaded. Choose a sample to retry.';
      $('viewer').setAttribute('aria-busy', 'false'); fail(error.message);
    }
  }
  $('play').addEventListener('click', async () => {
    if (!ready) return;
    if (!video.paused) { pause(); return; }
    if (video.ended || frameIndex >= clip.frameCount-1) video.currentTime = 0;
    try { await video.play(); updatePlay(); } catch (error) { fail(`Playback could not start: ${error.message}`); }
  });
  $('previous').addEventListener('click', () => seek(frameIndex - 1));
  $('next').addEventListener('click', () => seek(frameIndex + 1));
  $('timeline').addEventListener('input', event => seek(Number(event.target.value)));
  $('speed').addEventListener('change', event => { video.playbackRate = Number(event.target.value); });
  $('boxes').addEventListener('change', () => render(frameIndex));
  video.addEventListener('seeked', () => { if (ready) render(Math.floor(video.currentTime * clip.fps)); });
  video.addEventListener('play', updatePlay);
  video.addEventListener('pause', updatePlay);
  video.addEventListener('ended', () => { updatePlay(); if (ready) render(clip.frameCount-1); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
  window.addEventListener('pagehide', () => { pause(); cancelAnimationFrame(fallbackClock); });
  async function init() {
    const response = await fetch('demo-data.json?v=f5df847a3ed3');
    if (!response.ok) throw new Error(`Demo data returned HTTP ${response.status}.`);
    data = await response.json();
    if (data.schema !== 1 || !data.clips.length || data.models.length !== 2) throw new Error('Demo data is incomplete.');
    const architectures = {
      map50: [
        'RGB appearance and a one-channel motion mask pass through separate convolution and C2f blocks. Their features are concatenated after P1, at half the input resolution.',
        'A C2f block mixes the fused features, followed by the shared YOLOv8n backbone and spatial pyramid pooling (SPPF). The neck combines features across scales through upsampling, concatenation and C2f blocks.',
        'Four detection heads use P2, P3, P4 and P5 features at strides 4, 8, 16 and 32. P2 retains a finer feature grid for small drones. This variant has no CBAM attention.'
      ],
      map5095: [
        'RGB appearance and a one-channel motion mask pass through separate convolution and C2f blocks, then concatenate after P1, at half the input resolution.',
        'CBAM (Convolutional Block Attention Module) applies channel attention and spatial attention with a 7 × 7 kernel to the concatenated features, before the retained C2f fusion block.',
        'The shared YOLOv8n backbone, spatial pyramid pooling (SPPF) and multiscale neck feed two detection heads: P2 and P5, at strides 4 and 32. The intermediate P3 and P4 features still contribute to the neck, but do not have prediction heads.'
      ]
    };
    for (const item of data.models) {
      const button = document.createElement('button');
      button.type = 'button'; button.className = 'model-tab'; button.dataset.key = item.id;
      button.innerHTML = `<span class="model-badge">${item.badge}</span><strong>${item.name}</strong><span class="model-scores"><span class="${item.id === 'map50' ? 'leading' : ''}">${(item.ap50 * 100).toFixed(2)}%<small>mAP50</small></span><span class="${item.id === 'map5095' ? 'leading' : ''}">${(item.map5095 * 100).toFixed(2)}%<small>mAP50–95</small></span></span>`;
      button.addEventListener('click', () => setModel(item.id));
      const panel = document.createElement('div'); panel.className = 'model-panel';
      const details = document.createElement('details'); details.className = 'model-architecture';
      const summary = document.createElement('summary'); summary.textContent = 'Network architecture';
      summary.setAttribute('aria-label', `Network architecture: ${item.name}`);
      const body = document.createElement('div'); body.className = 'details-body';
      for (const text of architectures[item.id]) {
        const paragraph = document.createElement('p'); paragraph.textContent = text; body.append(paragraph);
      }
      details.append(summary, body); panel.append(button, details); $('model-tabs').append(panel);
    }
    for (const item of data.clips) {
      const button = document.createElement('button');
      button.type = 'button'; button.className = 'clip-tab'; button.dataset.key = item.id;
      const picture = document.createElement('img'); picture.src = item.poster; picture.alt = ''; picture.loading = 'lazy';
      const caption = document.createElement('span');
      const title = document.createElement('strong'); title.textContent = item.title;
      const meta = document.createElement('small'); meta.textContent = `${item.id} · ${item.duration.toFixed(1)} s`;
      caption.append(title, meta); button.append(picture, caption);
      button.addEventListener('click', () => selectClip(item.id)); $('clip-tabs').append(button);
    }
    setModel(model); await selectClip(data.clips[0].id);
  }
  init().catch(error => { $('loading').textContent = 'The demo could not load.'; fail(`${error.message} Open this page through a local HTTP server, not a file:// URL.`); });
})();
