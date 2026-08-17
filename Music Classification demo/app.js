const state = { data: null, track: null, attentionModel: "lstm" };
const $ = (selector) => document.querySelector(selector);

async function initialize() {
  const response = await fetch("data/demo.json");
  if (!response.ok) throw new Error(`Could not load demo data (${response.status})`);
  state.data = await response.json();
  const select = $("#song-select");
  state.data.tracks.forEach((track, index) => {
    const option = document.createElement("option");
    option.value = index;
    option.textContent = `${track.artist} — ${track.title}`;
    select.append(option);
  });
  select.addEventListener("change", () => renderTrack(state.data.tracks[Number(select.value)]));
  setupAttentionChoices();
  renderCredits();
  renderTrack(state.data.tracks[0]);
}

function setupAttentionChoices() {
  document.querySelectorAll(".attention-choice").forEach((button) => button.addEventListener("click", () => {
    document.querySelectorAll(".attention-choice").forEach((item) => item.classList.remove("active"));
    button.classList.add("active"); state.attentionModel = button.dataset.model; drawFeatures();
  }));
}

function renderTrack(track) {
  state.track = track;
  $("#song-heading").textContent = `${track.artist} — ${track.title}`;
  $("#actual-genre").textContent = track.dataset_genre;
  $("#track-meta").textContent = `FMA ${track.track_id} · Validation split · ${track.license}`;
  $("#audio-player").src = track.audio_path;
  const lstm = track.results.lstm, gru = track.results.gru;
  const agreement = $("#agreement");
  if (lstm.prediction === gru.prediction) {
    agreement.className = "notice";
    agreement.innerHTML = `The models agree: <strong>${lstm.prediction}</strong>`;
  } else {
    agreement.className = "notice disagree";
    agreement.innerHTML = `The models disagree: LSTM predicts <strong>${lstm.prediction}</strong>; GRU predicts <strong>${gru.prediction}</strong>.`;
  }
  $("#model-cards").replaceChildren(modelCard("LSTM", lstm, track.dataset_genre), modelCard("GRU", gru, track.dataset_genre));
  renderScores();
  drawFeatures();
}

function modelCard(name, result, referenceGenre) {
  const card = document.createElement("article"); card.className = `card model-card model-${name.toLowerCase()}`;
  const topScores = result.top_three.map(({genre, score}) => `<div class="top-score"><span>${genre}</span><div class="bar-track"><div class="bar-fill" style="width:${score * 100}%"></div></div><strong>${percent(score)}</strong></div>`).join("");
  card.innerHTML = `<div class="model-name"><h2>${name}</h2></div><div class="prediction">${result.prediction}</div><div class="match">${result.prediction === referenceGenre ? "Matches actual genre" : `Differs from actual genre: ${referenceGenre}`}</div>${topScores}`;
  return card;
}

function renderScores() {
  const chart = $("#score-chart"); chart.replaceChildren();
  state.data.genres.forEach((genre) => {
    const row = document.createElement("div"); row.className = "score-row";
    const lstm = state.track.results.lstm.probabilities[genre] * 100;
    const gru = state.track.results.gru.probabilities[genre] * 100;
    row.innerHTML = `<strong>${genre}</strong><div class="paired-bars" title="LSTM ${lstm.toFixed(1)}%, GRU ${gru.toFixed(1)}%"><div class="score-bar lstm" style="width:${lstm}%"></div><div class="score-bar gru" style="width:${gru}%"></div></div>`;
    chart.append(row);
  });
}

function drawFeatures() {
  if (!state.track) return;
  const canvas = $("#feature-canvas"), context = canvas.getContext("2d");
  const width = canvas.width, height = canvas.height, plot = {x: 76, y: 18, w: width - 96, h: height - 66};
  context.clearRect(0, 0, width, height); context.fillStyle = "#fff"; context.fillRect(0, 0, width, height);
  const matrix = state.track.features, timeSteps = matrix.length, bins = matrix[0].length;
  const image = context.createImageData(timeSteps, bins);
  for (let t = 0; t < timeSteps; t++) for (let b = 0; b < bins; b++) {
    const [r, g, blue] = magma(matrix[t][b] / 255); const pixel = ((bins - 1 - b) * timeSteps + t) * 4;
    image.data[pixel] = r; image.data[pixel + 1] = g; image.data[pixel + 2] = blue; image.data[pixel + 3] = 255;
  }
  const buffer = document.createElement("canvas"); buffer.width = timeSteps; buffer.height = bins;
  buffer.getContext("2d").putImageData(image, 0, 0); context.imageSmoothingEnabled = true;
  context.drawImage(buffer, plot.x, plot.y, plot.w, plot.h);
  const attention = state.track.results[state.attentionModel].attention;
  context.strokeStyle = "#f3a38f"; context.lineWidth = 3; context.beginPath();
  attention.forEach((value, index) => { const x = plot.x + index * plot.w / (attention.length - 1), y = plot.y + plot.h * (1 - value); index ? context.lineTo(x, y) : context.moveTo(x, y); });
  context.stroke(); context.fillStyle = "#62556b"; context.font = "14px system-ui";
  context.save(); context.translate(20, plot.y + plot.h / 2); context.rotate(-Math.PI / 2); context.textAlign = "center"; context.fillText("Mel feature bin", 0, 0); context.restore(); context.textAlign = "start";
  context.fillText("0s", plot.x, height - 18); context.fillText("15s", plot.x + plot.w / 2 - 10, height - 18); context.fillText("30s", plot.x + plot.w - 22, height - 18);
}

function magma(value) {
  const stops = [[0,[0,0,4]],[.25,[81,18,124]],[.5,[183,55,121]],[.75,[252,137,97]],[1,[252,253,191]]];
  const upper = stops.findIndex(([position]) => position >= value); if (upper <= 0) return stops[0][1];
  const [p1,c1] = stops[upper - 1], [p2,c2] = stops[upper], ratio = (value - p1) / (p2 - p1);
  return c1.map((channel, index) => Math.round(channel + (c2[index] - channel) * ratio));
}

function renderCredits() {
  $("#credits").innerHTML = state.data.tracks.map((track) => `<p><strong>${track.artist} — ${track.title}</strong> · FMA ${track.track_id} · ${track.license}</p>`).join("");
  $("#dataset-source").href = state.data.source_url;
}

function percent(value) { return `${(value * 100).toFixed(1)}%`; }

initialize().catch((error) => {
  document.body.innerHTML = `<main class="page-shell"><section class="card song-panel"><h1>Demo unavailable</h1><p>${error.message}</p><p>Serve the docs directory over HTTP rather than opening index.html directly.</p></section></main>`;
  console.error(error);
});
