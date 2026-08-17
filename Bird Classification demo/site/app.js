const state = { data: null, model: "proxynca", sampleId: null };
const $ = (selector) => document.querySelector(selector);
const percent = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;

function activeSample() { return state.data.samples.find((sample) => sample.id === state.sampleId); }

function renderSamples() {
  const strip = $("#sample-strip");
  strip.replaceChildren(...state.data.samples.map((sample, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sample-button";
    button.dataset.sample = sample.id;
    button.setAttribute("role", "listitem");
    button.setAttribute("aria-current", String(sample.id === state.sampleId));
    button.setAttribute("aria-label", `Sample ${index + 1}: ${sample.true_class}`);
    button.innerHTML = `<img src="${sample.image}" alt="" loading="lazy"><span>${String(index + 1).padStart(2,"0")}</span>`;
    button.addEventListener("click", () => { state.sampleId = sample.id; render(); });
    return button;
  }));
}

function renderProbabilities(items) {
  $("#probability-list").innerHTML = items.map((item) => `
    <li class="probability-item">
      <span class="rank">${String(item.rank).padStart(2,"0")}</span>
      <span>${item.class_name}</span>
      <span class="bar-track" aria-hidden="true"><span class="bar" style="width:${Math.max(item.probability * 100, 1)}%"></span></span>
      <span class="value">${percent(item.probability)}</span>
    </li>`).join("");
}

function renderNeighbors(neighbors) {
  $("#neighbor-grid").innerHTML = neighbors.map((item) => `
    <article class="neighbor-card">
      <img src="${item.image}" alt="Training-set ${item.class_name}" loading="lazy">
      <div class="neighbor-meta"><span class="neighbor-rank">NEIGHBOR ${String(item.rank).padStart(2,"0")}</span>
      <h3>${item.class_name}</h3><div class="neighbor-score"><span>cosine</span><strong>${item.similarity.toFixed(3)}</strong></div></div>
    </article>`).join("");
}

function render() {
  const sample = activeSample();
  const result = sample.results[state.model];
  const model = state.data.models[state.model];
  const sampleIndex = state.data.samples.indexOf(sample) + 1;
  document.querySelectorAll(".model-tab").forEach((tab) => tab.setAttribute("aria-checked", String(tab.dataset.model === state.model)));
  document.querySelectorAll(".sample-button").forEach((button) => button.setAttribute("aria-current", String(button.dataset.sample === state.sampleId)));
  $("#model-description").textContent = model.description;
  $("#sample-count").textContent = `${String(sampleIndex).padStart(2,"0")} / ${String(state.data.samples.length).padStart(2,"0")}`;
  $("#image-index").textContent = String(sampleIndex).padStart(2,"0");
  $("#query-image").src = sample.image;
  $("#query-image").alt = `${sample.true_class}, CUB test photograph`;
  $("#heatmap-image").src = result.heatmap;
  $("#heatmap-image").alt = `Grad-CAM overlay for ${result.predicted_class}`;
  $("#true-class").textContent = sample.true_class;
  $("#predicted-class").textContent = result.predicted_class;
  $("#confidence").textContent = percent(result.confidence);
  const status = $("#result-status");
  status.textContent = result.correct ? "Correct classification" : "Incorrect classification";
  status.classList.toggle("incorrect", !result.correct);
  renderProbabilities(result.top_classes);
  renderNeighbors(result.neighbors);
  $("#result-panel").setAttribute("aria-busy", "false");
}

async function init() {
  try {
    const response = await fetch("demo-data.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.data = await response.json();
    state.model = state.data.default_model;
    state.sampleId = state.data.default_sample;
    renderSamples();
    document.querySelectorAll(".model-tab").forEach((tab) => tab.addEventListener("click", () => { state.model = tab.dataset.model; render(); }));
    $("#heatmap-toggle").addEventListener("change", (event) => $("#heatmap-image").classList.toggle("hidden", !event.target.checked));
    render();
  } catch (error) {
    console.error(error);
    $("#error-banner").hidden = false;
    $("#result-panel").setAttribute("aria-busy", "false");
  }
}

init();
