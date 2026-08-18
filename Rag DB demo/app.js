"use strict";
let dataset = [];
const el = Object.fromEntries(["question-select","baseline-answer","baseline-time","rag-answer","rag-time","reference-answer","reference-source","evidence-list","evidence-note"].map(id => [id, document.getElementById(id)]));

function current() { return dataset.find(item => item.id === el["question-select"].value); }
function renderEvidence(chunks) {
  el["evidence-list"].replaceChildren();
  chunks.forEach(chunk => {
    const item = document.createElement("article"); item.className = "evidence-item";
    const rank = document.createElement("span"); rank.className = "rank"; rank.textContent = String(chunk.rank).padStart(2, "0");
    const source = document.createElement("div");
    const title = document.createElement("h3"); title.className = "source-title"; title.textContent = chunk.source_title;
    const meta = document.createElement("div"); meta.className = "source-meta"; meta.textContent = `${chunk.section_title || "Introduction"} · ${chunk.source_date}`;
    const link = document.createElement("a"); link.className = "source-link"; link.href = chunk.source_url; link.target = "_blank"; link.rel = "noopener"; link.textContent = "Open source ↗";
    source.append(title, meta, link);
    const passage = document.createElement("p"); passage.className = "passage"; passage.textContent = chunk.text;
    const similarity = document.createElement("div"); similarity.className = "similarity"; similarity.append("Similarity");
    const score = document.createElement("strong"); score.textContent = Number(chunk.similarity).toFixed(3); similarity.append(score);
    item.append(rank, source, passage, similarity); el["evidence-list"].append(item);
  });
}
function render() {
  const item = current(); if (!item) return; const rag = item.rag;
  el["baseline-answer"].textContent = item.baseline.answer; el["baseline-time"].textContent = `${item.baseline.generation_seconds.toFixed(2)}s`;
  el["rag-answer"].textContent = rag.answer; el["rag-time"].textContent = `${rag.generation_seconds.toFixed(2)}s`;
  el["reference-answer"].textContent = item.reference_answer; el["reference-source"].href = item.source_urls[0] || "#"; el["reference-source"].hidden = !item.source_urls.length;
  const visibleChunks = rag.retrieved_chunks.slice(0, 3);
  el["evidence-note"].textContent = `Showing ${visibleChunks.length} of ${rag.retrieved_chunks.length} retrieved passage${rag.retrieved_chunks.length === 1 ? "" : "s"}, ranked by cosine similarity.`;
  renderEvidence(visibleChunks);
}
async function initialize() {
  try {
    const response = await fetch("data/demo_results.json"); if (!response.ok) throw new Error(`HTTP ${response.status}`); dataset = await response.json();
    dataset.forEach((item, i) => { const option = document.createElement("option"); option.value = item.id; option.textContent = `${String(i + 1).padStart(2, "0")} — ${item.question}`; el["question-select"].append(option); });
    el["question-select"].addEventListener("change", render); render();
  } catch (error) { el["baseline-answer"].textContent = "Demo data could not be loaded."; el["rag-answer"].textContent = `Run demo/build_demo_data.py before deployment. (${error.message})`; }
}
initialize();
