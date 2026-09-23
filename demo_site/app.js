/* Belief-Tracking Debate Analyzer -- static reviewer demo.
 * Renders real, precomputed system output (data.json) client-side only.
 * No generation, no model calls: this page is read-only by design. */

const RHETORIC_LABELS = ["causal", "empirical", "emotional", "moral"];
const RHETORIC_COLORS = {
  causal: "#4f8dfd",
  empirical: "#33b679",
  emotional: "#e8734a",
  moral: "#b48ae8",
};
const LABEL_COLORS = {
  evidence_adoption: "#33b679",
  strategic_persuasion: "#e8734a",
  echo: "#b48ae8",
  anchoring: "#f2c94c",
  no_inflection: "#5a6472",
};

let TRANSCRIPTS = [];
let currentTranscript = null;
let currentTurnIndex = null;
let chart = null;

async function init() {
  const res = await fetch("data.json");
  TRANSCRIPTS = await res.json();
  renderList(TRANSCRIPTS);
  document.getElementById("search").addEventListener("input", onSearch);
}

function onSearch(e) {
  const q = e.target.value.trim().toLowerCase();
  const filtered = q
    ? TRANSCRIPTS.filter((t) => t.topic.toLowerCase().includes(q))
    : TRANSCRIPTS;
  renderList(filtered);
}

function renderList(list) {
  const ul = document.getElementById("debate-list");
  const countEl = document.getElementById("debate-count");
  countEl.textContent = `${list.length} of ${TRANSCRIPTS.length} debates`;
  ul.innerHTML = "";
  for (const t of list) {
    const li = document.createElement("li");
    li.textContent = t.topic;
    li.dataset.id = t.transcript_id;
    li.setAttribute("role", "option");
    if (currentTranscript && currentTranscript.transcript_id === t.transcript_id) {
      li.classList.add("active");
    }
    li.addEventListener("click", () => selectTranscript(t));
    ul.appendChild(li);
  }
}

function selectTranscript(t) {
  currentTranscript = t;
  currentTurnIndex = null;
  document.getElementById("empty-state").hidden = true;
  document.getElementById("debate-view").hidden = false;
  document.getElementById("debate-topic").textContent = `Panel 2 \u2014 ${t.topic}`;
  renderHero(t);
  renderList(
    document.getElementById("search").value.trim()
      ? TRANSCRIPTS.filter((x) =>
          x.topic.toLowerCase().includes(document.getElementById("search").value.trim().toLowerCase())
        )
      : TRANSCRIPTS
  );
  renderTranscript(t);
  renderStanceChart(t);
  renderMovement(t);
  renderWhyCard(null);
}

function sideOf(speaker) {
  return speaker.endsWith("_pro") ? "pro" : speaker.endsWith("_con") ? "con" : "?";
}

function speakerColor(speaker) {
  if (speaker.endsWith("_pro")) return "#33b679";
  if (speaker.endsWith("_con")) return "#e8734a";
  const palette = ["#5b8cff", "#f2c94c", "#b48ae8", "#33b679", "#e8734a"];
  const names = [...new Set(currentTranscript.turns.map((x) => x.speaker))].sort();
  const i = names.indexOf(speaker);
  return i >= 0 ? palette[i % palette.length] : "#e8eaed";
}

function renderHero(t) {
  const hero = document.getElementById("hero");
  const labels = {};
  for (const turn of t.turns) {
    labels[turn.attribution.label] = (labels[turn.attribution.label] || 0) + 1;
  }
  const nInflections = Object.entries(labels)
    .filter(([k, v]) => k !== "no_inflection")
    .reduce((s, [, v]) => s + v, 0);

  const kpis = [];
  const agreement = t.agreement || {};
  for (const spk of Object.keys(agreement).sort()) {
    const a = agreement[spk];
    if (a && "agreementPreTreatment" in a && "agreementPostTreatment" in a) {
      const side = sideOf(spk).toUpperCase();
      kpis.push(
        `<div class="kpi"><div class="v">${a.agreementPreTreatment}\u2192${a.agreementPostTreatment}</div>` +
        `<div class="l">${side} belief (1\u20135, stated)</div></div>`
      );
    }
  }
  const byLast = {};
  for (const turn of t.turns) byLast[turn.speaker] = turn.stance.S;
  const netShift = Object.values(byLast).reduce((s, v) => s + Math.abs(v), 0);
  kpis.push(`<div class="kpi"><div class="v">+${netShift.toFixed(2)}</div><div class="l">net system stance drift</div></div>`);

  const realMech = Object.entries(labels).filter(([k, v]) => k !== "no_inflection" && v > 0);
  const domMech = realMech.length
    ? realMech.sort((a, b) => b[1] - a[1])[0][0].replace(/_/g, " ")
    : "\u2014";
  kpis.push(`<div class="kpi"><div class="v">${domMech}</div><div class="l">dominant mechanism</div></div>`);

  const bits = Object.keys(agreement).sort().map((spk) => {
    const a = agreement[spk];
    if (!a || !("agreementPreTreatment" in a)) return "";
    const up = a.agreementPostTreatment >= a.agreementPreTreatment;
    const color = a.agreementPostTreatment > a.agreementPreTreatment ? "#33b679" : a.agreementPostTreatment < a.agreementPreTreatment ? "#e8734a" : "#9aa3b2";
    return `<span style="color:${color}">${sideOf(spk).toUpperCase()} ${a.agreementPreTreatment}\u2192${a.agreementPostTreatment}</span>`;
  }).filter(Boolean);
  const survey = bits.length
    ? `Stated belief change (DebateGPT survey, 1\u20135): ${bits.join(" \u00b7 ")}`
    : "Stated belief change: n/a for this debate (no DebateGPT survey data).";

  hero.innerHTML =
    `<div class="hero">` +
    `<h1>${escapeHtml(t.topic)}</h1>` +
    `<div class="badges">` +
    `<span class="badge-pill">transcript ${escapeHtml(String(t.transcript_id))}</span>` +
    `<span class="badge-pill">${t.turns.length} turns</span>` +
    `<span class="badge-pill">${nInflections} attributed inflection${nInflections === 1 ? "" : "s"}</span>` +
    `<span class="badge-pill">real DebateGPT human-human</span>` +
    `</div>` +
    `<p class="hero-tagline">Interactive demo for diagnosing persuasion dynamics in multi-agent debates.</p>` +
    `<hr>` +
    `<div class="kpis">${kpis.join("")}</div>` +
    `<div class="hero-survey">${survey}</div>` +
    `</div>`;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderMovement(t) {
  const host = document.getElementById("movement");
  const rows = [];
  for (let i = 0; i < t.turns.length; i++) {
    const prev = prevStanceForSpeaker(t, i);
    const delta = t.turns[i].stance.S - prev;
    if (Math.abs(delta) < 1e-9) continue;
    rows.push({ i, delta, color: speakerColor(t.turns[i].speaker) });
  }
  const maxAbs = rows.reduce((m, r) => Math.max(m, Math.abs(r.delta)), 0) || 1;
  host.innerHTML = rows.length
    ? `<div class="movement">` + rows.map((r) => {
        const w = Math.max((Math.abs(r.delta) / maxAbs) * 50, 1.5);
        const align = r.delta >= 0 ? "flex-start" : "flex-end";
        return `<div class="mv-row">` +
          `<span class="mv-label">turn ${r.i + 1}</span>` +
          `<div class="mv-track" style="justify-content:${align}"><div class="mv-bar" style="width:${w}%;background:${r.color}"></div></div>` +
          `<span class="mv-val">${r.delta >= 0 ? "+" : ""}${r.delta.toFixed(3)}</span></div>`;
      }).join("") + `</div>`
    : `<span style="color:#9aa3b2;font-size:12px">No per-turn movement.</span>`;
}

function renderTranscript(t) {
  const container = document.getElementById("transcript");
  container.innerHTML = "";
  t.turns.forEach((turn, i) => {
    const div = document.createElement("div");
    div.className = "turn";
    if (turn.attribution.label !== "no_inflection") div.classList.add("inflection");
    div.dataset.index = String(i);

    const head = document.createElement("div");
    head.className = "turn-head";
    const speaker = document.createElement("span");
    speaker.className = `turn-speaker spk-${sideOf(turn.speaker)}`;
    speaker.style.color = speakerColor(turn.speaker);
    speaker.textContent = turn.speaker;
    const label = document.createElement("span");
    label.className = "turn-label";
    label.textContent = turn.attribution.label.replace(/_/g, " ");
    label.style.borderColor = LABEL_COLORS[turn.attribution.label] || "#2a2f3a";
    label.style.color = LABEL_COLORS[turn.attribution.label] || "#9aa3b2";
    head.appendChild(speaker);
    head.appendChild(label);

    const text = document.createElement("div");
    text.className = "turn-text";
    text.textContent = turn.text;

    const tagbar = document.createElement("div");
    tagbar.className = "tagbar";
    const scores = turn.rhetoric.scores;
    const total = RHETORIC_LABELS.reduce((s, l) => s + scores[l], 0) || 1;
    for (const l of RHETORIC_LABELS) {
      const seg = document.createElement("span");
      seg.style.width = `${(scores[l] / total) * 100}%`;
      seg.style.background = RHETORIC_COLORS[l];
      seg.style.opacity = String(0.35 + 0.65 * scores[l]);
      tagbar.appendChild(seg);
    }

    div.appendChild(head);
    div.appendChild(text);
    div.appendChild(tagbar);

    const foot = document.createElement("div");
    foot.className = "turn-foot";
    for (const l of RHETORIC_LABELS) {
      const chip = document.createElement("span");
      chip.className = "metric";
      chip.textContent = `${l} ${scores[l].toFixed(2)}`;
      foot.appendChild(chip);
    }
    const qChip = document.createElement("span");
    qChip.className = "metric";
    qChip.textContent = `quality ${turn.quality.toFixed(2)}`;
    foot.appendChild(qChip);
    div.appendChild(foot);

    div.addEventListener("click", () => selectTurn(i));
    container.appendChild(div);
  });
}

function selectTurn(i) {
  currentTurnIndex = i;
  document.querySelectorAll(".turn").forEach((el) => el.classList.remove("selected"));
  const el = document.querySelector(`.turn[data-index="${i}"]`);
  if (el) el.classList.add("selected");
  renderWhyCard(i);
}

function prevStanceForSpeaker(t, index) {
  const speaker = t.turns[index].speaker;
  for (let j = index - 1; j >= 0; j--) {
    if (t.turns[j].speaker === speaker) return t.turns[j].stance.S;
  }
  return 0.0;
}

function renderWhyCard(index) {
  const card = document.getElementById("why-card");
  if (index === null || index === undefined) {
    card.className = "why-card why-card-empty";
    card.textContent = "Click any highlighted turn in the transcript to inspect its attribution signals.";
    return;
  }
  const t = currentTranscript;
  const turn = t.turns[index];
  const prevS = prevStanceForSpeaker(t, index);
  const nowS = turn.stance.S;
  const delta = nowS - prevS;

  card.className = "why-card";
  card.innerHTML = "";

  const row = document.createElement("div");
  row.className = "why-row";
  row.innerHTML = `<strong>${turn.speaker}</strong><span>turn ${index + 1} of ${t.turns.length}</span>`;
  card.appendChild(row);

  const stanceRow = document.createElement("div");
  stanceRow.className = "why-row";
  stanceRow.innerHTML = `<span>Stance</span><span>${prevS.toFixed(2)} \u2192 ${nowS.toFixed(2)} (\u0394 = ${delta >= 0 ? "+" : ""}${delta.toFixed(2)})</span>`;
  card.appendChild(stanceRow);

  const signals = document.createElement("div");
  signals.className = "why-signals";
  for (const [name, on] of Object.entries(turn.attribution.signals)) {
    const row2 = document.createElement("div");
    row2.className = "signal" + (on ? " on" : "");
    row2.innerHTML = `<span><span class="signal-dot"></span> ${name.replace(/_/g, " ")}</span><span>${on ? "yes" : "no"}</span>`;
    signals.appendChild(row2);
  }
  card.appendChild(signals);

  const mechanism = document.createElement("div");
  mechanism.className = "mechanism";
  mechanism.innerHTML = `\u21d2 Likely mechanism: <strong>${turn.attribution.label.replace(/_/g, " ").toUpperCase()}</strong> &nbsp; (confidence ${turn.attribution.confidence.toFixed(2)}, quality ${turn.quality.toFixed(2)})`;
  card.appendChild(mechanism);
}

function renderStanceChart(t) {
  const bySpeaker = {};
  t.turns.forEach((turn, i) => {
    if (!bySpeaker[turn.speaker]) bySpeaker[turn.speaker] = [];
    bySpeaker[turn.speaker].push({ x: i + 1, y: turn.stance.S });
  });

  const datasets = Object.entries(bySpeaker).map(([speaker, points], idx) => ({
    label: speaker,
    data: points,
    borderColor: speaker.endsWith("_pro") ? "#33b679" : speaker.endsWith("_con") ? "#e8734a" : ["#5b8cff", "#f2c94c"][idx % 2],
    backgroundColor: "transparent",
    tension: 0.25,
    pointRadius: 4,
    pointHoverRadius: 6,
  }));

  const ctx = document.getElementById("stance-chart").getContext("2d");
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      scales: {
        x: { type: "linear", title: { display: true, text: "turn", color: "#9aa3b2" }, ticks: { color: "#9aa3b2", stepSize: 1 }, grid: { color: "#2a2f3a" } },
        y: { min: -1, max: 1, title: { display: true, text: "S", color: "#9aa3b2" }, ticks: { color: "#9aa3b2" }, grid: { color: "#2a2f3a" } },
      },
      plugins: {
        legend: { labels: { color: "#e8eaed" } },
      },
      onClick: (evt, elements) => {
        if (!elements.length) return;
        const point = elements[0];
        const speaker = chart.data.datasets[point.datasetIndex].label;
        const turnNumber = chart.data.datasets[point.datasetIndex].data[point.index].x;
        const index = currentTranscript.turns.findIndex(
          (turn, i) => i + 1 === turnNumber && turn.speaker === speaker
        );
        if (index !== -1) selectTurn(index);
      },
    },
  });
}

init();
