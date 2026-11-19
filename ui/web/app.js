// VogueFusion web app. Talks only to the HTTP API on the same origin; no business logic here.
"use strict";

const POLL_MS = 1000;
const STATUS_REFRESH_MS = 3000;
const TERMINAL = ["succeeded", "failed", "dead"];
const WORKFLOW_LABELS = { v2: "VogueFusion", v1: "VogueFusion (no LoRA)" };
const SUBSTATES = {
  claimed: "Starting", uploading_references: "Uploading references", queued_in_comfyui: "Queued in ComfyUI",
  loading_model: "Loading model", encoding_prompt: "Encoding prompt", encoding_references: "Encoding references",
  sampling: "Sampling", decoding: "Decoding", saving: "Saving", fetching_outputs: "Fetching outputs",
  extracting_colours: "Extracting colours", solving: "Solving dye recipes",
};
const ICON = {
  upload: '<svg class="i" viewBox="0 0 24 24"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M12 12v6"/><path d="m15 15-3-3-3 3"/></svg>',
  trash: '<svg class="i" viewBox="0 0 24 24"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/></svg>',
  spin: '<svg class="i spin" viewBox="0 0 24 24"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>',
};

const state = {
  workflows: {},
  costing: null,   // /v1/costing: catalogue, fabric defaults, substrates
  dyeRgb: {},      // dye name -> [r, g, b]
  slots: [],       // Generate references: {url, sha, error}
  design: null,    // design being costed: {url, sha, error}
  quote: null,     // last quote result
  quoteKey: null,  // the request it answers
  busy: false,
  statusTimer: null,
};

// ---------------------------------------------------------------- helpers

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
const rgb255 = (srgb) => srgb.map((c) => Math.round(c * 255));
const money = (x) => `${state.costing && state.costing.currency === "INR" ? "₹" : ""}${x.toFixed(2)}`;

class Busy extends Error {
  constructor(retryAfter, depth) { super("busy"); this.retryAfter = retryAfter; this.depth = depth; }
}

async function call(method, path, body, headers = {}) {
  const opt = { method, headers: { ...headers } };
  if (body instanceof FormData) opt.body = body;
  else if (body !== undefined) { opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
  const r = await fetch(path, opt);
  let data = null;
  try { data = await r.json(); } catch (_) { /* not JSON */ }
  return { r, data };
}

function errorText(r, data) {
  const d = data && data.detail;
  return typeof d === "string" ? d : d ? JSON.stringify(d) : `HTTP ${r.status}`;
}

async function uploadImage(file) {
  const fd = new FormData();
  fd.append("file", file, file.name);
  const { r, data } = await call("POST", "/v1/references", fd);
  if (!r.ok) throw new Error(errorText(r, data));
  return data.sha256;
}

/** POST a job; returns its Location. Throws Busy on 429. */
async function submit(path, body) {
  const { r, data } = await call("POST", path, body, { "Idempotency-Key": crypto.randomUUID() });
  if (r.status === 429) throw new Busy(r.headers.get("Retry-After"), data && data.depth);
  if (!r.ok) throw new Error(errorText(r, data));
  return r.headers.get("Location");
}

/** Poll a job until it finishes, calling onUpdate with every snapshot. */
async function follow(location, onUpdate) {
  for (;;) {
    const { r, data } = await call("GET", location);
    if (!r.ok) throw new Error(errorText(r, data));
    onUpdate(data);
    if (TERMINAL.includes(data.state)) return data;
    await sleep(POLL_MS);
  }
}

function progress(job, t0, label) {
  const step = job.state === "queued" ? "Queued" : SUBSTATES[job.substate] || "Running";
  const secs = Math.round((Date.now() - t0) / 1000);
  return `${ICON.spin}${esc(label)}: ${esc(step)}…<span class="sub">${secs} s · attempt ${job.attempts || 1}</span>`;
}

/** Fill an output panel. The panel's corner chip shows only with placeholders. */
function show(el, html, cls = "placeholder") {
  el.className = cls;
  el.innerHTML = html;
  const chip = el.parentElement.querySelector(".chip");
  if (chip) chip.hidden = !cls.startsWith("placeholder");
}

function busyMessage(e, what) {
  return `The ${what} queue is full.<span class="sub">Try again in about ${esc(e.retryAfter)} s (${esc(e.depth)} jobs ahead).</span>`;
}

function setBusy(on) {
  state.busy = on;
  for (const id of ["generate", "extract", "calc"]) $(id).disabled = on;
}

// ---------------------------------------------------------------- theme

function setTheme(t) {
  if (t === "light") document.documentElement.dataset.theme = "light";
  else delete document.documentElement.dataset.theme;
  try { localStorage.setItem("vf-theme", t); } catch (_) { /* storage unavailable */ }
}
try { setTheme(localStorage.getItem("vf-theme") || "dark"); } catch (_) { setTheme("dark"); }
$("theme").onclick = () => setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");

// ---------------------------------------------------------------- routing

const VIEWS = ["generate", "colors", "dyes", "cost", "status"];

function route() {
  const view = VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "generate";
  for (const v of VIEWS) $(`view-${v}`).classList.toggle("active", v === view);
  for (const a of document.querySelectorAll("nav a")) {
    const on = a.dataset.view === view;
    a.classList.toggle("active", on);
    if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  }
  clearInterval(state.statusTimer);
  if (view === "status") { renderStatus(); state.statusTimer = setInterval(renderStatus, STATUS_REFRESH_MS); }
  if (view === "dyes") renderDyes();
  if (view === "cost") renderCostView();
}
window.addEventListener("hashchange", route);

// ---------------------------------------------------------------- upload widgets

let onPicked = null;
$("picker").onchange = (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (f && onPicked) onPicked(f);
};

function dropZone(onFile) {
  const d = document.createElement("div");
  d.className = "drop";
  d.tabIndex = 0;
  d.setAttribute("role", "button");
  d.innerHTML = "Drag Files to Upload " + ICON.upload;
  d.onclick = () => { onPicked = onFile; $("picker").click(); };
  d.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); d.click(); } };
  d.ondragover = (e) => { e.preventDefault(); d.classList.add("over"); };
  d.ondragleave = () => d.classList.remove("over");
  d.ondrop = (e) => {
    e.preventDefault();
    d.classList.remove("over");
    if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]);
  };
  return d;
}

function preview(item, alt, onRemove, large = false) {
  const p = document.createElement("div");
  p.className = "preview" + (large ? " large" : "");
  const status = item.error ? `<span class="status err">${esc(item.error)}</span>`
    : item.sha ? "" : '<span class="status">Uploading…</span>';
  p.innerHTML = `<img src="${item.url}" alt="${esc(alt)}">${status}<button class="remove">${ICON.trash} Remove image</button>`;
  p.querySelector("button").onclick = onRemove;
  return p;
}

// ---------------------------------------------------------------- Generate

function renderSlots() {
  const box = $("refs");
  box.innerHTML = "";
  state.slots.forEach((s, i) => {
    const label = document.createElement("label");
    label.className = "field";
    label.textContent = "LoadImage";
    box.appendChild(label);
    box.appendChild(s
      ? preview(s, `Reference ${i + 1}`, () => { URL.revokeObjectURL(s.url); state.slots[i] = null; renderSlots(); })
      : dropZone((f) => uploadSlot(i, f)));
  });
}

async function uploadSlot(i, file) {
  state.slots[i] = { url: URL.createObjectURL(file), sha: null, error: null };
  renderSlots();
  try {
    const sha = await uploadImage(file);
    if (state.slots[i]) state.slots[i].sha = sha;
  } catch (e) {
    if (state.slots[i]) state.slots[i].error = e.message;
  }
  renderSlots();
}

function applyWorkflow(name) {
  const w = state.workflows[name];
  $("form-title").textContent = WORKFLOW_LABELS[name] || name;
  $("negative").value = w.default_negative;
  const keep = state.slots.slice(0, w.reference_slots);
  state.slots = Array.from({ length: w.reference_slots }, (_, i) => keep[i] || null);
  renderSlots();
}

function renderKeywords(store) {
  const box = $("keywords");
  box.innerHTML = "";
  for (const kw of store.keywords || []) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = `+ ${kw}`;
    b.title = store.label || "";
    b.onclick = () => {
      const t = $("positive");
      t.value = t.value.trim() ? `${t.value.trim().replace(/,$/, "")}, ${kw}` : kw;
    };
    box.appendChild(b);
  }
}

async function generate() {
  if (state.busy) return;
  const out = $("gen-out");
  if (state.slots.some((s) => s && !s.sha && !s.error)) return show(out, "Wait for the reference uploads to finish.");
  const refs = state.slots.filter((s) => s && s.sha).map((s) => s.sha);
  if (!refs.length) return show(out, 'Add at least one reference image.<span class="sub">Use the LoadImage fields on the left.</span>');
  const prompt = $("positive").value.trim();
  if (!prompt) return show(out, "Enter a prompt in the first CLIPTextEncode field.");
  const name = $("workflow").value;
  const body = { workflow: name, prompt, negative: $("negative").value, batch: state.workflows[name].default_batch, references: refs };
  setBusy(true);
  const t0 = Date.now();
  show(out, ICON.spin + "Submitting…");
  try {
    const job = await follow(await submit("/v1/designs", body), (j) => show(out, progress(j, t0, "Generating")));
    if (job.state !== "succeeded") return show(out, `Generation ${esc(job.state)}<span class="sub">${esc(job.error || "")}</span>`, "placeholder err");
    showResults(job.result);
  } catch (e) {
    show(out, e instanceof Busy ? busyMessage(e, "generation") : `Error: ${esc(e.message)}`, e instanceof Busy ? "placeholder" : "placeholder err");
  } finally {
    setBusy(false);
  }
}

function showResults(res) {
  const out = $("gen-out");
  const cards = res.artifacts.map((sha, i) => `
    <div class="card">
      <img src="/v1/artifacts/${sha}" alt="Generated design ${i + 1}">
      <div class="actions">
        <button type="button" data-cost="${sha}">Cost this design</button>
        <a href="/v1/artifacts/${sha}" download="voguefusion_${res.seed}_${i + 1}.png">Download</a>
      </div>
    </div>`).join("");
  show(out, `<div class="grid">${cards}</div><span class="meta">seed ${esc(res.seed)} · ${esc(res.comfy_seconds.toFixed(1))} s</span>`, "");
  for (const b of out.querySelectorAll("[data-cost]")) b.onclick = () => { useDesign(b.dataset.cost); location.hash = "#colors"; };
}

// ---------------------------------------------------------------- Colors

function costBody(nColours) {
  return {
    artifact: state.design.sha,
    n_colours: nColours,
    gsm: Number($("gsm").value),
    length_m: Number($("length").value),
    width_m: Number($("width").value),
    substrate: $("substrate").value,
  };
}

function clearQuote() {
  state.quote = null;
  state.quoteKey = null;
  show($("colors-out"), state.design ? "Click Extract colors." : 'Pick a design, then click Extract colors.<span class="sub">Use a result from Generate, or upload any design image.</span>');
}

function useDesign(sha) {
  state.design = { url: `/v1/artifacts/${sha}`, sha, error: null };
  clearQuote();
  renderDesignSlot();
}

function renderDesignSlot() {
  const slot = $("design-slot");
  slot.innerHTML = "";
  slot.appendChild(state.design
    ? preview(state.design, "Design", () => { state.design = null; clearQuote(); renderDesignSlot(); }, true)
    : dropZone(uploadDesign));
}

async function uploadDesign(file) {
  state.design = { url: URL.createObjectURL(file), sha: null, error: null };
  clearQuote();
  renderDesignSlot();
  try {
    const sha = await uploadImage(file);
    if (state.design) state.design.sha = sha;
  } catch (e) {
    if (state.design) state.design.error = e.message;
  }
  renderDesignSlot();
}

/** Submit a quote and show progress in `out`. Returns the result or null. */
async function runQuote(body, out, label) {
  setBusy(true);
  const t0 = Date.now();
  show(out, ICON.spin + "Submitting…");
  try {
    const job = await follow(await submit("/v1/costs", body), (j) => show(out, progress(j, t0, label)));
    if (job.state !== "succeeded") {
      show(out, `${esc(label)} ${esc(job.state)}<span class="sub">${esc(job.error || "")}</span>`, "placeholder err");
      return null;
    }
    state.quote = job.result;
    state.quoteKey = JSON.stringify(body);
    return job.result;
  } catch (e) {
    show(out, e instanceof Busy ? busyMessage(e, "costing") : `Error: ${esc(e.message)}`, e instanceof Busy ? "placeholder" : "placeholder err");
    return null;
  } finally {
    setBusy(false);
  }
}

async function extract() {
  if (state.busy) return;
  const out = $("colors-out");
  if (!state.design) return show(out, "Upload a design or pick one from Generate first.");
  if (!state.design.sha) return show(out, state.design.error ? "The upload failed; remove it and try again." : "Wait for the upload to finish.");
  const q = await runQuote(costBody(Number($("ncolors").value)), out, "Extracting colors");
  if (q) renderColors(q);
}

function renderColors(q) {
  const max = Math.max(...q.colours.map((c) => c.area_fraction));
  const bars = q.colours.map((c) => `
    <div class="col"><span class="val">${pct(c.area_fraction)}</span>
      <div class="bar" style="height:${(c.area_fraction / max) * 100}%;background:${c.hex}"></div></div>`).join("");
  const labels = q.colours.map((_, i) => `<span>Color ${i + 1}</span>`).join("");
  const rows = q.colours.map((c, i) => {
    const [r, g, b] = rgb255(c.srgb);
    return `<div class="row"><span class="sw" style="background:${c.hex}"></span>
      <div class="grow"><b>Color ${i + 1}</b> <span class="muted">HEX</span> <code>${c.hex}</code>
      <span class="muted">· RGB</span> [${r} ${g} ${b}]</div><b>${pct(c.area_fraction, 2)}</b></div>`;
  }).join("");
  show($("colors-out"), `<div class="content">
      <h3>Color proportions</h3>
      <div class="chart">${bars}</div><div class="chart-x">${labels}</div>
      <div class="rows">${rows}</div>
      <div class="gen-row" style="justify-content:flex-start;gap:12px;padding-top:22px">
        <a class="btn small" href="#dyes">See dye recipes</a><a class="btn small" href="#cost">Calculate cost</a>
      </div>
      <p class="note">Colors are clustered in CIE Lab, so clusters follow perceived difference.</p>
    </div>`, "");
}

// ---------------------------------------------------------------- Dye Recipes

function renderDyes() {
  const out = $("dyes-out");
  const q = state.quote;
  if (!q) return show(out, 'No colors yet.<span class="sub">Extract colors from a design first.</span><a class="btn small" href="#colors">Go to Colors</a>');
  const tol = state.costing.delta_e_tolerance;
  const cards = q.colours.map((c, i) => {
    const total = c.dyes.reduce((s, d) => s + d.concentration_g_per_kg, 0) || 1;
    const dyes = c.dyes.map((d) => {
      const share = d.concentration_g_per_kg / total;
      const rgb = state.dyeRgb[d.name] || [128, 128, 128];
      return `<div class="dye"><span class="sw small" style="background:rgb(${rgb.join(",")})"></span>
        <span>${esc(d.name)}</span><span class="track"><i style="width:${share * 100}%"></i></span>
        <span class="muted">${d.concentration_g_per_kg.toFixed(2)} g/kg</span></div>`;
    }).join("");
    const badge = c.feasible
      ? `<span class="badge ok">ΔE ${c.delta_e.toFixed(2)} · within ${tol}</span>`
      : `<span class="badge bad">ΔE ${c.delta_e.toFixed(2)} · outside ${tol}</span>`;
    return `<div class="recipe">
      <header><b>Color ${i + 1}</b><span class="muted">${pct(c.area_fraction)} of area</span></header>
      <div class="pair">
        <div><span class="sw" style="background:${c.hex}"></span>Target<br><code>${c.hex}</code></div>
        <div><span class="sw" style="background:${c.predicted_hex}"></span>Predicted<br><code>${c.predicted_hex}</code></div>
      </div>
      ${badge}
      <div style="margin-top:12px">${dyes || '<span class="muted">No dye needed.</span>'}</div>
    </div>`;
  }).join("");
  show(out, `<div class="content">
      <p class="note" style="margin:0 0 18px">Each color gets the cheapest mix of up to 3 catalogue dyes whose
      predicted color (Kubelka-Munk on the chosen substrate) is within ΔE2000 ${tol} of it.
      Grams are per kg of fabric.</p>
      <div class="recipes">${cards}</div></div>`, "");
}

// ---------------------------------------------------------------- Cost

function initCostForm() {
  const c = state.costing;
  $("gsm").innerHTML = c.fabric.gsm_options.map((g) => `<option value="${g}">${g}</option>`).join("");
  $("gsm").value = c.fabric.default_gsm;
  $("length").value = c.fabric.default_length_m;
  $("width").value = c.fabric.default_width_m;
  $("substrate").innerHTML = c.substrates.map((s) => `<option value="${esc(s)}">${esc(s)}</option>`).join("");
  $("substrate").value = c.default_substrate;
  $("ncolors").value = c.default_n_colours;
  $("ncolors-val").textContent = c.default_n_colours;
}

function nColours() {
  return state.quoteKey ? JSON.parse(state.quoteKey).n_colours : Number($("ncolors").value);
}

function renderCostView() {
  const out = $("cost-out");
  if (!state.design || !state.design.sha) return show(out, 'No design yet.<span class="sub">Extract colors from a design first.</span><a class="btn small" href="#colors">Go to Colors</a>');
  if (state.quote && state.quoteKey === JSON.stringify(costBody(nColours()))) return renderCost(state.quote);
  show(out, "Click Calculate cost.");
}

async function calculate() {
  if (state.busy) return;
  if (!state.design || !state.design.sha) return renderCostView();
  const q = await runQuote(costBody(nColours()), $("cost-out"), "Calculating");
  if (q) renderCost(q);
}

function renderCost(q) {
  const body = JSON.parse(state.quoteKey);
  const byColour = q.colours.map((c, i) => `<div class="row"><span class="sw" style="background:${c.hex}"></span>
      <div class="grow"><b>Color ${i + 1}</b> <span class="muted">· ${pct(c.area_fraction)} of area</span></div>
      <b>${money(c.cost)}</b></div>`).join("");
  const mass = {};
  for (const c of q.colours) for (const d of c.dyes) mass[d.name] = (mass[d.name] || 0) + d.dye_kg;
  const totalKg = Object.values(mass).reduce((a, b) => a + b, 0) || 1;
  const byDye = Object.entries(mass).sort((a, b) => b[1] - a[1]).map(([name, kg]) => {
    const rgb = state.dyeRgb[name] || [128, 128, 128];
    return `<div class="dye"><span class="sw small" style="background:rgb(${rgb.join(",")})"></span>
      <span>${esc(name)}</span><span class="track"><i style="width:${(kg / totalKg) * 100}%"></i></span>
      <span class="muted">${pct(kg / totalKg)}</span></div>`;
  }).join("");
  const warn = q.all_feasible ? "" : '<p class="note" style="color:var(--danger)">Some colors cannot be matched within tolerance with this catalogue; their cost is for the closest match.</p>';
  show($("cost-out"), `<div class="content">
      <span class="muted">Dye cost for the entire design</span>
      <div class="total">${money(q.total_cost)}</div>
      <span class="muted">${body.length_m} m × ${body.width_m} m at ${body.gsm} GSM on ${esc(body.substrate)} =
        ${q.fabric_mass_kg.toFixed(3)} kg of fabric</span>
      ${warn}
      <div class="split">
        <div><h3>By color</h3><div class="rows" style="margin-top:0">${byColour}</div></div>
        <div><h3>Dye proportions</h3>${byDye}</div>
      </div>
      <p class="note">Catalogue dye colors are placeholders and the depth at which each dye reaches its
      catalogue color is assumed, so read the absolute figure as indicative and comparisons between
      designs as meaningful (docs/DECISIONS.md, ADR-005).</p>
    </div>`, "");
}

// ---------------------------------------------------------------- Status

function fmt(v) { return v === null || v === undefined ? "–" : v.toFixed(2); }

async function renderStatus() {
  const out = $("status-out");
  try {
    const [h, m] = await Promise.all([call("GET", "/healthz"), call("GET", "/v1/metrics")]);
    if (!h.r.ok || !m.r.ok) throw new Error("API returned an error");
    const dot = (ok) => `<span class="dot" style="background:${ok ? "var(--ok)" : "var(--danger)"}"></span>`;
    const names = { generate: "Generation (GPU worker)", cost: "Costing (CPU worker)" };
    const kinds = Object.entries(m.data).map(([kind, s]) => {
      const lat = [["Queue wait", s.queue_wait_s], ["Service", s.service_s], ["End to end", s.end_to_end_s]]
        .map(([n, v]) => `<tr><td>${n}</td><td>${fmt(v.p50)}</td><td>${fmt(v.p95)}</td><td>${fmt(v.max)}</td></tr>`).join("");
      const age = s.oldest_queued_age_s === null ? "–" : `${Math.round(s.oldest_queued_age_s)} s`;
      return `<h3>${esc(names[kind] || kind)}</h3>
        <div class="stats">
          <div class="stat"><span>Depth</span><b>${s.depth} / ${s.max_depth}</b></div>
          <div class="stat"><span>Succeeded</span><b>${s.counts.succeeded}</b></div>
          <div class="stat"><span>Failed + dead</span><b>${s.counts.failed + s.counts.dead}</b></div>
          <div class="stat"><span>Oldest queued</span><b>${age}</b></div>
        </div>
        <table><thead><tr><th>Latency (s), last ${s.window} jobs</th><th>p50</th><th>p95</th><th>max</th></tr></thead>
        <tbody>${lat}</tbody></table>`;
    }).join("");
    show(out, `<div class="content" style="width:100%">
        <p style="margin:0 0 22px">${dot(true)}API ${esc(h.data.status)}
          &nbsp;&nbsp;${dot(h.data.comfyui_reachable)}ComfyUI ${h.data.comfyui_reachable ? "reachable" : "unreachable"}</p>
        ${kinds}</div>`, "");
  } catch (e) {
    show(out, `API not reachable: ${esc(e.message)}`, "placeholder err");
  }
}

// ---------------------------------------------------------------- start

$("generate").onclick = generate;
$("extract").onclick = extract;
$("calc").onclick = calculate;
$("ncolors").oninput = (e) => { $("ncolors-val").textContent = e.target.value; };
for (const id of ["gsm", "length", "width", "substrate"]) $(id).onchange = renderCostView;

async function init() {
  try {
    const [w, c, s] = await Promise.all([call("GET", "/v1/workflows"), call("GET", "/v1/costing"), call("GET", "/v1/suggestions")]);
    if (!w.r.ok || !c.r.ok) throw new Error("API returned an error");
    state.workflows = w.data.workflows;
    state.costing = c.data;
    state.dyeRgb = Object.fromEntries(c.data.dyes.map((d) => [d.name, d.rgb]));
    const order = Object.keys(state.workflows).sort((a, b) => (a === "v2" ? -1 : b === "v2" ? 1 : a.localeCompare(b)));
    const sel = $("workflow");
    sel.innerHTML = order.map((n) => `<option value="${esc(n)}">${esc(WORKFLOW_LABELS[n] || n)}</option>`).join("");
    sel.onchange = () => applyWorkflow(sel.value);
    applyWorkflow(sel.value);
    initCostForm();
    if (s.r.ok) renderKeywords(s.data);
  } catch (e) {
    show($("gen-out"), `API not reachable: ${esc(e.message)}<span class="sub">Start it with scripts/vf.sh start</span>`, "placeholder err");
  }
  renderDesignSlot();
  route();
}
init();
