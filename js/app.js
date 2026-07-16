// gpuroom UI — vanilla ES module, no deps. Engine does all math.
import { QUANTS, KV_DTYPES, inferenceEstimate, finetuneEstimate, fitVerdict, maxContext, bestQuant } from "./engine.js";
import { MODELS, GPUS } from "../data/catalog.js";

const $ = (id) => document.getElementById(id);

// ---------- state (encoded in location.hash) ----------
const state = {
  model: "llama-3.1-8b", quant: "Q4_K_M", ctx: 16384, kv: "fp16",
  tab: "fit", gpuA: "rtx4090", gpuB: "m4pro-24", ft: "qlora",
};
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  for (const k of Object.keys(state)) if (p.has(k)) state[k] = k === "ctx" ? +p.get(k) : p.get(k);
  if (!MODELS.find(m => m.id === state.model)) state.model = MODELS[0].id;
  if (!QUANTS[state.quant]) state.quant = "Q4_K_M";
  if (!KV_DTYPES[state.kv]) state.kv = "fp16";
}
function writeHash() {
  history.replaceState(null, "", "#" + new URLSearchParams(state).toString());
}

const model = () => MODELS.find(m => m.id === state.model);
const gpu = (id) => GPUS.find(g => g.id === id) || GPUS[0];

// ---------- formatting ----------
const gib = (v) => v.toFixed(1);
const ctxFmt = (t) => t >= 1024 ? Math.round(t / 1024) + "k" : String(t);

// Context slider: 512 → model max on a power scale so small contexts get room.
const CTX_MIN = 512;
function sliderToCtx(pos) { // pos 0..100
  const max = model().maxContext;
  const v = CTX_MIN * Math.pow(max / CTX_MIN, pos / 100);
  return Math.max(CTX_MIN, Math.round(v / 512) * 512);
}
function ctxToSlider(ctx) {
  const max = model().maxContext;
  return Math.round(100 * Math.log(ctx / CTX_MIN) / Math.log(max / CTX_MIN));
}

// ---------- shared card renderer ----------
function gpuCard(g, totalGiB, est, showMaxCtx) {
  const v = fitVerdict(totalGiB, g);
  const usable = v.usable;
  const pct = (x) => Math.min(100, (x / usable) * 100).toFixed(2);
  const over = totalGiB > usable;
  const segs = est
    ? `<i class="w" style="width:${pct(est.weights)}%"></i>` +
      `<i class="k" style="width:${pct(est.kv ?? est.trainState ?? 0)}%"></i>` +
      `<i class="o" style="width:${pct((est.overhead ?? 0) + (est.activations ?? 0))}%"></i>`
    : `<i class="w" style="width:${pct(totalGiB)}%"></i>`;
  const labels = { fits: "FITS", tight: "TIGHT", offload: "OFFLOAD", no: "NO FIT" };
  let extra = "";
  if (showMaxCtx) {
    const mc = maxContext(model(), state.quant, g, state.kv);
    const bq = bestQuant(model(), g, state.ctx, state.kv);
    extra = `<div class="meta">max context: <b class="num">${mc ? ctxFmt(mc) : "—"}</b>` +
            ` · best quant @ ${ctxFmt(state.ctx)}: <b>${bq ? bq.quant : "none"}</b></div>`;
  }
  return `<div class="card">
    <h3>${g.name} <span class="pill ${v.status}">${labels[v.status]}</span></h3>
    <div class="sub">${g.vramGiB} GiB${g.unified ? ` unified · ~${Math.round((g.usableFrac ?? 1) * 100)}% usable` : ""}</div>
    <div class="bar">${over ? `<i class="x" style="width:100%"></i>` : segs}</div>
    <div class="meta"><b class="num">${gib(totalGiB)}</b> / ${gib(usable)} GiB · ${over ? `<b class="num">${gib(totalGiB - usable)}</b> GiB over` : `<b class="num">${gib(v.freeGiB)}</b> GiB free`}</div>
    ${extra}</div>`;
}

// ---------- tab renderers ----------
function renderFit() {
  const m = model();
  const est = inferenceEstimate(m, state.quant, state.ctx, state.kv);
  $("hero").innerHTML =
    `${m.name} · ${QUANTS[state.quant].label} · ${ctxFmt(state.ctx)} ctx needs <span class="val num">${gib(est.total)} GiB</span>` +
    `<small>weights ${gib(est.weights)} + KV cache ${gib(est.kv)} + overhead ${gib(est.overhead)} GiB</small>`;
  $("fitGrid").innerHTML = GPUS.map(g => gpuCard(g, est.total, est, true)).join("");
}

function renderCmp() {
  const m = model(), A = gpu(state.gpuA), B = gpu(state.gpuB);
  const best = {};
  for (const g of [A, B]) { const b = bestQuant(m, g, state.ctx, state.kv); best[g.id] = b && b.quant; }
  const labels = { fits: "FITS", tight: "TIGHT", offload: "OFFLOAD", no: "NO FIT" };
  const rows = Object.keys(QUANTS).map(q => {
    const est = inferenceEstimate(m, q, state.ctx, state.kv);
    const cell = (g) => {
      const v = fitVerdict(est.total, g);
      return `<td><span class="num">${gib(est.total)}</span> GiB <span class="pill ${v.status}">${labels[v.status]}</span></td>`;
    };
    const isBest = best[A.id] === q || best[B.id] === q;
    return `<tr${isBest ? ' class="best"' : ""}><td>${QUANTS[q].label}${isBest ? " ★" : ""}</td>${cell(A)}${cell(B)}</tr>`;
  }).join("");
  $("cmpTable").innerHTML =
    `<thead><tr><th>Quant</th><th>${A.name}</th><th>${B.name}</th></tr></thead><tbody>${rows}</tbody>`;
}

function renderFt() {
  const m = model();
  const est = finetuneEstimate(m, state.ft, Math.min(state.ctx, 8192));
  const box = (label, v) => `<div class="card"><div class="sub">${label}</div><div class="big num">${gib(v)}</div><div class="meta">GiB</div></div>`;
  $("ftBreak").innerHTML =
    box("Weights", est.weights) + box("Optimizer / train state", est.trainState) +
    box("Activations (ckpt)", est.activations) + box("Total", est.total);
  $("ftGrid").innerHTML = GPUS.map(g => gpuCard(g, est.total, est, false)).join("");
}

function render() {
  $("ctxVal").textContent = ctxFmt(state.ctx) + " tokens";
  document.querySelectorAll(".tabs [role=tab]").forEach(b =>
    b.setAttribute("aria-selected", String(b.dataset.tab === state.tab)));
  for (const t of ["fit", "cmp", "ft"]) $("tab-" + t).hidden = t !== state.tab;
  if (state.tab === "fit") renderFit();
  else if (state.tab === "cmp") renderCmp();
  else renderFt();
  writeHash();
}

// ---------- controls ----------
function fillSelect(el, items, value) {
  el.innerHTML = items.map(([v, label]) => `<option value="${v}"${v === value ? " selected" : ""}>${label}</option>`).join("");
}
function segControl(el, items, get, set) {
  el.innerHTML = items.map(([v, label]) =>
    `<button type="button" data-v="${v}" aria-pressed="${String(get() === v)}">${label}</button>`).join("");
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    set(b.dataset.v);
    el.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
    render();
  });
}

function init() {
  readHash();
  fillSelect($("model"), MODELS.map(m => [m.id, `${m.name} — ${m.paramsB}B`]), state.model);
  fillSelect($("quant"), Object.entries(QUANTS).map(([k, q]) => [k, q.label]), state.quant);
  fillSelect($("gpuA"), GPUS.map(g => [g.id, g.name]), state.gpuA);
  fillSelect($("gpuB"), GPUS.map(g => [g.id, g.name]), state.gpuB);
  segControl($("kvSeg"), Object.entries(KV_DTYPES).map(([k, d]) => [k, d.label]), () => state.kv, v => state.kv = v);
  segControl($("ftSeg"), [["full", "Full"], ["lora", "LoRA"], ["qlora", "QLoRA"]], () => state.ft, v => state.ft = v);

  state.ctx = Math.min(state.ctx, model().maxContext);
  $("ctx").value = ctxToSlider(state.ctx);

  $("model").addEventListener("change", e => {
    state.model = e.target.value;
    state.ctx = Math.min(state.ctx, model().maxContext);
    $("ctx").value = ctxToSlider(state.ctx);
    render();
  });
  $("quant").addEventListener("change", e => { state.quant = e.target.value; render(); });
  $("gpuA").addEventListener("change", e => { state.gpuA = e.target.value; render(); });
  $("gpuB").addEventListener("change", e => { state.gpuB = e.target.value; render(); });
  $("ctx").addEventListener("input", e => { state.ctx = sliderToCtx(+e.target.value); render(); });
  document.querySelector(".tabs").addEventListener("click", e => {
    const b = e.target.closest("[role=tab]"); if (!b) return;
    state.tab = b.dataset.tab; render();
  });
  $("share").addEventListener("click", async () => {
    writeHash();
    try { await navigator.clipboard.writeText(location.href); $("share").textContent = "Copied ✓"; }
    catch { $("share").textContent = location.href; }
    setTimeout(() => $("share").textContent = "Copy share link", 1600);
  });

  render();
}
init();
