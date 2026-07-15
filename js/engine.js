// gpuroom — VRAM fit engine. Pure functions, no dependencies.
// Works in browser (ES module) and Node (tests).

// Effective bits per weight, including GGUF metadata/scale overhead.
export const QUANTS = {
  FP16:    { bpw: 16.0,  label: "FP16 / BF16" },
  Q8_0:    { bpw: 8.5,   label: "Q8_0 (GGUF)" },
  Q6_K:    { bpw: 6.59,  label: "Q6_K (GGUF)" },
  Q5_K_M:  { bpw: 5.69,  label: "Q5_K_M (GGUF)" },
  Q4_K_M:  { bpw: 4.85,  label: "Q4_K_M (GGUF)" },
  GPTQ4:   { bpw: 4.5,   label: "GPTQ/AWQ 4-bit" },
  IQ4_XS:  { bpw: 4.25,  label: "IQ4_XS (GGUF)" },
  Q3_K_M:  { bpw: 3.91,  label: "Q3_K_M (GGUF)" },
  Q2_K:    { bpw: 2.63,  label: "Q2_K (GGUF)" },
};

export const KV_DTYPES = {
  fp16: { bytes: 2,   label: "FP16 KV" },
  q8:   { bytes: 1,   label: "Q8 KV" },
  q4:   { bytes: 0.5, label: "Q4 KV" },
};

const GiB = 1024 ** 3;

// Model weights footprint in GiB for a given quant.
export function weightsGiB(paramsB, quantKey) {
  const q = QUANTS[quantKey];
  if (!q) throw new Error(`unknown quant: ${quantKey}`);
  return (paramsB * 1e9 * q.bpw) / 8 / GiB;
}

// KV cache in GiB: 2 (K and V) * layers * kv_heads * head_dim * ctx * bytes.
export function kvCacheGiB(model, contextLen, kvDtype = "fp16") {
  const d = KV_DTYPES[kvDtype];
  if (!d) throw new Error(`unknown kv dtype: ${kvDtype}`);
  return (2 * model.layers * model.kvHeads * model.headDim * contextLen * d.bytes) / GiB;
}

// Fixed runtime overhead: CUDA/Metal context + activation scratch.
// ~0.75 GiB base + 5% of weights, matching llama.cpp observed behavior.
export function overheadGiB(weights) {
  return 0.75 + 0.05 * weights;
}

// Full inference estimate.
export function inferenceEstimate(model, quantKey, contextLen, kvDtype = "fp16") {
  const weights = weightsGiB(model.paramsB, quantKey);
  const kv = kvCacheGiB(model, contextLen, kvDtype);
  const overhead = overheadGiB(weights);
  return { weights, kv, overhead, total: weights + kv + overhead };
}

// Fine-tuning estimates (GiB). Mixed-precision conventions:
//  full:  weights fp16 (2B) + grads fp16 (2B) + Adam moments fp32 (8B) + master fp32 (4B) = 16 B/param
//  lora:  frozen fp16 weights + adapter params (~rank-dependent, ~1-2% at r=16) trained at 16 B/param
//  qlora: frozen 4-bit (NF4 ~0.55 B/param) + same adapter budget
export function finetuneEstimate(model, method, contextLen = 2048, loraPct = 0.015) {
  const p = model.paramsB * 1e9;
  const act = activationsGiB(model, contextLen);
  let weights, trainState;
  if (method === "full") {
    weights = (p * 2) / GiB;
    trainState = (p * 14) / GiB; // grads + Adam + fp32 master
  } else if (method === "lora") {
    weights = (p * 2) / GiB;
    trainState = (p * loraPct * 16) / GiB;
  } else if (method === "qlora") {
    weights = (p * 0.55) / GiB;
    trainState = (p * loraPct * 16) / GiB;
  } else {
    throw new Error(`unknown method: ${method}`);
  }
  const total = weights + trainState + act + overheadGiB(weights);
  return { weights, trainState, activations: act, total };
}

// Rough activation memory for batch=1 training with grad checkpointing.
function activationsGiB(model, contextLen) {
  const hidden = model.headDim * (model.heads || model.kvHeads * 4);
  return (model.layers * contextLen * hidden * 2 * 2) / GiB / 8; // ckpt ≈ /8
}

// Fit verdict for one GPU. usableFrac: unified-memory Macs cap GPU alloc (~75%).
export function fitVerdict(totalGiB, gpu) {
  const usable = gpu.vramGiB * (gpu.usableFrac ?? 1.0);
  const ratio = totalGiB / usable;
  let status;
  if (ratio <= 0.88) status = "fits";          // headroom
  else if (ratio <= 1.0) status = "tight";     // fits, no headroom
  else if (ratio <= 1.35) status = "offload";  // partial CPU offload viable
  else status = "no";
  return { status, usable, ratio, freeGiB: Math.max(0, usable - totalGiB) };
}

// Largest context that fits a GPU for a model+quant. Binary search, 512-step.
export function maxContext(model, quantKey, gpu, kvDtype = "fp16") {
  const usable = gpu.vramGiB * (gpu.usableFrac ?? 1.0);
  const fixed = weightsGiB(model.paramsB, quantKey);
  if (fixed + overheadGiB(fixed) >= usable) return 0;
  let lo = 0, hi = model.maxContext || 131072;
  while (hi - lo > 512) {
    const mid = Math.floor((lo + hi) / 2 / 512) * 512;
    const est = inferenceEstimate(model, quantKey, mid, kvDtype);
    if (est.total <= usable) lo = mid; else hi = mid;
  }
  return Math.min(lo, model.maxContext || lo);
}

// Best (highest-quality) quant that fits at a given context.
const QUALITY_ORDER = ["FP16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "GPTQ4", "IQ4_XS", "Q3_K_M", "Q2_K"];
export function bestQuant(model, gpu, contextLen, kvDtype = "fp16") {
  for (const q of QUALITY_ORDER) {
    const est = inferenceEstimate(model, quantKey(q), contextLen, kvDtype);
    if (fitVerdict(est.total, gpu).status === "fits" || fitVerdict(est.total, gpu).status === "tight") {
      return { quant: q, ...est };
    }
  }
  return null;
}
function quantKey(q) { return q; }
