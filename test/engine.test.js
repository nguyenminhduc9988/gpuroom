import { test } from "node:test";
import assert from "node:assert/strict";
import {
  QUANTS, weightsGiB, kvCacheGiB, inferenceEstimate, finetuneEstimate,
  fitVerdict, maxContext, bestQuant,
} from "../js/engine.js";
import { MODELS, GPUS } from "../data/catalog.js";

const m = (id) => MODELS.find((x) => x.id === id);
const g = (id) => GPUS.find((x) => x.id === id);

test("weights: llama-8B Q4_K_M lands near real GGUF file size (~4.6 GiB)", () => {
  const w = weightsGiB(8.03, "Q4_K_M");
  assert.ok(w > 4.2 && w < 5.0, `got ${w}`);
});

test("weights: 70B FP16 ≈ 131 GiB", () => {
  const w = weightsGiB(70.6, "FP16");
  assert.ok(w > 128 && w < 135, `got ${w}`);
});

test("kv cache: llama-8B @ 8k fp16 ≈ 1 GiB (GQA, 8 kv heads)", () => {
  const kv = kvCacheGiB(m("llama-3.1-8b"), 8192, "fp16");
  assert.ok(kv > 0.9 && kv < 1.1, `got ${kv}`);
});

test("kv cache: q4 KV is 1/4 of fp16", () => {
  const model = m("qwen2.5-14b");
  assert.ok(Math.abs(kvCacheGiB(model, 16384, "q4") - kvCacheGiB(model, 16384, "fp16") / 4) < 1e-9);
});

test("inference: 8B Q4_K_M @ 8k fits a 12GB card with headroom", () => {
  const est = inferenceEstimate(m("llama-3.1-8b"), "Q4_K_M", 8192);
  const v = fitVerdict(est.total, g("rtx3060-12"));
  assert.equal(v.status, "fits");
});

test("inference: 70B Q4_K_M does NOT fit a 24GB card", () => {
  const est = inferenceEstimate(m("llama-3.3-70b"), "Q4_K_M", 4096);
  const v = fitVerdict(est.total, g("rtx4090"));
  assert.equal(v.status, "no");
});

test("inference: 32B Q4_K_M @ 8k fits a 24GB card", () => {
  const est = inferenceEstimate(m("qwen2.5-32b"), "Q4_K_M", 8192);
  const v = fitVerdict(est.total, g("rtx4090"));
  assert.ok(v.status === "fits" || v.status === "tight", v.status);
});

test("unified memory: Mac usable share is capped", () => {
  const v = fitVerdict(10, g("m4-16"));
  assert.equal(v.usable, 12); // 16 * 0.75
});

test("maxContext: monotonic in VRAM and never exceeds model limit", () => {
  const model = m("llama-3.1-8b");
  const small = maxContext(model, "Q4_K_M", g("rtx3060-12"));
  const big = maxContext(model, "Q4_K_M", g("rtx4090"));
  assert.ok(big >= small);
  assert.ok(big <= model.maxContext);
  assert.ok(small > 8192, `12GB should give >8k ctx on 8B Q4, got ${small}`);
});

test("maxContext: 0 when weights alone overflow", () => {
  assert.equal(maxContext(m("qwen2.5-72b"), "FP16", g("rtx4060")), 0);
});

test("bestQuant: 24GB card runs 8B at FP16, 32B at ~Q4/Q5", () => {
  const b8 = bestQuant(m("llama-3.1-8b"), g("rtx4090"), 8192);
  assert.equal(b8.quant, "FP16");
  const b32 = bestQuant(m("qwen2.5-32b"), g("rtx4090"), 8192);
  assert.ok(["Q5_K_M", "Q4_K_M", "GPTQ4"].includes(b32.quant), b32.quant);
});

test("bestQuant: null when nothing fits", () => {
  assert.equal(bestQuant(m("qwen2.5-72b"), g("rtx4060"), 8192), null);
});

test("finetune: full 8B needs ~130+ GiB; QLoRA 8B fits 24GB", () => {
  const full = finetuneEstimate(m("llama-3.1-8b"), "full");
  assert.ok(full.total > 120, `got ${full.total}`);
  const qlora = finetuneEstimate(m("llama-3.1-8b"), "qlora");
  assert.ok(qlora.total < 24, `got ${qlora.total}`);
});

test("finetune: rejects unknown method; engine rejects unknown quant/kv", () => {
  assert.throws(() => finetuneEstimate(m("phi-4"), "dpo"));
  assert.throws(() => weightsGiB(7, "Q9"));
  assert.throws(() => kvCacheGiB(m("phi-4"), 1024, "int1"));
});

test("catalog sanity: all models have positive arch fields, all quants have bpw", () => {
  for (const model of MODELS) {
    for (const k of ["paramsB", "layers", "kvHeads", "headDim", "maxContext"]) {
      assert.ok(model[k] > 0, `${model.id}.${k}`);
    }
  }
  for (const [k, q] of Object.entries(QUANTS)) assert.ok(q.bpw > 0 && q.bpw <= 16, k);
});
