// gpuroom — model architectures (from published HF configs) and GPU specs.
// paramsB = total params in billions; MoE models list ALL experts (must be resident).

export const MODELS = [
  { id: "llama-3.2-1b",   name: "Llama 3.2 1B",          paramsB: 1.24, layers: 16,  heads: 32, kvHeads: 8,  headDim: 64,  maxContext: 131072 },
  { id: "llama-3.2-3b",   name: "Llama 3.2 3B",          paramsB: 3.21, layers: 28,  heads: 24, kvHeads: 8,  headDim: 128, maxContext: 131072 },
  { id: "llama-3.1-8b",   name: "Llama 3.1 8B",          paramsB: 8.03, layers: 32,  heads: 32, kvHeads: 8,  headDim: 128, maxContext: 131072 },
  { id: "llama-3.3-70b",  name: "Llama 3.3 70B",         paramsB: 70.6, layers: 80,  heads: 64, kvHeads: 8,  headDim: 128, maxContext: 131072 },
  { id: "qwen2.5-7b",     name: "Qwen2.5 7B",            paramsB: 7.62, layers: 28,  heads: 28, kvHeads: 4,  headDim: 128, maxContext: 131072 },
  { id: "qwen2.5-14b",    name: "Qwen2.5 14B",           paramsB: 14.7, layers: 48,  heads: 40, kvHeads: 8,  headDim: 128, maxContext: 131072 },
  { id: "qwen2.5-32b",    name: "Qwen2.5 32B",           paramsB: 32.8, layers: 64,  heads: 40, kvHeads: 8,  headDim: 128, maxContext: 131072 },
  { id: "qwen2.5-72b",    name: "Qwen2.5 72B",           paramsB: 72.7, layers: 80,  heads: 64, kvHeads: 8,  headDim: 128, maxContext: 131072 },
  { id: "qwq-32b",        name: "QwQ 32B",               paramsB: 32.8, layers: 64,  heads: 40, kvHeads: 8,  headDim: 128, maxContext: 131072 },
  { id: "mistral-7b",     name: "Mistral 7B v0.3",       paramsB: 7.25, layers: 32,  heads: 32, kvHeads: 8,  headDim: 128, maxContext: 32768 },
  { id: "mixtral-8x7b",   name: "Mixtral 8x7B (MoE)",    paramsB: 46.7, layers: 32,  heads: 32, kvHeads: 8,  headDim: 128, maxContext: 32768, moe: true },
  { id: "gemma-2-9b",     name: "Gemma 2 9B",            paramsB: 9.24, layers: 42,  heads: 16, kvHeads: 8,  headDim: 256, maxContext: 8192 },
  { id: "gemma-2-27b",    name: "Gemma 2 27B",           paramsB: 27.2, layers: 46,  heads: 32, kvHeads: 16, headDim: 128, maxContext: 8192 },
  { id: "phi-4",          name: "Phi-4 14B",             paramsB: 14.7, layers: 40,  heads: 40, kvHeads: 10, headDim: 128, maxContext: 16384 },
  { id: "r1-distill-32b", name: "DeepSeek R1 Distill 32B", paramsB: 32.8, layers: 64, heads: 40, kvHeads: 8, headDim: 128, maxContext: 131072 },
  { id: "r1-distill-8b",  name: "DeepSeek R1 Distill 8B",  paramsB: 8.03, layers: 32, heads: 32, kvHeads: 8, headDim: 128, maxContext: 131072 },
];

export const GPUS = [
  { id: "rtx3060-12", name: "RTX 3060 12GB",      vramGiB: 12 },
  { id: "rtx3090",    name: "RTX 3090 24GB",      vramGiB: 24 },
  { id: "rtx4060",    name: "RTX 4060 8GB",       vramGiB: 8 },
  { id: "rtx4060ti16",name: "RTX 4060 Ti 16GB",   vramGiB: 16 },
  { id: "rtx4070",    name: "RTX 4070 12GB",      vramGiB: 12 },
  { id: "rtx4070tis", name: "RTX 4070 Ti S 16GB", vramGiB: 16 },
  { id: "rtx4080",    name: "RTX 4080 16GB",      vramGiB: 16 },
  { id: "rtx4090",    name: "RTX 4090 24GB",      vramGiB: 24 },
  { id: "rtx5080",    name: "RTX 5080 16GB",      vramGiB: 16 },
  { id: "rtx5090",    name: "RTX 5090 32GB",      vramGiB: 32 },
  { id: "a100-40",    name: "A100 40GB",          vramGiB: 40 },
  { id: "a100-80",    name: "A100 80GB",          vramGiB: 80 },
  { id: "h100",       name: "H100 80GB",          vramGiB: 80 },
  { id: "l4",         name: "L4 24GB",            vramGiB: 24 },
  { id: "t4",         name: "T4 16GB",            vramGiB: 16 },
  { id: "rx7900xtx",  name: "RX 7900 XTX 24GB",   vramGiB: 24 },
  // Apple unified memory: macOS caps GPU-allocatable share (~75% default).
  { id: "m4-16",      name: "Mac M4 16GB",        vramGiB: 16,  usableFrac: 0.75, unified: true },
  { id: "m4pro-24",   name: "Mac M4 Pro 24GB",    vramGiB: 24,  usableFrac: 0.75, unified: true },
  { id: "m4pro-48",   name: "Mac M4 Pro 48GB",    vramGiB: 48,  usableFrac: 0.75, unified: true },
  { id: "m4max-64",   name: "Mac M4 Max 64GB",    vramGiB: 64,  usableFrac: 0.75, unified: true },
  { id: "m3ultra-128",name: "Mac M3 Ultra 128GB", vramGiB: 128, usableFrac: 0.8,  unified: true },
];
