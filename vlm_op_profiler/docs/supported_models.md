# Supported models

Models known to work end-to-end with vlm-op-profiler.  Update this file
when a model is validated or when a known issue is resolved.

## Status legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Fully validated: trace.jsonl produced, layer categories ≥ 95% non-`other` |
| 🔧 | Runs but has known classification gaps — see notes |
| ❌ | Does not work — blocked on a known issue |
| 🔲 | Not yet tested |

---

## Validated models

| Model | Architecture | GGUF source | Status | Notes |
|-------|-------------|------------|--------|-------|
| Llama-3.2-1B-Instruct (text only) | Llama 3.2 | `bartowski/Llama-3.2-1B-Instruct-GGUF` | ✅ | Phase 3: 2012 records, 0 unclassified. All ops (MUL_MAT, FLASH_ATTN_EXT, RMS_NORM, ROPE, GLU, SET_ROWS, VIEW, PERMUTE, ADD, GET_ROWS, CPY) classified. Phase 7: passes all 10 structural + analytical MAC cross-checks on every CI run. |
| LLaVA-1.6 Mistral 7B | LLaVA / CLIP + Mistral | `cjpais/llava-1.6-mistral-7b-gguf` | 🔲 | |
| LLaVA-1.6 Vicuna 13B | LLaVA / CLIP + Vicuna | HuggingFace | 🔲 | |
| Qwen2-VL 7B Instruct | Qwen2-VL | `bartowski/Qwen2-VL-7B-Instruct-GGUF` | 🔲 | |
| Llama-3.2-11B-Vision | Llama 3.2 Vision | `lmstudio-community/Llama-3.2-11B-Vision-Instruct-GGUF` | 🔲 | Gated — needs `HF_TOKEN` in `.env`. |
| MiniCPM-V 2.6 | MiniCPM-V | `openbmb/MiniCPM-V-2_6-gguf` | 🔲 | |
| Pixtral 12B | Pixtral | `bartowski/pixtral-12b-GGUF` | 🔲 | |
| Phi-3.5-Vision Instruct | Phi-3.5-V | `bartowski/Phi-3.5-vision-instruct-GGUF` | 🔲 | |
| SmolVLM-Instruct | SmolVLM | `ggml-org/SmolVLM-Instruct-GGUF` | 🔧 | Phase 3: 751 prefill records captured before inference error; 0 unclassified. Phase 7: warmup trace passes all 10 structural + analytical MAC cross-checks (text body matches Llama-1.7B architecture). Inference itself fails on image-token injection — see notes. |
| Idefics3-8B | Idefics3 / SMOLLM | `bartowski/Idefics3-8B-Llama3-GGUF` | 🔲 | |

### Edge / physical-AI / robotics models

These ship in the `edge` suite of `scripts/fetch_models_hf.py`. They target
Jetson-class and Qualcomm AI SoC deployments and dominate the physical-AI
landscape, so they're carried separately from the general-purpose suite.

| Model | Architecture | GGUF source | Status | Notes |
|-------|-------------|------------|--------|-------|
| InternVL2-2B | InternVL2 / InternViT + InternLM2 | `bartowski/InternVL2-2B-GGUF` | 🔲 | Most widely used VLM in robotics deployments. |
| InternVL2-2B (Q8_0) | InternVL2 | `bartowski/InternVL2-2B-GGUF` | 🔲 | int8 weights — exercises tensor-core int8 paths. |
| Qwen2.5-VL-3B-Instruct | Qwen2.5-VL | `bartowski/Qwen2.5-VL-3B-Instruct-GGUF` | 🔲 | 3B variant; common edge fine-tune base. |
| moondream2 | moondream2 (~2B) | `vikhyatk/moondream2` | 🔲 | Embedded-vision focused; upstream ships f16 only (quantize locally for IQ3_M / Q4_K_M). |
| PaliGemma 2 (3B) | SigLIP + Gemma 2 | `bartowski/paligemma2-3b-pt-224-GGUF` | 🔲 | Google robotics-lineage VLM; 224 patch variant. |
| Florence-2 base | Florence-2 (~0.23B) | `bartowski/Florence-2-base-ft-GGUF` | 🔲 | No LLM body — grounding / detection / OCR-with-location; different MAC profile. mmproj is N/A. May not run under `llama-mtmd-cli`. |
| Gemma 3 4B it | Gemma 3 multimodal | `bartowski/gemma-3-4b-it-GGUF` | 🔲 | Compact Google multimodal; common edge fine-tune base. |
| SmolVLM-Instruct (Q8_0) | SmolVLM | `ggml-org/SmolVLM-Instruct-GGUF` | 🔲 | int8 SmolVLM — primary fixture for int8 dot-product profiling. |
| SmolVLM-Instruct (IQ3_M) | SmolVLM | `ggml-org/SmolVLM-Instruct-GGUF` | 🔲 | Sub-4 GB variant for ≤ 4 GB embedded targets. |
| Phi-3.5-vision (Q8_0) | Phi-3.5-V | `bartowski/Phi-3.5-vision-instruct-GGUF` | 🔲 | int8 Phi-3.5-V — different attention shape from SmolVLM. |

---

## Architecture-specific notes

### SmolVLM-Instruct (ggml-org/SmolVLM-Instruct-GGUF)

**Status: ❌ blocked — `invalid token[6] = -1` during prompt evaluation.**

The warmup pass (run with an empty prompt) completes successfully and produces
a valid `trace.jsonl` with correct phase labels and MUL_MAT classification.
However, the actual inference pass with `--image` fails during the first decode
batch with:

```
E init: invalid token[6] = -1
E decode: failed to initialize batch
E failed to eval chunk 0
```

Root cause: when `llama-mtmd-cli` injects image patch tokens into the prompt
batch, one token has ID `-1` (unknown/out-of-vocabulary). This appears to be a
mismatch between the image tokenizer expected by the current `llama.cpp` commit
and the format used by the `ggml-org` GGUF file.

**Workaround:** the warmup graphs do capture a representative prefill forward
pass (one full forward through all 32 transformer layers), providing enough
data for architecture analysis even without a successful decode step.

**Resolution:** test again after bumping the `llama.cpp` submodule — the
`ggml-org` team updates SmolVLM support in sync with new llama.cpp releases.

### LLaVA family
- Weight tensor names use `blk.<N>.attn_q.weight` etc. for the LLM body.
- Vision encoder tensors follow `vision_model.encoder.layers.<N>.<suffix>`.
- Projector tensors follow `mm_projector.<N>.weight`.

### Qwen2-VL / Qwen2.5-VL
- Vision encoder uses `visual.blocks.<N>.<suffix>` naming.
- Rotary embeddings for vision are separate tensors; classified as `vision_attn`.

### Llama 3.2 Vision
- Uses a cross-attention bridge; projector tensors named `cross_attn.*`.
- Attention variant differs from standard Llama; update `layer_classifier.cpp`
  if `other` categories appear in cross-attention layers.

### MiniCPM-V
- Resampler projector uses `resampler.*` naming; add pattern to `layer_classifier.cpp`.

### Phi-3.5-Vision
- Vision embedding uses `vision_embed_tokens.*`.

### InternVL2-2B
- Vision tower is InternViT-300M; tensors typically `vision_model.encoder.layers.<N>.*`.
- Projector is an MLP under `mlp1.*` / `mm_projector.*` depending on conversion script.
- Robotics-deployed: prioritise validating that classifier coverage is ≥ 95% before
  collecting MAC totals for hardware-design work.

### moondream2
- Single-stack model (no separate llama-arch decoder); vision tower is SigLIP-style.
- Upstream `vikhyatk/moondream2` ships f16 weights only — quantize locally with
  `llama-quantize` to evaluate IQ3_M / Q4_K_M MAC patterns.

### PaliGemma 2
- SigLIP vision encoder + Gemma 2 decoder. Vision tensors under `vision_tower.*`.
- 224 and 448 patch variants exist; we list the 224 variant by default. Switch
  to 448 if profiling higher-res inputs.

### Florence-2
- Sub-1B spatial-AI model with no autoregressive decoder. The whole model is the
  vision stack, so `--mmproj` is N/A and `llama-mtmd-cli` may not load it directly;
  treated here as exploratory until upstream support stabilises.

### Gemma 3 (vision)
- Uses Gemma 3's native vision input path. Vision tensors typically prefixed
  `vision_model.*`; classifier may need new patterns once a successful run lands.

### Quantization variants on edge

The `edge` suite carries multiple quantizations per architecture so the
profiler captures the dominant low-precision dot-product pattern on the target
hardware:

| Quant | Target hardware | Why |
|-------|------------------|-----|
| `Q4_K_M` | All edge SoCs (baseline) | Default profile — int4 weights, fp32 accumulation. |
| `Q8_0` | Jetson Orin tensor cores, Qualcomm AI 100 | int8 weight path; matches the int8 × int8 → int32 dot-product on these targets. |
| `IQ3_M` | ≤ 4 GB embedded (Jetson Nano-class) | Sub-4-bit weights for memory-constrained deployments. |

---

## Op-name pattern reference

Sourced from real Llama-3.2-1B and SmolVLM-Instruct traces (Phase 7
validation). When adding a new model, grep `trace.jsonl` for these prefixes
to confirm the classifier already covers them; the `other` percentage should
remain < 5% of total MACs.

| Pattern (substring in `tensor->name`) | Op (typical) | Layer category | Origin |
|---------------------------------------|--------------|----------------|--------|
| `blk.<N>.attn_q.weight`               | `MUL_MAT`    | `attn_qkv`     | Llama / SmolVLM body |
| `blk.<N>.attn_k.weight`               | `MUL_MAT`    | `attn_qkv`     | Llama / SmolVLM body |
| `blk.<N>.attn_v.weight`               | `MUL_MAT`    | `attn_qkv`     | Llama / SmolVLM body |
| `blk.<N>.attn_output.weight`          | `MUL_MAT`    | `attn_out`     | Llama / SmolVLM body |
| `blk.<N>.ffn_gate.weight`             | `MUL_MAT`    | `ffn_gate`     | SwiGLU (Llama, SmolVLM, …) |
| `blk.<N>.ffn_up.weight`               | `MUL_MAT`    | `ffn_up`       | SwiGLU |
| `blk.<N>.ffn_down.weight`             | `MUL_MAT`    | `ffn_down`     | SwiGLU |
| `blk.<N>.attn_norm.weight`            | `RMS_NORM`   | `norm`         | Pre-attention RMSNorm |
| `blk.<N>.ffn_norm.weight`             | `RMS_NORM`   | `norm`         | Pre-FFN RMSNorm |
| `cache_k`, `cache_v`                  | `SET_ROWS` / `VIEW` / `PERMUTE` | `attn_qkv` | KV-cache update ops |
| `kq_mask`                             | `CPY`        | `attn_qkv`     | Causal-mask copy |
| `__fattn__`                           | `FLASH_ATTN_EXT` | `attn_out` | Fused attention (per-graph anonymous tensor) |
| `ffn_inp`                             | `ADD`        | `attn_out`     | Residual at FFN input |
| `l_out`                               | `ADD`        | `ffn_down`     | Residual at layer output |
| `ffn_swiglu`                          | `GLU`        | `ffn_gate`     | SwiGLU activation |
| `output.weight`                       | `MUL_MAT`    | `lm_head`      | LM head projection (tied for Llama-3.2) |
| `output_norm.weight`                  | `RMS_NORM`   | `norm`         | Final norm before LM head |
| `embd`                                | `GET_ROWS`   | `embd`         | Token embedding lookup |
| `vision_model.encoder.layers.<N>.*`   | `MUL_MAT` / `LAYER_NORM` | `vision_*` | CLIP / LLaVA / SmolVLM vision tower |
| `mm_projector.<N>.weight`             | `MUL_MAT`    | `projector`    | Cross-modal MLP projector |
| `visual.blocks.<N>.*`                 | `MUL_MAT`    | `vision_*`     | Qwen2-VL vision tower |
| `cross_attn.*`                        | `MUL_MAT`    | `projector`    | Llama-3.2-Vision cross-attention bridge |
| `resampler.*`                         | `MUL_MAT`    | `projector`    | MiniCPM-V resampler |
| `vision_embed_tokens.*`               | `GET_ROWS` / `MUL_MAT` | `vision_*` | Phi-3.5-V vision embedding |

Op-only fallbacks (used when `name` is empty) — see `classify_by_op` in
`src/layer_classifier.cpp`:

| ggml op       | Layer category |
|---------------|----------------|
| `FLASH_ATTN_EXT` | `attn_out`  |
| `ROPE`           | `attn_qkv`  |
| `SOFT_MAX`       | `attn_qkv`  |
| `RMS_NORM`, `LAYER_NORM`, `GROUP_NORM` | `norm` |
| `GET_ROWS`       | `embd`      |
| `GLU`            | `ffn_gate`  |

## Preprocessing quirks

- **Empty-prompt warmup**: `llama-cli` / `llama-mtmd-cli` run an "empty"
  forward pass (`--no-warmup` to disable) before processing the real prompt.
  The interceptor records these graphs as one or more `prefill` rows. They
  are *not* spurious and *should* be counted — they exercise the same MAC
  formula as a real prefill — but the per-token totals are smaller. Phase 7
  validation accounts for this by checking `macs == 2·m·n·k` per row instead
  of relying on a global prompt-length estimate.
- **GQA / MQA shape conventions**: K/V projection weights have
  `ne = [d_model, d_model · n_kv_heads / n_heads]`. The MAC formula falls
  out of the recorded shapes — no special-casing needed — but means K/V
  contribute less than Q to `attn_qkv` MACs. Llama-3.2-1B (8 KV / 32 Q)
  has K/V MACs at exactly ¼ of Q.
- **`MUL_MAT` shape order**: ggml convention is src0 = weights, src1 =
  activations, both with the contraction dimension at `ne[0]`. M (sequence
  length) is `src1->ne[1]`, **not** `dst->ne[1]`. The graph walker assumes
  this and is consistent with every Llama-style architecture observed so
  far. Vision-encoder MUL_MATs follow the same convention.
- **Tied LM head**: Llama-3.2-1B and SmolVLM both reuse the embedding
  weight matrix for the LM head; the LM-head MUL_MAT still appears in the
  trace (its weight tensor is `output.weight`) but is allocation-aliased to
  `token_embd.weight`. Total MAC accounting is unaffected.

## Adding a new model

1. Run the profiler with `--steps 1` to get a short trace.
2. Check `trace.jsonl` for `"layer_category": "other"` entries.
3. Inspect the `name` field for those entries to discover the tensor-name prefix.
4. Add the new pattern to the table in `src/layer_classifier.cpp` (and to the
   reference table above).
5. Re-run and verify `other` percentage drops to < 5% of total MACs.
6. (Recommended) add an `ArchSpec` entry in `scripts/validate.py` and run
   `make validate VALIDATE_PATH=results/<model> VALIDATE_ARCH=<key>` —
   passing the analytical cross-check is a strong signal that the model
   has been profiled correctly.
7. Add the model to this file with status ✅.
