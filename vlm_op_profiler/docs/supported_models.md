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
| Llama-3.2-1B-Instruct (text only) | Llama 3.2 | `bartowski/Llama-3.2-1B-Instruct-GGUF` | ✅ | Phase 3: 2012 records, 0 unclassified. All ops (MUL_MAT, FLASH_ATTN_EXT, RMS_NORM, ROPE, GLU, SET_ROWS, VIEW, PERMUTE, ADD, GET_ROWS, CPY) classified. |
| LLaVA-1.6 Mistral 7B | LLaVA / CLIP + Mistral | `cjpais/llava-1.6-mistral-7b-gguf` | 🔲 | |
| LLaVA-1.6 Vicuna 13B | LLaVA / CLIP + Vicuna | HuggingFace | 🔲 | |
| Qwen2-VL 7B Instruct | Qwen2-VL | `Qwen/Qwen2-VL-7B-Instruct-GGUF` | 🔲 | |
| Qwen2.5-VL 7B Instruct | Qwen2.5-VL | HuggingFace | 🔲 | |
| Llama-3.2-11B-Vision | Llama 3.2 Vision | HuggingFace | 🔲 | |
| MiniCPM-V 2.6 | MiniCPM-V | HuggingFace | 🔲 | |
| Pixtral 12B | Pixtral | HuggingFace | 🔲 | |
| Phi-3.5-Vision Instruct | Phi-3.5-V | HuggingFace | 🔲 | |
| SmolVLM-Instruct | SmolVLM | `ggml-org/SmolVLM-Instruct-GGUF` | 🔧 | Phase 3: 751 prefill records captured before inference error; 0 unclassified. Inference fails on image token injection (see notes). |
| Idefics3-8B | Idefics3 / SMOLLM | HuggingFace | 🔲 | |

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

---

## Adding a new model

1. Run the profiler with `--steps 1` to get a short trace.
2. Check `trace.jsonl` for `"layer_category": "other"` entries.
3. Inspect the `name` field for those entries to discover the tensor-name prefix.
4. Add the new pattern to the table in `src/layer_classifier.cpp`.
5. Re-run and verify `other` percentage drops to < 5% of total MACs.
6. Add the model to this file with status ✅.
