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
| LLaVA-1.6 Mistral 7B | LLaVA / CLIP + Mistral | HuggingFace `cmp-nct/gguf-llava-v1.6-mistral-7b` | 🔲 | |
| LLaVA-1.6 Vicuna 13B | LLaVA / CLIP + Vicuna | HuggingFace | 🔲 | |
| Qwen2-VL 7B Instruct | Qwen2-VL | HuggingFace `Qwen/Qwen2-VL-7B-Instruct-GGUF` | 🔲 | |
| Qwen2.5-VL 7B Instruct | Qwen2.5-VL | HuggingFace | 🔲 | |
| Llama-3.2-11B-Vision | Llama 3.2 Vision | HuggingFace | 🔲 | |
| MiniCPM-V 2.6 | MiniCPM-V | HuggingFace | 🔲 | |
| Pixtral 12B | Pixtral | HuggingFace | 🔲 | |
| Phi-3.5-Vision Instruct | Phi-3.5-V | HuggingFace | 🔲 | |
| SmolVLM | SmolVLM | HuggingFace | 🔲 | |
| Idefics3-8B | Idefics3 / SMOLLM | HuggingFace | 🔲 | |

---

## Architecture-specific notes

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
