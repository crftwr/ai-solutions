// layer_classifier.cpp
//
// Heuristic tensor-name → layer-category mapping.
//
// The mapping table covers the naming conventions used by the major VLM
// architectures supported by llama.cpp.  Architecture-specific patterns
// are grouped by comment block; add new architectures within those blocks.
//
// Naming conventions we rely on (from llama.cpp model loading):
//   LLaMA/Mistral backbone:   blk.<N>.<suffix>
//   Vision encoder (LLaVA):   vision_model.encoder.layers.<N>.<suffix>
//   Projector:                mm_projector or mm.<suffix>
//   Qwen2-VL vision:          visual.blocks.<N>.<suffix>
//   Phi-3.5 vision:           vision_embed_tokens.<suffix>
//
// Phase 3 fills this file with a complete table.  For now the stubs return
// "other" so that Phase 0/1 produce valid (if unclassified) trace.jsonl.

#include "layer_classifier.h"

#include <array>

// ---------------------------------------------------------------------------
// Pattern table
//
// Each entry is { substr_in_name, category }.  The table is checked in order;
// the first match wins.  All comparisons are case-sensitive substring searches.
// ---------------------------------------------------------------------------
struct NamePattern {
    std::string_view substr;
    std::string_view category;
};

// clang-format off
static constexpr std::array<NamePattern, 32> kNamePatterns = {{
    // ---- LLM backbone (blk.<N>.<suffix>) ----
    { "attn_q",        "attn_qkv"   },
    { "attn_k",        "attn_qkv"   },
    { "attn_v",        "attn_qkv"   },
    { "attn_qkv",      "attn_qkv"   },  // fused QKV (e.g. Falcon, Phi)
    { "attn_output",   "attn_out"   },
    { "attn_out",      "attn_out"   },
    { "ffn_gate",      "ffn_gate"   },
    { "ffn_up",        "ffn_up"     },
    { "ffn_down",      "ffn_down"   },
    { "ffn_fc1",       "ffn_up"     },  // non-gated FFN first linear
    { "ffn_fc2",       "ffn_down"   },  // non-gated FFN second linear
    { "attn_norm",     "norm"       },
    { "ffn_norm",      "norm"       },
    { "output_norm",   "norm"       },
    { "norm",          "norm"       },
    { "output.weight", "lm_head"    },  // LLaMA lm_head
    { "lm_head",       "lm_head"    },

    // ---- Vision encoder (LLaVA / CLIP) ----
    { "vision_model.encoder.layers", "vision_attn" },  // TODO: sub-classify attn vs mlp in Phase 3
    { "visual.blocks",               "vision_attn" },  // Qwen2-VL

    // ---- Vision convolutions ----
    { "patch_embedding",  "vision_conv" },
    { "conv_proj",        "vision_conv" },
    { "conv1",            "vision_conv" },

    // ---- VLM projector ----
    { "mm_projector",     "projector" },
    { "mm.",              "projector" },
    { "vision_proj",      "projector" },
    { "multi_modal_projector", "projector" },
    { "language_model",   "other"     },  // skip — will be caught by blk.* patterns first

    // ---- Embeddings ----
    { "token_embd",       "embd"   },
    { "pos_embd",         "embd"   },
    { "embed_tokens",     "embd"   },
}};
// clang-format on

// ---------------------------------------------------------------------------
// classify_layer
// ---------------------------------------------------------------------------
std::string classify_layer(std::string_view tensor_name, std::string_view /*op_name*/) {
    for (const auto & p : kNamePatterns) {
        if (tensor_name.find(p.substr) != std::string_view::npos) {
            return std::string(p.category);
        }
    }
    // Phase 3: op-based fallback (e.g. SOFT_MAX → attn, RMS_NORM → norm)
    return "other";
}
