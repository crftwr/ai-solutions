// layer_classifier.cpp
//
// Heuristic tensor-name → layer-category mapping.
//
// Two-stage classification:
//
// Stage 1 — name-substring table (kNamePatterns, checked in order, first match
//   wins).  Covers both static weight-tensor names (set by the model loader,
//   e.g. "blk.0.attn_q.weight") and dynamic computation-graph names assigned
//   by llama.cpp at graph-build time (e.g. "Qcur-5", "cache_k_l3 (view)").
//
// Stage 2 — op-type fallback (classify_by_op), applied when the name does not
//   match any pattern.  Covers ops whose semantics are unambiguous regardless
//   of the tensor name (GLU → ffn_gate, ROPE → attn_qkv, etc.).
//
// If neither stage matches, the node is labelled "other".
//
// Naming conventions we rely on (from llama.cpp model loading and graph build):
//   LLaMA/Mistral backbone:   blk.<N>.<suffix>
//   llama.cpp dynamic names:  Qcur-<N>, Kcur-<N>, cache_k_l<N> (view), …
//   Vision encoder (LLaVA):   vision_model.encoder.layers.<N>.<suffix>
//   Projector:                mm_projector or mm.<suffix>
//   Qwen2-VL vision:          visual.blocks.<N>.<suffix>
//   Phi-3.5 vision:           vision_embed_tokens.<suffix>

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
static constexpr std::array<NamePattern, 48> kNamePatterns = {{
    // ---- LLM backbone (blk.<N>.<suffix>, standard GGUF naming) ----
    { "attn_q",        "attn_qkv"   },
    { "attn_k",        "attn_qkv"   },
    { "attn_v",        "attn_qkv"   },
    { "attn_qkv",      "attn_qkv"   },  // fused QKV (e.g. Falcon, Phi)

    // ---- LLM backbone: llama.cpp dynamic tensor names (e.g. SmolVLM, Qwen3) ----
    // These are the *output* tensors from the Q/K/V projections, not weight names.
    // Named by llama.cpp as "Qcur-<layer>", "Kcur-<layer>", "Vcur-<layer>".
    { "Qcur",          "attn_qkv"   },
    { "Kcur",          "attn_qkv"   },
    { "Vcur",          "attn_qkv"   },

    // ---- KV cache (SET_ROWS / VIEW / PERMUTE on the KV cache tensors) ----
    // Named "cache_k_l<N> (view)", "cache_v_l<N> (view)", etc.
    { "cache_k",       "attn_qkv"   },
    { "cache_v",       "attn_qkv"   },

    // ---- Attention mask (CPY to causal/ALiBi mask buffer) ----
    { "kq_mask",       "attn_qkv"   },

    // ---- Fused flash attention (FLASH_ATTN_EXT) ----
    // Named "__fattn__-<layer>" by llama.cpp's graph builder.
    { "__fattn__",     "attn_out"   },

    // ---- Attention output ----
    { "attn_output",   "attn_out"   },
    { "attn_out",      "attn_out"   },
    { "kqv_out",       "attn_out"   },  // post-flash-attn concatenated KQV

    // ---- Residual connections (ADD) ----
    // "ffn_inp-<N>"  = residual add at the FFN input boundary (= attn output + skip).
    // "l_out-<N>"    = residual add at the layer output boundary (= FFN output + skip).
    { "ffn_inp",       "attn_out"   },
    { "l_out",         "ffn_down"   },

    // ---- FFN (gated SwiGLU: gate, up, down) ----
    { "ffn_gate",      "ffn_gate"   },
    { "ffn_up",        "ffn_up"     },
    { "ffn_down",      "ffn_down"   },
    { "ffn_fc1",       "ffn_up"     },  // non-gated FFN first linear
    { "ffn_fc2",       "ffn_down"   },  // non-gated FFN second linear
    // ffn_out-<layer> is the down-projection output in SmolVLM/modern llama.cpp
    { "ffn_out",       "ffn_down"   },
    { "ffn_proj",      "ffn_down"   },  // e.g. MiniCPM
    // SwiGLU activation tensor (element-wise gate × up; no weights)
    { "ffn_swiglu",    "ffn_gate"   },

    // ---- Normalization ----
    { "attn_norm",     "norm"       },
    { "ffn_norm",      "norm"       },
    { "output_norm",   "norm"       },
    { "norm",          "norm"       },  // also matches "norm-<layer>"

    // ---- LM head ----
    { "result_output", "lm_head"    },  // SmolVLM / modern llama.cpp logit output
    { "output.weight", "lm_head"    },  // LLaMA weight name
    { "lm_head",       "lm_head"    },

    // ---- Vision encoder (LLaVA / CLIP style) ----
    { "vision_model",          "vision_attn" },  // covers .encoder.layers.N.* etc.
    { "visual.blocks",         "vision_attn" },  // Qwen2-VL
    { "vision_encoder",        "vision_attn" },
    { "vit.",                  "vision_attn" },  // generic ViT prefix

    // ---- Vision convolutions (patch embed) ----
    { "patch_embedding",  "vision_conv" },
    { "conv_proj",        "vision_conv" },
    { "patch_embd",       "vision_conv" },
    { "conv1",            "vision_conv" },

    // ---- VLM projector ----
    { "mm_projector",          "projector"  },
    { "mm.",                   "projector"  },
    { "vision_proj",           "projector"  },
    { "multi_modal_projector", "projector"  },
    { "image_newline",         "projector"  },

    // ---- Embeddings ----
    { "token_embd",    "embd"   },
    { "pos_embd",      "embd"   },
    { "embed_tokens",  "embd"   },
    { "embd",          "embd"   },  // bare "embd" (e.g. SmolVLM GET_ROWS output)
}};
// clang-format on

// ---------------------------------------------------------------------------
// classify_by_op — stage-2 fallback when no name pattern matches.
//
// Only used for ops whose semantic category is unambiguous regardless of the
// specific tensor name.  Returns "" if the op type is ambiguous.
// ---------------------------------------------------------------------------
static std::string_view classify_by_op(std::string_view op_name) {
    // Rotary position embedding — always applied to Q and K projections.
    if (op_name == "ROPE")          { return "attn_qkv"; }
    // Attention score normalisation (softmax over QK^T / flash-attn internals).
    if (op_name == "SOFT_MAX")      { return "attn_qkv"; }
    // Gated linear unit variants — all appear inside the FFN gate branch.
    if (op_name == "GLU")           { return "ffn_gate"; }
    // Embedding table lookup — GET_ROWS on unnamed nodes is always an embd op.
    if (op_name == "GET_ROWS")      { return "embd"; }
    // Normalisation ops without a matching weight name (e.g. RoPE-norm, QK-norm).
    if (op_name == "RMS_NORM")      { return "norm"; }
    if (op_name == "LAYER_NORM")    { return "norm"; }
    if (op_name == "GROUP_NORM")    { return "norm"; }
    return "";
}

// ---------------------------------------------------------------------------
// classify_layer
// ---------------------------------------------------------------------------
std::string classify_layer(std::string_view tensor_name, std::string_view op_name) {
    // Stage 1: name-substring table.
    for (const auto & p : kNamePatterns) {
        if (tensor_name.find(p.substr) != std::string_view::npos) {
            return std::string(p.category);
        }
    }
    // Stage 2: op-type fallback.
    const std::string_view by_op = classify_by_op(op_name);
    if (!by_op.empty()) {
        return std::string(by_op);
    }
    return "other";
}
