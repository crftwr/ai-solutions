// layer_classifier.h
//
// Maps (tensor_name, op_name) -> layer category string.
//
// Layer categories match the values used in trace.jsonl / report.csv:
//
//   attn_qkv      Q, K, V projection weights (and biases) in attention layers
//   attn_out      attention output projection
//   ffn_gate      FFN gate projection (SwiGLU / GeGLU gate branch)
//   ffn_up        FFN up projection (or first linear in non-gated FFN)
//   ffn_down      FFN down projection
//   norm          layer norm / RMS norm weights and ops
//   lm_head       language-model head projection
//   embd          embedding lookup
//   vision_conv   convolution ops in the vision encoder
//   vision_attn   attention ops in the vision encoder
//   vision_mlp    MLP ops in the vision encoder
//   projector     VLM cross-modal projector (e.g. MLP or cross-attn bridge)
//   other         anything that doesn't match a known pattern

#pragma once

#include <string>
#include <string_view>

// ---------------------------------------------------------------------------
// classify_layer
//
// tensor_name: value of ggml_tensor::name for the node (may be empty).
// op_name:     ggml_op_name() string (e.g. "MUL_MAT", "ADD", …).
//
// Returns one of the category strings listed above.
// ---------------------------------------------------------------------------
std::string classify_layer(std::string_view tensor_name, std::string_view op_name);
