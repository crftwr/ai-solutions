// graph_walker.h
//
// Per-node statistics extracted from a ggml_cgraph.  Every executed ggml node
// gets one NodeStats record.
//
// MAC accounting (Phase 2) covers:
//   MUL_MAT, MUL_MAT_ID  → M/N/K populated, macs = 2*M*N*K
//   FLASH_ATTN_EXT        → macs = QK + AV cost; M/N/K = 0 (fused op)
//   CONV_2D               → M/N/K = output positions/channels/kernel volume
//   SSM_CONV, SSM_SCAN    → macs from Mamba depthwise conv / selective scan
//   RWKV_WKV6, RWKV_WKV7 → macs from RWKV linear attention kernel
//   All other ops         → macs = 0 (no significant multiply-accumulate)
//
// This header is used by both backend_stats.cpp (producer) and aggregate.py's
// companion C++ test (consumer via FFI).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Maximum number of dimensions we track (matches GGML_MAX_DIMS == 4).
static constexpr int PROFSTATS_MAX_DIMS = 4;

// ---------------------------------------------------------------------------
// NodeStats — one record per executed ggml node
// ---------------------------------------------------------------------------
struct NodeStats {
    // Graph position
    uint64_t graph_id  = 0;  ///< monotonically increasing graph counter
    uint32_t node_idx  = 0;  ///< index within this graph's node array

    // Op identity
    std::string op;           ///< ggml_op_name() for this node's op
    std::string name;         ///< tensor->name (may be empty)

    // Data types
    std::string src0_type;    ///< ggml_type_name(src[0]->type), or "" if no src[0]
    std::string src1_type;    ///< ggml_type_name(src[1]->type), or "" if no src[1]
    std::string dst_type;     ///< ggml_type_name(dst->type)

    // Shapes (ne[0..3] in ggml convention: ne[0] = innermost)
    int64_t src0_ne[PROFSTATS_MAX_DIMS] = {};
    int64_t src1_ne[PROFSTATS_MAX_DIMS] = {};
    int64_t dst_ne[PROFSTATS_MAX_DIMS]  = {};

    // MatMul dimensions (valid only for MUL_MAT / MUL_MAT_ID)
    int64_t m   = 0;
    int64_t n   = 0;
    int64_t k   = 0;
    int64_t macs = 0;  ///< 2*M*N*K for MatMul/MoE; attention/conv/SSM/RWKV formulas in graph_walker.cpp

    // Classification (populated by layer_classifier.h and phase_tracker.h)
    std::string layer_category;  ///< e.g. "attn_qkv", "ffn_down", "vision_mlp"
    std::string phase;           ///< "prefill" | "decode" | ""
};

// Forward declaration — defined in ggml.h, included by graph_walker.cpp.
struct ggml_tensor;
struct ggml_cgraph;

// ---------------------------------------------------------------------------
// walk_graph — extract NodeStats for every node in cgraph.
//
// graph_id is caller-supplied (incremented once per graph_compute call).
// Returned vector has one entry per cgraph->nodes[i] that has op != NONE.
// ---------------------------------------------------------------------------
std::vector<NodeStats> walk_graph(const struct ggml_cgraph * cgraph, uint64_t graph_id);
