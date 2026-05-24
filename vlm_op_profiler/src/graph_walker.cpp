// graph_walker.cpp
//
// Implements walk_graph(): iterates every node in a ggml_cgraph and extracts
// a NodeStats record.  MAC counting for MUL_MAT is implemented in Phase 2;
// this file contains correct dtype/shape extraction as of Phase 1.
//
// ggml convention for MUL_MAT(A, B) -> C:
//   src[0] = A  (weights), shape (K, N, ...)  -- ne[0]=K, ne[1]=N
//   src[1] = B  (activations), shape (K, M, ...) -- ne[0]=K, ne[1]=M
//   dst    = C, shape (N, M, ...)
//
// Therefore: K = src[0]->ne[0]
//            N = src[0]->ne[1]
//            M = src[1]->ne[1]   (= number of tokens in the batch)

#include "graph_walker.h"
#include "layer_classifier.h"

#include <vector>
#include <cstring>

// ggml headers (from llama.cpp submodule)
#include "ggml.h"

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static void copy_ne(int64_t dst[PROFSTATS_MAX_DIMS], const struct ggml_tensor * t) {
    if (!t) { return; }
    for (int i = 0; i < PROFSTATS_MAX_DIMS; ++i) {
        dst[i] = t->ne[i];
    }
}

static std::string safe_type_name(const struct ggml_tensor * t) {
    if (!t) { return ""; }
    const char * n = ggml_type_name(t->type);
    return n ? n : "UNKNOWN";
}

// ---------------------------------------------------------------------------
// walk_graph
// ---------------------------------------------------------------------------
std::vector<NodeStats> walk_graph(const struct ggml_cgraph * cgraph, uint64_t graph_id) {
    std::vector<NodeStats> records;
    if (!cgraph) { return records; }

    const int n_nodes = ggml_graph_n_nodes(const_cast<struct ggml_cgraph *>(cgraph));
    records.reserve(static_cast<size_t>(n_nodes));

    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor * node = ggml_graph_node(
            const_cast<struct ggml_cgraph *>(cgraph), i);
        if (!node || node->op == GGML_OP_NONE) { continue; }

        NodeStats s;
        s.graph_id  = graph_id;
        s.node_idx  = static_cast<uint32_t>(i);

        // Op name
        const char * op_name = ggml_op_name(node->op);
        s.op = op_name ? op_name : "UNKNOWN";

        // Tensor name
        if (node->name[0] != '\0') {
            s.name = std::string(node->name);
        }

        // Destination dtype + shape
        s.dst_type = safe_type_name(node);
        copy_ne(s.dst_ne, node);

        // Source dtypes + shapes
        struct ggml_tensor * src0 = node->src[0];
        struct ggml_tensor * src1 = node->src[1];

        s.src0_type = safe_type_name(src0);
        s.src1_type = safe_type_name(src1);
        copy_ne(s.src0_ne, src0);
        copy_ne(s.src1_ne, src1);

        // MAC count (Phase 2: extended to more ops)
        if (node->op == GGML_OP_MUL_MAT && src0 && src1) {
            // ne[0] is the innermost dimension; for weights ne[0]=K, ne[1]=N.
            const int64_t K = src0->ne[0];
            const int64_t N = src0->ne[1];
            const int64_t M = src1->ne[1];
            s.k    = K;
            s.n    = N;
            s.m    = M;
            s.macs = 2LL * M * N * K;
        }
        // TODO Phase 2: GGML_OP_MUL_MAT_ID, GGML_OP_CONV_*, GGML_OP_RWKV_WKV

        // Layer category (Phase 3)
        s.layer_category = classify_layer(s.name, s.op);

        records.push_back(std::move(s));
    }

    return records;
}
