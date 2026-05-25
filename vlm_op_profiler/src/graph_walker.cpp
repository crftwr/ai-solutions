// graph_walker.cpp
//
// Implements walk_graph(): iterates every node in a ggml_cgraph and extracts
// a NodeStats record.  MAC counting covers:
//
//   MUL_MAT(A, B) -> C  [Phase 1]
//     src[0] = A (weights):      ne = [K, N, ...]
//     src[1] = B (activations):  ne = [K, M, ...]
//     dst    = C:                ne = [N, M, ...]
//     macs = 2 * M * N * K
//
//   MUL_MAT_ID(as, b, ids) -> C  [Phase 2 — MoE routed matmul]
//     src[0] = as (expert weight stack): [K, N, n_expert]
//     src[1] = b  (activations):         [K, n_expert_used, n_tokens]
//     src[2] = ids (routing indices):    [n_expert_used, n_tokens]  (i32)
//     macs = 2 * K * N * n_expert_used * n_tokens
//
//   FLASH_ATTN_EXT(q, k, v, mask)  [Phase 2 — fused QK·V attention]
//     src[0] = q: [D, n_q,  Sq,  B]
//     src[1] = k: [D, n_kv, Skv, B]
//     src[2] = v: [Dv,n_kv, Skv, B]
//     macs = 2 * n_q * B * Sq * Skv * (D + Dv)   (QK + AV matmuls)
//
//   CONV_2D(kernel, data)  [Phase 2 — 2-D convolution]
//     src[0] = kernel: [KW, KH, IC, OC]
//     src[1] = data:   [W,  H,  IC, N]
//     dst:             [OW, OH, OC, N]
//     macs = 2 * OW * OH * N * OC * KW * KH * IC
//
//   SSM_CONV(sx, c)  [Phase 2 — Mamba depthwise conv on rolling state]
//     src[0] = sx: [d_conv-1+n_t, d_inner, n_s]
//     src[1] = c:  [d_conv, d_inner]
//     dst:         [d_inner, n_t, n_s]
//     macs = 2 * d_inner * n_t * n_s * d_conv
//
//   SSM_SCAN(s, x, dt, A, B, C, ids)  [Phase 2 — Mamba selective state scan]
//     src[0] = s (state): [d_state, head_dim, n_head]
//     src[1] = x (input): [head_dim, n_head, n_seq_tokens, n_seqs]
//     Per (head, token): B·x outer product (d_state×head_dim) +
//                        C·state contraction (d_state×head_dim)
//     macs = 2 * d_state * head_dim * n_head * n_seq_tokens * n_seqs
//
//   RWKV_WKV6 / RWKV_WKV7  [Phase 2 — RWKV recurrent attention]
//     src[0] = k (WKV6) / r (WKV7): [S, H, n_tokens]
//     Per (head, token): k⊗v outer product (S²) + r·state contraction (S²)
//     macs = 2 * S² * H * n_tokens

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

        // ---------------------------------------------------------------------------
        // MAC counting
        // ---------------------------------------------------------------------------
        if (node->op == GGML_OP_MUL_MAT && src0 && src1) {
            // ne[0] is the innermost dimension; for weights ne[0]=K, ne[1]=N.
            const int64_t K = src0->ne[0];
            const int64_t N = src0->ne[1];
            const int64_t M = src1->ne[1];
            s.k    = K;
            s.n    = N;
            s.m    = M;
            s.macs = 2LL * M * N * K;

        } else if (node->op == GGML_OP_MUL_MAT_ID && src0 && src1) {
            // Routed (MoE) matmul.  ids tensor is at src[2].
            // as  (src0): [K, N, n_expert]
            // b   (src1): [K, n_expert_used, n_tokens]
            // ids (src2): [n_expert_used, n_tokens]  (i32)
            const struct ggml_tensor * ids = node->src[2];
            const int64_t K            = src0->ne[0];
            const int64_t N            = src0->ne[1];
            const int64_t n_exp_used   = ids  ? ids->ne[0]  : src1->ne[1];
            const int64_t n_tokens     = src1->ne[2];
            s.k    = K;
            s.n    = N;
            s.m    = n_exp_used * n_tokens;
            s.macs = 2LL * s.m * N * K;

        } else if (node->op == GGML_OP_FLASH_ATTN_EXT && src0 && src1) {
            // Fused flash attention (Q·K^T then A·V).
            // q (src0): [D,  n_q,  Sq,  B]
            // k (src1): [D,  n_kv, Skv, B]
            // v (src2): [Dv, n_kv, Skv, B]
            const struct ggml_tensor * v_tensor = node->src[2];
            const int64_t D   = src0->ne[0];   // query head dim
            const int64_t n_q = src0->ne[1];   // query heads
            const int64_t Sq  = src0->ne[2];   // query seq len
            const int64_t B   = src0->ne[3];   // batch
            const int64_t Skv = src1->ne[2];   // KV seq len
            const int64_t Dv  = v_tensor ? v_tensor->ne[0] : D;
            // QK matmul: 2*Sq*Skv*D per (head, batch)
            // AV matmul: 2*Sq*Dv*Skv per (head, batch)
            s.macs = 2LL * n_q * B * Sq * Skv * (D + Dv);
            // m/n/k left at 0: not a single rectangular matmul

        } else if (node->op == GGML_OP_CONV_2D && src0 && src1) {
            // 2-D convolution.
            // kernel (src0): [KW, KH, IC, OC]
            // data   (src1): [W,  H,  IC, N]
            // dst:           [OW, OH, OC, N]
            const int64_t KW = src0->ne[0];
            const int64_t KH = src0->ne[1];
            const int64_t IC = src0->ne[2];
            const int64_t OW = node->ne[0];
            const int64_t OH = node->ne[1];
            const int64_t OC = node->ne[2];
            const int64_t N  = node->ne[3];
            // Treat as (OW*OH*N) × OC output, inner dim KW*KH*IC.
            s.m    = OW * OH * N;
            s.n    = OC;
            s.k    = KW * KH * IC;
            s.macs = 2LL * s.m * s.n * s.k;

        } else if (node->op == GGML_OP_SSM_CONV && src0 && src1) {
            // Mamba depthwise rolling conv.
            // sx  (src0): [d_conv-1+n_t, d_inner, n_s]
            // c   (src1): [d_conv,       d_inner]
            // dst:        [d_inner, n_t, n_s]
            const int64_t d_inner = node->ne[0];
            const int64_t n_t     = node->ne[1];
            const int64_t n_s     = node->ne[2];
            const int64_t d_conv  = src1->ne[0];
            s.macs = 2LL * d_inner * n_t * n_s * d_conv;

        } else if (node->op == GGML_OP_SSM_SCAN && src0 && src1) {
            // Mamba selective state scan.
            // s (state, src0): [d_state, head_dim, n_head]
            // x (input, src1): [head_dim, n_head, n_seq_tokens, n_seqs]
            // Per (head, token): B·x outer product + C·state contraction
            //   each costs d_state × head_dim → 2 × d_state × head_dim per step.
            const int64_t d_state  = src0->ne[0];
            const int64_t head_dim = src1->ne[0];
            const int64_t n_head   = src1->ne[1];
            const int64_t n_t      = src1->ne[2];
            const int64_t n_seqs   = src1->ne[3];
            s.macs = 2LL * d_state * head_dim * n_head * n_t * n_seqs;

        } else if ((node->op == GGML_OP_RWKV_WKV6 ||
                    node->op == GGML_OP_RWKV_WKV7) && src0) {
            // RWKV linear attention (WKV kernel).
            // WKV6: src[0]=k, src[1]=v, src[2]=r, ...
            // WKV7: src[0]=r, src[1]=w, src[2]=k, src[3]=v, ...
            // k (WKV6) / r (WKV7) carry the shape we need: [S, H, n_tokens]
            //   S = head dim, H = num heads, n_tokens = tokens this call
            // Per (head, token): k⊗v outer product (S²) + r·state (S²)
            const int64_t S        = src0->ne[0];   // head dim
            const int64_t H        = src0->ne[1];   // num heads
            const int64_t n_tokens = src0->ne[2];
            s.macs = 2LL * S * S * H * n_tokens;
        }

        // Layer category (Phase 3)
        s.layer_category = classify_layer(s.name, s.op);

        records.push_back(std::move(s));
    }

    return records;
}
