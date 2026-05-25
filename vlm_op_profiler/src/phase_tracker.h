// phase_tracker.h
//
// Tracks whether the current ggml graph is part of:
//   - the vision-encoder forward pass (VLM image encoder),
//   - the prefill (prompt processing) pass, or
//   - the decode (single-token autoregressive generation) pass.
//
// Detection heuristic (Phase 3/4):
//   - If the graph contains any GGML_OP_CONV_2D node we treat it as a
//     vision-encoder graph (Phase 4). LLM bodies in llama.cpp do not use
//     CONV_2D — it is exclusive to image patch embedding.
//   - Otherwise, the largest M dimension across MUL_MAT nodes decides:
//     M > 1 → prefill, M == 1 → decode.

#pragma once

#include <cstdint>
#include <string>

// ---------------------------------------------------------------------------
// PhaseTracker
// ---------------------------------------------------------------------------
class PhaseTracker {
public:
    PhaseTracker()  = default;
    ~PhaseTracker() = default;

    // Call once per graph_compute invocation, before NodeStats are finalised.
    // Returns "prefill" or "decode".
    //
    // max_m: the maximum M dimension observed across all MUL_MAT nodes in the
    //         graph (0 if the graph contains no MUL_MAT nodes).
    std::string classify(int64_t max_m);

    // Tag this graph as the vision-encoder forward pass. Increments the
    // vision-encode counter and returns "vision_encode".
    std::string classify_vision_encode();

    // Reset state (e.g. at the start of a new inference request).
    void reset();

    // Total counts since last reset().
    uint64_t prefill_count()       const { return prefill_count_;       }
    uint64_t decode_count()        const { return decode_count_;        }
    uint64_t vision_encode_count() const { return vision_encode_count_; }

private:
    uint64_t prefill_count_       = 0;
    uint64_t decode_count_        = 0;
    uint64_t vision_encode_count_ = 0;
    bool     seen_prefill_        = false;
};
