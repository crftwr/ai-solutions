// phase_tracker.h
//
// Tracks whether the current ggml graph is part of the prefill (prompt
// processing) or decode (single-token autoregressive generation) phase.
//
// Detection heuristic (Phase 3):
//   - The very first graph_compute call after a prompt is loaded is prefill.
//     Subsequent calls with a batch size of 1 are decode.
//   - "Batch size" is read from the KV-cache / sequence length embedded in the
//     graph by looking at the largest M dimension across MUL_MAT nodes: if
//     M > 1 → prefill, else → decode.
//
// Phase 0/1: always returns "prefill" as a safe placeholder.

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

    // Reset state (e.g. at the start of a new inference request).
    void reset();

    // Total prefill / decode graph count since last reset().
    uint64_t prefill_count() const { return prefill_count_; }
    uint64_t decode_count()  const { return decode_count_;  }

private:
    uint64_t prefill_count_ = 0;
    uint64_t decode_count_  = 0;
    bool     seen_prefill_  = false;
};
