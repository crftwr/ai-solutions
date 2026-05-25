// phase_tracker.cpp
//
// Implements the heuristics described in phase_tracker.h.

#include "phase_tracker.h"

std::string PhaseTracker::classify(int64_t max_m) {
    // Phase 3 heuristic: M > 1 → prefill, M == 1 → decode.
    // Treat anything without a prior prefill as prefill, and subsequent
    // single-token graphs as decode.
    if (!seen_prefill_ || max_m > 1) {
        seen_prefill_ = true;
        ++prefill_count_;
        return "prefill";
    }
    ++decode_count_;
    return "decode";
}

std::string PhaseTracker::classify_vision_encode() {
    ++vision_encode_count_;
    return "vision_encode";
}

void PhaseTracker::reset() {
    prefill_count_       = 0;
    decode_count_        = 0;
    vision_encode_count_ = 0;
    seen_prefill_        = false;
}
