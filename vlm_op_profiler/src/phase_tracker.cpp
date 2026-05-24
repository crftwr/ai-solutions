// phase_tracker.cpp
//
// Phase 0/1 stub — always classifies as "prefill".
// Phase 3 implements the heuristic described in phase_tracker.h.

#include "phase_tracker.h"

std::string PhaseTracker::classify(int64_t max_m) {
    // Phase 3 heuristic: M > 1 → prefill, M == 1 → decode.
    // For now treat anything without a prior prefill as prefill, and
    // subsequent single-token graphs as decode.
    if (!seen_prefill_ || max_m > 1) {
        seen_prefill_ = true;
        ++prefill_count_;
        return "prefill";
    }
    ++decode_count_;
    return "decode";
}

void PhaseTracker::reset() {
    prefill_count_ = 0;
    decode_count_  = 0;
    seen_prefill_  = false;
}
