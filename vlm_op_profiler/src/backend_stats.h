// backend_stats.h
//
// Public C API for the profiling backend.
//
// The backend is loaded into a llama.cpp process via DYLD_INSERT_LIBRARIES
// (macOS) or LD_PRELOAD (Linux).  A __attribute__((constructor)) function
// registers it with ggml before main() runs.
//
// Configuration is through environment variables so that the vlm_op_profiler
// CLI wrapper can configure the backend without linking against it:
//
//   PROFSTATS_OUT_DIR                Output directory  (default: ./profstats_out)
//   PROFSTATS_INNER_BACKEND          Inner backend name: "cpu" | "metal" | "cuda"
//                                    (default: "cpu"; auto-selects best if unset)
//   PROFSTATS_MAX_STEPS              Cap on decode graphs to record (default: 0 = all)
//   PROFSTATS_INCLUDE_VISION_ENCODE  "1" to also record vision-encoder graphs
//                                    (graphs containing CONV_2D); default skips them
//                                    so trace.jsonl reflects the LLM body only.
//
// The functions below can also be called programmatically from a host that
// links directly against libbackend_stats (used in unit tests).

#pragma once

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Configuration API (call before ggml backend registration runs; idempotent)
// ---------------------------------------------------------------------------

/// Set the output directory.  The path is not created here; the backend
/// creates it on first use.
void profstats_set_output_dir(const char * path);

/// Override the inner backend name (must match a registered ggml backend name,
/// e.g. "CPU", "Metal", "CUDA0").
void profstats_set_inner_backend_name(const char * name);

/// Cap the number of decode-phase graphs to record (0 = unlimited).
void profstats_set_max_steps(size_t n);

// ---------------------------------------------------------------------------
// Registration (called automatically from the shared-library constructor)
// ---------------------------------------------------------------------------

/// Register the profiling backend with ggml.  Safe to call multiple times;
/// subsequent calls are no-ops.
void profstats_register(void);

#ifdef __cplusplus
}
#endif
