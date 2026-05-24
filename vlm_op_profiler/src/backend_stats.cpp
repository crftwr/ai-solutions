// backend_stats.cpp
//
// Profiling ggml backend wrapper.
//
// Architecture
// ============
// This file implements a ggml backend (ggml_backend_i / ggml_backend_device_i /
// ggml_backend_reg_i) that wraps any other registered backend ("inner backend").
// The wrapper intercepts graph_compute(), walks every node in the cgraph to
// record statistics (via graph_walker.h), then forwards the graph to the inner
// backend so that computation proceeds normally.
//
// Load mechanism
// ==============
// The shared library is injected via DYLD_INSERT_LIBRARIES (macOS) or
// LD_PRELOAD (Linux).  A __attribute__((constructor)) calls profstats_register()
// which calls ggml_backend_register() so the profiling backend appears in ggml's
// backend list before any inference code runs.
//
// The vlm_op_profiler CLI (Phase 4) subsequently tells the llama.cpp scheduler
// to route graphs through the profiling backend by passing its backend handle
// as the first (highest-priority) backend to ggml_backend_sched_new().
//
// Phase 0: the backend compiles and registers; graph_compute simply forwards
// to the inner backend.  Statistics collection begins in Phase 1.
//
// NOTE: no exceptions are thrown from the ggml backend hot-path (matches ggml
// convention).  Use return-value error codes or early returns instead.

#define GGML_BACKEND_SHARED
#define GGML_BACKEND_BUILD

#include "backend_stats.h"
#include "graph_walker.h"
#include "phase_tracker.h"

// ggml public headers
#include "ggml.h"
#include "ggml-backend.h"

// ggml internal header — needed to define ggml_backend_i, ggml_backend, etc.
// This header is in ggml/src/ (not the public include/) and is intentionally
// not part of ggml's installed interface; we access it via the submodule tree.
#include "ggml-backend-impl.h"

#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Global configuration (written at init time, read-only during inference)
// ---------------------------------------------------------------------------
static std::string  g_output_dir         = "profstats_out";
static std::string  g_inner_backend_name = "";  // empty = auto-select
static std::size_t  g_max_steps          = 0;   // 0 = unlimited
static std::once_flag g_register_once;

// ---------------------------------------------------------------------------
// Stats state
// ---------------------------------------------------------------------------
struct StatsState {
    uint64_t         graph_counter = 0;
    PhaseTracker     phase_tracker;
    std::vector<NodeStats> pending;   // records not yet flushed
    std::mutex       mu;
};

// One global stats state; shared across all uses of the profiling backend.
static StatsState g_stats;

// ---------------------------------------------------------------------------
// JSONL serialisation helpers (Phase 1: full output; Phase 0: stubs)
// ---------------------------------------------------------------------------
static void write_int64_array(std::ostream & out, const int64_t ne[PROFSTATS_MAX_DIMS]) {
    out << '[';
    for (int i = 0; i < PROFSTATS_MAX_DIMS; ++i) {
        if (i) { out << ','; }
        out << ne[i];
    }
    out << ']';
}

// Escape a JSON string (minimal: escapes \ and ").
static void write_json_string(std::ostream & out, const std::string & s) {
    out << '"';
    for (char c : s) {
        if (c == '"' || c == '\\') { out << '\\'; }
        out << c;
    }
    out << '"';
}

static void flush_records(const std::string & out_dir,
                          std::vector<NodeStats> & records) {
    if (records.empty()) { return; }

    // Ensure output directory exists.
    // TODO Phase 1: use std::filesystem::create_directories (requires C++17).
    // For now, use system("mkdir -p …") as a placeholder.
    {
        std::string cmd = "mkdir -p " + out_dir;
        (void)std::system(cmd.c_str());
    }

    const std::string path = out_dir + "/trace.jsonl";
    std::ofstream f(path, std::ios::app);
    if (!f.is_open()) {
        fprintf(stderr, "[profstats] WARNING: cannot open %s for writing\n",
                path.c_str());
        records.clear();
        return;
    }

    for (const auto & s : records) {
        // Build one JSON object per record on a single line.
        f << "{";
        f << "\"step\":"       << s.graph_id  << ',';
        f << "\"graph_id\":"   << s.graph_id  << ',';
        f << "\"node_idx\":"   << s.node_idx  << ',';
        f << "\"phase\":";       write_json_string(f, s.phase);          f << ',';
        f << "\"op\":";          write_json_string(f, s.op);             f << ',';
        f << "\"name\":";        write_json_string(f, s.name);           f << ',';
        f << "\"layer_category\":"; write_json_string(f, s.layer_category); f << ',';
        f << "\"src0\":{\"type\":"; write_json_string(f, s.src0_type);
        f << ",\"ne\":"; write_int64_array(f, s.src0_ne); f << "},";
        f << "\"src1\":{\"type\":"; write_json_string(f, s.src1_type);
        f << ",\"ne\":"; write_int64_array(f, s.src1_ne); f << "},";
        f << "\"dst\":{\"type\":" ; write_json_string(f, s.dst_type);
        f << ",\"ne\":"; write_int64_array(f, s.dst_ne);  f << "},";
        f << "\"m\":"    << s.m    << ',';
        f << "\"n\":"    << s.n    << ',';
        f << "\"k\":"    << s.k    << ',';
        f << "\"macs\":" << s.macs;
        f << "}\n";
    }

    records.clear();
}

// ---------------------------------------------------------------------------
// ggml_backend_i implementation — the profiling backend "stream"
// ---------------------------------------------------------------------------

static const char * stats_backend_get_name(ggml_backend_t /*backend*/) {
    return "profstats";
}

static void stats_backend_free(ggml_backend_t backend) {
    // The backend is a global singleton; nothing to free.
    (void)backend;
}

static enum ggml_status stats_backend_graph_compute(ggml_backend_t backend,
                                                     struct ggml_cgraph * cgraph) {
    // Retrieve the inner backend from our context.
    ggml_backend_t inner = static_cast<ggml_backend_t>(backend->context);

    // --- Phase 1: statistics collection ---
    const uint64_t gid = ++g_stats.graph_counter;
    std::vector<NodeStats> records = walk_graph(cgraph, gid);

    // Determine phase from maximum M dimension.
    int64_t max_m = 0;
    for (const auto & r : records) { if (r.m > max_m) { max_m = r.m; } }
    const std::string phase = g_stats.phase_tracker.classify(max_m);
    for (auto & r : records) { r.phase = phase; }

    // Flush to trace.jsonl.
    {
        std::lock_guard<std::mutex> lk(g_stats.mu);
        g_stats.pending.insert(g_stats.pending.end(),
                                records.begin(), records.end());
        flush_records(g_output_dir, g_stats.pending);
    }

    // Check step cap.
    if (g_max_steps > 0 && g_stats.phase_tracker.decode_count() >= g_max_steps) {
        fprintf(stderr, "[profstats] reached max_steps=%zu; stopping.\n",
                g_max_steps);
        // Returning GGML_STATUS_ABORTED signals the caller to stop.
        return GGML_STATUS_ABORTED;
    }

    // Delegate to the inner backend.
    if (!inner) {
        fprintf(stderr, "[profstats] WARNING: no inner backend; graph not computed.\n");
        return GGML_STATUS_SUCCESS;
    }
    return ggml_backend_graph_compute(inner, cgraph);
}

static struct ggml_backend_i kStatsBackendIface = {
    /* get_name          */ stats_backend_get_name,
    /* free              */ stats_backend_free,
    /* set_tensor_async  */ nullptr,
    /* get_tensor_async  */ nullptr,
    /* set_tensor_2d_async */ nullptr,
    /* get_tensor_2d_async */ nullptr,
    /* cpy_tensor_async  */ nullptr,
    /* synchronize       */ nullptr,
    /* graph_plan_create */ nullptr,
    /* graph_plan_free   */ nullptr,
    /* graph_plan_update */ nullptr,
    /* graph_plan_compute*/ nullptr,
    /* graph_compute     */ stats_backend_graph_compute,
    /* event_record      */ nullptr,
    /* event_wait        */ nullptr,
    /* graph_optimize    */ nullptr,
};

// ---------------------------------------------------------------------------
// ggml_backend_device_i — single "profstats" device
// ---------------------------------------------------------------------------

static const char * stats_dev_get_name(ggml_backend_dev_t /*dev*/) {
    return "profstats";
}

static const char * stats_dev_get_description(ggml_backend_dev_t /*dev*/) {
    return "vlm-op-profiler statistics wrapper";
}

static void stats_dev_get_memory(ggml_backend_dev_t /*dev*/,
                                  size_t * free, size_t * total) {
    if (free)  { *free  = 0; }
    if (total) { *total = 0; }
}

static enum ggml_backend_dev_type stats_dev_get_type(ggml_backend_dev_t /*dev*/) {
    // Treat as a CPU-class accelerator so the scheduler considers it.
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void stats_dev_get_props(ggml_backend_dev_t dev,
                                 struct ggml_backend_dev_props * props) {
    props->name        = stats_dev_get_name(dev);
    props->description = stats_dev_get_description(dev);
    props->memory_free = 0;
    props->memory_total= 0;
    props->type        = stats_dev_get_type(dev);
    props->device_id   = nullptr;
    props->caps        = { false, false, false, false };
}

static ggml_backend_t stats_dev_init_backend(ggml_backend_dev_t dev,
                                              const char * /*params*/) {
    // Look up the inner backend by name (or best available).
    ggml_backend_t inner = nullptr;
    if (!g_inner_backend_name.empty()) {
        inner = ggml_backend_init_by_name(g_inner_backend_name.c_str(), nullptr);
        if (!inner) {
            fprintf(stderr, "[profstats] WARNING: inner backend '%s' not found; falling back to best.\n",
                    g_inner_backend_name.c_str());
        }
    }
    if (!inner) {
        inner = ggml_backend_init_best();
    }

    // Allocate the backend struct.
    struct ggml_backend * b = new ggml_backend{};
    // GUID for the profiling backend — arbitrary but unique bytes.
    // "ps" → 0x70, 0x73 (ASCII 'p','s' for "profstats")
    static ggml_guid kStatsGuid = {
        0x70, 0x73, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01
    };
    b->guid    = &kStatsGuid;
    b->iface   = kStatsBackendIface;
    b->device  = dev;
    b->context = inner;   // inner backend lives in context
    return b;
}

static ggml_backend_buffer_type_t stats_dev_get_buffer_type(ggml_backend_dev_t /*dev*/) {
    // Delegate buffer allocation to the CPU backend so weights are accessible.
    return ggml_backend_cpu_buffer_type();
}

static bool stats_dev_supports_op(ggml_backend_dev_t /*dev*/,
                                   const struct ggml_tensor * /*op*/) {
    // The profiling backend can handle any op (it delegates to the inner backend).
    return true;
}

static bool stats_dev_supports_buft(ggml_backend_dev_t /*dev*/,
                                     ggml_backend_buffer_type_t /*buft*/) {
    return true;
}

static struct ggml_backend_device_i kStatsDeviceIface = {
    /* get_name          */ stats_dev_get_name,
    /* get_description   */ stats_dev_get_description,
    /* get_memory        */ stats_dev_get_memory,
    /* get_type          */ stats_dev_get_type,
    /* get_props         */ stats_dev_get_props,
    /* init_backend      */ stats_dev_init_backend,
    /* get_buffer_type   */ stats_dev_get_buffer_type,
    /* get_host_buffer_type */ nullptr,
    /* buffer_from_host_ptr */ nullptr,
    /* supports_op       */ stats_dev_supports_op,
    /* supports_buft     */ stats_dev_supports_buft,
    /* offload_op        */ nullptr,
    /* event_new         */ nullptr,
    /* event_free        */ nullptr,
    /* event_synchronize */ nullptr,
};

static struct ggml_backend_device kStatsDevice = {
    /* iface   */ kStatsDeviceIface,
    /* reg     */ nullptr,     // filled in by stats_backend_reg() below
    /* context */ nullptr,
};

// ---------------------------------------------------------------------------
// ggml_backend_reg_i — the profiling backend registry entry
// ---------------------------------------------------------------------------

static const char * stats_reg_get_name(ggml_backend_reg_t /*reg*/) {
    return "profstats";
}

static size_t stats_reg_get_device_count(ggml_backend_reg_t /*reg*/) {
    return 1;
}

static ggml_backend_dev_t stats_reg_get_device(ggml_backend_reg_t /*reg*/,
                                                 size_t index) {
    return (index == 0) ? &kStatsDevice : nullptr;
}

static void * stats_reg_get_proc_address(ggml_backend_reg_t /*reg*/,
                                          const char * /*name*/) {
    return nullptr;
}

static struct ggml_backend_reg_i kStatsRegIface = {
    /* get_name         */ stats_reg_get_name,
    /* get_device_count */ stats_reg_get_device_count,
    /* get_device       */ stats_reg_get_device,
    /* get_proc_address */ stats_reg_get_proc_address,
};

static struct ggml_backend_reg kStatsReg = {
    /* api_version */ GGML_BACKEND_API_VERSION,
    /* iface       */ kStatsRegIface,
    /* context     */ nullptr,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void profstats_set_output_dir(const char * path) {
    if (path) { g_output_dir = path; }
}

void profstats_set_inner_backend_name(const char * name) {
    if (name) { g_inner_backend_name = name; }
}

void profstats_set_max_steps(size_t n) {
    g_max_steps = n;
}

void profstats_register(void) {
    // Read config from environment (may be overridden by explicit calls above).
    const char * env_dir = std::getenv("PROFSTATS_OUT_DIR");
    if (env_dir && *env_dir) { g_output_dir = env_dir; }

    const char * env_inner = std::getenv("PROFSTATS_INNER_BACKEND");
    if (env_inner && *env_inner) { g_inner_backend_name = env_inner; }

    const char * env_steps = std::getenv("PROFSTATS_MAX_STEPS");
    if (env_steps && *env_steps) { g_max_steps = static_cast<size_t>(std::atoi(env_steps)); }

    // Fix up the device's back-pointer to the registry.
    kStatsDevice.reg = &kStatsReg;

    // Register with ggml.
    ggml_backend_register(&kStatsReg);

    fprintf(stderr, "[profstats] registered; out_dir=%s inner=%s max_steps=%zu\n",
            g_output_dir.c_str(),
            g_inner_backend_name.empty() ? "(auto)" : g_inner_backend_name.c_str(),
            g_max_steps);
}

// ---------------------------------------------------------------------------
// Shared-library constructor — runs before main() when preloaded
// ---------------------------------------------------------------------------
__attribute__((constructor))
static void profstats_init(void) {
    profstats_register();
}

// ---------------------------------------------------------------------------
// Dynamic-loading entry point (used when loaded via ggml_backend_load())
// ---------------------------------------------------------------------------
GGML_BACKEND_DL_IMPL([]() -> ggml_backend_reg_t {
    profstats_register();
    return &kStatsReg;
})
