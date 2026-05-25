// backend_stats.cpp
//
// Intercepts ggml_backend_sched_graph_compute (and its _async variant) via:
//   Linux :  LD_PRELOAD symbol shadowing + dlsym(RTLD_NEXT, …)
//   macOS :  DYLD_INSERT_LIBRARIES + DYLD_INTERPOSE section
//
// The interception is entirely transparent: statistics are collected from the
// cgraph before the graph is forwarded to the real scheduler, so inference
// output is bit-identical to an un-profiled run.
//
// Configuration (read once at first call):
//   PROFSTATS_OUT_DIR                output directory     (default: ./profstats_out)
//   PROFSTATS_MAX_STEPS              cap on decode graphs (default: 0 = unlimited)
//   PROFSTATS_INCLUDE_VISION_ENCODE  if "1", also record vision-encoder graphs
//                                    (graphs containing CONV_2D); default skips them

#include "backend_stats.h"
#include "graph_walker.h"
#include "phase_tracker.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <atomic>
#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#ifndef _WIN32
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// ---------------------------------------------------------------------------
// Configuration — initialised once from environment variables
// ---------------------------------------------------------------------------
static std::string   g_output_dir            = "profstats_out";
static std::size_t   g_max_steps             = 0;
static bool          g_include_vision_encode = false;
static bool          g_config_done           = false;
static std::mutex    g_config_mu;

static void ensure_config() {
    std::lock_guard<std::mutex> lk(g_config_mu);
    if (g_config_done) return;
    g_config_done = true;

    const char * dir     = std::getenv("PROFSTATS_OUT_DIR");
    const char * steps   = std::getenv("PROFSTATS_MAX_STEPS");
    const char * vis_env = std::getenv("PROFSTATS_INCLUDE_VISION_ENCODE");
    if (dir   && *dir)   g_output_dir = dir;
    if (steps && *steps) g_max_steps  = static_cast<std::size_t>(std::atoi(steps));
    if (vis_env && *vis_env) {
        // "1", "true", "yes", "on" → enable
        g_include_vision_encode =
            (vis_env[0] == '1' || vis_env[0] == 't' || vis_env[0] == 'T' ||
             vis_env[0] == 'y' || vis_env[0] == 'Y' ||
             (vis_env[0] == 'o' && (vis_env[1] == 'n' || vis_env[1] == 'N')));
    }

    // Create output directory (best-effort)
    std::string cmd = "mkdir -p '" + g_output_dir + "'";
    (void)std::system(cmd.c_str());

    fprintf(stderr,
            "[profstats] interceptor active — out_dir=%s  max_steps=%zu"
            "  include_vision_encode=%d\n",
            g_output_dir.c_str(), g_max_steps,
            g_include_vision_encode ? 1 : 0);
}

// ---------------------------------------------------------------------------
// Global stats state
// ---------------------------------------------------------------------------
static std::atomic<uint64_t> g_graph_counter{0};
static PhaseTracker           g_phase_tracker;
static std::vector<NodeStats> g_pending;
static std::mutex             g_stats_mu;

// ---------------------------------------------------------------------------
// JSONL helpers (same as before; kept here to avoid pulling in a separate TU)
// ---------------------------------------------------------------------------
static void write_ne(std::ostream & out, const int64_t ne[PROFSTATS_MAX_DIMS]) {
    out << '[';
    for (int i = 0; i < PROFSTATS_MAX_DIMS; ++i) {
        if (i) out << ',';
        out << ne[i];
    }
    out << ']';
}

static void write_str(std::ostream & out, const std::string & s) {
    out << '"';
    for (char c : s) {
        if (c == '"' || c == '\\') out << '\\';
        out << c;
    }
    out << '"';
}

static void flush_pending(const std::string & dir, std::vector<NodeStats> & records) {
    if (records.empty()) return;

    const std::string path = dir + "/trace.jsonl";
    std::ofstream f(path, std::ios::app);
    if (!f.is_open()) {
        fprintf(stderr, "[profstats] WARNING: cannot open %s\n", path.c_str());
        records.clear();
        return;
    }

    for (const auto & s : records) {
        f << '{';
        f << "\"step\":"           << s.graph_id  << ',';
        f << "\"phase\":";          write_str(f, s.phase);          f << ',';
        f << "\"graph_id\":"       << s.graph_id  << ',';
        f << "\"node_idx\":"       << s.node_idx  << ',';
        f << "\"op\":";             write_str(f, s.op);             f << ',';
        f << "\"name\":";           write_str(f, s.name);           f << ',';
        f << "\"layer_category\":"; write_str(f, s.layer_category); f << ',';
        f << "\"src0\":{\"type\":"; write_str(f, s.src0_type);
        f << ",\"ne\":";            write_ne(f, s.src0_ne);         f << "},";
        f << "\"src1\":{\"type\":"; write_str(f, s.src1_type);
        f << ",\"ne\":";            write_ne(f, s.src1_ne);         f << "},";
        f << "\"dst\":{\"type\":";  write_str(f, s.dst_type);
        f << ",\"ne\":";            write_ne(f, s.dst_ne);          f << "},";
        f << "\"m\":"    << s.m    << ',';
        f << "\"n\":"    << s.n    << ',';
        f << "\"k\":"    << s.k    << ',';
        f << "\"macs\":" << s.macs;
        f << "}\n";
    }
    records.clear();
}

// ---------------------------------------------------------------------------
// Core interception logic — called from both the sync and async overrides
// ---------------------------------------------------------------------------
static void record_graph(const struct ggml_cgraph * graph) {
    ensure_config();

    const uint64_t gid = ++g_graph_counter;
    std::vector<NodeStats> records = walk_graph(graph, gid);
    if (records.empty()) return;

    // Vision-encoder graphs are identified by the presence of CONV_2D nodes,
    // which llama.cpp uses only for image patch embedding. By default we skip
    // them so trace.jsonl reflects the LLM body only; set
    // PROFSTATS_INCLUDE_VISION_ENCODE=1 to include them.
    bool is_vision_encode = false;
    int64_t max_m = 0;
    for (const auto & r : records) {
        if (r.op == "CONV_2D") is_vision_encode = true;
        if (r.m  > max_m)      max_m = r.m;
    }

    if (is_vision_encode && !g_include_vision_encode) {
        return;
    }

    std::lock_guard<std::mutex> lk(g_stats_mu);
    const std::string phase = is_vision_encode
        ? g_phase_tracker.classify_vision_encode()
        : g_phase_tracker.classify(max_m);
    for (auto & r : records) r.phase = phase;

    g_pending.insert(g_pending.end(), records.begin(), records.end());
    flush_pending(g_output_dir, g_pending);

    // Enforce step cap (decode only).
    if (g_max_steps > 0 && g_phase_tracker.decode_count() >= g_max_steps) {
        fprintf(stderr,
                "[profstats] max_steps=%zu reached after %" PRIu64 " decode graphs; "
                "subsequent graphs will not be recorded.\n",
                g_max_steps, g_phase_tracker.decode_count());
        // Do not abort — let inference finish; just stop recording.
        g_max_steps = 0;  // one-shot; disable further checks
    }
}

// ---------------------------------------------------------------------------
// Public configuration API (called from vlm_op_profiler or unit tests)
// ---------------------------------------------------------------------------
void profstats_set_output_dir(const char * path) {
    if (path) {
        std::lock_guard<std::mutex> lk(g_config_mu);
        g_output_dir   = path;
        g_config_done  = false;  // force re-init with new path
    }
}

void profstats_set_inner_backend_name(const char * /*name*/) {
    // Not used in the interposer approach (inner backend is the real scheduler).
}

void profstats_set_max_steps(size_t n) {
    std::lock_guard<std::mutex> lk(g_config_mu);
    g_max_steps = n;
}

void profstats_register(void) {
    // No-op in the interposer approach; backend registration is not needed.
    // Kept for API compatibility with the original design.
    ensure_config();
}

// ---------------------------------------------------------------------------
// Function interposition
//
// We intercept TWO functions:
//   ggml_backend_sched_graph_compute        (synchronous)
//   ggml_backend_sched_graph_compute_async  (asynchronous; llama.cpp may use this)
// ---------------------------------------------------------------------------

// ---- Original function pointers (resolved once, lazily) ------------------

typedef enum ggml_status (*fn_sched_compute_t)(
    ggml_backend_sched_t, struct ggml_cgraph *);

static fn_sched_compute_t resolve_orig_compute() {
    static fn_sched_compute_t fn = nullptr;
    if (!fn) {
        fn = reinterpret_cast<fn_sched_compute_t>(
            dlsym(RTLD_NEXT, "ggml_backend_sched_graph_compute"));
        if (!fn) {
            fprintf(stderr,
                "[profstats] FATAL: dlsym(RTLD_NEXT, ggml_backend_sched_graph_compute) "
                "returned NULL: %s\n", dlerror());
        }
    }
    return fn;
}

static fn_sched_compute_t resolve_orig_compute_async() {
    static fn_sched_compute_t fn = nullptr;
    if (!fn) {
        fn = reinterpret_cast<fn_sched_compute_t>(
            dlsym(RTLD_NEXT, "ggml_backend_sched_graph_compute_async"));
        // async variant may not exist in all builds — tolerate nullptr
    }
    return fn;
}

// ---- Our implementations -------------------------------------------------

static enum ggml_status profstats_sched_compute_impl(
    ggml_backend_sched_t sched, struct ggml_cgraph * graph)
{
    record_graph(graph);
    auto orig = resolve_orig_compute();
    return orig ? orig(sched, graph) : GGML_STATUS_SUCCESS;
}

static enum ggml_status profstats_sched_compute_async_impl(
    ggml_backend_sched_t sched, struct ggml_cgraph * graph)
{
    record_graph(graph);
    auto orig = resolve_orig_compute_async();
    return orig ? orig(sched, graph) : GGML_STATUS_SUCCESS;
}

// ---- Platform-specific wiring --------------------------------------------

#ifdef __APPLE__

// macOS: inject via __DATA,__interpose section.
// dyld processes this before main(); dlsym(RTLD_NEXT,…) skips the interpose
// and returns the original symbol from the next image.

#define DYLD_INTERPOSE(_repl, _orig)                                  \
    __attribute__((used))                                             \
    static struct { const void * r; const void * o; }                \
    _interpose_##_orig                                                \
    __attribute__((section("__DATA,__interpose"))) = {               \
        (const void *)reinterpret_cast<uintptr_t>(&_repl),            \
        (const void *)reinterpret_cast<uintptr_t>(&_orig)             \
    }

DYLD_INTERPOSE(profstats_sched_compute_impl,       ggml_backend_sched_graph_compute)
DYLD_INTERPOSE(profstats_sched_compute_async_impl, ggml_backend_sched_graph_compute_async)

#else   // Linux / other POSIX

// Linux: export functions with the same names as the ggml originals.
// LD_PRELOAD puts our library earlier in the linker search order, so
// calls from libllama.so → ggml_backend_sched_graph_compute resolve here
// instead.  We use dlsym(RTLD_NEXT,…) to call through to the real ggml.

extern "C" {

GGML_BACKEND_API
enum ggml_status ggml_backend_sched_graph_compute(
    ggml_backend_sched_t sched, struct ggml_cgraph * graph)
{
    return profstats_sched_compute_impl(sched, graph);
}

GGML_BACKEND_API
enum ggml_status ggml_backend_sched_graph_compute_async(
    ggml_backend_sched_t sched, struct ggml_cgraph * graph)
{
    return profstats_sched_compute_async_impl(sched, graph);
}

} // extern "C"

#endif  // __APPLE__

// ---------------------------------------------------------------------------
// Constructor: prints a banner so users know the library was loaded
// ---------------------------------------------------------------------------
__attribute__((constructor))
static void profstats_on_load() {
    // Don't call ensure_config() here — PROFSTATS_OUT_DIR might not be set
    // yet (the caller sets it after LD_PRELOAD takes effect).
    fprintf(stderr, "[profstats] library loaded (pid=%d)\n",
            (int)getpid());
}
