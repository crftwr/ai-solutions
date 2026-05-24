// vlm_op_profiler.cpp
//
// Process wrapper around llama-mtmd-cli.
//
// What it does:
//   1. Parses profiler-specific flags (--out-dir, --steps, etc.).
//   2. Locates libbackend_stats.so and llama-mtmd-cli relative to itself.
//   3. Creates the output directory.
//   4. Sets LD_PRELOAD / DYLD_INSERT_LIBRARIES so libbackend_stats is injected.
//   5. Sets PROFSTATS_OUT_DIR and PROFSTATS_MAX_STEPS for the interceptor.
//   6. exec's llama-mtmd-cli with the remaining (model / image / prompt) args.
//
// The profiler-specific flags are consumed here; everything else is forwarded
// to llama-mtmd-cli unchanged.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __linux__
#  include <limits.h>  // PATH_MAX
#endif
#ifdef __APPLE__
#  include <mach-o/dyld.h>
#  include <limits.h>
#endif

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------
static constexpr const char * VERSION = "0.1.0-dev";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool file_exists(const std::string & p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

static bool is_executable(const std::string & p) {
    return access(p.c_str(), X_OK) == 0;
}

// Return the directory containing our own executable, with trailing '/'.
static std::string self_dir() {
    char buf[4096] = {};

#ifdef __linux__
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0) {
        buf[len] = '\0';
        char * slash = strrchr(buf, '/');
        if (slash) { slash[1] = '\0'; return buf; }
    }
#elif defined(__APPLE__)
    uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) == 0) {
        char * slash = strrchr(buf, '/');
        if (slash) { slash[1] = '\0'; return buf; }
    }
#endif
    return "./";
}

// Search candidate paths for an executable; return first hit.
static std::string find_binary(const std::string & name,
                                const std::vector<std::string> & candidates) {
    for (const auto & c : candidates) {
        if (is_executable(c)) return c;
    }
    // Fall back to PATH
    const char * path_env = std::getenv("PATH");
    if (path_env) {
        std::string path_str(path_env);
        size_t start = 0;
        while (start < path_str.size()) {
            size_t end = path_str.find(':', start);
            if (end == std::string::npos) end = path_str.size();
            std::string candidate = path_str.substr(start, end - start) + "/" + name;
            if (is_executable(candidate)) return candidate;
            start = end + 1;
        }
    }
    return {};
}

// Recursively create directories (like mkdir -p).
static bool mkdirp(const std::string & path) {
    if (path.empty()) return false;
    struct stat st;
    if (stat(path.c_str(), &st) == 0) return true;  // already exists

    // Create parent first.
    size_t slash = path.rfind('/');
    if (slash != std::string::npos && slash > 0) {
        if (!mkdirp(path.substr(0, slash))) return false;
    }

    if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "vlm-op-profiler: mkdir(%s): %s\n",
                path.c_str(), strerror(errno));
        return false;
    }
    return true;
}

// Prepend our library to LD_PRELOAD (or DYLD_INSERT_LIBRARIES on macOS),
// preserving any pre-existing value.
static void prepend_preload(const std::string & lib_path) {
#ifdef __APPLE__
    const char * key = "DYLD_INSERT_LIBRARIES";
#else
    const char * key = "LD_PRELOAD";
#endif
    const char * existing = std::getenv(key);
    std::string value = lib_path;
    if (existing && *existing) value += std::string(":") + existing;
    setenv(key, value.c_str(), /*overwrite=*/1);

#ifdef __APPLE__
    // Needed for DYLD_INSERT_LIBRARIES to work across two-level namespace libs.
    setenv("DYLD_FORCE_FLAT_NAMESPACE", "1", 1);
#endif
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
static void print_usage(const char * prog) {
    printf(
        "vlm-op-profiler %s\n"
        "\n"
        "Collect ggml tensor-operation statistics from llama.cpp while running\n"
        "a vision-language model.  Outputs trace.jsonl under --out-dir.\n"
        "\n"
        "Usage:\n"
        "  %s [profiler-options] -- [llama-mtmd-cli options]\n"
        "  %s [profiler-options] --model <model.gguf> [--image <img>] [prompt]\n"
        "\n"
        "Profiler options (consumed here, not forwarded):\n"
        "  --out-dir  <path>    Output directory (default: ./profstats_out)\n"
        "  --steps    <N>       Stop recording after N decode graphs (0=unlimited)\n"
        "  --mtmd-cli <path>    Explicit path to llama-mtmd-cli\n"
        "  --include-vision-encode\n"
        "                       Also record the image-encoder graph separately\n"
        "  -h, --help           Print this message\n"
        "  --version            Print version\n"
        "\n"
        "All other flags are forwarded to llama-mtmd-cli unchanged.\n"
        "\n"
        "Environment variables (also settable here as flags):\n"
        "  PROFSTATS_OUT_DIR    Same as --out-dir\n"
        "  PROFSTATS_MAX_STEPS  Same as --steps\n"
        "\n"
        "Examples:\n"
        "  # VLM with image:\n"
        "  %s --out-dir results/test \\\n"
        "       --model  models/llava-1.6.Q4_K_M.gguf \\\n"
        "       --mmproj models/llava-1.6-mmproj.gguf \\\n"
        "       --image  photo.jpg \\\n"
        "       'Describe this image.'\n"
        "\n"
        "  # Text-only (no image):\n"
        "  %s --out-dir results/text \\\n"
        "       --model models/mistral-7b.Q4_K_M.gguf \\\n"
        "       -p 'The capital of France is'\n",
        VERSION, prog, prog, prog, prog);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    // ---- Parse profiler-specific flags ------------------------------------
    std::string out_dir          = "profstats_out";
    std::string mtmd_cli_path;
    std::size_t max_steps        = 0;
    bool        include_vision   = false;
    bool        help             = false;
    bool        show_version     = false;

    // Collect args that are NOT profiler-specific (forwarded to llama-mtmd-cli).
    std::vector<const char *> forward_args;

    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];

        if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            help = true;
        } else if (strcmp(a, "--version") == 0) {
            show_version = true;
        } else if (strcmp(a, "--out-dir") == 0 && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (strcmp(a, "--steps") == 0 && i + 1 < argc) {
            max_steps = static_cast<std::size_t>(std::atoi(argv[++i]));
        } else if (strcmp(a, "--mtmd-cli") == 0 && i + 1 < argc) {
            mtmd_cli_path = argv[++i];
        } else if (strcmp(a, "--include-vision-encode") == 0) {
            include_vision = true;  // TODO Phase 4
        } else if (strcmp(a, "--") == 0) {
            // Everything after '--' goes straight to llama-mtmd-cli.
            for (int j = i + 1; j < argc; ++j) {
                forward_args.push_back(argv[j]);
            }
            break;
        } else {
            forward_args.push_back(a);
        }
    }

    if (help) { print_usage(argv[0]); return 0; }
    if (show_version) { printf("vlm-op-profiler %s\n", VERSION); return 0; }

    if (forward_args.empty()) {
        // No model args: just show usage.
        print_usage(argv[0]);
        return 0;
    }

    // ---- Locate binaries and library -------------------------------------
    const std::string dir = self_dir();

    // libbackend_stats: must be alongside this binary.
#ifdef __APPLE__
    const std::string lib_name = "libbackend_stats.dylib";
#else
    const std::string lib_name = "libbackend_stats.so";
#endif
    const std::string lib_path = dir + lib_name;
    if (!file_exists(lib_path)) {
        fprintf(stderr,
            "vlm-op-profiler: cannot find %s (looked in %s).\n"
            "Ensure make docker-build has completed.\n",
            lib_name.c_str(), dir.c_str());
        return 1;
    }

    // llama-mtmd-cli: explicit flag, alongside binary, /app/, or PATH.
    if (mtmd_cli_path.empty()) {
        mtmd_cli_path = find_binary("llama-mtmd-cli", {
            dir + "llama-mtmd-cli",
            "/app/llama-mtmd-cli",
            dir + "bin/llama-mtmd-cli",
        });
    }
    if (mtmd_cli_path.empty()) {
        fprintf(stderr,
            "vlm-op-profiler: cannot find llama-mtmd-cli.\n"
            "Pass --mtmd-cli <path> or ensure it is on PATH.\n");
        return 1;
    }

    // ---- Create output directory -----------------------------------------
    if (!mkdirp(out_dir)) return 1;

    // ---- Configure the interceptor via environment -----------------------
    setenv("PROFSTATS_OUT_DIR", out_dir.c_str(), 1);
    {
        char buf[32];
        snprintf(buf, sizeof(buf), "%zu", max_steps);
        setenv("PROFSTATS_MAX_STEPS", buf, 1);
    }
    (void)include_vision;  // Phase 4: PROFSTATS_INCLUDE_VISION_ENCODE

    // Inject our interceptor library into the child process.
    prepend_preload(lib_path);

    // ---- Build exec argument list ----------------------------------------
    std::vector<char *> exec_argv;
    exec_argv.push_back(const_cast<char *>(mtmd_cli_path.c_str()));
    for (const char * a : forward_args) {
        exec_argv.push_back(const_cast<char *>(a));
    }
    exec_argv.push_back(nullptr);

    fprintf(stderr,
            "vlm-op-profiler: exec %s  (out_dir=%s  steps=%zu)\n",
            mtmd_cli_path.c_str(), out_dir.c_str(), max_steps);

    // exec replaces this process; on success it does not return.
    execv(mtmd_cli_path.c_str(), exec_argv.data());

    // Only reached on error.
    fprintf(stderr, "vlm-op-profiler: execv(%s): %s\n",
            mtmd_cli_path.c_str(), strerror(errno));
    return 1;
}
