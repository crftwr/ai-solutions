// vlm_op_profiler.cpp
//
// Thin wrapper around llama-mtmd-cli that:
//   1. Parses profiler-specific flags (--out-dir, --steps, --include-vision-encode).
//   2. Sets DYLD_INSERT_LIBRARIES (macOS) or LD_PRELOAD (Linux) to inject
//      libbackend_stats into the child process.
//   3. Passes remaining flags through to llama-mtmd-cli.
//
// Phase 0: prints usage and exits.  Backend injection implemented in Phase 4.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Version / identity
// ---------------------------------------------------------------------------
static constexpr const char * VERSION = "0.1.0-dev";

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
static void print_usage(const char * prog) {
    printf(
        "vlm-op-profiler %s\n"
        "\n"
        "Collect ggml tensor-operation statistics from llama.cpp while running\n"
        "a vision-language model.  Outputs trace.jsonl, report.csv, report.md,\n"
        "and run_meta.json under --out-dir.\n"
        "\n"
        "Usage:\n"
        "  %s [profiler-options] --model <model.gguf> --mmproj <mmproj.gguf>\n"
        "       [--image <image>] [prompt]\n"
        "\n"
        "Profiler options:\n"
        "  --out-dir <path>          Directory for results (default: ./profstats_out)\n"
        "  --steps <N>               Cap decode at N tokens (default: 256)\n"
        "  --include-vision-encode   Also profile the image-encoder graph\n"
        "  --inner-backend <name>    ggml backend to wrap: cpu | metal | cuda\n"
        "                            (default: cpu; auto-detect if not set)\n"
        "  --mtmd-cli <path>         Path to llama-mtmd-cli binary\n"
        "                            (default: searches PATH and build/bin/)\n"
        "  -h, --help                Print this message and exit\n"
        "  --version                 Print version and exit\n"
        "\n"
        "All other flags are forwarded to llama-mtmd-cli unchanged.\n"
        "Run 'llama-mtmd-cli --help' for the full list.\n"
        "\n"
        "Example:\n"
        "  %s --out-dir results/llava-test \\\n"
        "       --model  models/llava-v1.6-mistral-7b.Q4_K_M.gguf \\\n"
        "       --mmproj models/llava-v1.6-mistral-7b-mmproj.gguf \\\n"
        "       --image  docs/example.jpg \\\n"
        "       'Describe this image in detail.'\n",
        VERSION, prog, prog);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 0;
    }

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "--version") == 0) {
            printf("vlm-op-profiler %s\n", VERSION);
            return 0;
        }
    }

    // Phase 0 stub — backend injection and child-process exec not yet implemented.
    fprintf(stderr,
            "vlm-op-profiler: backend injection not yet implemented (Phase 0 skeleton).\n"
            "Re-run after Phase 4 is complete, or use 'make run-suite' once build is ready.\n");
    return 1;
}
