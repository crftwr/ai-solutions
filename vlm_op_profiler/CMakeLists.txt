cmake_minimum_required(VERSION 3.22)
project(vlm-op-profiler VERSION 0.1.0 LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Verify submodule is initialised
# ---------------------------------------------------------------------------
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    message(FATAL_ERROR
        "llama.cpp submodule not found.\n"
        "Run from the ai-solutions repo root:\n"
        "  git submodule update --init --recursive")
endif()

# ---------------------------------------------------------------------------
# llama.cpp subproject
#
# Key options:
#   GGML_BACKEND_DL=ON  — enables GGML_BACKEND_DL macro so that backends can
#                         export ggml_backend_init() for dynamic loading.
#   LLAMA_BUILD_TESTS=OFF — avoid pulling in test deps.
#   LLAMA_BUILD_EXAMPLES=ON — builds llama-mtmd-cli which is our inner binary.
# ---------------------------------------------------------------------------
set(GGML_BACKEND_DL   ON  CACHE BOOL "Enable dynamic backend loading" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Skip llama.cpp tests" FORCE)
set(LLAMA_BUILD_EXAMPLES ON CACHE BOOL "Build llama-mtmd-cli" FORCE)

add_subdirectory(third_party/llama.cpp)

# ---------------------------------------------------------------------------
# Helper: consistent include paths for our code
# ---------------------------------------------------------------------------
set(GGML_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/src"   # ggml-backend-impl.h
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include"
)

# ---------------------------------------------------------------------------
# libbackend_stats  (shared library — loaded via DYLD_INSERT_LIBRARIES / LD_PRELOAD)
# ---------------------------------------------------------------------------
add_library(backend_stats SHARED
    src/backend_stats.cpp
    src/graph_walker.cpp
    src/layer_classifier.cpp
    src/phase_tracker.cpp
)

target_include_directories(backend_stats
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${GGML_INCLUDE_DIRS}
)

# Link against ggml (provides ggml_backend_register, ggml_graph_node, etc.)
target_link_libraries(backend_stats PRIVATE ggml)

# Ensure we get the GGML_BACKEND_DL macro so GGML_BACKEND_DL_IMPL expands.
target_compile_definitions(backend_stats PRIVATE GGML_BACKEND_DL)

set_target_properties(backend_stats PROPERTIES
    OUTPUT_NAME "backend_stats"
    PREFIX      "lib"
)

# On macOS, set rpath so the dylib can find ggml when preloaded.
if(APPLE)
    set_target_properties(backend_stats PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_RPATH   "${CMAKE_BINARY_DIR}/third_party/llama.cpp/ggml/src"
    )
endif()

# ---------------------------------------------------------------------------
# vlm-op-profiler  (CLI process wrapper — no direct llama linkage required)
#
# The CLI's job is to set up the environment (DYLD_INSERT_LIBRARIES, output
# path env vars) and exec llama-mtmd-cli.  It is a standalone C++17 binary.
# ---------------------------------------------------------------------------
add_executable(vlm-op-profiler
    cli/vlm_op_profiler.cpp
)

target_compile_features(vlm-op-profiler PRIVATE cxx_std_17)

# Place the binary alongside the shared library for easy discovery.
set_target_properties(vlm-op-profiler PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ---------------------------------------------------------------------------
# Optional: clang-format check target
# ---------------------------------------------------------------------------
find_program(CLANG_FORMAT_BIN clang-format)
if(CLANG_FORMAT_BIN)
    file(GLOB_RECURSE ALL_CPP_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/cli/*.cpp"
    )
    add_custom_target(clang-format-check
        COMMAND ${CLANG_FORMAT_BIN} --dry-run --Werror ${ALL_CPP_SOURCES}
        COMMENT "Checking clang-format compliance"
    )
    add_custom_target(clang-format-fix
        COMMAND ${CLANG_FORMAT_BIN} -i ${ALL_CPP_SOURCES}
        COMMENT "Auto-formatting C++ sources"
    )
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "vlm-op-profiler ${PROJECT_VERSION} configured")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Output dir : ${CMAKE_BINARY_DIR}")
message(STATUS "  Backend DL : ${GGML_BACKEND_DL}")
