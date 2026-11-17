# Verum-Bespoke-Singularity — CMake build.
#
# The Makefile is the primary build; this file exists so Windows users can
# build with MSVC (no MinGW needed) and so any CMake-based workflow works.
# Same sources, same flags, same C11 standard as the Makefile.
#
#   Linux/macOS:  cmake -S . -B build && cmake --build build
#   Windows MSVC: cmake -S . -B build   (Visual Studio generator, VS2019+)
#                 cmake --build build --config Release
#   Windows MinGW cross-compile from Linux:
#                 cmake -S . -B build-mingw -DCMAKE_SYSTEM_NAME=Windows \
#                       -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc
#                 cmake --build build-mingw
#
# MSVC status: documented, untested on a real Windows machine — the
# windows-msvc CI job is the first MSVC run. MinGW via CMake is verified
# (cross-compiled and run under Wine); see PORTABLE.md.

cmake_minimum_required(VERSION 3.16)
project(verum-slice C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(VBS_SRC
    src/main.c
    src/tokenizer.c
    src/vbs_gguf.c
    src/vbs_graph.c
    src/vbs_backend_cuda.c
    src/vbs_backend_hip.c
    src/vbs_backend_vulkan.c
    src/vbs_backend_cpu.c
    src/vbs_backend_select.c
    src/vbs_backend_test.c
    src/vbs_slice_executor.c
    src/vbs_engine.c
    src/vbs_profiler.c
    src/vbs_fabric.c
    src/vbs_planner.c
    src/vbs_specializer.c
    src/vbs_specializer_bridge.c
    src/vbs_learn.c
    src/vbs_train.c
    src/vbs_block.c
    src/vbs_autonomy.c
    src/vbs_scheduler.c
    src/vbs_autotune.c
    src/vbs_cpu.c
    src/vbs_dequant.c
    src/vbs_sampling.c
    src/vbs_hotpath.c
    src/vbs_cc.c
    src/vbs_loop.c
    src/vbs_fuse.c
    src/vbs_tune.c
    src/compiler/lexer.c
    src/compiler/parser.c
    src/compiler/typecheck.c
    src/compiler/driver.c
    src/compiler/ptx_emit.c
    src/compiler/ptx_validate.c
    src/compiler/msl_emit.c
    src/compiler/spirv_emit.c
    src/compiler/spirv_validate.c
    src/compiler/gcn_emit.c
    src/compiler/gcn_validate.c
    src/compiler/native_emit_stub.c
)

if(APPLE)
    list(APPEND VBS_SRC src/vbs_backend_metal.c)
endif()

add_executable(verum-slice ${VBS_SRC})
target_include_directories(verum-slice PRIVATE include)

# Same defines as the Makefile. _GNU_SOURCE exposes clock_gettime/strtok_r
# on glibc; harmless on MSVC/MinGW.
target_compile_definitions(verum-slice PRIVATE
    _GNU_SOURCE
    VBS_ENABLE_CUDA=1
    VBS_ENABLE_HIP=1
)

if(APPLE)
    target_compile_definitions(verum-slice PRIVATE VBS_ENABLE_METAL=1)
    target_link_libraries(verum-slice PRIVATE
        "-framework Foundation" "-framework Metal" "-framework CoreGraphics" objc)
endif()

if(MSVC)
    # Silence the strcpy/fopen deprecation warnings (C4996); the code is
    # plain C11 and these calls are safe here.
    target_compile_definitions(verum-slice PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(verum-slice PRIVATE /W3)
else()
    target_compile_options(verum-slice PRIVATE -Wall -Wextra -O3)
endif()

if(UNIX)
    target_link_libraries(verum-slice PRIVATE dl m pthread)
elseif(MINGW)
    target_link_libraries(verum-slice PRIVATE m)
    # Export the exe's symbols and emit an import library so the JIT
    # specializer can link its DLLs against the bridge functions
    # (same as the Makefile `mingw` target).
    target_link_options(verum-slice PRIVATE
        -Wl,--export-all-symbols -Wl,--out-implib,libverum-slice.dll.a)
endif()
