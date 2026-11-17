CC ?= $(shell command -v clang 2>/dev/null || command -v gcc 2>/dev/null || echo cc)
CFLAGS := -std=c11 -Wall -Wextra -O3 -Iinclude -D_GNU_SOURCE
LDFLAGS := -ldl -lm -lpthread

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

SRC := src/main.c \
       src/tokenizer.c \
       src/vbs_gguf.c \
       src/vbs_graph.c \
       src/vbs_backend_cuda.c \
       src/vbs_backend_hip.c \
       src/vbs_backend_vulkan.c \
       src/vbs_backend_cpu.c \
       src/vbs_backend_select.c \
       src/vbs_backend_test.c \
       src/vbs_slice_executor.c \
       src/vbs_engine.c \
       src/vbs_profiler.c \
       src/vbs_fabric.c \
       src/vbs_planner.c \
       src/vbs_specializer.c \
       src/vbs_specializer_bridge.c \
       src/vbs_learn.c \
       src/vbs_train.c \
       src/vbs_block.c \
       src/vbs_autonomy.c \
       src/vbs_scheduler.c \
       src/vbs_autotune.c \
       src/vbs_cpu.c \
       src/vbs_dequant.c \
       src/vbs_sampling.c \
       src/vbs_hotpath.c \
       src/vbs_cc.c \
       src/vbs_loop.c \
       src/vbs_fuse.c \
       src/vbs_tune.c \
       src/compiler/lexer.c \
       src/compiler/parser.c \
       src/compiler/typecheck.c \
       src/compiler/driver.c \
       src/compiler/ptx_emit.c \
       src/compiler/ptx_validate.c \
       src/compiler/msl_emit.c \
       src/compiler/spirv_emit.c \
       src/compiler/spirv_validate.c \
       src/compiler/gcn_emit.c \
       src/compiler/gcn_validate.c \
       src/compiler/native_emit_stub.c

ifeq ($(UNAME_S),Darwin)
    CFLAGS += -DVBS_ENABLE_METAL=1
    LDFLAGS += -framework Foundation -framework Metal -framework CoreGraphics -lobjc
    SRC += src/vbs_backend_metal.c
endif

CFLAGS += -DVBS_ENABLE_CUDA=1 -DVBS_ENABLE_HIP=1

BIN := verum-slice
CPU_BIN := vbs-cpu
GGUF_TEST := gguf_e2e_test
HOT_REF_SO := libvbs_hot_ref.so

.PHONY: all clean test test-d mingw

all: $(BIN) $(CPU_BIN) $(HOT_REF_SO)

$(BIN): $(SRC) $(wildcard include/*.h src/*.h)
	$(CC) $(CFLAGS) $(SRC) $(LDFLAGS) -o $@

# Reference hot-kernel module: the same arithmetic as the built-in table,
# packaged as a loadable .so for the loop's swap/rollback path.
$(HOT_REF_SO): src/hotkernels/vbs_hot_ref.c src/hotkernels/vbs_hot_ref_impl.h include/vbs_hotpath.h
	$(CC) -std=c11 -O3 -shared -fPIC -Iinclude -Isrc/hotkernels src/hotkernels/vbs_hot_ref.c -o $@

# Standalone CPU reference inference CLI: reads prompt token IDs from
# stdin, writes one JSON result. Has its own main(), so it cannot link
# into verum-slice. The CPU path dispatches through the hot-kernel table
# and the model fuser, so those modules (and their deps) ride along.
CPU_SRC := src/vbs_cpu.c src/vbs_gguf.c src/vbs_dequant.c src/tokenizer.c \
       src/vbs_hotpath.c src/vbs_fuse.c src/vbs_cc.c src/vbs_tune.c
$(CPU_BIN): src/vbs_cpu_main.c $(CPU_SRC) $(wildcard include/*.h)
	$(CC) $(CFLAGS) src/vbs_cpu_main.c $(CPU_SRC) -lm -o $@

# Regression test: loader + dequantizer + CPU forward on a tiny synthetic
# GGUF built in memory (no model file needed). Covers Q8_K/Q4_K/Q6_K/Q5_K/
# Q5_0 conversion at load plus the native Q8_0/Q4_0/F32 paths.
$(GGUF_TEST): tests/gguf_e2e_test.c $(CPU_SRC) $(wildcard include/*.h)
	$(CC) $(CFLAGS) tests/gguf_e2e_test.c $(CPU_SRC) -lm -o $@

# Windows (MinGW): cross-compile from Linux with mingw-w64 installed, or run
# natively under MSYS2/MinGW on Windows (CC=x86_64-w64-mingw32-gcc is the
# cross compiler name; under MSYS2 plain `make CC=gcc BIN=verum-slice.exe`
# also works). No -ldl/-lpthread on Windows: dynamic loading is LoadLibraryA
# and the engine uses no pthreads. --export-all-symbols plus --out-implib
# produce libverum-slice.dll.a next to the exe; the JIT specializer links
# its DLLs against that import library so the bridge symbols resolve at
# load time (see vbs_specializer_compile). Ship the .dll.a next to the exe.
mingw:
	$(MAKE) CC=x86_64-w64-mingw32-gcc LDFLAGS="-lm -Wl,--export-all-symbols -Wl,--out-implib,libverum-slice.dll.a" BIN=verum-slice.exe all

clean:
	rm -f verum-slice verum-slice.exe libverum-slice.dll.a $(CPU_BIN) $(CPU_BIN).exe $(GGUF_TEST) $(HOT_REF_SO) *.o

test: $(BIN) $(GGUF_TEST)
	./$(BIN) --help > /dev/null 2>&1
	./$(BIN) --dequant | grep -q "ALL PASS"
	./$(BIN) --sample | grep -q "ALL PASS"
	./$(GGUF_TEST) | grep -q "ALL PASS"
	./$(BIN) --backends > /dev/null 2>&1
	./$(BIN) --backend-test | grep -q "ALL PASS"
	VK_ICD_FILENAMES=/nonexistent/no-icd-here.json ./$(BIN) --backend-test | grep -q "ALL PASS"
	@echo "make test: all model-free self-checks passed"

# Phase D: recursive loop + fused kernel + tune profile, end to end on a
# synthetic tiny model. Needs python3 (model generator) and objcopy.
# Invoked via sh so it works even if the patch lost the script's +x bit.
test-d: $(BIN) $(HOT_REF_SO)
	sh tools/run_phase_d_demo.sh
