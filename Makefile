# Verum-Bespoke-Singularity — primary build (make).
#
# Produces three artifacts:
#   verum-slice       the engine CLI (everything: CPU + GPU backends)
#   vbs-cpu           standalone CPU inference CLI (own main, small link)
#   libvbs_hot_ref.so reference hot-kernel module for the loop's swap path
#
# GPU backends are runtime-optional: CUDA/HIP/Metal/Vulkan are dlopen'd,
# so the build needs no GPU, no SDK, and no vendor headers. The same
# source list is mirrored in CMakeLists.txt (the MSVC/MinGW build);
# keep the two in step when adding files.
#
#   make            build all three artifacts
#   make test       model-free self-checks (no model file, no GPU)
#   make test-d     Phase D end-to-end demo (needs python3 + objcopy)
#   make mingw      Windows cross-compile (mingw-w64)
#   make clean      remove build artifacts

CC ?= $(shell command -v clang 2>/dev/null || command -v gcc 2>/dev/null || echo cc)
CFLAGS := -std=c11 -Wall -Wextra -O3 -Iinclude -D_GNU_SOURCE
LDFLAGS := -ldl -lm -lpthread

UNAME_S := $(shell uname -s)

# Engine sources, grouped by area. Order is link order, not important
# beyond keeping related files together for readability.
CORE_SRC := src/main.c \
       src/tokenizer.c \
       src/vbs_model.c \
       src/vbs_graph.c \
       src/vbs_cpu.c \
       src/vbs_dequant.c \
       src/vbs_sampling.c

BACKEND_SRC := src/vbs_backend_cpu.c \
       src/vbs_backend_gpu.c \
       src/vbs_gpu_vendors.c \
       src/driver/vbs_drv.c \
       src/driver/vbs_drv_turing_enc.c \
       src/driver/vbs_drv_turing.c \
       src/driver/vbs_drv_rdna2_enc.c \
       src/driver/vbs_drv_rdna2.c \
       src/driver/vbs_drv_alchemist_enc.c \
       src/driver/vbs_drv_alchemist.c \
       src/driver/vbs_backend_native.c \
       src/vbs_backend_select.c \
       src/vbs_backend_test.c \
       src/vbs_specializer.c \
       src/vbs_specializer_bridge.c

ENGINE_SRC := src/vbs_engine.c \
       src/vbs_slice_executor.c \
       src/vbs_fabric.c \
       src/vbs_planner.c \
       src/vbs_profiler.c

LEARN_SRC := src/vbs_learn.c \
       src/vbs_train.c \
       src/vbs_block.c \
       src/vbs_autonomy.c \
       src/vbs_scheduler.c \
       src/vbs_autotune.c

LOOP_SRC := src/vbs_hotpath.c \
       src/vbs_cc.c \
       src/vbs_loop.c \
       src/vbs_fuse.c \
       src/vbs_tune.c

COMPILER_SRC := src/compiler/lexer.c \
       src/compiler/parser.c \
       src/compiler/typecheck.c \
       src/compiler/driver.c \
       src/compiler/emit_vendors.c \
       src/compiler/kernel_emit.c \
       src/compiler/kernel_validate.c \

SRC := $(CORE_SRC) $(BACKEND_SRC) $(ENGINE_SRC) $(LEARN_SRC) $(LOOP_SRC) $(COMPILER_SRC)

ifeq ($(UNAME_S),Darwin)
    # The Metal section of src/vbs_gpu_vendors.c is enabled by this
    # define alone; the frameworks below are what it links against.
    CFLAGS += -DVBS_ENABLE_METAL=1
    LDFLAGS += -framework Foundation -framework Metal -framework CoreGraphics -lobjc
endif

CFLAGS += -DVBS_ENABLE_CUDA=1 -DVBS_ENABLE_HIP=1

BIN := verum-slice
CPU_BIN := vbs-cpu
MODEL_TEST := model_e2e_test
HOT_REF_SO := libvbs_hot_ref.so

.PHONY: all clean test test-d mingw

all: $(BIN) $(CPU_BIN) $(HOT_REF_SO)

$(BIN): $(SRC) $(wildcard include/*.h src/*.h)
	$(CC) $(CFLAGS) $(SRC) $(LDFLAGS) -o $@

# Reference hot-kernel module: the same arithmetic as the built-in table,
# packaged as a loadable .so for the loop's swap/rollback path.
$(HOT_REF_SO): src/hotkernels/vbs_hot_ref.c src/hotkernels/vbs_hot_ref_impl.h include/vbs_hotpath.h
	$(CC) -std=c11 -O3 -shared -fPIC -Iinclude -Isrc/hotkernels src/hotkernels/vbs_hot_ref.c -o $@

# Sources shared by the CPU CLI and the model regression test. The CPU
# path dispatches through the hot-kernel table and the model fuser, so
# those modules (and their deps) ride along.
CPU_SRC := src/vbs_cpu.c src/vbs_model.c src/vbs_dequant.c src/tokenizer.c \
       src/vbs_hotpath.c src/vbs_fuse.c src/vbs_cc.c src/vbs_tune.c
$(CPU_BIN): src/vbs_cpu_main.c $(CPU_SRC) $(wildcard include/*.h)
	$(CC) $(CFLAGS) src/vbs_cpu_main.c $(CPU_SRC) -lm -o $@

# Regression test: loader + dequantizer + CPU forward on a tiny synthetic
# model built in memory (no model file needed). Covers Q8_K/Q4_K/Q6_K/Q5_K/
# Q5_0 conversion at load plus the native Q8_0/Q4_0/F32 paths.
$(MODEL_TEST): tests/model_e2e_test.c $(CPU_SRC) $(wildcard include/*.h)
	$(CC) $(CFLAGS) tests/model_e2e_test.c $(CPU_SRC) -lm -o $@

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
	rm -f verum-slice verum-slice.exe libverum-slice.dll.a $(CPU_BIN) $(CPU_BIN).exe $(MODEL_TEST) $(HOT_REF_SO) *.o

test: $(BIN) $(MODEL_TEST)
	./$(BIN) --help > /dev/null 2>&1
	./$(BIN) --dequant | grep -q "ALL PASS"
	./$(BIN) --sample | grep -q "ALL PASS"
	./$(MODEL_TEST) | grep -q "ALL PASS"
	./$(BIN) --backends > /dev/null 2>&1
	./$(BIN) --backend-test | grep -q "ALL PASS"
	VK_ICD_FILENAMES=/nonexistent/no-icd-here.json ./$(BIN) --backend-test | grep -q "ALL PASS"
	@echo "make test: all model-free self-checks passed"

# Phase D: recursive loop + fused kernel + tune profile, end to end on a
# synthetic tiny model. Needs python3 (model generator) and objcopy.
# Invoked via sh so it works even if the patch lost the script's +x bit.
test-d: $(BIN) $(HOT_REF_SO)
	sh tools/run_phase_d_demo.sh
