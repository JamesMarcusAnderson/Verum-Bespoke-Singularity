# Auto-detect compiler: prefer clang, fallback to gcc, or respect CC environment override
CC ?= $(shell command -v clang 2>/dev/null || command -v gcc 2>/dev/null || echo cc)

CFLAGS := -std=c11 -Wall -Wextra -O3 -Iinclude -D_GNU_SOURCE
LDFLAGS := -ldl -lm -lpthread

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Core sources (compiled on all platforms)
SRC := src/main.c \
       src/tokenizer.c \
       src/vbs_gguf.c \
       src/vbs_graph.c \
       src/vbs_kernels_ref.c \
       src/vbs_backend_cpu.c \
       src/vbs_backend_cuda.c \
       src/vbs_backend_hip.c \
       src/vbs_slice_executor.c \
       src/vbs_engine.c \
       src/vbs_profiler.c \
       src/vbs_fabric.c \
       src/vbs_planner.c \
       src/vbs_specializer.c

# Platform-specific backend & flags
ifeq ($(UNAME_S),Darwin)
    CFLAGS += -DVBS_ENABLE_METAL=1
    LDFLAGS += -framework Foundation -framework Metal -framework CoreGraphics -lobjc
    SRC += src/vbs_backend_metal.c
endif

# Dynamic CUDA / HIP runtime loading flags
CFLAGS += -DVBS_ENABLE_CUDA=1 -DVBS_ENABLE_HIP=1

# Architecture-specific SIMD optimization flags
ifeq ($(UNAME_M),x86_64)
    CFLAGS += -mavx2 -mfma -DVBS_HAVE_AVX2=1
endif
ifeq ($(UNAME_M),arm64)
    CFLAGS += -DVBS_HAVE_NEON=1
endif
ifeq ($(UNAME_M),aarch64)
    CFLAGS += -DVBS_HAVE_NEON=1
endif

BIN := verum-slice

.PHONY: all clean

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(SRC) $(LDFLAGS) -o $@

clean:
	rm -f $(BIN) *.o
