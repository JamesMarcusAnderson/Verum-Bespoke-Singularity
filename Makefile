CC := clang
CFLAGS := -std=c11 -Wall -Wextra -O3 -Iinclude -D_GNU_SOURCE
LDFLAGS := -ldl -lm

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Enable backends based on platform
ifeq ($(UNAME_S),Darwin)
    CFLAGS += -DVBS_ENABLE_METAL=1
    LDFLAGS += -framework Foundation -framework Metal -framework CoreGraphics -lobjc
    METAL_SRC = src/vbs_backend_metal.c
endif

# NVIDIA and AMD backends are dynamically loaded at runtime (no link-time deps)
CFLAGS += -DVBS_ENABLE_CUDA=1 -DVBS_ENABLE_HIP=1

# SIMD intrinsics
ifeq ($(UNAME_M),x86_64)
    CFLAGS += -mavx2 -mfma -DVBS_HAVE_AVX2=1
endif
ifeq ($(UNAME_M),arm64)
    CFLAGS += -DVBS_HAVE_NEON=1
endif
ifeq ($(UNAME_M),aarch64)
    CFLAGS += -DVBS_HAVE_NEON=1
endif

SRC := src/main.c \
       src/vbs_gguf.c \
       src/vbs_graph.c \
       src/vbs_kernels_ref.c \
       src/vbs_backend_cpu.c \
       src/vbs_backend_cuda.c \
       src/vbs_backend_hip.c \
       src/vbs_slice_executor.c \
       src/vbs_engine.c \
       src/tokenizer.c \
       $(METAL_SRC)

BIN := verum-slice

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(SRC) $(LDFLAGS) -o $@

clean:
	rm -f $(BIN) *.o
