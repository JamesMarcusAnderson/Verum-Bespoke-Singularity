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
       src/vbs_scheduler.c

ifeq ($(UNAME_S),Darwin)
    CFLAGS += -DVBS_ENABLE_METAL=1
    LDFLAGS += -framework Foundation -framework Metal -framework CoreGraphics -lobjc
    SRC += src/vbs_backend_metal.c
endif

CFLAGS += -DVBS_ENABLE_CUDA=1 -DVBS_ENABLE_HIP=1

BIN := verum-slice

.PHONY: all clean

all: $(BIN)

$(BIN): $(SRC) $(wildcard include/*.h src/*.h)
	$(CC) $(CFLAGS) $(SRC) $(LDFLAGS) -o $@

clean:
	rm -f $(BIN) *.o
