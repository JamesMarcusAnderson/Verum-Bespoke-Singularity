CC := clang
CFLAGS := -std=c11 -Wall -Wextra -Werror -O3 -Iinclude -D_GNU_SOURCE
LDFLAGS := -framework Foundation -framework Metal -framework CoreGraphics -lobjc

SRC := src/main.c \
       src/tokenizer.c \
       src/vbs_gguf.c \
       src/vbs_graph.c \
       src/vbs_backend.c \
       src/vbs_metal.c \
       src/vbs_slice_executor.c \
       src/vbs_engine.c

BIN := verum-slice

.PHONY: all clean

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(SRC) $(LDFLAGS) -o $@

clean:
	rm -f $(BIN)
