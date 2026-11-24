# Tests for the Turing shader path (SASS assembler + vecadd kernel).
# Standalone build: needs only a C11 compiler. No GPU, no CUDA
# toolkit, no hardware access — every test is encoding logic.
#
#   make        build + run all tests
#   make list   regenerate ../vecadd-listing.txt from the assembler
#   make clean  remove build artifacts

CC ?= cc
CFLAGS := -std=c11 -Wall -Wextra -Werror -O2 -I.. -I../..
# QMD checks reuse the sibling Turing driver's builder (pure C11,
# no ioctls) plus the framework command-buffer helpers it links
# against; both live one directory up. vbs_drv.c auto-registers
# the Alchemist driver on first probe (register_builtin_vendors),
# so the Alchemist sources must link here too or the build fails
# with an undefined reference to vbs_drv_alchemist_register.
DRV_SRCS := ../../vbs_drv_turing_enc.c ../../vbs_drv.c \
	../../vbs_drv_alchemist_enc.c ../../vbs_drv_alchemist.c

SHADER_SRCS := ../vbs_turing_sass.c ../vbs_turing_vecadd.c

TEST_BIN := test_turing_sass
LIST_BIN := vbs_turing_sass_list

.PHONY: all list clean

all: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): test_turing_sass.c $(SHADER_SRCS) $(DRV_SRCS) \
		../vbs_turing_sass.h ../vbs_turing_sass_opcodes.h \
		../vbs_turing_vecadd.h ../../vbs_drv_turing.h
	$(CC) $(CFLAGS) test_turing_sass.c $(SHADER_SRCS) $(DRV_SRCS) \
		-o $@

$(LIST_BIN): ../vbs_turing_sass_list.c $(SHADER_SRCS) \
		../vbs_turing_sass.h ../vbs_turing_sass_opcodes.h \
		../vbs_turing_vecadd.h
	$(CC) $(CFLAGS) ../vbs_turing_sass_list.c $(SHADER_SRCS) -o $@

list: $(LIST_BIN)
	./$(LIST_BIN) > ../vecadd-listing.txt
	@echo "wrote ../vecadd-listing.txt"

clean:
	rm -f $(TEST_BIN) $(LIST_BIN)
