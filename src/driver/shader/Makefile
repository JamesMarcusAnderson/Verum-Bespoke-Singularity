# Verum shader assemblers — standalone build (AMD RDNA 2).
#
# Builds the RDNA 2 (gfx1030) assembler + vector-add kernel and
# runs the unit tests. Standalone: no GPU needed, every test
# passes on a machine with no /dev/dri.
#
#   make          build the tests and the listing tool
#   make test     build and run all tests (must print ALL PASS each)
#   make listing  regenerate vbs_rdna2_vecadd_listing.txt
#   make clean    remove build outputs
#
# Pure C11, -Wall -Wextra -Werror: zero warnings is the bar.
# This Makefile covers the AMD (vbs_rdna2_*) files only; the
# NVIDIA (vbs_turing_*) files in this directory are built
# elsewhere and are not touched here.

CC ?= cc
CFLAGS := -std=c11 -Wall -Wextra -Werror -O2 -I. -Itests

RDNA2_SRC := vbs_rdna2_asm.c vbs_rdna2_vecadd.c
RDNA2_OBJ := $(RDNA2_SRC:.c=.o)

TESTS := tests/test_rdna2_asm tests/test_rdna2_vecadd
LIST_TOOL := vbs_rdna2_vecadd_list

.PHONY: all test listing clean

all: $(TESTS) $(LIST_TOOL)

%.o: %.c vbs_rdna2_isa.h vbs_rdna2_asm.h vbs_rdna2_vecadd.h
	$(CC) $(CFLAGS) -c $< -o $@

tests/test_rdna2_asm: tests/test_rdna2_asm.c $(RDNA2_OBJ) tests/test_util.h
	$(CC) $(CFLAGS) tests/test_rdna2_asm.c $(RDNA2_OBJ) -o $@

tests/test_rdna2_vecadd: tests/test_rdna2_vecadd.c $(RDNA2_OBJ) tests/test_util.h
	$(CC) $(CFLAGS) tests/test_rdna2_vecadd.c $(RDNA2_OBJ) -o $@

$(LIST_TOOL): vbs_rdna2_vecadd_list.c $(RDNA2_OBJ)
	$(CC) $(CFLAGS) vbs_rdna2_vecadd_list.c $(RDNA2_OBJ) -o $@

test: $(TESTS)
	./tests/test_rdna2_asm
	./tests/test_rdna2_vecadd
	@echo "rdna2 shader tests: ALL PASS"

listing: $(LIST_TOOL)
	./$(LIST_TOOL) > vbs_rdna2_vecadd_listing.txt

clean:
	rm -f $(RDNA2_OBJ) $(TESTS) $(LIST_TOOL)
