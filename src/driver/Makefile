# Verum native GPU driver stack — standalone build.
#
# Builds the framework + Turing driver + RDNA 2 driver + Alchemist driver +
# engine backend wiring into libvbsdrv.a and runs the unit tests. This Makefile is
# standalone: the top-level Verum Makefile/CMakeLists integration is
# a separate step (exact diffs in the driver report).
#
#   make        build the library and all tests
#   make test   build and run all tests (must print ALL PASS each)
#   make clean  remove build outputs
#
# Pure C11, -Wall -Wextra -Werror: zero warnings is the bar.
# No GPU is needed: every test passes on a machine with no /dev/dri.

CC ?= cc
AR ?= ar
CFLAGS := -std=c11 -Wall -Wextra -Werror -O2 -I. -I../../include -D_GNU_SOURCE

LIB := libvbsdrv.a
LIB_SRC := vbs_drv.c vbs_drv_turing_enc.c vbs_drv_turing.c vbs_drv_rdna2_enc.c vbs_drv_rdna2.c vbs_drv_alchemist_enc.c vbs_drv_alchemist.c vbs_backend_native.c
LIB_OBJ := $(LIB_SRC:.c=.o)

TESTS := tests/test_qmd tests/test_pushbuf tests/test_valloc tests/test_probe tests/test_backend tests/test_rdna2_pm4 tests/test_rdna2_probe tests/test_alchemist_enc tests/test_alchemist_probe

.PHONY: all test clean

all: $(LIB) $(TESTS)

$(LIB): $(LIB_OBJ)
	$(AR) rcs $@ $(LIB_OBJ)

%.o: %.c vbs_drv.h vbs_drv_turing.h vbs_drv_rdna2.h vbs_drv_alchemist.h
	$(CC) $(CFLAGS) -c $< -o $@

tests/test_qmd: tests/test_qmd.c $(LIB)
	$(CC) $(CFLAGS) tests/test_qmd.c $(LIB) -o $@

tests/test_pushbuf: tests/test_pushbuf.c $(LIB)
	$(CC) $(CFLAGS) tests/test_pushbuf.c $(LIB) -o $@

tests/test_valloc: tests/test_valloc.c $(LIB)
	$(CC) $(CFLAGS) tests/test_valloc.c $(LIB) -o $@

tests/test_probe: tests/test_probe.c $(LIB)
	$(CC) $(CFLAGS) tests/test_probe.c $(LIB) -o $@

tests/test_backend: tests/test_backend.c $(LIB)
	$(CC) $(CFLAGS) tests/test_backend.c $(LIB) -o $@

tests/test_rdna2_pm4: tests/test_rdna2_pm4.c $(LIB)
	$(CC) $(CFLAGS) tests/test_rdna2_pm4.c $(LIB) -o $@

tests/test_rdna2_probe: tests/test_rdna2_probe.c $(LIB)
	$(CC) $(CFLAGS) tests/test_rdna2_probe.c $(LIB) -o $@

tests/test_alchemist_enc: tests/test_alchemist_enc.c $(LIB)
	$(CC) $(CFLAGS) tests/test_alchemist_enc.c $(LIB) -o $@

tests/test_alchemist_probe: tests/test_alchemist_probe.c $(LIB)
	$(CC) $(CFLAGS) tests/test_alchemist_probe.c $(LIB) -o $@

test: $(TESTS)
	./tests/test_qmd
	./tests/test_pushbuf
	./tests/test_valloc
	./tests/test_probe
	./tests/test_backend
	./tests/test_rdna2_pm4
	./tests/test_rdna2_probe
	./tests/test_alchemist_enc
	./tests/test_alchemist_probe
	@echo "driver tests: ALL PASS"

clean:
	rm -f $(LIB) $(LIB_OBJ) $(TESTS)
