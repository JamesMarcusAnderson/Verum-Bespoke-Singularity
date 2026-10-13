MACOSX_DEPLOYMENT_TARGET ?= 13.0
CC := xcrun --sdk macosx clang
CFLAGS := -fobjc-arc -fmodules -Wall -Wextra -Werror -O2 -mmacosx-version-min=$(MACOSX_DEPLOYMENT_TARGET)
FRAMEWORKS := -framework Foundation -framework Metal
SRC := $(wildcard Sources/*.m)

.PHONY: all clean check
all: verum-slice

verum-slice: $(SRC)
	$(CC) $(CFLAGS) -I Sources $(SRC) $(FRAMEWORKS) -o $@

check:
	$(CC) $(CFLAGS) -fsyntax-only -I Sources $(SRC) $(FRAMEWORKS)

clean:
	$(RM) verum-slice

