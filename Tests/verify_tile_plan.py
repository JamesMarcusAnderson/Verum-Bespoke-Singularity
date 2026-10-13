#!/usr/bin/env python3
"""Reference-check the executor's bounded 2-D partition invariant."""

GEOMETRY = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2),
    "Q4_0": (32, 18), "Q5_0": (32, 22), "Q8_0": (32, 34),
    "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144),
    "Q5_K": (256, 176), "Q6_K": (256, 210), "Q8_K": (256, 292),
}

def plan(rows, cols, window, geom):
    block, packed = geom
    assert cols % block == 0
    max_blocks = window // packed
    max_cols = max(block, (max_blocks // max(1, rows)) * block)
    max_cols = min(max_cols, cols)
    max_cols -= max_cols % block
    if not max_cols:
        max_cols = block
    seen = 0
    for cs in range(0, cols, max_cols):
        cc = min(max_cols, cols - cs)
        packed_row = (cc // block) * packed
        max_rows = window // packed_row
        assert max_rows
        for rs in range(0, rows, max_rows):
            rc = min(max_rows, rows - rs)
            size = rc * packed_row
            assert 0 < size <= window
            seen += rc * cc
    assert seen == rows * cols

for name, geom in GEOMETRY.items():
    # A tensor far larger than the simulated 32 MiB HBM window, including a
    # row that itself exceeds the window.
    block, _ = geom
    plan(262_144, block * 1_000_000, 32 * 1024 * 1024, geom)
    print("PASS", name)
print("PASS every planned packed tile is bounded and covers the tensor exactly")

