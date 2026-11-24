// matvec_common.glsl — shared skeleton for the matvec_* kernels.
//
// One workgroup of 32 lanes per output row. Lane `l` covers columns
// l, l+32, ... of its row, each lane keeps a partial dot product, and
// the group then tree-reduces the 32 partials in shared memory; lane
// 0 writes (or accumulates into) y[row_off + row].
//
// Push constants, identical for every matvec kernel: cols, rows,
// col_off, row_off, accumulate, row_base. `row_base` is the
// weight-row offset of this dispatch chunk: the backend splits
// dispatches larger than 65535 workgroups (the compute API's minimum
// guaranteed workgroup-count limit) into chunks, each with its own
// row_base, so the weight row is row_base + gl_WorkGroupID.x. The
// weight-row stride comes from `cols` and the weight type, so one
// kernel serves both the bridge path (full resident tensor) and the
// slice-executor path (packed tile in the staging buffer).

layout(local_size_x = 32) in;
shared float partial[32];

layout(push_constant) uniform PC {
    uint cols; uint rows; uint col_off; uint row_off; uint accumulate;
    uint row_base;
} pc;

// Reduce `sum` across the workgroup and store the row result.
// `y` must be the output buffer (float array) in scope.
void matvec_store(float sum, uint row) {
    uint lane = gl_LocalInvocationID.x;
    partial[lane] = sum;
    barrier();
    for (uint step = 16u; step > 0u; step >>= 1) {
        if (lane < step) partial[lane] += partial[lane + step];
        barrier();
    }
    if (lane == 0u) {
        uint slot = pc.row_off + pc.row_base + row;
        if (pc.accumulate != 0u) y[slot] += partial[0];
        else y[slot] = partial[0];
    }
}
