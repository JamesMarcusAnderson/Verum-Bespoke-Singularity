// matvec_common.glsl — shared skeleton for the matvec_* kernels.
//
// Warp-per-row, mirroring the CUDA matvec kernels: workgroup size 32, one
// workgroup per row, lane `l` covers columns l, l+32, ...; shared-memory
// tree reduction; lane 0 writes y[row_off + row] (or accumulates into it).
// Push constants (all matvec kernels): cols, rows, col_off, row_off,
// accumulate, row_base. `row_base` is the weight-row offset of this
// chunk: the backend splits dispatches larger than 65535 workgroups
// (the Vulkan minimum maxComputeWorkGroupCount[0]) into chunks, each
// with its own row_base; the weight row is row_base + gl_WorkGroupID.x. The weight row stride is derived from `cols` and the type,
// so the same kernel serves both the bridge path (full resident tensor)
// and the slice-executor path (packed tile in the staging buffer).

layout(local_size_x = 32) in;
shared float red[32];

layout(push_constant) uniform PC {
    uint cols; uint rows; uint col_off; uint row_off; uint accumulate;
    uint row_base;
} pc;

// Tree-reduce `sum` across the 32 lanes and write y[row_off + row].
// `y` must be the out buffer (float array) in scope.
void matvec_finish(float sum, uint row) {
    red[gl_LocalInvocationID.x] = sum;
    barrier();
    if (gl_LocalInvocationID.x < 16u) red[gl_LocalInvocationID.x] += red[gl_LocalInvocationID.x + 16u];
    barrier();
    if (gl_LocalInvocationID.x < 8u) red[gl_LocalInvocationID.x] += red[gl_LocalInvocationID.x + 8u];
    barrier();
    if (gl_LocalInvocationID.x < 4u) red[gl_LocalInvocationID.x] += red[gl_LocalInvocationID.x + 4u];
    barrier();
    if (gl_LocalInvocationID.x < 2u) red[gl_LocalInvocationID.x] += red[gl_LocalInvocationID.x + 2u];
    barrier();
    if (gl_LocalInvocationID.x == 0u) {
        float v = red[0] + red[1];
        uint o = pc.row_off + pc.row_base + row;
        if (pc.accumulate != 0u) y[o] += v; else y[o] = v;
    }
}
