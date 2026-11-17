#ifndef VBS_BACKEND_TEST_H
#define VBS_BACKEND_TEST_H

#ifdef __cplusplus
extern "C" {
#endif

/* Backend self-test (--backend-test): exercises every available backend
 * through the ops table (alloc/copy, all primitives, linear tiles in
 * both calling conventions, quantized types), checks selection and the
 * CPU fallback, and validates the SPIR-V path (emitter + precompiled
 * blobs). Prints one line per check plus a summary; returns 0 when
 * every check passes, 1 otherwise. Model-free: needs no GGUF file and
 * no GPU (the Vulkan section runs only when a Vulkan device is
 * present; the CPU section always runs). */
int vbs_backend_selftest(void);

#ifdef __cplusplus
}
#endif

#endif /* VBS_BACKEND_TEST_H */
