/* native_emit_stub.c — placeholder native emitters.
 *
 * driver.c's vbs_compile() calls x86_64_emit_function /
 * aarch64_emit_function, which do not exist yet in the repo (the
 * Dropbox native emitters were written against a different AST and are
 * HELD for a rewrite). Without these symbols the compiler driver
 * cannot link at all — which would block the GPU emitters
 * (vbs_compile_to_spirv/ptx/msl/gcn) that live in the same file.
 *
 * These stubs return -1 ("native emit not available"), so vbs_compile()
 * fails cleanly at runtime instead of failing the link. They are WEAK
 * symbols (GCC/Clang on ELF/Mach-O): when the real native emitters land,
 * their strong definitions override these automatically and this file can
 * be deleted. On Windows (PE/COFF) weak definitions do NOT satisfy strong
 * references under MinGW ld, so the stubs are plain strong definitions
 * there — the real emitters must then replace this file instead of
 * coexisting with it.
 */

#include "../../include/vbs_compiler.h"

#if (defined(__GNUC__) || defined(__clang__)) && !defined(_WIN32)
#define VBS_WEAK __attribute__((weak))
#else
#define VBS_WEAK
#endif

VBS_WEAK int x86_64_emit_function(const ast_function_t *fn,
                                  void **out_fn, size_t *out_size) {
    (void)fn; (void)out_fn; (void)out_size;
    /* Native x86_64 emit not available yet (held emitters are being
     * rewritten against the vbs_compiler.h AST). */
    return -1;
}

VBS_WEAK int aarch64_emit_function(const ast_function_t *fn,
                                   void **out_fn, size_t *out_size) {
    (void)fn; (void)out_fn; (void)out_size;
    /* Native AArch64 emit not available yet (held emitters are being
     * rewritten against the vbs_compiler.h AST). */
    return -1;
}
