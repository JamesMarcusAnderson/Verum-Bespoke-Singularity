/* vbs_cc.h — system-compiler invocation for generated code.
 *
 * The recursive loop (candidate kernels) and the fuser (per-model
 * fused kernels) both compile generated C11 with the system
 * compiler, the same way the GPU specializer does
 * (vbs_specializer_compile). This helper is the single choke point:
 * no shell is involved (fork/execvp with an argv array), stderr is
 * captured and returned, and the caller decides what to do with a
 * failure.
 *
 * This is deliberately NOT the embedded vbs_compile() JIT: the
 * native emitters it needs (x86_64 / aarch64) are not written yet,
 * so vbs_compile() reports unavailable. When the native-emitter
 * rewrite lands, this helper is the seam to replace: candidates
 * and fused kernels are plain C11 sources, and either compiler can
 * consume them.
 */
#ifndef VBS_CC_H
#define VBS_CC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compile one C source file into a shared object.
 *   cc: compiler binary (NULL means vbs_cc_default())
 *   extra_argv: extra flags, e.g. {"-mavx2", "-Iinclude", "-Isrc/hotkernels"}
 * Returns 0 on success. On failure returns the compiler's exit
 * status (or -1 on spawn/wait failure) and fills err_out with the
 * captured stderr (truncated to err_cap). */
int vbs_cc_compile_so(const char *cc, const char *src, const char *so,
                      const char *const *extra_argv, size_t n_extra,
                      char *err_out, size_t err_cap);

/* The compiler to use: $CC when set and non-empty, else "cc". */
const char *vbs_cc_default(void);

#ifdef __cplusplus
}
#endif

#endif /* VBS_CC_H */
