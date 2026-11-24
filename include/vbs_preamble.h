/* vbs_preamble.h — the prologue every Verum source shares.
 *
 * One place for the C library headers nearly every file needs and
 * for the C++ linkage guards, so no file repeats the same six-line
 * prologue. A header or source file includes this first, then its
 * own project headers, then opens VBS_EXTERN_C_BEGIN (headers
 * only) around its declarations.
 *
 * Files that need library headers beyond these (errno.h, stdarg.h,
 * limits.h, ...) include those directly.
 */
#ifndef VBS_PREAMBLE_H
#define VBS_PREAMBLE_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __cplusplus
#define VBS_EXTERN_C_BEGIN extern "C" {
#define VBS_EXTERN_C_END }
#else
#define VBS_EXTERN_C_BEGIN
#define VBS_EXTERN_C_END
#endif

#endif /* VBS_PREAMBLE_H */
