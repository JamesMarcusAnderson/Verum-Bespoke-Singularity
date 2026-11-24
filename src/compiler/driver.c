/* driver.c — Verum embedded C11 compiler, top-level pipeline.
 *
 * One function per output target; every target shares the same front
 * end (lex, parse, typecheck) and differs only in the emitter it hands
 * the checked AST to. Nothing here shells out to an external tool:
 * no clang, no gcc, no assembler, no linker.
 *
 * Targets:
 *   native  — machine code for the host CPU (x86_64 / aarch64)
 *   PTX     — text for the CUDA-style driver path
 *   MSL     — text for the Metal-style driver path
 *   kernel  — binary GPU kernel module (SPIR-V format)
 *   GCN     — binary words for the AMD ISA path
 */

#include "../../include/vbs_compiler.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* GPU emitters (defined in the emit_* files). */
extern int ptx_emit_function(const ast_function_t *fn, char *out, size_t out_sz);
extern int msl_emit_function(const ast_function_t *fn, char *out, size_t out_sz);
extern int kernel_emit_function(const ast_function_t *fn, uint32_t *out, size_t *out_words);
extern int gcn_emit_function(const ast_function_t *fn, uint32_t *out, size_t *out_words);

/* Run the shared front end: lex, parse, typecheck. Returns the checked
 * AST on success (caller frees it), NULL on failure with err_out set.
 * The tag names the target in parse-failure messages. */
static ast_function_t *front_end(const char *src, size_t src_len,
                                 const char *tag, char *err_out, size_t err_sz) {
    vbs_lexer_t *lexer = vbs_lexer_create(src, src_len);
    if (!lexer) {
        snprintf(err_out, err_sz, "driver: lexer alloc failed");
        return NULL;
    }
    vbs_parser_t *parser = vbs_parser_create(lexer);
    if (!parser) {
        vbs_lexer_destroy(lexer);
        snprintf(err_out, err_sz, "driver: parser alloc failed");
        return NULL;
    }
    ast_function_t *fn = vbs_parse_function(parser);
    vbs_parser_destroy(parser);
    vbs_lexer_destroy(lexer);
    if (!fn) {
        if (tag) snprintf(err_out, err_sz, "driver: parse failed (%s)", tag);
        else snprintf(err_out, err_sz, "driver: parse failed");
        return NULL;
    }
    if (vbs_typecheck(fn, err_out, err_sz) != 0) {
        ast_function_free(fn);
        return NULL;
    }
    return fn;
}

/* Compile C11 source to native machine code. The native emitters
 * (x86_64 / aarch64) are not written yet, so this always reports
 * unavailable — but it still runs the front end first, so a caller
 * gets a real parse/type error for bad source instead of a blanket
 * refusal. When the emitters land they plug in here: on success
 * *out_fn holds the executable buffer and *out_size its length, and
 * the caller releases them with vbs_jit_free(). */
int vbs_compile(const char *src, size_t src_len,
                void **out_fn, size_t *out_size,
                char *err_out, size_t err_sz) {
    *out_fn = NULL;
    *out_size = 0;

    ast_function_t *fn = front_end(src, src_len, NULL, err_out, err_sz);
    if (!fn) return -1;
    ast_function_free(fn);

    snprintf(err_out, err_sz,
             "driver: native emit not available (native emitters pending)");
    return -1;
}

/* Compile C11 source to PTX text for the CUDA-style dispatch path. */
int vbs_compile_to_ptx(const char *src, size_t src_len,
                       char *ptx_out, size_t ptx_sz,
                       char *err_out, size_t err_sz) {
    ast_function_t *fn = front_end(src, src_len, "ptx", err_out, err_sz);
    if (!fn) return -1;
    int rc = ptx_emit_function(fn, ptx_out, ptx_sz);
    ast_function_free(fn);
    return rc;
}

/* Compile C11 source to MSL text for the Metal-style dispatch path. */
int vbs_compile_to_msl(const char *src, size_t src_len,
                       char *msl_out, size_t msl_sz,
                       char *err_out, size_t err_sz) {
    ast_function_t *fn = front_end(src, src_len, "msl", err_out, err_sz);
    if (!fn) return -1;
    int rc = msl_emit_function(fn, msl_out, msl_sz);
    ast_function_free(fn);
    return rc;
}

/* Compile C11 source to a binary GPU kernel module (SPIR-V format).
 * *out_words carries the output capacity in words on entry and the
 * number of words written on success. */
int vbs_compile_to_kernel(const char *src, size_t src_len,
                          uint32_t *kernel_out, size_t *out_words,
                          char *err_out, size_t err_sz) {
    ast_function_t *fn = front_end(src, src_len, "kernel", err_out, err_sz);
    if (!fn) return -1;
    int rc = kernel_emit_function(fn, kernel_out, out_words);
    ast_function_free(fn);
    return rc;
}

/* Compile C11 source to GCN words for the AMD ISA path. */
int vbs_compile_to_gcn(const char *src, size_t src_len,
                       uint32_t *gcn_out, size_t *out_words,
                       char *err_out, size_t err_sz) {
    ast_function_t *fn = front_end(src, src_len, "gcn", err_out, err_sz);
    if (!fn) return -1;
    int rc = gcn_emit_function(fn, gcn_out, out_words);
    ast_function_free(fn);
    return rc;
}
