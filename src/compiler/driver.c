/* =============================================================================
 * driver.c
 * Verum-Bespoke-Singularity embedded C11 compiler — top-level pipeline
 *
 * Ties the full compiler pipeline together:
 *   source text → lexer → parser → typecheck → native emit → fn pointer
 *
 * This is the single entry point vbs_jit.c calls.
 * It never touches external tools — no clang, no gcc, no as, no ld.
 * =========================================================================== */

#include "../../include/vbs_compiler.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* -------------------------------------------------------------------------- */
/* Platform emit dispatch                                                      */
/* -------------------------------------------------------------------------- */

#if defined(__aarch64__)
    extern int aarch64_emit_function(const ast_function_t *fn,
                                     void **out_fn, size_t *out_size);
    #define NATIVE_EMIT aarch64_emit_function
#else
    extern int x86_64_emit_function(const ast_function_t *fn,
                                    void **out_fn, size_t *out_size);
    #define NATIVE_EMIT x86_64_emit_function
#endif

/* -------------------------------------------------------------------------- */
/* GPU IR emit dispatch                                                        */
/* -------------------------------------------------------------------------- */

extern int ptx_emit_function   (const ast_function_t *fn,
                                 char *out, size_t out_sz);
extern int msl_emit_function   (const ast_function_t *fn,
                                 char *out, size_t out_sz);
extern int spirv_emit_function (const ast_function_t *fn,
                                 uint32_t *out, size_t *out_words);
extern int gcn_emit_function   (const ast_function_t *fn,
                                 uint32_t *out, size_t *out_words);

/* -------------------------------------------------------------------------- */
/* Public: vbs_compile                                                         */
/*                                                                             */
/* Compiles C11 source to native machine code in one shot.                    */
/* Returns 0 on success, -1 on any error.                                     */
/* out_fn and out_size are set on success — caller must vbs_jit_free() them.  */
/* -------------------------------------------------------------------------- */

int vbs_compile(const char *src, size_t src_len,
                void **out_fn, size_t *out_size,
                char *err_out, size_t err_sz)
{
    *out_fn   = NULL;
    *out_size = 0;

    /* ---- 1. Lex ---------------------------------------------------------- */
    vbs_lexer_t *lexer = vbs_lexer_create(src, src_len);
    if (!lexer) {
        snprintf(err_out, err_sz, "driver: lexer alloc failed");
        return -1;
    }

    /* ---- 2. Parse -------------------------------------------------------- */
    vbs_parser_t *parser = vbs_parser_create(lexer);
    if (!parser) {
        vbs_lexer_destroy(lexer);
        snprintf(err_out, err_sz, "driver: parser alloc failed");
        return -1;
    }

    ast_function_t *fn = vbs_parse_function(parser);
    vbs_parser_destroy(parser);
    vbs_lexer_destroy(lexer);

    if (!fn) {
        snprintf(err_out, err_sz, "driver: parse failed");
        return -1;
    }

    /* ---- 3. Typecheck ---------------------------------------------------- */
    if (vbs_typecheck(fn, err_out, err_sz) != 0) {
        ast_function_free(fn);
        return -1;
    }

    /* ---- 4. Native codegen ----------------------------------------------- */
    int rc = NATIVE_EMIT(fn, out_fn, out_size);
    ast_function_free(fn);

    if (rc != 0) {
        snprintf(err_out, err_sz, "driver: native emit failed (rc=%d)", rc);
        return -1;
    }

    fprintf(stdout,
        "[VBS] compiler: compiled '%.*s' -> %zu bytes @ %p\n",
        64, src, *out_size, *out_fn);

    return 0;
}

/* -------------------------------------------------------------------------- */
/* Public: vbs_compile_to_ptx                                                 */
/* Compiles C11 source to PTX text for CUDA dispatch                          */
/* -------------------------------------------------------------------------- */

int vbs_compile_to_ptx(const char *src, size_t src_len,
                        char *ptx_out, size_t ptx_sz,
                        char *err_out, size_t err_sz)
{
    vbs_lexer_t    *lexer  = vbs_lexer_create(src, src_len);
    vbs_parser_t   *parser = vbs_parser_create(lexer);
    ast_function_t *fn     = vbs_parse_function(parser);
    vbs_parser_destroy(parser);
    vbs_lexer_destroy(lexer);
    if (!fn) { snprintf(err_out, err_sz, "driver: parse failed (ptx)"); return -1; }
    if (vbs_typecheck(fn, err_out, err_sz) != 0) { ast_function_free(fn); return -1; }
    int rc = ptx_emit_function(fn, ptx_out, ptx_sz);
    ast_function_free(fn);
    return rc;
}

/* -------------------------------------------------------------------------- */
/* Public: vbs_compile_to_msl                                                 */
/* Compiles C11 source to MSL text for Metal dispatch                         */
/* -------------------------------------------------------------------------- */

int vbs_compile_to_msl(const char *src, size_t src_len,
                        char *msl_out, size_t msl_sz,
                        char *err_out, size_t err_sz)
{
    vbs_lexer_t    *lexer  = vbs_lexer_create(src, src_len);
    vbs_parser_t   *parser = vbs_parser_create(lexer);
    ast_function_t *fn     = vbs_parse_function(parser);
    vbs_parser_destroy(parser);
    vbs_lexer_destroy(lexer);
    if (!fn) { snprintf(err_out, err_sz, "driver: parse failed (msl)"); return -1; }
    if (vbs_typecheck(fn, err_out, err_sz) != 0) { ast_function_free(fn); return -1; }
    int rc = msl_emit_function(fn, msl_out, msl_sz);
    ast_function_free(fn);
    return rc;
}

/* -------------------------------------------------------------------------- */
/* Public: vbs_compile_to_spirv                                                */
/* -------------------------------------------------------------------------- */

int vbs_compile_to_spirv(const char *src, size_t src_len,
                          uint32_t *spirv_out, size_t *out_words,
                          char *err_out, size_t err_sz)
{
    vbs_lexer_t    *lexer  = vbs_lexer_create(src, src_len);
    vbs_parser_t   *parser = vbs_parser_create(lexer);
    ast_function_t *fn     = vbs_parse_function(parser);
    vbs_parser_destroy(parser);
    vbs_lexer_destroy(lexer);
    if (!fn) { snprintf(err_out, err_sz, "driver: parse failed (spirv)"); return -1; }
    if (vbs_typecheck(fn, err_out, err_sz) != 0) { ast_function_free(fn); return -1; }
    int rc = spirv_emit_function(fn, spirv_out, out_words);
    ast_function_free(fn);
    return rc;
}

/* -------------------------------------------------------------------------- */
/* Public: vbs_compile_to_gcn                                                 */
/* -------------------------------------------------------------------------- */

int vbs_compile_to_gcn(const char *src, size_t src_len,
                        uint32_t *gcn_out, size_t *out_words,
                        char *err_out, size_t err_sz)
{
    vbs_lexer_t    *lexer  = vbs_lexer_create(src, src_len);
    vbs_parser_t   *parser = vbs_parser_create(lexer);
    ast_function_t *fn     = vbs_parse_function(parser);
    vbs_parser_destroy(parser);
    vbs_lexer_destroy(lexer);
    if (!fn) { snprintf(err_out, err_sz, "driver: parse failed (gcn)"); return -1; }
    if (vbs_typecheck(fn, err_out, err_sz) != 0) { ast_function_free(fn); return -1; }
    int rc = gcn_emit_function(fn, gcn_out, out_words);
    ast_function_free(fn);
    return rc;
}