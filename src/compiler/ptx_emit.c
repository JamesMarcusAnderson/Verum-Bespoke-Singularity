/* =============================================================================
 * ptx_emit.c
 * Verum-Bespoke-Singularity — AST → PTX text emitter (CUDA)
 *
 * Emits PTX 7.0 text from an ast_function_t.
 * The PTX string is handed to the CUDA driver via cuModuleLoadDataEx()
 * in vbs_backend_cuda.c — no nvcc, no cicc, no fatbin.
 *
 * Target: sm_80 (Ampere) — works on any sm_70+ (Volta/Turing/Ampere/Ada)
 * =========================================================================== */

#include "../../include/vbs_compiler.h"
#include <string.h>
#include <stdio.h>

/* -------------------------------------------------------------------------- */
/* Output buffer helpers                                                       */
/* -------------------------------------------------------------------------- */

typedef struct { char *buf; size_t pos; size_t cap; } pbuf_t;

static void pb_init(pbuf_t *b, char *buf, size_t cap)
    { b->buf = buf; b->pos = 0; b->cap = cap; }

static void pb_write(pbuf_t *b, const char *s)
{
    size_t n = strlen(s);
    if (b->pos + n + 1 < b->cap) {
        memcpy(b->buf + b->pos, s, n);
        b->pos += n;
        b->buf[b->pos] = '\0';
    }
}

static void pb_fmt(pbuf_t *b, const char *fmt, ...)
{
    char tmp[512];
    va_list ap;
    va_start(ap, fmt); vsnprintf(tmp, sizeof(tmp), fmt, ap); va_end(ap);
    pb_write(b, tmp);
}

#include <stdarg.h>

/* -------------------------------------------------------------------------- */
/* Type → PTX type string                                                      */
/* -------------------------------------------------------------------------- */

static const char *ptx_type(ast_type_t t)
{
    if (t.ptr_depth > 0) return ".u64";
    switch (t.kind) {
        case AST_TY_INT:    return ".s32";
        case AST_TY_FLOAT:  return ".f32";
        case AST_TY_DOUBLE: return ".f64";
        case AST_TY_CHAR:   return ".s8";
        case AST_TY_SIZE_T: return ".u64";
        default:            return ".s32";
    }
}

static const char *ptx_reg_prefix(ast_type_t t)
{
    if (t.ptr_depth > 0) return "%rd";
    switch (t.kind) {
        case AST_TY_FLOAT:  return "%f";
        case AST_TY_DOUBLE: return "%fd";
        default:            return "%r";
    }
}

/* -------------------------------------------------------------------------- */
/* Expression emitter                                                          */
/* -------------------------------------------------------------------------- */

static int g_tmp;

static void emit_expr_ptx(pbuf_t *b, ast_node_t *n, const char *dst_reg)
{
    if (!n) return;
    switch (n->kind) {
        case AST_INT_LIT:
            pb_fmt(b, "    mov.s32 %s, %lld;\n", dst_reg, n->u.int_val);
            break;
        case AST_FLOAT_LIT:
            pb_fmt(b, "    mov.f32 %s, 0f%08X;\n", dst_reg,
                   *(unsigned int *)&n->u.float_val);
            break;
        case AST_VAR_REF:
            pb_fmt(b, "    mov.s32 %s, %%%s;\n", dst_reg, n->u.var.name);
            break;
        case AST_BINOP: {
            char lreg[32], rreg[32];
            snprintf(lreg, sizeof(lreg), "%%tmp%d", g_tmp++);
            snprintf(rreg, sizeof(rreg), "%%tmp%d", g_tmp++);
            pb_fmt(b, "    .reg .f32 %s;\n", lreg);
            pb_fmt(b, "    .reg .f32 %s;\n", rreg);
            emit_expr_ptx(b, n->u.binop.lhs, lreg);
            emit_expr_ptx(b, n->u.binop.rhs, rreg);
            const char *op = "add";
            switch (n->u.binop.op) {
                case '+': op = "add"; break;
                case '-': op = "sub"; break;
                case '*': op = "mul"; break;
                case '/': op = "div.rn"; break;
                default:  op = "add"; break;
            }
            pb_fmt(b, "    %s.f32 %s, %s, %s;\n", op, dst_reg, lreg, rreg);
            break;
        }
        default:
            pb_fmt(b, "    mov.s32 %s, 0;\n", dst_reg);
            break;
    }
}

/* -------------------------------------------------------------------------- */
/* Statement emitter                                                           */
/* -------------------------------------------------------------------------- */

static void emit_stmt_ptx(pbuf_t *b, ast_node_t *n);

static void emit_stmt_ptx(pbuf_t *b, ast_node_t *n)
{
    if (!n) return;
    switch (n->kind) {
        case AST_BLOCK:
            for (int i = 0; i < n->u.block.n; i++)
                emit_stmt_ptx(b, n->u.block.stmts[i]);
            break;
        case AST_DECL: {
            const char *ty = ptx_type(n->u.decl.type);
            pb_fmt(b, "    .reg %s %%%s;\n", ty, n->u.decl.name);
            if (n->u.decl.init) {
                char dst[64];
                snprintf(dst, sizeof(dst), "%%%s", n->u.decl.name);
                emit_expr_ptx(b, n->u.decl.init, dst);
            }
            break;
        }
        case AST_RETURN:
            if (n->u.ret.value) {
                char tmp[32];
                snprintf(tmp, sizeof(tmp), "%%retval%d", g_tmp++);
                pb_fmt(b, "    .reg .f32 %s;\n", tmp);
                emit_expr_ptx(b, n->u.ret.value, tmp);
            }
            pb_write(b, "    ret;\n");
            break;
        case AST_FOR: {
            int lbl = g_tmp++;
            pb_fmt(b, "    bra.uni FOR_COND_%d;\n", lbl);
            pb_fmt(b, "FOR_BODY_%d:\n", lbl);
            emit_stmt_ptx(b, n->u.for_s.body);
            if (n->u.for_s.post) emit_stmt_ptx(b, n->u.for_s.post);
            pb_fmt(b, "FOR_COND_%d:\n", lbl);
            /* Simplified: always branch back — full cond eval omitted */
            pb_fmt(b, "    bra.uni FOR_BODY_%d;\n", lbl);
            pb_fmt(b, "FOR_END_%d:\n", lbl);
            break;
        }
        case AST_EXPR_STMT:
        case AST_ASSIGN: {
            char tmp[32];
            snprintf(tmp, sizeof(tmp), "%%etmp%d", g_tmp++);
            pb_fmt(b, "    .reg .f32 %s;\n", tmp);
            emit_expr_ptx(b, n->u.expr_stmt.e ? n->u.expr_stmt.e : n, tmp);
            break;
        }
        default: break;
    }
}

/* -------------------------------------------------------------------------- */
/* Public API                                                                  */
/* -------------------------------------------------------------------------- */

int ptx_emit_function(const ast_function_t *fn, char *out, size_t out_sz)
{
    pbuf_t b;
    pb_init(&b, out, out_sz);
    g_tmp = 0;

    pb_write(&b, ".version 7.0\n");
    pb_write(&b, ".target sm_80\n");
    pb_write(&b, ".address_size 64\n\n");

    pb_fmt(&b, ".visible .entry %s(\n", fn->name);
    for (int i = 0; i < fn->param_count; i++) {
        const char *ty = ptx_type(fn->params[i].type);
        pb_fmt(&b, "    .param %s param_%s%s\n",
               ty, fn->params[i].name,
               i + 1 < fn->param_count ? "," : "");
    }
    pb_write(&b, ")\n{\n");

    /* Load params from .param space into registers */
    for (int i = 0; i < fn->param_count; i++) {
        const char *ty  = ptx_type(fn->params[i].type);
        pb_fmt(&b, "    .reg %s %%%s;\n", ty, fn->params[i].name);
        pb_fmt(&b, "    ld.param%s %%%s, [param_%s];\n",
               ty, fn->params[i].name, fn->params[i].name);
    }
    pb_write(&b, "\n");

    emit_stmt_ptx(&b, fn->body);

    pb_write(&b, "    ret;\n}\n");
    return (b.pos > 0 && b.pos < out_sz) ? 0 : -1;
}