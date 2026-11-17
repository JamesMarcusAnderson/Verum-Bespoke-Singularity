/* =============================================================================
 * msl_emit.c
 * Verum-Bespoke-Singularity — AST → MSL text emitter (Metal)
 *
 * Emits Metal Shading Language source from an ast_function_t.
 * The MSL string is handed to MTLDevice newLibraryWithSource:options:error:
 * in vbs_backend_metal.c — no xcrun, no metallib, no offline compilation.
 * =========================================================================== */

#include "../../include/vbs_compiler.h"
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

typedef struct { char *buf; size_t pos; size_t cap; } mbuf_t;

static void mb_init(mbuf_t *b, char *buf, size_t cap)
    { b->buf = buf; b->pos = 0; b->cap = cap; }

static void mb_write(mbuf_t *b, const char *s)
{
    size_t n = strlen(s);
    if (b->pos + n + 1 < b->cap) {
        memcpy(b->buf + b->pos, s, n);
        b->pos += n;
        b->buf[b->pos] = '\0';
    }
}

static void mb_fmt(mbuf_t *b, const char *fmt, ...)
{
    char tmp[512];
    va_list ap;
    va_start(ap, fmt); vsnprintf(tmp, sizeof(tmp), fmt, ap); va_end(ap);
    mb_write(b, tmp);
}

static const char *msl_type(ast_type_t t)
{
    if (t.ptr_depth > 0) {
        switch (t.kind) {
            case AST_TY_FLOAT:  return "device float*";
            case AST_TY_INT:    return "device int*";
            default:            return "device void*";
        }
    }
    switch (t.kind) {
        case AST_TY_VOID:   return "void";
        case AST_TY_INT:    return "int";
        case AST_TY_FLOAT:  return "float";
        case AST_TY_DOUBLE: return "float"; /* Metal has no double by default */
        case AST_TY_CHAR:   return "char";
        case AST_TY_SIZE_T: return "uint";
        default:            return "int";
    }
}

static void emit_expr_msl(mbuf_t *b, ast_node_t *n)
{
    if (!n) return;
    switch (n->kind) {
        case AST_INT_LIT:   mb_fmt(b, "%lld", n->u.int_val);   break;
        case AST_FLOAT_LIT: mb_fmt(b, "%ff",  n->u.float_val); break;
        case AST_VAR_REF:   mb_write(b, n->u.var.name);        break;
        case AST_SUBSCRIPT:
            emit_expr_msl(b, n->u.binop.lhs);
            mb_write(b, "[");
            emit_expr_msl(b, n->u.binop.rhs);
            mb_write(b, "]");
            break;
        case AST_BINOP:
            emit_expr_msl(b, n->u.binop.lhs);
            switch (n->u.binop.op) {
                case '+': mb_write(b, " + "); break;
                case '-': mb_write(b, " - "); break;
                case '*': mb_write(b, " * "); break;
                case '/': mb_write(b, " / "); break;
                case '%': mb_write(b, " % "); break;
                case '<': mb_write(b, " < "); break;
                case '>': mb_write(b, " > "); break;
                case 'L': mb_write(b, " <= "); break;
                case 'G': mb_write(b, " >= "); break;
                case '=': mb_write(b, " == "); break;
                case '!': mb_write(b, " != "); break;
                default:  mb_write(b, " + "); break;
            }
            emit_expr_msl(b, n->u.binop.rhs);
            break;
        case AST_UNARY:
            mb_fmt(b, "%c", n->u.unary.op);
            emit_expr_msl(b, n->u.unary.operand);
            break;
        case AST_ASSIGN:
            emit_expr_msl(b, n->u.binop.lhs);
            mb_write(b, " = ");
            emit_expr_msl(b, n->u.binop.rhs);
            break;
        case AST_CALL:
            mb_fmt(b, "%s(", n->u.call.name);
            for (int i = 0; i < n->u.call.argc; i++) {
                if (i) mb_write(b, ", ");
                emit_expr_msl(b, n->u.call.args[i]);
            }
            mb_write(b, ")");
            break;
        default: break;
    }
}

static void emit_stmt_msl(mbuf_t *b, ast_node_t *n, int indent)
{
    if (!n) return;
    char pad[64] = {0};
    for (int i = 0; i < indent && i < 60; i++) pad[i] = ' ';

    switch (n->kind) {
        case AST_BLOCK:
            mb_write(b, " {\n");
            for (int i = 0; i < n->u.block.n; i++)
                emit_stmt_msl(b, n->u.block.stmts[i], indent + 4);
            mb_fmt(b, "%s}", pad);
            break;
        case AST_DECL:
            mb_fmt(b, "%s%s %s", pad, msl_type(n->u.decl.type), n->u.decl.name);
            if (n->u.decl.init) {
                mb_write(b, " = ");
                emit_expr_msl(b, n->u.decl.init);
            }
            mb_write(b, ";\n");
            break;
        case AST_RETURN:
            mb_fmt(b, "%sreturn", pad);
            if (n->u.ret.value) {
                mb_write(b, " ");
                emit_expr_msl(b, n->u.ret.value);
            }
            mb_write(b, ";\n");
            break;
        case AST_IF:
            mb_fmt(b, "%sif (", pad);
            emit_expr_msl(b, n->u.if_s.cond);
            mb_write(b, ")");
            emit_stmt_msl(b, n->u.if_s.then_body, indent);
            if (n->u.if_s.else_body) {
                mb_write(b, " else");
                emit_stmt_msl(b, n->u.if_s.else_body, indent);
            }
            mb_write(b, "\n");
            break;
        case AST_FOR:
            mb_fmt(b, "%sfor (", pad);
            if (n->u.for_s.init) emit_stmt_msl(b, n->u.for_s.init, 0);
            else mb_write(b, "; ");
            if (n->u.for_s.cond) emit_expr_msl(b, n->u.for_s.cond);
            mb_write(b, "; ");
            if (n->u.for_s.post) emit_expr_msl(b, n->u.for_s.post);
            mb_write(b, ")");
            emit_stmt_msl(b, n->u.for_s.body, indent);
            mb_write(b, "\n");
            break;
        case AST_WHILE:
            mb_fmt(b, "%swhile (", pad);
            emit_expr_msl(b, n->u.while_s.cond);
            mb_write(b, ")");
            emit_stmt_msl(b, n->u.while_s.body, indent);
            mb_write(b, "\n");
            break;
        case AST_EXPR_STMT:
            mb_write(b, pad);
            emit_expr_msl(b, n->u.expr_stmt.e);
            mb_write(b, ";\n");
            break;
        default: break;
    }
}

int msl_emit_function(const ast_function_t *fn, char *out, size_t out_sz)
{
    mbuf_t b;
    mb_init(&b, out, out_sz);

    mb_write(&b, "#include <metal_stdlib>\nusing namespace metal;\n\n");
    mb_fmt(&b,   "kernel void %s(\n", fn->name);

    for (int i = 0; i < fn->param_count; i++) {
        mb_fmt(&b, "    %s %s [[buffer(%d)]]%s\n",
               msl_type(fn->params[i].type),
               fn->params[i].name, i,
               (i + 1 < fn->param_count) ? "," : ",");
    }
    mb_write(&b, "    uint3 tid [[thread_position_in_grid]])\n");
    emit_stmt_msl(&b, fn->body, 0);
    mb_write(&b, "\n");

    return (b.pos > 0 && b.pos < out_sz) ? 0 : -1;
}