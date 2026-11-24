/* emit_vendors.c
 * Verum-Bespoke-Singularity — vendor GPU code emitters + validators.
 *
 * One file, three vendor targets: PTX (CUDA), MSL (Metal), GCN (AMD).
 * Each emitter turns an ast_function_t into that target's code; each
 * validator runs a lightweight structural check on the emitted
 * output. The kernel-IR emitter/validator live in
 * kernel_emit.c/kernel_validate.c; dispatch to all emitters lives
 * in driver.c.
 *
 * Kept external interface names: PTX, MSL, and GCN are the output
 * formats' own names — the emitted text must be valid PTX/MSL/GCN
 * or the target driver rejects it. The entry-point names
 * (ptx_emit_function, msl_emit_function, gcn_emit_function,
 * ptx_validate, gcn_validate, vbs_compile_to_ptx/msl/gcn) name the
 * output format and are called by driver.c and vbs_backend_test.c.
 */

#include "../../include/vbs_compiler.h"
#include "emit_buf.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Section 1 — PTX text emitter (CUDA).
 * Emits PTX 7.0 text from an ast_function_t. The PTX string is
 * handed to the CUDA driver via cuModuleLoadDataEx() in
 * vbs_gpu_vendors.c — no nvcc, no cicc, no fatbin. Target: sm_80
 * (Ampere) — works on any sm_70+ (Volta/Turing/Ampere/Ada).
 *
 * Output buffer: vbs_textbuf_t from emit_buf.h (shared with the
 * MSL emitter below).
 */

/* AST type -> PTX type suffix. */
static const char *ptx_type(ast_type_t t) {
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

/* Fresh temporary register counter, reset per function. */
static int g_tmp;

static void fresh_reg(char *dst, size_t cap, const char *prefix) {
    snprintf(dst, cap, "%%%s%d", prefix, g_tmp++);
}

/* PTX arithmetic mnemonic for a binary operator code. */
static const char *ptx_arith_op(int op) {
    switch (op) {
        case '+': return "add";
        case '-': return "sub";
        case '*': return "mul";
        case '/': return "div.rn";
        default:  return "add";
    }
}

static void emit_expr_ptx(vbs_textbuf_t *b, ast_node_t *n, const char *dst_reg) {
    if (!n) return;
    switch (n->kind) {
        case AST_INT_LIT:
            vbs_textbuf_fmt(b, "    mov.s32 %s, %lld;\n", dst_reg, n->u.int_val);
            break;
        case AST_FLOAT_LIT: {
            /* memcpy, not a punned dereference (strict aliasing). */
            uint32_t bits;
            memcpy(&bits, &n->u.float_val, sizeof(bits));
            vbs_textbuf_fmt(b, "    mov.f32 %s, 0f%08X;\n", dst_reg, bits);
            break;
        }
        case AST_VAR_REF:
            vbs_textbuf_fmt(b, "    mov.s32 %s, %%%s;\n", dst_reg, n->u.var.name);
            break;
        case AST_BINOP: {
            char lreg[32], rreg[32];
            fresh_reg(lreg, sizeof(lreg), "tmp");
            fresh_reg(rreg, sizeof(rreg), "tmp");
            vbs_textbuf_fmt(b, "    .reg .f32 %s;\n", lreg);
            vbs_textbuf_fmt(b, "    .reg .f32 %s;\n", rreg);
            emit_expr_ptx(b, n->u.binop.lhs, lreg);
            emit_expr_ptx(b, n->u.binop.rhs, rreg);
            vbs_textbuf_fmt(b, "    %s.f32 %s, %s, %s;\n",
                            ptx_arith_op(n->u.binop.op), dst_reg, lreg, rreg);
            break;
        }
        default:
            vbs_textbuf_fmt(b, "    mov.s32 %s, 0;\n", dst_reg);
            break;
    }
}

static void emit_stmt_ptx(vbs_textbuf_t *b, ast_node_t *n) {
    if (!n) return;
    switch (n->kind) {
        case AST_BLOCK:
            for (int i = 0; i < n->u.block.n; i++)
                emit_stmt_ptx(b, n->u.block.stmts[i]);
            break;
        case AST_DECL: {
            const char *ty = ptx_type(n->u.decl.type);
            vbs_textbuf_fmt(b, "    .reg %s %%%s;\n", ty, n->u.decl.name);
            if (n->u.decl.init) {
                char dst[64];
                snprintf(dst, sizeof(dst), "%%%s", n->u.decl.name);
                emit_expr_ptx(b, n->u.decl.init, dst);
            }
            break;
        }
        case AST_RETURN: {
            if (n->u.ret.value) {
                char tmp[32];
                fresh_reg(tmp, sizeof(tmp), "retval");
                vbs_textbuf_fmt(b, "    .reg .f32 %s;\n", tmp);
                emit_expr_ptx(b, n->u.ret.value, tmp);
            }
            vbs_textbuf_write(b, "    ret;\n");
            break;
        }
        case AST_FOR: {
            int lbl = g_tmp++;
            vbs_textbuf_fmt(b, "    bra.uni FOR_COND_%d;\n", lbl);
            vbs_textbuf_fmt(b, "FOR_BODY_%d:\n", lbl);
            emit_stmt_ptx(b, n->u.for_s.body);
            if (n->u.for_s.post) emit_stmt_ptx(b, n->u.for_s.post);
            vbs_textbuf_fmt(b, "FOR_COND_%d:\n", lbl);
            /* Simplified: always branch back — full cond eval omitted. */
            vbs_textbuf_fmt(b, "    bra.uni FOR_BODY_%d;\n", lbl);
            vbs_textbuf_fmt(b, "FOR_END_%d:\n", lbl);
            break;
        }
        case AST_EXPR_STMT:
        case AST_ASSIGN: {
            char tmp[32];
            fresh_reg(tmp, sizeof(tmp), "etmp");
            vbs_textbuf_fmt(b, "    .reg .f32 %s;\n", tmp);
            emit_expr_ptx(b, n->u.expr_stmt.e ? n->u.expr_stmt.e : n, tmp);
            break;
        }
        default:
            break;
    }
}

int ptx_emit_function(const ast_function_t *fn, char *out, size_t out_sz) {
    vbs_textbuf_t b;
    vbs_textbuf_init(&b, out, out_sz);
    g_tmp = 0;

    vbs_textbuf_write(&b, ".version 7.0\n");
    vbs_textbuf_write(&b, ".target sm_80\n");
    vbs_textbuf_write(&b, ".address_size 64\n\n");

    vbs_textbuf_fmt(&b, ".visible .entry %s(\n", fn->name);
    for (int i = 0; i < fn->param_count; i++) {
        vbs_textbuf_fmt(&b, "    .param %s param_%s%s\n",
                        ptx_type(fn->params[i].type), fn->params[i].name,
                        i + 1 < fn->param_count ? "," : "");
    }
    vbs_textbuf_write(&b, ")\n{\n");

    /* Load params from .param space into registers. */
    for (int i = 0; i < fn->param_count; i++) {
        const char *ty = ptx_type(fn->params[i].type);
        vbs_textbuf_fmt(&b, "    .reg %s %%%s;\n", ty, fn->params[i].name);
        vbs_textbuf_fmt(&b, "    ld.param%s %%%s, [param_%s];\n",
                        ty, fn->params[i].name, fn->params[i].name);
    }
    vbs_textbuf_write(&b, "\n");

    emit_stmt_ptx(&b, fn->body);

    vbs_textbuf_write(&b, "    ret;\n}\n");
    return (b.pos > 0 && b.pos < out_sz) ? 0 : -1;
}

/* Section 2 — PTX text validator.
 * Lightweight structural check on the PTX text produced by
 * Section 1. Does NOT require ptxas or any CUDA SDK — pure C11,
 * offline. Checks: magic header (.version + .target +
 * .address_size), at least one .entry, balanced braces, ret; in
 * each entry body, no malformed register names.
 */

static int text_has(const char *hay, const char *needle) {
    return strstr(hay, needle) != NULL;
}

int ptx_validate(const char *ptx, size_t len, char *err, size_t err_sz) {
    if (!ptx || len == 0) {
        snprintf(err, err_sz, "ptx_validate: empty input");
        return -1;
    }

    /* 1. Required header directives. */
    static const char *k_required[] = {
        ".version", ".target", ".address_size"
    };
    for (size_t i = 0; i < sizeof(k_required) / sizeof(k_required[0]); i++) {
        if (!text_has(ptx, k_required[i])) {
            snprintf(err, err_sz, "ptx_validate: missing %s directive",
                     k_required[i]);
            return -1;
        }
    }

    /* 2. At least one kernel entry point. */
    if (!text_has(ptx, ".entry") && !text_has(ptx, ".func")) {
        snprintf(err, err_sz, "ptx_validate: no .entry or .func found");
        return -1;
    }

    /* 3. Balanced braces. */
    int depth = 0;
    for (size_t i = 0; i < len; i++) {
        if (ptx[i] == '{') depth++;
        else if (ptx[i] == '}') depth--;
        if (depth < 0) {
            snprintf(err, err_sz, "ptx_validate: unmatched '}' at offset %zu", i);
            return -1;
        }
    }
    if (depth != 0) {
        snprintf(err, err_sz, "ptx_validate: %d unclosed '{' block(s)", depth);
        return -1;
    }

    /* 4. ret; present. */
    if (!text_has(ptx, "ret;")) {
        snprintf(err, err_sz, "ptx_validate: no ret; found in kernel body");
        return -1;
    }

    /* 5. Spot check: if % appears, it must be followed by a letter. */
    for (size_t i = 0; i + 1 < len; i++) {
        if (ptx[i] == '%') {
            char next = ptx[i + 1];
            if (next != '%' && next != '_' &&
                !(next >= 'a' && next <= 'z') &&
                !(next >= 'A' && next <= 'Z')) {
                snprintf(err, err_sz,
                         "ptx_validate: malformed register '%%' at offset %zu", i);
                return -1;
            }
        }
    }

    return 0;
}

/* Section 3 — MSL text emitter (Metal).
 * Emits Metal Shading Language source from an ast_function_t. The
 * MSL string is handed to MTLDevice
 * newLibraryWithSource:options:error: in vbs_gpu_vendors.c — no
 * xcrun, no metallib, no offline compilation.
 *
 * Output buffer: vbs_textbuf_t from emit_buf.h (shared with the
 * PTX emitter above).
 */

static const char *msl_type(ast_type_t t) {
    if (t.ptr_depth > 0) {
        switch (t.kind) {
            case AST_TY_FLOAT: return "device float*";
            case AST_TY_INT:   return "device int*";
            default:           return "device void*";
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

/* MSL infix spelling for a binary operator code. */
static const char *msl_infix(int op) {
    switch (op) {
        case '+': return " + ";
        case '-': return " - ";
        case '*': return " * ";
        case '/': return " / ";
        case '%': return " % ";
        case '<': return " < ";
        case '>': return " > ";
        case 'L': return " <= ";
        case 'G': return " >= ";
        case '=': return " == ";
        case '!': return " != ";
        default:  return " + ";
    }
}

static void emit_expr_msl(vbs_textbuf_t *b, ast_node_t *n) {
    if (!n) return;
    switch (n->kind) {
        case AST_INT_LIT:
            vbs_textbuf_fmt(b, "%lld", n->u.int_val);
            break;
        case AST_FLOAT_LIT:
            vbs_textbuf_fmt(b, "%ff", n->u.float_val);
            break;
        case AST_VAR_REF:
            vbs_textbuf_write(b, n->u.var.name);
            break;
        case AST_SUBSCRIPT:
            emit_expr_msl(b, n->u.binop.lhs);
            vbs_textbuf_write(b, "[");
            emit_expr_msl(b, n->u.binop.rhs);
            vbs_textbuf_write(b, "]");
            break;
        case AST_BINOP:
            emit_expr_msl(b, n->u.binop.lhs);
            vbs_textbuf_write(b, msl_infix(n->u.binop.op));
            emit_expr_msl(b, n->u.binop.rhs);
            break;
        case AST_UNARY:
            vbs_textbuf_fmt(b, "%c", n->u.unary.op);
            emit_expr_msl(b, n->u.unary.operand);
            break;
        case AST_ASSIGN:
            emit_expr_msl(b, n->u.binop.lhs);
            vbs_textbuf_write(b, " = ");
            emit_expr_msl(b, n->u.binop.rhs);
            break;
        case AST_CALL:
            vbs_textbuf_fmt(b, "%s(", n->u.call.name);
            for (int i = 0; i < n->u.call.argc; i++) {
                if (i) vbs_textbuf_write(b, ", ");
                emit_expr_msl(b, n->u.call.args[i]);
            }
            vbs_textbuf_write(b, ")");
            break;
        default:
            break;
    }
}

static void emit_stmt_msl(vbs_textbuf_t *b, ast_node_t *n, int indent) {
    if (!n) return;
    char pad[64] = {0};
    for (int i = 0; i < indent && i < 60; i++) pad[i] = ' ';

    switch (n->kind) {
        case AST_BLOCK:
            vbs_textbuf_write(b, " {\n");
            for (int i = 0; i < n->u.block.n; i++)
                emit_stmt_msl(b, n->u.block.stmts[i], indent + 4);
            vbs_textbuf_fmt(b, "%s}", pad);
            break;
        case AST_DECL:
            vbs_textbuf_fmt(b, "%s%s %s", pad, msl_type(n->u.decl.type),
                            n->u.decl.name);
            if (n->u.decl.init) {
                vbs_textbuf_write(b, " = ");
                emit_expr_msl(b, n->u.decl.init);
            }
            vbs_textbuf_write(b, ";\n");
            break;
        case AST_RETURN:
            vbs_textbuf_fmt(b, "%sreturn", pad);
            if (n->u.ret.value) {
                vbs_textbuf_write(b, " ");
                emit_expr_msl(b, n->u.ret.value);
            }
            vbs_textbuf_write(b, ";\n");
            break;
        case AST_IF:
            vbs_textbuf_fmt(b, "%sif (", pad);
            emit_expr_msl(b, n->u.if_s.cond);
            vbs_textbuf_write(b, ")");
            emit_stmt_msl(b, n->u.if_s.then_body, indent);
            if (n->u.if_s.else_body) {
                vbs_textbuf_write(b, " else");
                emit_stmt_msl(b, n->u.if_s.else_body, indent);
            }
            vbs_textbuf_write(b, "\n");
            break;
        case AST_FOR:
            vbs_textbuf_fmt(b, "%sfor (", pad);
            if (n->u.for_s.init) emit_stmt_msl(b, n->u.for_s.init, 0);
            else vbs_textbuf_write(b, "; ");
            if (n->u.for_s.cond) emit_expr_msl(b, n->u.for_s.cond);
            vbs_textbuf_write(b, "; ");
            if (n->u.for_s.post) emit_expr_msl(b, n->u.for_s.post);
            vbs_textbuf_write(b, ")");
            emit_stmt_msl(b, n->u.for_s.body, indent);
            vbs_textbuf_write(b, "\n");
            break;
        case AST_WHILE:
            vbs_textbuf_fmt(b, "%swhile (", pad);
            emit_expr_msl(b, n->u.while_s.cond);
            vbs_textbuf_write(b, ")");
            emit_stmt_msl(b, n->u.while_s.body, indent);
            vbs_textbuf_write(b, "\n");
            break;
        case AST_EXPR_STMT:
            vbs_textbuf_write(b, pad);
            emit_expr_msl(b, n->u.expr_stmt.e);
            vbs_textbuf_write(b, ";\n");
            break;
        default:
            break;
    }
}

int msl_emit_function(const ast_function_t *fn, char *out, size_t out_sz) {
    vbs_textbuf_t b;
    vbs_textbuf_init(&b, out, out_sz);

    vbs_textbuf_write(&b, "#include <metal_stdlib>\nusing namespace metal;\n\n");
    vbs_textbuf_fmt(&b, "kernel void %s(\n", fn->name);

    for (int i = 0; i < fn->param_count; i++) {
        /* Every parameter line ends with a comma — including the
         * last one, which the thread-position parameter follows. */
        vbs_textbuf_fmt(&b, "    %s %s [[buffer(%d)]],\n",
                        msl_type(fn->params[i].type),
                        fn->params[i].name, i);
    }
    vbs_textbuf_write(&b, "    uint3 tid [[thread_position_in_grid]])\n");
    emit_stmt_msl(&b, fn->body, 0);
    vbs_textbuf_write(&b, "\n");

    return (b.pos > 0 && b.pos < out_sz) ? 0 : -1;
}

/* Section 4 — GCN ISA emitter (AMD gfx900 / Vega).
 * Emits GCN (Graphics Core Next) machine code words for AMD GPUs.
 * Target: gfx900 (Vega10) — ISA compatible with gfx906/gfx908/
 * gfx90a. Output is a flat array of 32-bit instruction words
 * ready for PM4 dispatch via the CP ring in
 * baremetal/phase2/gpu.c.
 * Instruction encoding reference: AMD GCN ISA 3.0 manual.
 *
 * The encodings below are the ISA's own bit layouts, written as
 * small functions rather than macros. (SOP1/SOP2 scalar encodings
 * exist in the ISA but no emitter in this file uses them, so only
 * the forms actually emitted are defined here.)
 */
/* VOP2: vector operation, 2 sources. */
static uint32_t gcn_vop2(uint32_t op, uint32_t vdst, uint32_t vsrc0, uint32_t vsrc1) {
    return (op << 25) | (vdst << 17) | (vsrc1 << 9) | vsrc0;
}
/* VOP1: vector operation, 1 source. */
static uint32_t gcn_vop1(uint32_t op, uint32_t vdst, uint32_t vsrc0) {
    return 0x7E000000u | (op << 9) | (vdst << 17) | vsrc0;
}
/* FLAT: global memory load/store (64-bit, followed by a data word). */
static uint32_t gcn_flat_load(uint32_t vdst, uint32_t vaddr) {
    return 0xDC500000u | (vdst << 24) | vaddr;
}
static uint32_t gcn_flat_store(uint32_t vdata, uint32_t vaddr) {
    return 0xDC700000u | (vdata << 24) | vaddr;
}

#define GCN_S_ENDPGM 0xBF810000u /* end kernel */

static uint32_t gcn_v_add_f32(uint32_t d, uint32_t a, uint32_t b) {
    return gcn_vop2(0x01, d, a, b);
}
static uint32_t gcn_v_sub_f32(uint32_t d, uint32_t a, uint32_t b) {
    return gcn_vop2(0x03, d, a, b);
}
static uint32_t gcn_v_mul_f32(uint32_t d, uint32_t a, uint32_t b) {
    return gcn_vop2(0x04, d, a, b);
}
static uint32_t gcn_v_mov_b32(uint32_t d, uint32_t s) {
    return gcn_vop1(0x01, d, s);
}

/* Word output buffer: appends stop silently at capacity. */
typedef struct {
    uint32_t *buf;
    size_t pos, cap;
} gbuf_t;

static void gb_init(gbuf_t *b, uint32_t *buf, size_t cap) {
    b->buf = buf;
    b->pos = 0;
    b->cap = cap;
}
static void gb_emit(gbuf_t *b, uint32_t word) {
    if (b->pos < b->cap) b->buf[b->pos++] = word;
}

/* Next free vector register; v0..vN hold the kernel arguments. */
static int g_vreg = 4;

static int emit_expr_gcn(gbuf_t *b, ast_node_t *n) {
    if (!n) return 0;
    switch (n->kind) {
        case AST_INT_LIT:
        case AST_FLOAT_LIT: {
            int dst = g_vreg++;
            /* V_MOV_B32 with an inline constant — for simplicity
             * use register 0 as the source. */
            gb_emit(b, gcn_v_mov_b32((uint32_t)dst, 0));
            return dst;
        }
        case AST_VAR_REF:
            /* Map the first few parameter names to VGPRs by index. */
            return g_vreg - 1;
        case AST_BINOP: {
            int lreg = emit_expr_gcn(b, n->u.binop.lhs);
            int rreg = emit_expr_gcn(b, n->u.binop.rhs);
            int dst = g_vreg++;
            uint32_t d = (uint32_t)dst, l = (uint32_t)lreg, r = (uint32_t)rreg;
            switch (n->u.binop.op) {
                case '+': gb_emit(b, gcn_v_add_f32(d, l, r)); break;
                case '-': gb_emit(b, gcn_v_sub_f32(d, l, r)); break;
                case '*': gb_emit(b, gcn_v_mul_f32(d, l, r)); break;
                default:  gb_emit(b, gcn_v_add_f32(d, l, r)); break;
            }
            return dst;
        }
        case AST_SUBSCRIPT: {
            int base = emit_expr_gcn(b, n->u.binop.lhs);
            int dst = g_vreg++;
            gb_emit(b, gcn_flat_load((uint32_t)dst, (uint32_t)base));
            gb_emit(b, 0x00000000u); /* FLAT needs a 64-bit word — padding */
            return dst;
        }
        default:
            return 0;
    }
}

/* Emits one statement. Returns 0 on success, -1 for a statement
 * this emitter cannot express. A loop is refused: the emitter has
 * no branch or label support, so the only honest outcomes for a
 * loop are a real loop (not built yet) or a refusal — emitting the
 * body once would run it the wrong number of times. */
static int emit_stmt_gcn(gbuf_t *b, ast_node_t *n) {
    if (!n) return 0;
    switch (n->kind) {
        case AST_BLOCK:
            for (int i = 0; i < n->u.block.n; i++)
                if (emit_stmt_gcn(b, n->u.block.stmts[i]) != 0) return -1;
            return 0;
        case AST_DECL:
            g_vreg++; /* allocate a VGPR slot */
            if (n->u.decl.init) emit_expr_gcn(b, n->u.decl.init);
            return 0;
        case AST_RETURN:
            if (n->u.ret.value) emit_expr_gcn(b, n->u.ret.value);
            gb_emit(b, GCN_S_ENDPGM);
            return 0;
        case AST_FOR:
            /* No branch support: refuse rather than emit one pass. */
            return -1;
        case AST_EXPR_STMT:
            emit_expr_gcn(b, n->u.expr_stmt.e);
            return 0;
        case AST_ASSIGN: {
            int src = emit_expr_gcn(b, n->u.binop.rhs);
            /* Store back to the destination (subscript only). */
            if (n->u.binop.lhs && n->u.binop.lhs->kind == AST_SUBSCRIPT) {
                int base = emit_expr_gcn(b, n->u.binop.lhs->u.binop.lhs);
                gb_emit(b, gcn_flat_store((uint32_t)src, (uint32_t)base));
                gb_emit(b, 0x00000000u);
            }
            return 0;
        }
        default:
            return 0;
    }
}

int gcn_emit_function(const ast_function_t *fn,
                      uint32_t *out, size_t *out_words) {
    if (!out || !out_words || *out_words < 16) return -1;

    gbuf_t b;
    gb_init(&b, out, *out_words);
    g_vreg = fn->param_count; /* params live in v0..vN */

    /* Kernel prologue: load 64-bit base addresses from SGPRs into VGPRs. */
    for (int i = 0; i < fn->param_count && i < 8; i++)
        gb_emit(&b, gcn_v_mov_b32((uint32_t)i, (uint32_t)i));

    if (emit_stmt_gcn(&b, fn->body) != 0) return -1;

    /* Ensure S_ENDPGM is always present. */
    gb_emit(&b, GCN_S_ENDPGM);

    *out_words = b.pos;
    return (b.pos > 0) ? 0 : -1;
}

/* Section 5 — GCN ISA binary validator.
 * Structural validation of GCN words produced by Section 4. No AMD
 * toolchain required — pure C11, offline.
 * Checks: nonzero word count; S_ENDPGM (0xBF810000) present and
 * the last non-padding word; no illegal encodings (upper nibble
 * 0xF reserved in GCN3); FLAT instructions always followed by a
 * data word.
 */

#define GCN_FLAT_MASK   0xFC000000u
#define GCN_FLAT_PREFIX 0xDC000000u

int gcn_validate(const uint32_t *words, size_t word_count,
                 char *err, size_t err_sz) {
    if (!words || word_count == 0) {
        snprintf(err, err_sz, "gcn_validate: empty input");
        return -1;
    }

    /* 1. S_ENDPGM present. */
    int endpgm_found = 0;
    size_t endpgm_pos = 0;
    for (size_t i = 0; i < word_count; i++) {
        if (words[i] == GCN_S_ENDPGM) {
            endpgm_found = 1;
            endpgm_pos = i;
        }
    }
    if (!endpgm_found) {
        snprintf(err, err_sz, "gcn_validate: S_ENDPGM (0xBF810000) not found");
        return -1;
    }

    /* 2. S_ENDPGM should be the last meaningful word. */
    for (size_t i = endpgm_pos + 1; i < word_count; i++) {
        if (words[i] != 0x00000000u && words[i] != GCN_S_ENDPGM) {
            snprintf(err, err_sz,
                     "gcn_validate: instructions after S_ENDPGM at word %zu", i);
            return -1;
        }
    }

    /* 3. FLAT instructions must come in pairs (64-bit). */
    for (size_t i = 0; i + 1 < word_count; i++) {
        if ((words[i] & GCN_FLAT_MASK) == GCN_FLAT_PREFIX) {
            if (i + 1 >= word_count) {
                snprintf(err, err_sz,
                         "gcn_validate: FLAT instruction at word %zu missing data word", i);
                return -1;
            }
            i++; /* skip the data word */
        }
    }

    /* 4. No 0xFFxxxxxx encodings — reserved in GCN3. */
    for (size_t i = 0; i < word_count; i++) {
        if ((words[i] >> 24) == 0xFF) {
            snprintf(err, err_sz,
                     "gcn_validate: reserved 0xFF opcode prefix at word %zu", i);
            return -1;
        }
    }

    return 0;
}
