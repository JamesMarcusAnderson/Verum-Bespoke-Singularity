/* =============================================================================
 * gcn_emit.c
 * Verum-Bespoke-Singularity — AST → GCN ISA emitter (AMD gfx900 / Vega)
 *
 * Emits GCN (Graphics Core Next) machine code words for AMD GPUs.
 * Target: gfx900 (Vega10) — ISA compatible with gfx906/gfx908/gfx90a.
 * Output is a flat array of 32-bit instruction words ready for PM4 dispatch
 * via the CP ring in skynet/phase2/gpu.c.
 *
 * Instruction encoding reference: AMD GCN ISA 3.0 manual
 * =========================================================================== */

#include "../../include/vbs_compiler.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>

/* -------------------------------------------------------------------------- */
/* GCN instruction encodings (scalar and vector)                              */
/* -------------------------------------------------------------------------- */

/* SOP1: scalar operation, 1 src */
#define GCN_SOP1(op, sdst, ssrc0) \
    (0xBE800000u | ((op) << 8) | ((sdst) << 16) | (ssrc0))

/* SOP2: scalar operation, 2 src */
#define GCN_SOP2(op, sdst, ssrc0, ssrc1) \
    (((op) << 23) | ((sdst) << 16) | ((ssrc1) << 8) | (ssrc0))

/* VOP2: vector operation, 2 src */
#define GCN_VOP2(op, vdst, vsrc0, vsrc1) \
    (((op) << 25) | ((vdst) << 17) | ((vsrc1) << 9) | (vsrc0))

/* VOP1: vector operation, 1 src */
#define GCN_VOP1(op, vdst, vsrc0) \
    (0x7E000000u | ((op) << 9) | ((vdst) << 17) | (vsrc0))

/* FLAT: global memory load/store */
#define GCN_FLAT_LOAD_DWORD(vdst, vaddr)  \
    (0xDC500000u | ((vdst) << 24) | (vaddr))
#define GCN_FLAT_STORE_DWORD(vdata, vaddr) \
    (0xDC700000u | ((vdata) << 24) | (vaddr))

/* S_ENDPGM: end kernel */
#define GCN_S_ENDPGM  0xBF810000u

/* V_ADD_F32: vdst = vsrc0 + vsrc1 */
#define GCN_V_ADD_F32(vdst, vsrc0, vsrc1)  GCN_VOP2(0x01, vdst, vsrc0, vsrc1)
/* V_MUL_F32 */
#define GCN_V_MUL_F32(vdst, vsrc0, vsrc1)  GCN_VOP2(0x04, vdst, vsrc0, vsrc1)
/* V_SUB_F32 */
#define GCN_V_SUB_F32(vdst, vsrc0, vsrc1)  GCN_VOP2(0x03, vdst, vsrc0, vsrc1)
/* V_MOV_B32: move immediate or register */
#define GCN_V_MOV_B32(vdst, vsrc)          GCN_VOP1(0x01, vdst, vsrc)
/* S_MOV_B32: scalar move */
#define GCN_S_MOV_B32(sdst, ssrc)          GCN_SOP1(0x03, sdst, ssrc)
/* S_ADD_U32 */
#define GCN_S_ADD_U32(sdst, ssrc0, ssrc1)  GCN_SOP2(0x00, sdst, ssrc0, ssrc1)

/* VGPR / SGPR base indices */
#define VGPR(n)  (n)
#define SGPR(n)  (n)

/* -------------------------------------------------------------------------- */
/* Output buffer                                                               */
/* -------------------------------------------------------------------------- */

typedef struct { uint32_t *buf; size_t pos; size_t cap; } gbuf_t;

static void gb_init(gbuf_t *b, uint32_t *buf, size_t cap)
    { b->buf = buf; b->pos = 0; b->cap = cap; }

static void gb_emit(gbuf_t *b, uint32_t word)
    { if (b->pos < b->cap) b->buf[b->pos++] = word; }

/* -------------------------------------------------------------------------- */
/* Expression emitter                                                          */
/* -------------------------------------------------------------------------- */

static int g_vreg = 4;   /* v0-v3 reserved for kernel args */

static int emit_expr_gcn(gbuf_t *b, ast_node_t *n)
{
    if (!n) return VGPR(0);
    switch (n->kind) {
        case AST_INT_LIT:
        case AST_FLOAT_LIT: {
            int dst = g_vreg++;
            /* V_MOV_B32 with inline constant — for simplicity use reg 0 */
            gb_emit(b, GCN_V_MOV_B32(VGPR(dst), 0));
            return VGPR(dst);
        }
        case AST_VAR_REF: {
            /* Map first few param names to VGPRs by index */
            return VGPR(g_vreg - 1);
        }
        case AST_BINOP: {
            int lreg = emit_expr_gcn(b, n->u.binop.lhs);
            int rreg = emit_expr_gcn(b, n->u.binop.rhs);
            int dst  = g_vreg++;
            switch (n->u.binop.op) {
                case '+': gb_emit(b, GCN_V_ADD_F32(VGPR(dst), lreg, rreg)); break;
                case '-': gb_emit(b, GCN_V_SUB_F32(VGPR(dst), lreg, rreg)); break;
                case '*': gb_emit(b, GCN_V_MUL_F32(VGPR(dst), lreg, rreg)); break;
                default:  gb_emit(b, GCN_V_ADD_F32(VGPR(dst), lreg, rreg)); break;
            }
            return VGPR(dst);
        }
        case AST_SUBSCRIPT: {
            int base = emit_expr_gcn(b, n->u.binop.lhs);
            /* FLAT_LOAD_DWORD: load from global memory */
            int dst = g_vreg++;
            gb_emit(b, GCN_FLAT_LOAD_DWORD(VGPR(dst), base));
            gb_emit(b, 0x00000000u); /* FLAT needs 64-bit word — padding */
            return VGPR(dst);
        }
        default:
            return VGPR(0);
    }
}

/* -------------------------------------------------------------------------- */
/* Statement emitter                                                           */
/* -------------------------------------------------------------------------- */

static void emit_stmt_gcn(gbuf_t *b, ast_node_t *n)
{
    if (!n) return;
    switch (n->kind) {
        case AST_BLOCK:
            for (int i = 0; i < n->u.block.n; i++)
                emit_stmt_gcn(b, n->u.block.stmts[i]);
            break;
        case AST_DECL:
            g_vreg++;   /* allocate VGPR slot */
            if (n->u.decl.init) emit_expr_gcn(b, n->u.decl.init);
            break;
        case AST_RETURN:
            if (n->u.ret.value) emit_expr_gcn(b, n->u.ret.value);
            gb_emit(b, GCN_S_ENDPGM);
            break;
        case AST_FOR:
            if (n->u.for_s.init) emit_stmt_gcn(b, n->u.for_s.init);
            /* GCN loop: simplified linear unroll placeholder */
            emit_stmt_gcn(b, n->u.for_s.body);
            break;
        case AST_EXPR_STMT:
            emit_expr_gcn(b, n->u.expr_stmt.e);
            break;
        case AST_ASSIGN: {
            int src = emit_expr_gcn(b, n->u.binop.rhs);
            /* FLAT_STORE_DWORD back to destination */
            if (n->u.binop.lhs && n->u.binop.lhs->kind == AST_SUBSCRIPT) {
                int base = emit_expr_gcn(b, n->u.binop.lhs->u.binop.lhs);
                gb_emit(b, GCN_FLAT_STORE_DWORD(VGPR(src), base));
                gb_emit(b, 0x00000000u);
            }
            break;
        }
        default: break;
    }
}

/* -------------------------------------------------------------------------- */
/* Public API                                                                  */
/* -------------------------------------------------------------------------- */

int gcn_emit_function(const ast_function_t *fn,
                       uint32_t *out, size_t *out_words)
{
    if (!out || !out_words || *out_words < 16) return -1;

    gbuf_t b;
    gb_init(&b, out, *out_words);
    g_vreg = fn->param_count;  /* params live in v0..vN */

    /* Kernel prologue: load 64-bit base addresses from SGPRs into VGPRs */
    for (int i = 0; i < fn->param_count && i < 8; i++)
        gb_emit(&b, GCN_V_MOV_B32(VGPR(i), SGPR(i)));

    emit_stmt_gcn(&b, fn->body);

    /* Ensure S_ENDPGM is always present */
    gb_emit(&b, GCN_S_ENDPGM);

    *out_words = b.pos;
    return (b.pos > 0) ? 0 : -1;
}