/* kernel_emit.c — AST to binary GPU kernel module.
 *
 * Emits a valid SPIR-V 1.5 module for one compute entry point,
 * straight from the SPIR-V specification: a five-word header, the
 * capability and memory model, an entry point with a 1x1x1 local size,
 * the type and constant declarations, and two functions — the source
 * function itself, with its body compiled from the AST, and a void()
 * wrapper that serves as the entry point and calls it with zero
 * arguments. The wrapper exists because a compute entry point must
 * take no parameters and return void, while the source function may
 * take parameters and return a value. The word stream goes directly
 * to the driver's shader module loader — no external assembler, no SDK.
 *
 * Supported subset (straight-line code, int32 and float32 only):
 *   types: int, float. double, char, size_t and pointers are refused.
 *   expressions: integer and float literals, variable reads, the
 *     arithmetic operators + - * / (with int-to-float promotion),
 *     unary minus, logical not on integers, assignment.
 *   statements: blocks, declarations (with or without an initialiser),
 *     expression statements, returns.
 *   names must be unique across the whole function: a local that
 *     shadows a parameter or another local is refused.
 * Refused cleanly with -1, with no module produced: control flow
 * (if/for/while), calls, subscripts, comparisons, logical and
 * bitwise operators, %, and anything outside the supported types.
 * A refused source never yields a module that claims to be something
 * it is not; the caller reports the refusal.
 *
 * Layout notes: constants are collected in a pre-pass because the
 * format puts all constants before the functions; local variables
 * are likewise declared up front because the format requires every
 * variable declaration to lead the entry block. The ID bound in the
 * header is patched after emission to one past the largest ID used,
 * and a module that would overrun the caller's buffer is refused
 * rather than truncated.
 *
 * SPIR-V is the module format's name; its magic word and version are
 * the format's constants, kept literal below. The opcode numbers
 * were cross-checked against the engine's precompiled modules in
 * kernels/gpu_kernels.spv.h; the two conversion opcodes do not occur
 * in any of those modules and come from the specification.
 */

#include "../../include/vbs_compiler.h"
#include <string.h>
#include <stdint.h>
#include <stdio.h>

/* Format constants (SPIR-V specification). */
#define KERNEL_MAGIC    0x07230203u
#define KERNEL_VERSION  0x00010500u   /* 1.5 */
#define KERNEL_GENERATOR 0x56425300u  /* 'VBS\0' */

/* Opcodes used by the emitter (SPIR-V specification). */
enum {
    OP_CAPABILITY   = 17,
    OP_MEMORY_MODEL = 14,
    OP_ENTRY_POINT  = 15,
    OP_EXEC_MODE    = 16,
    OP_TYPE_VOID    = 19,
    OP_TYPE_BOOL    = 20,
    OP_TYPE_INT     = 21,
    OP_TYPE_FLOAT   = 22,
    OP_TYPE_POINTER = 32,
    OP_TYPE_FUNC    = 33,
    OP_CONSTANT     = 43,
    OP_FUNCTION     = 54,
    OP_FUNCTION_PARAM = 55,
    OP_FUNCTION_END = 56,
    OP_FUNCTION_CALL = 57,
    OP_VARIABLE     = 59,
    OP_LOAD         = 61,
    OP_STORE        = 62,
    OP_CONVERT_F_TO_S = 109,
    OP_CONVERT_S_TO_F = 110,
    OP_S_NEGATE     = 126,
    OP_F_NEGATE     = 127,
    OP_I_ADD        = 128,
    OP_F_ADD        = 129,
    OP_I_SUB        = 130,
    OP_F_SUB        = 131,
    OP_I_MUL        = 132,
    OP_F_MUL        = 133,
    OP_S_DIV        = 135,
    OP_F_DIV        = 136,
    OP_SELECT       = 169,
    OP_I_EQUAL      = 170,
    OP_LABEL        = 248,
    OP_RETURN       = 253,
    OP_RETURN_VALUE = 254
};

/* Operand values (SPIR-V specification). */
enum {
    CAP_SHADER     = 1,   /* Shader capability */
    ADDR_LOGICAL   = 0,   /* Logical addressing model */
    MEM_GLSL450    = 1,   /* GLSL450 memory model */
    EXEC_GLCOMPUTE = 5,   /* GLCompute execution model */
    MODE_LOCAL_SIZE = 17, /* LocalSize execution mode */
    STORAGE_FUNCTION = 7  /* Function storage class */
};

/* Emitter capacities. */
#define EMIT_MAX_VARS   (AST_MAX_PARAMS + AST_MAX_STMTS)
#define EMIT_MAX_CONSTS 2048

/* Word stream writer: appends to the caller's buffer and records an
 * overflow instead of silently dropping words past the capacity. */
typedef struct {
    uint32_t *words;
    size_t used;
    size_t cap;
    int overflow;
} stream_t;

static void stream_word(stream_t *s, uint32_t w) {
    if (s->used < s->cap) s->words[s->used++] = w;
    else s->overflow = 1;
}

/* Start one instruction: the first word packs the word count in the
 * high half and the opcode in the low half. */
static void stream_op(stream_t *s, uint32_t opcode, uint32_t count) {
    stream_word(s, (count << 16) | opcode);
}

/* Words a NUL-terminated string occupies when packed four bytes per
 * word, little-endian, NUL-padded. */
static size_t packed_string_words(const char *s) {
    size_t n = strlen(s) + 1;
    return (n + 3) / 4;
}

static void stream_string(stream_t *s, const char *text) {
    size_t n = strlen(text) + 1;
    size_t total = packed_string_words(text) * 4;
    for (size_t off = 0; off < total; off += 4) {
        uint32_t w = 0;
        for (int b = 0; b < 4; b++) {
            size_t idx = off + (size_t)b;
            uint8_t c = idx < n ? (uint8_t)text[idx] : 0;
            w |= (uint32_t)c << (b * 8);
        }
        stream_word(s, w);
    }
}

/* One collected constant: the literal's bits plus the result id it
 * is assigned when the constants section is emitted. */
typedef struct {
    uint32_t id;
    int is_float;
    uint32_t bits;
} const_entry_t;

/* One variable (parameter or local): the AST name, the OpVariable
 * result id, and its kind. Ids are assigned when the variables are
 * emitted at the head of the entry block. */
typedef struct {
    const char *name;
    uint32_t var;
    ast_type_kind_t kind;
} var_entry_t;

typedef struct {
    stream_t *s;
    uint32_t next_id;
    uint32_t t_void, t_bool, t_int, t_float;
    uint32_t tp_int, tp_float;
    uint32_t fn_type;
    uint32_t wrap_type;
    uint32_t fn_id;
    uint32_t wrap_id;
    uint32_t param_ids[AST_MAX_PARAMS];
    ast_type_kind_t ret_kind;
    var_entry_t vars[EMIT_MAX_VARS];
    int n_vars;
    const_entry_t consts[EMIT_MAX_CONSTS];
    int n_consts;
    uint32_t c_int_zero, c_int_one, c_float_zero;
    int failed;
    int returned;
} emit_t;

static uint32_t new_id(emit_t *e) { return e->next_id++; }

static uint32_t type_id(emit_t *e, ast_type_kind_t kind) {
    switch (kind) {
        case AST_TY_VOID: return e->t_void;
        case AST_TY_INT: return e->t_int;
        case AST_TY_FLOAT: return e->t_float;
        default: return 0;
    }
}

static uint32_t ptr_type_id(emit_t *e, ast_type_kind_t kind) {
    return kind == AST_TY_FLOAT ? e->tp_float : e->tp_int;
}

/* The emitter supports int and float values only; anything else in
 * a signature or declaration is refused. */
static int supported_type(ast_type_t t) {
    if (t.ptr_depth != 0) return 0;
    return t.kind == AST_TY_INT || t.kind == AST_TY_FLOAT ||
           t.kind == AST_TY_VOID;
}

static int find_var(emit_t *e, const char *name) {
    for (int i = 0; i < e->n_vars; i++)
        if (strcmp(e->vars[i].name, name) == 0) return i;
    return -1;
}

static void add_var(emit_t *e, const char *name, ast_type_kind_t kind) {
    if (e->failed) return;
    if (find_var(e, name) >= 0 || e->n_vars >= EMIT_MAX_VARS) {
        e->failed = 1;
        return;
    }
    e->vars[e->n_vars].name = name;
    e->vars[e->n_vars].var = 0;
    e->vars[e->n_vars].kind = kind;
    e->n_vars++;
}

/* Collect the declarations in a statement subtree. Nested blocks are
 * flattened into the one variable table, so a repeated name anywhere
 * in the function is a refusal. */
static void collect_decls(emit_t *e, const ast_node_t *n) {
    if (!n || e->failed) return;
    switch (n->kind) {
        case AST_BLOCK:
            for (int i = 0; i < n->u.block.n; i++)
                collect_decls(e, n->u.block.stmts[i]);
            break;
        case AST_DECL:
            if (!supported_type(n->u.decl.type) ||
                n->u.decl.type.kind == AST_TY_VOID)
                e->failed = 1;
            else
                add_var(e, n->u.decl.name, n->u.decl.type.kind);
            break;
        case AST_IF:
            collect_decls(e, n->u.if_s.then_body);
            collect_decls(e, n->u.if_s.else_body);
            break;
        case AST_FOR:
            collect_decls(e, n->u.for_s.init);
            collect_decls(e, n->u.for_s.body);
            break;
        case AST_WHILE:
            collect_decls(e, n->u.while_s.body);
            break;
        default:
            break;
    }
}

/* The index of a collected constant, or -1. Ids are assigned only
 * when the constants section is emitted, so dedup keys on the
 * index, not the id. */
static int const_index(emit_t *e, int is_float, uint32_t bits) {
    for (int i = 0; i < e->n_consts; i++)
        if (e->consts[i].is_float == is_float && e->consts[i].bits == bits)
            return i;
    return -1;
}

static uint32_t const_find(emit_t *e, int is_float, uint32_t bits) {
    int i = const_index(e, is_float, bits);
    return i >= 0 ? e->consts[i].id : 0;
}

static void const_add(emit_t *e, int is_float, uint32_t bits) {
    if (e->failed) return;
    if (const_index(e, is_float, bits) >= 0) return;
    if (e->n_consts >= EMIT_MAX_CONSTS) {
        e->failed = 1;
        return;
    }
    e->consts[e->n_consts].id = 0;
    e->consts[e->n_consts].is_float = is_float;
    e->consts[e->n_consts].bits = bits;
    e->n_consts++;
}

static uint32_t float_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return bits;
}

/* Collect the literals in an expression subtree. Control-flow and
 * call subtrees are walked too: collecting their literals is
 * harmless, and the statement walk refuses them later. */
static void collect_expr_consts(emit_t *e, const ast_node_t *n) {
    if (!n || e->failed) return;
    switch (n->kind) {
        case AST_INT_LIT:
            const_add(e, 0, (uint32_t)(int32_t)n->u.int_val);
            break;
        case AST_FLOAT_LIT:
            const_add(e, 1, float_bits(n->u.float_val));
            break;
        case AST_BINOP:
        case AST_ASSIGN:
        case AST_SUBSCRIPT:
            collect_expr_consts(e, n->u.binop.lhs);
            collect_expr_consts(e, n->u.binop.rhs);
            break;
        case AST_UNARY:
            collect_expr_consts(e, n->u.unary.operand);
            break;
        case AST_CALL:
            for (int i = 0; i < n->u.call.argc; i++)
                collect_expr_consts(e, n->u.call.args[i]);
            break;
        default:
            break;
    }
}

static void collect_stmt_consts(emit_t *e, const ast_node_t *n) {
    if (!n || e->failed) return;
    switch (n->kind) {
        case AST_BLOCK:
            for (int i = 0; i < n->u.block.n; i++)
                collect_stmt_consts(e, n->u.block.stmts[i]);
            break;
        case AST_DECL:
            collect_expr_consts(e, n->u.decl.init);
            break;
        case AST_RETURN:
            collect_expr_consts(e, n->u.ret.value);
            break;
        case AST_EXPR_STMT:
            collect_expr_consts(e, n->u.expr_stmt.e);
            break;
        case AST_IF:
            collect_expr_consts(e, n->u.if_s.cond);
            collect_stmt_consts(e, n->u.if_s.then_body);
            collect_stmt_consts(e, n->u.if_s.else_body);
            break;
        case AST_FOR:
            collect_stmt_consts(e, n->u.for_s.init);
            collect_expr_consts(e, n->u.for_s.cond);
            collect_expr_consts(e, n->u.for_s.post);
            collect_stmt_consts(e, n->u.for_s.body);
            break;
        case AST_WHILE:
            collect_expr_consts(e, n->u.while_s.cond);
            collect_stmt_consts(e, n->u.while_s.body);
            break;
        default:
            collect_expr_consts(e, n);
            break;
    }
}

/* Convert a value between int and float; a same-kind value passes
 * through untouched. */
static uint32_t convert(emit_t *e, uint32_t val,
                        ast_type_kind_t from, ast_type_kind_t to) {
    if (from == to) return val;
    if (from == AST_TY_INT && to == AST_TY_FLOAT) {
        uint32_t r = new_id(e);
        stream_op(e->s, OP_CONVERT_S_TO_F, 4);
        stream_word(e->s, e->t_float);
        stream_word(e->s, r);
        stream_word(e->s, val);
        return r;
    }
    if (from == AST_TY_FLOAT && to == AST_TY_INT) {
        uint32_t r = new_id(e);
        stream_op(e->s, OP_CONVERT_F_TO_S, 4);
        stream_word(e->s, e->t_int);
        stream_word(e->s, r);
        stream_word(e->s, val);
        return r;
    }
    e->failed = 1;
    return 0;
}

static uint32_t emit_expr(emit_t *e, const ast_node_t *n,
                          ast_type_kind_t *kind_out);

/* Arithmetic opcode for one operator and result kind, or 0 for an
 * operator outside the subset. */
static uint32_t arith_opcode(int op, ast_type_kind_t kind) {
    int is_float = kind == AST_TY_FLOAT;
    switch (op) {
        case '+': return is_float ? OP_F_ADD : OP_I_ADD;
        case '-': return is_float ? OP_F_SUB : OP_I_SUB;
        case '*': return is_float ? OP_F_MUL : OP_I_MUL;
        case '/': return is_float ? OP_F_DIV : OP_S_DIV;
        default: return 0;
    }
}

static uint32_t emit_binop(emit_t *e, const ast_node_t *n,
                           ast_type_kind_t *kind_out) {
    uint32_t opcode = arith_opcode(n->u.binop.op, n->resolved_type.kind);
    if (!opcode || n->resolved_type.kind == AST_TY_VOID) {
        e->failed = 1;
        return 0;
    }
    ast_type_kind_t lk, rk;
    uint32_t l = emit_expr(e, n->u.binop.lhs, &lk);
    uint32_t r = emit_expr(e, n->u.binop.rhs, &rk);
    if (e->failed) return 0;
    /* Usual promotion: an int side converts to float when the other
     * side is float, matching the typechecker's result kind. */
    ast_type_kind_t result = n->resolved_type.kind;
    l = convert(e, l, lk, result);
    r = convert(e, r, rk, result);
    if (e->failed) return 0;
    uint32_t out = new_id(e);
    stream_op(e->s, opcode, 5);
    stream_word(e->s, type_id(e, result));
    stream_word(e->s, out);
    stream_word(e->s, l);
    stream_word(e->s, r);
    *kind_out = result;
    return out;
}

static uint32_t emit_unary(emit_t *e, const ast_node_t *n,
                           ast_type_kind_t *kind_out) {
    ast_type_kind_t ok;
    uint32_t v = emit_expr(e, n->u.unary.operand, &ok);
    if (e->failed) return 0;
    if (n->u.unary.op == '-') {
        uint32_t opcode = ok == AST_TY_FLOAT ? OP_F_NEGATE : OP_S_NEGATE;
        uint32_t out = new_id(e);
        stream_op(e->s, opcode, 4);
        stream_word(e->s, type_id(e, ok));
        stream_word(e->s, out);
        stream_word(e->s, v);
        *kind_out = ok;
        return out;
    }
    if (n->u.unary.op == '!') {
        /* Logical not on an integer: (v == 0) ? 1 : 0. A float
         * operand is refused. */
        if (ok != AST_TY_INT) {
            e->failed = 1;
            return 0;
        }
        uint32_t is_zero = new_id(e);
        stream_op(e->s, OP_I_EQUAL, 5);
        stream_word(e->s, e->t_bool);
        stream_word(e->s, is_zero);
        stream_word(e->s, v);
        stream_word(e->s, e->c_int_zero);
        uint32_t out = new_id(e);
        stream_op(e->s, OP_SELECT, 6);
        stream_word(e->s, e->t_int);
        stream_word(e->s, out);
        stream_word(e->s, is_zero);
        stream_word(e->s, e->c_int_one);
        stream_word(e->s, e->c_int_zero);
        *kind_out = AST_TY_INT;
        return out;
    }
    e->failed = 1;
    return 0;
}

static uint32_t emit_assign(emit_t *e, const ast_node_t *n,
                            ast_type_kind_t *kind_out) {
    if (n->u.binop.lhs->kind != AST_VAR_REF) {
        e->failed = 1;
        return 0;
    }
    int idx = find_var(e, n->u.binop.lhs->u.var.name);
    if (idx < 0) {
        e->failed = 1;
        return 0;
    }
    ast_type_kind_t rk;
    uint32_t r = emit_expr(e, n->u.binop.rhs, &rk);
    if (e->failed) return 0;
    r = convert(e, r, rk, e->vars[idx].kind);
    if (e->failed) return 0;
    stream_op(e->s, OP_STORE, 3);
    stream_word(e->s, e->vars[idx].var);
    stream_word(e->s, r);
    *kind_out = e->vars[idx].kind;
    return r;
}

static uint32_t emit_expr(emit_t *e, const ast_node_t *n,
                          ast_type_kind_t *kind_out) {
    if (!n || e->failed) {
        e->failed = 1;
        return 0;
    }
    switch (n->kind) {
        case AST_INT_LIT: {
            uint32_t id = const_find(e, 0, (uint32_t)(int32_t)n->u.int_val);
            if (!id) e->failed = 1;
            *kind_out = AST_TY_INT;
            return id;
        }
        case AST_FLOAT_LIT: {
            uint32_t id = const_find(e, 1, float_bits(n->u.float_val));
            if (!id) e->failed = 1;
            *kind_out = AST_TY_FLOAT;
            return id;
        }
        case AST_VAR_REF: {
            int idx = find_var(e, n->u.var.name);
            if (idx < 0) {
                e->failed = 1;
                return 0;
            }
            uint32_t out = new_id(e);
            stream_op(e->s, OP_LOAD, 4);
            stream_word(e->s, type_id(e, e->vars[idx].kind));
            stream_word(e->s, out);
            stream_word(e->s, e->vars[idx].var);
            *kind_out = e->vars[idx].kind;
            return out;
        }
        case AST_BINOP:
            return emit_binop(e, n, kind_out);
        case AST_UNARY:
            return emit_unary(e, n, kind_out);
        case AST_ASSIGN:
            return emit_assign(e, n, kind_out);
        default:
            /* Calls, subscripts and anything else are refused. */
            e->failed = 1;
            return 0;
    }
}

static void emit_stmt(emit_t *e, const ast_node_t *n) {
    if (!n || e->failed || e->returned) return;
    switch (n->kind) {
        case AST_BLOCK:
            for (int i = 0; i < n->u.block.n && !e->failed && !e->returned; i++)
                emit_stmt(e, n->u.block.stmts[i]);
            break;
        case AST_DECL: {
            int idx = find_var(e, n->u.decl.name);
            if (idx < 0 || !n->u.decl.init) break;
            ast_type_kind_t ik;
            uint32_t v = emit_expr(e, n->u.decl.init, &ik);
            if (e->failed) break;
            v = convert(e, v, ik, e->vars[idx].kind);
            if (e->failed) break;
            stream_op(e->s, OP_STORE, 3);
            stream_word(e->s, e->vars[idx].var);
            stream_word(e->s, v);
            break;
        }
        case AST_EXPR_STMT: {
            ast_type_kind_t k;
            (void)emit_expr(e, n->u.expr_stmt.e, &k);
            break;
        }
        case AST_ASSIGN: {
            ast_type_kind_t k;
            (void)emit_expr(e, n, &k);
            break;
        }
        case AST_RETURN: {
            if (!n->u.ret.value) {
                stream_op(e->s, OP_RETURN, 1);
            } else {
                if (e->ret_kind == AST_TY_VOID) {
                    e->failed = 1;
                    break;
                }
                ast_type_kind_t vk;
                uint32_t v = emit_expr(e, n->u.ret.value, &vk);
                if (e->failed) break;
                v = convert(e, v, vk, e->ret_kind);
                if (e->failed) break;
                stream_op(e->s, OP_RETURN_VALUE, 2);
                stream_word(e->s, v);
            }
            e->returned = 1;
            break;
        }
        default:
            /* Control flow and anything else are refused. */
            e->failed = 1;
            break;
    }
}

int kernel_emit_function(const ast_function_t *fn,
                         uint32_t *out, size_t *out_words) {
    if (!fn || !fn->name || !fn->body || !out || !out_words ||
        *out_words < 64)
        return -1;
    if (!supported_type(fn->return_type) ||
        fn->param_count < 0 || fn->param_count > AST_MAX_PARAMS)
        return -1;

    stream_t s = { out, 0, *out_words, 0 };
    emit_t e;
    memset(&e, 0, sizeof(e));
    e.s = &s;
    e.next_id = 1;
    e.ret_kind = fn->return_type.kind;

    /* Pre-pass: parameters, then every declaration in the body. */
    for (int i = 0; i < fn->param_count; i++) {
        if (!supported_type(fn->params[i].type) ||
            fn->params[i].type.kind == AST_TY_VOID) {
            e.failed = 1;
            break;
        }
        add_var(&e, fn->params[i].name, fn->params[i].type.kind);
    }
    collect_decls(&e, fn->body);

    /* Pre-pass: literals, plus the zeros the wrapper calls with and
     * the constants logical-not is built from. */
    const_add(&e, 0, 0);
    const_add(&e, 0, 1);
    const_add(&e, 1, 0);
    collect_stmt_consts(&e, fn->body);
    if (e.failed) return -1;

    /* Allocate the ids the header section references: types, the two
     * function types, and the two function ids. */
    e.t_void = new_id(&e);
    e.t_bool = new_id(&e);
    e.t_int = new_id(&e);
    e.t_float = new_id(&e);
    e.tp_int = new_id(&e);
    e.tp_float = new_id(&e);
    e.fn_type = new_id(&e);
    e.wrap_type = new_id(&e);
    e.fn_id = new_id(&e);
    e.wrap_id = new_id(&e);

    size_t name_len = packed_string_words(fn->name);

    /* Header: magic, version, generator, bound (patched at the end),
     * schema. */
    stream_word(&s, KERNEL_MAGIC);
    stream_word(&s, KERNEL_VERSION);
    stream_word(&s, KERNEL_GENERATOR);
    stream_word(&s, 0);
    stream_word(&s, 0);

    /* OpCapability Shader */
    stream_op(&s, OP_CAPABILITY, 2);
    stream_word(&s, CAP_SHADER);

    /* OpMemoryModel Logical GLSL450 */
    stream_op(&s, OP_MEMORY_MODEL, 3);
    stream_word(&s, ADDR_LOGICAL);
    stream_word(&s, MEM_GLSL450);

    /* OpEntryPoint GLCompute <wrapper> "<name>" */
    stream_op(&s, OP_ENTRY_POINT, (uint32_t)(3 + name_len));
    stream_word(&s, EXEC_GLCOMPUTE);
    stream_word(&s, e.wrap_id);
    stream_string(&s, fn->name);

    /* OpExecutionMode <wrapper> LocalSize 1 1 1 */
    stream_op(&s, OP_EXEC_MODE, 6);
    stream_word(&s, e.wrap_id);
    stream_word(&s, MODE_LOCAL_SIZE);
    stream_word(&s, 1);
    stream_word(&s, 1);
    stream_word(&s, 1);

    /* Types: void, bool, int32, float32, and the Function-storage
     * pointers the variables use. */
    stream_op(&s, OP_TYPE_VOID, 2);
    stream_word(&s, e.t_void);

    stream_op(&s, OP_TYPE_BOOL, 2);
    stream_word(&s, e.t_bool);

    stream_op(&s, OP_TYPE_INT, 4);
    stream_word(&s, e.t_int);
    stream_word(&s, 32);
    stream_word(&s, 1);

    stream_op(&s, OP_TYPE_FLOAT, 3);
    stream_word(&s, e.t_float);
    stream_word(&s, 32);

    stream_op(&s, OP_TYPE_POINTER, 4);
    stream_word(&s, e.tp_int);
    stream_word(&s, STORAGE_FUNCTION);
    stream_word(&s, e.t_int);

    stream_op(&s, OP_TYPE_POINTER, 4);
    stream_word(&s, e.tp_float);
    stream_word(&s, STORAGE_FUNCTION);
    stream_word(&s, e.t_float);

    /* Function type of the source function: return type, then one
     * parameter type per parameter. */
    stream_op(&s, OP_TYPE_FUNC, (uint32_t)(3 + fn->param_count));
    stream_word(&s, e.fn_type);
    stream_word(&s, type_id(&e, e.ret_kind));
    for (int i = 0; i < fn->param_count; i++)
        stream_word(&s, type_id(&e, fn->params[i].type.kind));

    /* Function type of the wrapper: void(). */
    stream_op(&s, OP_TYPE_FUNC, 3);
    stream_word(&s, e.wrap_type);
    stream_word(&s, e.t_void);

    /* Constants, in collection order. */
    for (int i = 0; i < e.n_consts; i++) {
        e.consts[i].id = new_id(&e);
        stream_op(&s, OP_CONSTANT, 4);
        stream_word(&s, e.consts[i].is_float ? e.t_float : e.t_int);
        stream_word(&s, e.consts[i].id);
        stream_word(&s, e.consts[i].bits);
    }
    e.c_int_zero = const_find(&e, 0, 0);
    e.c_int_one = const_find(&e, 0, 1);
    e.c_float_zero = const_find(&e, 1, 0);

    /* The source function: parameters, one entry block, variables
     * first, then the parameter stores, then the body. */
    stream_op(&s, OP_FUNCTION, 5);
    stream_word(&s, type_id(&e, e.ret_kind));
    stream_word(&s, e.fn_id);
    stream_word(&s, 0);
    stream_word(&s, e.fn_type);

    for (int i = 0; i < fn->param_count; i++) {
        e.param_ids[i] = new_id(&e);
        stream_op(&s, OP_FUNCTION_PARAM, 3);
        stream_word(&s, type_id(&e, fn->params[i].type.kind));
        stream_word(&s, e.param_ids[i]);
    }

    stream_op(&s, OP_LABEL, 2);
    stream_word(&s, new_id(&e));

    for (int i = 0; i < e.n_vars; i++) {
        e.vars[i].var = new_id(&e);
        stream_op(&s, OP_VARIABLE, 4);
        stream_word(&s, ptr_type_id(&e, e.vars[i].kind));
        stream_word(&s, e.vars[i].var);
        stream_word(&s, STORAGE_FUNCTION);
    }

    for (int i = 0; i < fn->param_count; i++) {
        stream_op(&s, OP_STORE, 3);
        stream_word(&s, e.vars[i].var);
        stream_word(&s, e.param_ids[i]);
    }

    emit_stmt(&e, fn->body);
    if (e.failed) return -1;

    /* A body that falls off the end returns: void, or a zero of the
     * return kind for a value function. */
    if (!e.returned) {
        if (e.ret_kind == AST_TY_VOID) {
            stream_op(&s, OP_RETURN, 1);
        } else {
            uint32_t zero = e.ret_kind == AST_TY_FLOAT
                ? e.c_float_zero : e.c_int_zero;
            stream_op(&s, OP_RETURN_VALUE, 2);
            stream_word(&s, zero);
        }
    }
    stream_op(&s, OP_FUNCTION_END, 1);

    /* The wrapper: the entry point. It calls the source function
     * with a zero for every parameter and drops the result. */
    stream_op(&s, OP_FUNCTION, 5);
    stream_word(&s, e.t_void);
    stream_word(&s, e.wrap_id);
    stream_word(&s, 0);
    stream_word(&s, e.wrap_type);

    stream_op(&s, OP_LABEL, 2);
    stream_word(&s, new_id(&e));

    /* The call's result id is required even when the source function
     * returns void; it is simply never read. */
    stream_op(&s, OP_FUNCTION_CALL, (uint32_t)(4 + fn->param_count));
    stream_word(&s, type_id(&e, e.ret_kind));
    stream_word(&s, new_id(&e));
    stream_word(&s, e.fn_id);
    for (int i = 0; i < fn->param_count; i++)
        stream_word(&s, fn->params[i].type.kind == AST_TY_FLOAT
            ? e.c_float_zero : e.c_int_zero);

    stream_op(&s, OP_RETURN, 1);
    stream_op(&s, OP_FUNCTION_END, 1);

    if (s.overflow) return -1;
    out[3] = e.next_id;
    *out_words = s.used;
    return 0;
}
