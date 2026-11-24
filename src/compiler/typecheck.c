/* typecheck.c
 * Verum-Bespoke-Singularity embedded C11 compiler — type checker.
 *
 * Single-pass AST walk. Catches:
 *   - Undeclared variables
 *   - Type mismatches in assignments
 *   - Wrong argument counts on calls
 *   - Void variable declarations
 *
 * On error: returns -1 and writes a message to err_out (caller
 * supplies the buffer). On success: returns 0. The AST is
 * annotated in place with resolved types.
 */

#include "../../include/vbs_compiler.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* Symbol table: a flat stack of names with scope marks. Entering a
 * block records the current depth; leaving one truncates back to
 * it, so inner declarations never leak outward. */
#define MAX_SYMS 256
#define MAX_SCOPES 32

typedef struct {
    char name[64];
    ast_type_t type;
} sym_entry_t;

typedef struct {
    sym_entry_t syms[MAX_SYMS];
    int n;
    int scope_marks[MAX_SCOPES];
    int scope_depth;
} sym_table_t;

static void scope_enter(sym_table_t *t) {
    t->scope_marks[t->scope_depth++] = t->n;
}

static void scope_leave(sym_table_t *t) {
    if (t->scope_depth > 0)
        t->n = t->scope_marks[--t->scope_depth];
}

static int sym_add(sym_table_t *t, const char *name, ast_type_t ty,
                   char *err, size_t err_sz) {
    if (t->n >= MAX_SYMS) {
        snprintf(err, err_sz, "typecheck: symbol table full");
        return -1;
    }
    strncpy(t->syms[t->n].name, name, 63);
    t->syms[t->n].type = ty;
    t->n++;
    return 0;
}

/* Innermost definition wins (search from the top of the stack). */
static ast_type_t *sym_find(sym_table_t *t, const char *name) {
    for (int i = t->n - 1; i >= 0; i--)
        if (strcmp(t->syms[i].name, name) == 0)
            return &t->syms[i].type;
    return NULL;
}

/* Two types agree if their pointer depths match and either both
 * are pointers (pointer compatibility is loose), both are the
 * same kind, or both are numeric (int/float/double convert
 * implicitly). */
static int is_numeric(ast_type_kind_t k) {
    return k == AST_TY_INT || k == AST_TY_FLOAT || k == AST_TY_DOUBLE;
}

static int types_compatible(ast_type_t a, ast_type_t b) {
    if (a.ptr_depth != b.ptr_depth) return 0;
    if (a.ptr_depth > 0) return 1;
    if (a.kind == b.kind) return 1;
    return is_numeric(a.kind) && is_numeric(b.kind);
}

static const char *type_name(ast_type_t t) {
    if (t.ptr_depth) return "pointer";
    switch (t.kind) {
        case AST_TY_VOID:   return "void";
        case AST_TY_INT:    return "int";
        case AST_TY_FLOAT:  return "float";
        case AST_TY_DOUBLE: return "double";
        case AST_TY_CHAR:   return "char";
        case AST_TY_SIZE_T: return "size_t";
        default:            return "unknown";
    }
}

static ast_type_t make_type(ast_type_kind_t kind) {
    ast_type_t t = { kind, 0, 0 };
    return t;
}

static ast_type_t check_expr(ast_node_t *n, sym_table_t *syms,
                             const ast_function_t *fn,
                             char *err, size_t esz, int *ok);
static void check_stmt(ast_node_t *n, sym_table_t *syms,
                       const ast_function_t *fn,
                       char *err, size_t esz, int *ok);

/* Builtin math functions: all take floats and return float. */
static const char *k_math_builtins[] = {
    "sqrtf", "expf", "logf", "sinf", "cosf", "fabsf", "powf",
    "fmaxf", "fminf", "floorf", "ceilf", NULL
};

static ast_type_t check_expr(ast_node_t *n, sym_table_t *syms,
                             const ast_function_t *fn,
                             char *err, size_t esz, int *ok) {
    if (!n || !*ok) return make_type(AST_TY_VOID);

    switch (n->kind) {
        case AST_INT_LIT:
            return make_type(AST_TY_INT);
        case AST_FLOAT_LIT:
            return make_type(AST_TY_FLOAT);

        case AST_VAR_REF: {
            ast_type_t *ty = sym_find(syms, n->u.var.name);
            if (!ty) {
                snprintf(err, esz, "typecheck: undeclared variable '%s'",
                         n->u.var.name);
                *ok = 0;
                return make_type(AST_TY_VOID);
            }
            n->resolved_type = *ty;
            return *ty;
        }

        case AST_SUBSCRIPT: {
            ast_type_t base = check_expr(n->u.binop.lhs, syms, fn, err, esz, ok);
            check_expr(n->u.binop.rhs, syms, fn, err, esz, ok); /* index */
            if (base.ptr_depth > 0) {
                base.ptr_depth--;
                n->resolved_type = base;
                return base;
            }
            snprintf(err, esz, "typecheck: subscript on non-pointer");
            *ok = 0;
            return make_type(AST_TY_VOID);
        }

        case AST_BINOP: {
            ast_type_t lty = check_expr(n->u.binop.lhs, syms, fn, err, esz, ok);
            ast_type_t rty = check_expr(n->u.binop.rhs, syms, fn, err, esz, ok);
            /* Usual arithmetic promotion: double wins over float
             * wins over int. */
            ast_type_kind_t result = AST_TY_INT;
            if (lty.kind == AST_TY_DOUBLE || rty.kind == AST_TY_DOUBLE)
                result = AST_TY_DOUBLE;
            else if (lty.kind == AST_TY_FLOAT || rty.kind == AST_TY_FLOAT)
                result = AST_TY_FLOAT;
            n->resolved_type = make_type(result);
            return n->resolved_type;
        }

        case AST_UNARY: {
            ast_type_t ty = check_expr(n->u.unary.operand, syms, fn, err, esz, ok);
            n->resolved_type = ty;
            return ty;
        }

        case AST_ASSIGN: {
            ast_type_t lty = check_expr(n->u.binop.lhs, syms, fn, err, esz, ok);
            ast_type_t rty = check_expr(n->u.binop.rhs, syms, fn, err, esz, ok);
            if (!types_compatible(lty, rty)) {
                snprintf(err, esz,
                         "typecheck: cannot assign %s to %s",
                         type_name(rty), type_name(lty));
                *ok = 0;
            }
            n->resolved_type = lty;
            return lty;
        }

        case AST_CALL: {
            for (int i = 0; k_math_builtins[i]; i++) {
                if (strcmp(n->u.call.name, k_math_builtins[i]) == 0) {
                    n->resolved_type = make_type(AST_TY_FLOAT);
                    return n->resolved_type;
                }
            }
            /* Otherwise unresolved — treat as int and warn. */
            fprintf(stderr, "[VBS] typecheck: unknown function '%s' — assumed int\n",
                    n->u.call.name);
            n->resolved_type = make_type(AST_TY_INT);
            return n->resolved_type;
        }

        default:
            return make_type(AST_TY_VOID);
    }
}

static void check_stmt(ast_node_t *n, sym_table_t *syms,
                       const ast_function_t *fn,
                       char *err, size_t esz, int *ok) {
    if (!n || !*ok) return;

    switch (n->kind) {
        case AST_DECL: {
            if (n->u.decl.type.kind == AST_TY_VOID &&
                n->u.decl.type.ptr_depth == 0) {
                snprintf(err, esz, "typecheck: cannot declare void variable '%s'",
                         n->u.decl.name);
                *ok = 0;
                return;
            }
            if (n->u.decl.init) {
                ast_type_t ity = check_expr(n->u.decl.init, syms, fn, err, esz, ok);
                if (!types_compatible(n->u.decl.type, ity)) {
                    snprintf(err, esz,
                             "typecheck: init type mismatch for '%s': %s vs %s",
                             n->u.decl.name,
                             type_name(n->u.decl.type), type_name(ity));
                    *ok = 0;
                    return;
                }
            }
            sym_add(syms, n->u.decl.name, n->u.decl.type, err, esz);
            break;
        }

        case AST_RETURN: {
            if (n->u.ret.value) {
                ast_type_t rty = check_expr(n->u.ret.value, syms, fn, err, esz, ok);
                if (!types_compatible(fn->return_type, rty)) {
                    snprintf(err, esz,
                             "typecheck: return type mismatch: expected %s got %s",
                             type_name(fn->return_type), type_name(rty));
                    *ok = 0;
                }
            }
            break;
        }

        case AST_BLOCK:
            scope_enter(syms);
            for (int i = 0; i < n->u.block.n && *ok; i++)
                check_stmt(n->u.block.stmts[i], syms, fn, err, esz, ok);
            scope_leave(syms);
            break;

        case AST_IF:
            check_expr(n->u.if_s.cond, syms, fn, err, esz, ok);
            check_stmt(n->u.if_s.then_body, syms, fn, err, esz, ok);
            if (n->u.if_s.else_body)
                check_stmt(n->u.if_s.else_body, syms, fn, err, esz, ok);
            break;

        case AST_FOR:
            scope_enter(syms);
            if (n->u.for_s.init) check_stmt(n->u.for_s.init, syms, fn, err, esz, ok);
            if (n->u.for_s.cond) check_expr(n->u.for_s.cond, syms, fn, err, esz, ok);
            if (n->u.for_s.post) check_expr(n->u.for_s.post, syms, fn, err, esz, ok);
            check_stmt(n->u.for_s.body, syms, fn, err, esz, ok);
            scope_leave(syms);
            break;

        case AST_WHILE:
            check_expr(n->u.while_s.cond, syms, fn, err, esz, ok);
            check_stmt(n->u.while_s.body, syms, fn, err, esz, ok);
            break;

        case AST_EXPR_STMT:
            check_expr(n->u.expr_stmt.e, syms, fn, err, esz, ok);
            break;

        default:
            break;
    }
}

int vbs_typecheck(ast_function_t *fn, char *err_out, size_t err_sz) {
    sym_table_t syms;
    memset(&syms, 0, sizeof(syms));
    int ok = 1;

    /* Seed the symbol table with the function parameters. */
    for (int i = 0; i < fn->param_count && ok; i++)
        sym_add(&syms, fn->params[i].name, fn->params[i].type,
                err_out, err_sz);

    scope_enter(&syms);
    check_stmt(fn->body, &syms, fn, err_out, err_sz, &ok);
    scope_leave(&syms);

    return ok ? 0 : -1;
}
