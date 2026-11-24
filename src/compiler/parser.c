/* parser.c
 * Verum-Bespoke-Singularity embedded C11 compiler — recursive
 * descent parser.
 *
 * Parses the C11 subset produced by the LLM optimizer into an AST.
 * Input:  token stream from lexer.c
 * Output: ast_function_t — passed directly to the emitters
 *
 * Grammar (simplified):
 *   function   := type IDENT '(' param_list ')' block
 *   param_list := (type IDENT (',' type IDENT)*)?
 *   block      := '{' stmt* '}'
 *   stmt       := decl_stmt | assign_stmt | return_stmt
 *               | if_stmt | for_stmt | while_stmt | expr_stmt
 *   expr       := assignment | logical_or
 */

#include "../../include/vbs_compiler.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Parser state: a one-token lookahead over the lexer, plus the
 * first error recorded (later errors are suppressed). */
struct vbs_parser {
    vbs_lexer_t *lexer;
    vbs_token_t  current;
    vbs_token_t  previous;
    int          had_error;
    char         error_msg[256];
};

/* Record the first error; later ones are ignored so the caller
 * sees the root cause, not the cascade. */
static void parse_fail(vbs_parser_t *p, const char *msg) {
    if (!p->had_error) {
        snprintf(p->error_msg, sizeof(p->error_msg),
                 "parse error at line %d col %d: %s",
                 p->current.line, p->current.col, msg);
        p->had_error = 1;
    }
}

static vbs_token_t step(vbs_parser_t *p) {
    p->previous = p->current;
    p->current = vbs_lexer_next(p->lexer);
    return p->previous;
}

static int at_kind(vbs_parser_t *p, vbs_tok_kind_t k) {
    return p->current.kind == k;
}

static int accept(vbs_parser_t *p, vbs_tok_kind_t k) {
    if (!at_kind(p, k)) return 0;
    step(p);
    return 1;
}

/* Expect a token: on mismatch, record the error but still advance
 * (so parsing continues and reports the first problem only). */
static vbs_token_t require(vbs_parser_t *p, vbs_tok_kind_t k, const char *msg) {
    if (p->current.kind != k) parse_fail(p, msg);
    return step(p);
}

/* Allocate a zeroed node of the given kind. */
static ast_node_t *new_node(ast_node_kind_t kind) {
    ast_node_t *n = calloc(1, sizeof(*n));
    if (n) n->kind = kind;
    return n;
}

/* Copy a token's source slice into a NUL-terminated string. */
static char *token_text(vbs_token_t t) {
    char *s = malloc(t.len + 1);
    if (s) {
        memcpy(s, t.start, t.len);
        s[t.len] = '\0';
    }
    return s;
}

/* Type keyword -> AST type kind. */
static int type_keyword_kind(vbs_tok_kind_t k, ast_type_kind_t *out) {
    switch (k) {
        case TOK_VOID:   *out = AST_TY_VOID;   return 1;
        case TOK_INT:    *out = AST_TY_INT;    return 1;
        case TOK_FLOAT:  *out = AST_TY_FLOAT;  return 1;
        case TOK_DOUBLE: *out = AST_TY_DOUBLE; return 1;
        case TOK_CHAR:   *out = AST_TY_CHAR;   return 1;
        case TOK_SIZE_T: *out = AST_TY_SIZE_T; return 1;
        default: return 0;
    }
}

static ast_type_t parse_type(vbs_parser_t *p) {
    ast_type_t ty = { AST_TY_VOID, 0, 0 };

    if (accept(p, TOK_CONST)) ty.is_const = 1;

    ast_type_kind_t kind;
    if (type_keyword_kind(p->current.kind, &kind)) {
        ty.kind = kind;
        step(p);
    } else {
        parse_fail(p, "expected type");
        step(p);
    }

    while (accept(p, TOK_STAR)) ty.ptr_depth++;
    return ty;
}

static ast_node_t *parse_expr(vbs_parser_t *p);
static ast_node_t *parse_stmt(vbs_parser_t *p);
static ast_node_t *parse_block(vbs_parser_t *p);

/* Binary operator table: token kind, precedence (higher binds
 * tighter), and the single-character code stored in the AST node.
 * Multi-character operators map to otherwise-unused codes that the
 * emitters translate back ('=' for ==, '!' for !=, 'L' for <=,
 * 'G' for >=, '{' for <<, '}' for >>). */
typedef struct {
    vbs_tok_kind_t kind;
    int prec;
    int code;
} binop_entry_t;

static const binop_entry_t k_binops[] = {
    { TOK_PIPE_PIPE, 1, '|' },
    { TOK_AMP_AMP,   2, '&' },
    { TOK_EQ_EQ,     3, '=' },
    { TOK_BANG_EQ,   3, '!' },
    { TOK_LT,        4, '<' },
    { TOK_GT,        4, '>' },
    { TOK_LT_EQ,     4, 'L' },
    { TOK_GT_EQ,     4, 'G' },
    { TOK_SHL,       5, '{' },
    { TOK_SHR,       5, '}' },
    { TOK_PLUS,      6, '+' },
    { TOK_MINUS,     6, '-' },
    { TOK_STAR,      7, '*' },
    { TOK_SLASH,     7, '/' },
    { TOK_PERCENT,   7, '%' },
};

static const binop_entry_t *find_binop(vbs_tok_kind_t k) {
    for (size_t i = 0; i < sizeof(k_binops) / sizeof(k_binops[0]); i++)
        if (k_binops[i].kind == k) return &k_binops[i];
    return NULL;
}

static ast_node_t *parse_primary(vbs_parser_t *p) {
    if (accept(p, TOK_INT_LIT)) {
        ast_node_t *n = new_node(AST_INT_LIT);
        n->u.int_val = strtoll(p->previous.start, NULL, 10);
        return n;
    }
    if (accept(p, TOK_FLOAT_LIT)) {
        ast_node_t *n = new_node(AST_FLOAT_LIT);
        n->u.float_val = strtof(p->previous.start, NULL);
        return n;
    }
    if (accept(p, TOK_IDENT)) {
        char *name = token_text(p->previous);
        if (accept(p, TOK_LPAREN)) {
            /* Function call. */
            ast_node_t *n = new_node(AST_CALL);
            n->u.call.name = name;
            n->u.call.argc = 0;
            while (!at_kind(p, TOK_RPAREN) && !at_kind(p, TOK_EOF)) {
                n->u.call.args[n->u.call.argc++] = parse_expr(p);
                if (!accept(p, TOK_COMMA)) break;
            }
            require(p, TOK_RPAREN, "expected ')' after call args");
            return n;
        }
        /* Variable reference, possibly subscripted. */
        ast_node_t *n = new_node(AST_VAR_REF);
        n->u.var.name = name;
        if (accept(p, TOK_LBRACKET)) {
            ast_node_t *idx = parse_expr(p);
            require(p, TOK_RBRACKET, "expected ']'");
            ast_node_t *sub = new_node(AST_SUBSCRIPT);
            sub->u.binop.lhs = n;
            sub->u.binop.rhs = idx;
            return sub;
        }
        return n;
    }
    if (accept(p, TOK_LPAREN)) {
        ast_node_t *n = parse_expr(p);
        require(p, TOK_RPAREN, "expected ')'");
        return n;
    }
    if (accept(p, TOK_MINUS)) {
        ast_node_t *n = new_node(AST_UNARY);
        n->u.unary.op = '-';
        n->u.unary.operand = parse_primary(p);
        return n;
    }
    if (accept(p, TOK_BANG)) {
        ast_node_t *n = new_node(AST_UNARY);
        n->u.unary.op = '!';
        n->u.unary.operand = parse_primary(p);
        return n;
    }
    parse_fail(p, "unexpected token in expression");
    step(p);
    return new_node(AST_INT_LIT);
}

/* Precedence-climbing binary expression parser. */
static ast_node_t *parse_binop(vbs_parser_t *p, int min_prec) {
    ast_node_t *lhs = parse_primary(p);
    for (;;) {
        const binop_entry_t *op = find_binop(p->current.kind);
        if (!op || op->prec < min_prec) break;
        step(p);
        ast_node_t *rhs = parse_binop(p, op->prec + 1);
        ast_node_t *node = new_node(AST_BINOP);
        node->u.binop.op = op->code;
        node->u.binop.lhs = lhs;
        node->u.binop.rhs = rhs;
        lhs = node;
    }
    return lhs;
}

/* Compound-assignment table: token kind -> plain operator code. */
typedef struct {
    vbs_tok_kind_t kind;
    int code;
} compound_entry_t;

static const compound_entry_t k_compound[] = {
    { TOK_PLUS_EQ,  '+' },
    { TOK_MINUS_EQ, '-' },
    { TOK_STAR_EQ,  '*' },
    { TOK_SLASH_EQ, '/' },
};

static ast_node_t *parse_expr(vbs_parser_t *p) {
    ast_node_t *lhs = parse_binop(p, 0);

    if (accept(p, TOK_EQ)) {
        ast_node_t *rhs = parse_expr(p);
        ast_node_t *node = new_node(AST_ASSIGN);
        node->u.binop.lhs = lhs;
        node->u.binop.rhs = rhs;
        return node;
    }

    /* Compound assignment: x += e becomes x = (x + e). The lhs
     * node is shared between the assignment and the binop. */
    for (size_t i = 0; i < sizeof(k_compound) / sizeof(k_compound[0]); i++) {
        if (p->current.kind == k_compound[i].kind) {
            step(p);
            ast_node_t *rhs = parse_expr(p);
            ast_node_t *binop = new_node(AST_BINOP);
            binop->u.binop.op = k_compound[i].code;
            binop->u.binop.lhs = lhs;
            binop->u.binop.rhs = rhs;
            ast_node_t *node = new_node(AST_ASSIGN);
            node->u.binop.lhs = lhs;
            node->u.binop.rhs = binop;
            return node;
        }
    }

    return lhs;
}

static ast_node_t *parse_decl_stmt(vbs_parser_t *p) {
    ast_type_t ty = parse_type(p);
    vbs_token_t name = require(p, TOK_IDENT, "expected variable name");
    ast_node_t *n = new_node(AST_DECL);
    n->u.decl.type = ty;
    n->u.decl.name = token_text(name);
    n->u.decl.init = NULL;
    if (accept(p, TOK_EQ))
        n->u.decl.init = parse_expr(p);
    require(p, TOK_SEMI, "expected ';' after declaration");
    return n;
}

static ast_node_t *parse_return_stmt(vbs_parser_t *p) {
    ast_node_t *n = new_node(AST_RETURN);
    n->u.ret.value = NULL;
    if (!at_kind(p, TOK_SEMI))
        n->u.ret.value = parse_expr(p);
    require(p, TOK_SEMI, "expected ';' after return");
    return n;
}

static ast_node_t *parse_if_stmt(vbs_parser_t *p) {
    require(p, TOK_LPAREN, "expected '(' after if");
    ast_node_t *cond = parse_expr(p);
    require(p, TOK_RPAREN, "expected ')' after if condition");
    ast_node_t *then_b = parse_block(p);
    ast_node_t *else_b = NULL;
    if (accept(p, TOK_ELSE)) {
        if (at_kind(p, TOK_IF)) {
            step(p);
            else_b = parse_if_stmt(p);
        } else {
            else_b = parse_block(p);
        }
    }
    ast_node_t *n = new_node(AST_IF);
    n->u.if_s.cond = cond;
    n->u.if_s.then_body = then_b;
    n->u.if_s.else_body = else_b;
    return n;
}

static ast_node_t *parse_for_stmt(vbs_parser_t *p) {
    require(p, TOK_LPAREN, "expected '(' after for");
    ast_node_t *init = NULL;
    if (!at_kind(p, TOK_SEMI))
        init = parse_stmt(p); /* consumes its own ';' */
    else
        require(p, TOK_SEMI, "expected ';'");
    ast_node_t *cond = NULL;
    if (!at_kind(p, TOK_SEMI)) cond = parse_expr(p);
    require(p, TOK_SEMI, "expected ';' in for");
    ast_node_t *post = NULL;
    if (!at_kind(p, TOK_RPAREN)) post = parse_expr(p);
    require(p, TOK_RPAREN, "expected ')' after for clauses");
    ast_node_t *body = parse_block(p);
    ast_node_t *n = new_node(AST_FOR);
    n->u.for_s.init = init;
    n->u.for_s.cond = cond;
    n->u.for_s.post = post;
    n->u.for_s.body = body;
    return n;
}

static ast_node_t *parse_while_stmt(vbs_parser_t *p) {
    require(p, TOK_LPAREN, "expected '(' after while");
    ast_node_t *cond = parse_expr(p);
    require(p, TOK_RPAREN, "expected ')' after while condition");
    ast_node_t *body = parse_block(p);
    ast_node_t *n = new_node(AST_WHILE);
    n->u.while_s.cond = cond;
    n->u.while_s.body = body;
    return n;
}

static int starts_declaration(vbs_tok_kind_t k) {
    switch (k) {
        case TOK_INT: case TOK_FLOAT: case TOK_DOUBLE:
        case TOK_CHAR: case TOK_VOID: case TOK_SIZE_T:
        case TOK_CONST: case TOK_STATIC:
            return 1;
        default:
            return 0;
    }
}

static ast_node_t *parse_stmt(vbs_parser_t *p) {
    if (p->had_error) return NULL;

    if (accept(p, TOK_RETURN)) return parse_return_stmt(p);
    if (accept(p, TOK_IF)) return parse_if_stmt(p);
    if (accept(p, TOK_FOR)) return parse_for_stmt(p);
    if (accept(p, TOK_WHILE)) return parse_while_stmt(p);
    if (at_kind(p, TOK_LBRACE)) return parse_block(p);

    if (starts_declaration(p->current.kind))
        return parse_decl_stmt(p);

    ast_node_t *e = parse_expr(p);
    require(p, TOK_SEMI, "expected ';' after expression statement");
    ast_node_t *s = new_node(AST_EXPR_STMT);
    s->u.expr_stmt.e = e;
    return s;
}

static ast_node_t *parse_block(vbs_parser_t *p) {
    require(p, TOK_LBRACE, "expected '{'");
    ast_node_t *blk = new_node(AST_BLOCK);
    blk->u.block.n = 0;
    while (!at_kind(p, TOK_RBRACE) && !at_kind(p, TOK_EOF) && !p->had_error) {
        if (blk->u.block.n < AST_MAX_STMTS) {
            blk->u.block.stmts[blk->u.block.n++] = parse_stmt(p);
        } else {
            parse_fail(p, "too many statements in block");
            break;
        }
    }
    require(p, TOK_RBRACE, "expected '}'");
    return blk;
}

vbs_parser_t *vbs_parser_create(vbs_lexer_t *l) {
    vbs_parser_t *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->lexer = l;
    p->current = vbs_lexer_next(l);
    return p;
}

void vbs_parser_destroy(vbs_parser_t *p) {
    free(p);
}

ast_function_t *vbs_parse_function(vbs_parser_t *p) {
    /* Optional 'static' (accepted, not recorded). */
    accept(p, TOK_STATIC);

    ast_type_t ret_type = parse_type(p);
    vbs_token_t name = require(p, TOK_IDENT, "expected function name");

    ast_function_t *fn = calloc(1, sizeof(*fn));
    if (!fn) return NULL;
    fn->name = token_text(name);
    fn->return_type = ret_type;

    require(p, TOK_LPAREN, "expected '(' in function definition");
    fn->param_count = 0;
    while (!at_kind(p, TOK_RPAREN) && !at_kind(p, TOK_EOF) && !p->had_error) {
        if (fn->param_count >= AST_MAX_PARAMS) {
            parse_fail(p, "too many parameters");
            break;
        }
        ast_type_t pty = parse_type(p);
        vbs_token_t pnam = require(p, TOK_IDENT, "expected param name");
        fn->params[fn->param_count].type = pty;
        fn->params[fn->param_count].name = token_text(pnam);
        fn->param_count++;
        if (!accept(p, TOK_COMMA)) break;
    }
    require(p, TOK_RPAREN, "expected ')' after params");

    fn->body = parse_block(p);

    if (p->had_error) {
        fprintf(stderr, "[VBS] %s\n", p->error_msg);
        free(fn->name);
        free(fn);
        return NULL;
    }
    return fn;
}

void ast_function_free(ast_function_t *fn) {
    if (!fn) return;
    free(fn->name);
    for (int i = 0; i < fn->param_count; i++)
        free(fn->params[i].name);
    /* Note: full recursive AST free omitted for brevity —
       add if long-running sessions accumulate many patches */
    free(fn);
}
