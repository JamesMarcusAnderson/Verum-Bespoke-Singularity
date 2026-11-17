/* =============================================================================
 * parser.c
 * Verum-Bespoke-Singularity embedded C11 compiler — recursive descent parser
 *
 * Parses the C11 subset produced by the LLM optimizer into an AST.
 * Input:  token stream from lexer.c
 * Output: ast_function_t — passed directly to x86_64_emit.c / aarch64_emit.c
 *
 * Grammar (simplified):
 *   function   := type IDENT '(' param_list ')' block
 *   param_list := (type IDENT (',' type IDENT)*)?
 *   block      := '{' stmt* '}'
 *   stmt       := decl_stmt | assign_stmt | return_stmt
 *               | if_stmt | for_stmt | while_stmt | expr_stmt
 *   expr       := assignment | logical_or
 * =========================================================================== */

#include "../../include/vbs_compiler.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* -------------------------------------------------------------------------- */
/* Parser state                                                                */
/* -------------------------------------------------------------------------- */

struct vbs_parser {
    vbs_lexer_t  *lexer;
    vbs_token_t   current;
    vbs_token_t   previous;
    int           had_error;
    char          error_msg[256];
};

/* -------------------------------------------------------------------------- */
/* Internal helpers                                                            */
/* -------------------------------------------------------------------------- */

static void parser_error(vbs_parser_t *p, const char *msg)
{
    if (!p->had_error) {
        snprintf(p->error_msg, sizeof(p->error_msg),
                 "parse error at line %d col %d: %s",
                 p->current.line, p->current.col, msg);
        p->had_error = 1;
    }
}

static vbs_token_t advance_tok(vbs_parser_t *p)
{
    p->previous = p->current;
    p->current  = vbs_lexer_next(p->lexer);
    return p->previous;
}

static int check(vbs_parser_t *p, vbs_tok_kind_t k) { return p->current.kind == k; }

static int match(vbs_parser_t *p, vbs_tok_kind_t k)
{
    if (!check(p, k)) return 0;
    advance_tok(p);
    return 1;
}

static vbs_token_t expect(vbs_parser_t *p, vbs_tok_kind_t k, const char *msg)
{
    if (p->current.kind != k) { parser_error(p, msg); }
    return advance_tok(p);
}

/* -------------------------------------------------------------------------- */
/* AST node allocators                                                         */
/* -------------------------------------------------------------------------- */

static ast_node_t *alloc_node(ast_node_kind_t kind)
{
    ast_node_t *n = calloc(1, sizeof(*n));
    if (n) n->kind = kind;
    return n;
}

static char *tok_to_str(vbs_token_t t)
{
    char *s = malloc(t.len + 1);
    if (s) { memcpy(s, t.start, t.len); s[t.len] = '\0'; }
    return s;
}

/* -------------------------------------------------------------------------- */
/* Type parsing                                                                */
/* -------------------------------------------------------------------------- */

static ast_type_t parse_type(vbs_parser_t *p)
{
    ast_type_t ty = { AST_TY_VOID, 0, 0 };

    if (match(p, TOK_CONST)) ty.is_const = 1;

    switch (p->current.kind) {
        case TOK_VOID:   ty.kind = AST_TY_VOID;   advance_tok(p); break;
        case TOK_INT:    ty.kind = AST_TY_INT;     advance_tok(p); break;
        case TOK_FLOAT:  ty.kind = AST_TY_FLOAT;   advance_tok(p); break;
        case TOK_DOUBLE: ty.kind = AST_TY_DOUBLE;  advance_tok(p); break;
        case TOK_CHAR:   ty.kind = AST_TY_CHAR;    advance_tok(p); break;
        case TOK_SIZE_T: ty.kind = AST_TY_SIZE_T;  advance_tok(p); break;
        default:
            parser_error(p, "expected type");
            advance_tok(p);
            break;
    }

    while (match(p, TOK_STAR)) ty.ptr_depth++;
    return ty;
}

/* -------------------------------------------------------------------------- */
/* Forward declarations for recursive grammar                                 */
/* -------------------------------------------------------------------------- */

static ast_node_t *parse_expr(vbs_parser_t *p);
static ast_node_t *parse_stmt(vbs_parser_t *p);
static ast_node_t *parse_block(vbs_parser_t *p);

/* -------------------------------------------------------------------------- */
/* Expression parsing (Pratt / precedence climb)                              */
/* -------------------------------------------------------------------------- */

static ast_node_t *parse_primary(vbs_parser_t *p)
{
    /* Integer literal */
    if (match(p, TOK_INT_LIT)) {
        ast_node_t *n = alloc_node(AST_INT_LIT);
        n->u.int_val  = strtoll(p->previous.start, NULL, 10);
        return n;
    }
    /* Float literal */
    if (match(p, TOK_FLOAT_LIT)) {
        ast_node_t *n  = alloc_node(AST_FLOAT_LIT);
        n->u.float_val = strtof(p->previous.start, NULL);
        return n;
    }
    /* Identifier (variable ref or function call) */
    if (match(p, TOK_IDENT)) {
        char *name = tok_to_str(p->previous);
        if (match(p, TOK_LPAREN)) {
            /* Function call */
            ast_node_t *n   = alloc_node(AST_CALL);
            n->u.call.name  = name;
            n->u.call.argc  = 0;
            while (!check(p, TOK_RPAREN) && !check(p, TOK_EOF)) {
                n->u.call.args[n->u.call.argc++] = parse_expr(p);
                if (!match(p, TOK_COMMA)) break;
            }
            expect(p, TOK_RPAREN, "expected ')' after call args");
            return n;
        }
        /* Array subscript */
        ast_node_t *n  = alloc_node(AST_VAR_REF);
        n->u.var.name  = name;
        if (match(p, TOK_LBRACKET)) {
            ast_node_t *idx   = parse_expr(p);
            expect(p, TOK_RBRACKET, "expected ']'");
            ast_node_t *sub   = alloc_node(AST_SUBSCRIPT);
            sub->u.binop.lhs  = n;
            sub->u.binop.rhs  = idx;
            return sub;
        }
        return n;
    }
    /* Grouped expression */
    if (match(p, TOK_LPAREN)) {
        ast_node_t *n = parse_expr(p);
        expect(p, TOK_RPAREN, "expected ')'");
        return n;
    }
    /* Unary */
    if (match(p, TOK_MINUS)) {
        ast_node_t *n   = alloc_node(AST_UNARY);
        n->u.unary.op   = '-';
        n->u.unary.operand = parse_primary(p);
        return n;
    }
    if (match(p, TOK_BANG)) {
        ast_node_t *n   = alloc_node(AST_UNARY);
        n->u.unary.op   = '!';
        n->u.unary.operand = parse_primary(p);
        return n;
    }
    parser_error(p, "unexpected token in expression");
    advance_tok(p);
    return alloc_node(AST_INT_LIT);
}

static int get_binop_prec(vbs_tok_kind_t k)
{
    switch (k) {
        case TOK_PIPE_PIPE:  return 1;
        case TOK_AMP_AMP:    return 2;
        case TOK_EQ_EQ:
        case TOK_BANG_EQ:    return 3;
        case TOK_LT: case TOK_GT:
        case TOK_LT_EQ: case TOK_GT_EQ: return 4;
        case TOK_SHL: case TOK_SHR:     return 5;
        case TOK_PLUS: case TOK_MINUS:  return 6;
        case TOK_STAR: case TOK_SLASH:
        case TOK_PERCENT:               return 7;
        default: return -1;
    }
}

static int tok_to_binop_char(vbs_tok_kind_t k)
{
    switch (k) {
        case TOK_PLUS:      return '+';
        case TOK_MINUS:     return '-';
        case TOK_STAR:      return '*';
        case TOK_SLASH:     return '/';
        case TOK_PERCENT:   return '%';
        case TOK_LT:        return '<';
        case TOK_GT:        return '>';
        case TOK_EQ_EQ:     return '='; /* handled by codegen as == */
        case TOK_BANG_EQ:   return '!'; /* handled by codegen as != */
        case TOK_LT_EQ:     return 'L'; /* <=  */
        case TOK_GT_EQ:     return 'G'; /* >=  */
        case TOK_AMP_AMP:   return '&';
        case TOK_PIPE_PIPE: return '|';
        case TOK_SHL:       return '{'; /* << */
        case TOK_SHR:       return '}'; /* >> */
        default:            return '?';
    }
}

static ast_node_t *parse_binop(vbs_parser_t *p, int min_prec)
{
    ast_node_t *lhs = parse_primary(p);
    for (;;) {
        int prec = get_binop_prec(p->current.kind);
        if (prec < min_prec) break;
        vbs_tok_kind_t op_tok = p->current.kind;
        advance_tok(p);
        ast_node_t *rhs  = parse_binop(p, prec + 1);
        ast_node_t *node = alloc_node(AST_BINOP);
        node->u.binop.op  = tok_to_binop_char(op_tok);
        node->u.binop.lhs = lhs;
        node->u.binop.rhs = rhs;
        lhs = node;
    }
    return lhs;
}

static ast_node_t *parse_expr(vbs_parser_t *p)
{
    ast_node_t *lhs = parse_binop(p, 0);

    /* Assignment */
    if (match(p, TOK_EQ)) {
        ast_node_t *rhs   = parse_expr(p);
        ast_node_t *node  = alloc_node(AST_ASSIGN);
        node->u.binop.lhs = lhs;
        node->u.binop.rhs = rhs;
        return node;
    }
    /* Compound assignment += -= *= /= */
    vbs_tok_kind_t op = p->current.kind;
    if (op == TOK_PLUS_EQ || op == TOK_MINUS_EQ ||
        op == TOK_STAR_EQ || op == TOK_SLASH_EQ) {
        advance_tok(p);
        ast_node_t *rhs    = parse_expr(p);
        ast_node_t *binop  = alloc_node(AST_BINOP);
        binop->u.binop.op  = (op == TOK_PLUS_EQ)  ? '+' :
                             (op == TOK_MINUS_EQ) ? '-' :
                             (op == TOK_STAR_EQ)  ? '*' : '/';
        binop->u.binop.lhs = lhs;
        binop->u.binop.rhs = rhs;
        ast_node_t *node   = alloc_node(AST_ASSIGN);
        node->u.binop.lhs  = lhs;
        node->u.binop.rhs  = binop;
        return node;
    }

    return lhs;
}

/* -------------------------------------------------------------------------- */
/* Statement parsing                                                           */
/* -------------------------------------------------------------------------- */

static ast_node_t *parse_decl_stmt(vbs_parser_t *p)
{
    ast_type_t  ty   = parse_type(p);
    vbs_token_t name = expect(p, TOK_IDENT, "expected variable name");
    ast_node_t *n    = alloc_node(AST_DECL);
    n->u.decl.type   = ty;
    n->u.decl.name   = tok_to_str(name);
    n->u.decl.init   = NULL;
    if (match(p, TOK_EQ))
        n->u.decl.init = parse_expr(p);
    expect(p, TOK_SEMI, "expected ';' after declaration");
    return n;
}

static ast_node_t *parse_return_stmt(vbs_parser_t *p)
{
    ast_node_t *n = alloc_node(AST_RETURN);
    n->u.ret.value = NULL;
    if (!check(p, TOK_SEMI))
        n->u.ret.value = parse_expr(p);
    expect(p, TOK_SEMI, "expected ';' after return");
    return n;
}

static ast_node_t *parse_if_stmt(vbs_parser_t *p)
{
    expect(p, TOK_LPAREN, "expected '(' after if");
    ast_node_t *cond = parse_expr(p);
    expect(p, TOK_RPAREN, "expected ')' after if condition");
    ast_node_t *then_b = parse_block(p);
    ast_node_t *else_b = NULL;
    if (match(p, TOK_ELSE))
        else_b = check(p, TOK_IF) ? (advance_tok(p), parse_if_stmt(p))
                                  : parse_block(p);
    ast_node_t *n       = alloc_node(AST_IF);
    n->u.if_s.cond      = cond;
    n->u.if_s.then_body = then_b;
    n->u.if_s.else_body = else_b;
    return n;
}

static ast_node_t *parse_for_stmt(vbs_parser_t *p)
{
    expect(p, TOK_LPAREN, "expected '(' after for");
    ast_node_t *init = NULL;
    if (!check(p, TOK_SEMI))
        init = parse_stmt(p);
    else
        expect(p, TOK_SEMI, "expected ';'");
    ast_node_t *cond = NULL;
    if (!check(p, TOK_SEMI)) cond = parse_expr(p);
    expect(p, TOK_SEMI, "expected ';' in for");
    ast_node_t *post = NULL;
    if (!check(p, TOK_RPAREN)) post = parse_expr(p);
    expect(p, TOK_RPAREN, "expected ')' after for clauses");
    ast_node_t *body  = parse_block(p);
    ast_node_t *n     = alloc_node(AST_FOR);
    n->u.for_s.init   = init;
    n->u.for_s.cond   = cond;
    n->u.for_s.post   = post;
    n->u.for_s.body   = body;
    return n;
}

static ast_node_t *parse_while_stmt(vbs_parser_t *p)
{
    expect(p, TOK_LPAREN, "expected '(' after while");
    ast_node_t *cond = parse_expr(p);
    expect(p, TOK_RPAREN, "expected ')' after while condition");
    ast_node_t *body  = parse_block(p);
    ast_node_t *n     = alloc_node(AST_WHILE);
    n->u.while_s.cond = cond;
    n->u.while_s.body = body;
    return n;
}

static ast_node_t *parse_stmt(vbs_parser_t *p)
{
    if (p->had_error) return NULL;

    if (match(p, TOK_RETURN)) return parse_return_stmt(p);
    if (match(p, TOK_IF))     return parse_if_stmt(p);
    if (match(p, TOK_FOR))    return parse_for_stmt(p);
    if (match(p, TOK_WHILE))  return parse_while_stmt(p);
    if (check(p, TOK_LBRACE)) return parse_block(p);

    /* Declaration or expression statement */
    vbs_tok_kind_t k = p->current.kind;
    if (k == TOK_INT   || k == TOK_FLOAT  || k == TOK_DOUBLE ||
        k == TOK_CHAR  || k == TOK_VOID   || k == TOK_SIZE_T ||
        k == TOK_CONST || k == TOK_STATIC)
        return parse_decl_stmt(p);

    ast_node_t *e = parse_expr(p);
    expect(p, TOK_SEMI, "expected ';' after expression statement");
    ast_node_t *s    = alloc_node(AST_EXPR_STMT);
    s->u.expr_stmt.e = e;
    return s;
}

static ast_node_t *parse_block(vbs_parser_t *p)
{
    expect(p, TOK_LBRACE, "expected '{'");
    ast_node_t *blk  = alloc_node(AST_BLOCK);
    blk->u.block.n   = 0;
    while (!check(p, TOK_RBRACE) && !check(p, TOK_EOF) && !p->had_error) {
        if (blk->u.block.n < AST_MAX_STMTS)
            blk->u.block.stmts[blk->u.block.n++] = parse_stmt(p);
        else { parser_error(p, "too many statements in block"); break; }
    }
    expect(p, TOK_RBRACE, "expected '}'");
    return blk;
}

/* -------------------------------------------------------------------------- */
/* Public API                                                                  */
/* -------------------------------------------------------------------------- */

vbs_parser_t *vbs_parser_create(vbs_lexer_t *l)
{
    vbs_parser_t *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->lexer   = l;
    p->current = vbs_lexer_next(l);
    return p;
}

void vbs_parser_destroy(vbs_parser_t *p) { free(p); }

ast_function_t *vbs_parse_function(vbs_parser_t *p)
{
    /* Optional 'static' */
    int is_static = match(p, TOK_STATIC);
    (void)is_static;

    ast_type_t ret_type = parse_type(p);
    vbs_token_t name    = expect(p, TOK_IDENT, "expected function name");

    ast_function_t *fn  = calloc(1, sizeof(*fn));
    if (!fn) return NULL;
    fn->name            = tok_to_str(name);
    fn->return_type     = ret_type;

    expect(p, TOK_LPAREN, "expected '(' in function definition");
    fn->param_count = 0;
    while (!check(p, TOK_RPAREN) && !check(p, TOK_EOF) && !p->had_error) {
        if (fn->param_count >= AST_MAX_PARAMS) {
            parser_error(p, "too many parameters");
            break;
        }
        ast_type_t  pty  = parse_type(p);
        vbs_token_t pnam = expect(p, TOK_IDENT, "expected param name");
        fn->params[fn->param_count].type = pty;
        fn->params[fn->param_count].name = tok_to_str(pnam);
        fn->param_count++;
        if (!match(p, TOK_COMMA)) break;
    }
    expect(p, TOK_RPAREN, "expected ')' after params");

    fn->body = parse_block(p);

    if (p->had_error) {
        fprintf(stderr, "[VBS] %s\n", p->error_msg);
        free(fn->name);
        free(fn);
        return NULL;
    }
    return fn;
}

void ast_function_free(ast_function_t *fn)
{
    if (!fn) return;
    free(fn->name);
    for (int i = 0; i < fn->param_count; i++)
        free(fn->params[i].name);
    /* Note: full recursive AST free omitted for brevity —
       add if long-running sessions accumulate many patches */
    free(fn);
}