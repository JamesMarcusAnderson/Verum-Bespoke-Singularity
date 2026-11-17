/* =============================================================================
 * lexer.c
 * Verum-Bespoke-Singularity embedded C11 compiler — lexer / tokenizer
 *
 * Tokenizes the C11 subset the LLM is permitted to generate:
 *   - Types:      void, int, float, double, char, size_t, const, static
 *   - Control:    if, else, for, while, return
 *   - Operators:  + - * / % = == != < > <= >= && || ! & | ^ ~ << >>
 *   - Literals:   integer, float, string
 *   - Punctuation: ( ) { } [ ] , ; .
 *
 * No preprocessor. No #include. No macros. The LLM is the optimizer.
 * =========================================================================== */

#include "../../include/vbs_compiler.h"
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>

/* -------------------------------------------------------------------------- */
/* Keyword table                                                               */
/* -------------------------------------------------------------------------- */

typedef struct { const char *word; vbs_tok_kind_t kind; } keyword_entry_t;

static const keyword_entry_t kKeywords[] = {
    { "void",     TOK_VOID     },
    { "int",      TOK_INT      },
    { "float",    TOK_FLOAT    },
    { "double",   TOK_DOUBLE   },
    { "char",     TOK_CHAR     },
    { "size_t",   TOK_SIZE_T   },
    { "const",    TOK_CONST    },
    { "static",   TOK_STATIC   },
    { "if",       TOK_IF       },
    { "else",     TOK_ELSE     },
    { "for",      TOK_FOR      },
    { "while",    TOK_WHILE    },
    { "return",   TOK_RETURN   },
    { NULL,       TOK_EOF      },
};

/* -------------------------------------------------------------------------- */
/* Lexer state                                                                 */
/* -------------------------------------------------------------------------- */

struct vbs_lexer {
    const char *src;
    size_t      pos;
    size_t      len;
    int         line;
    int         col;
};

vbs_lexer_t *vbs_lexer_create(const char *src, size_t len)
{
    vbs_lexer_t *l = calloc(1, sizeof(*l));
    if (!l) return NULL;
    l->src  = src;
    l->len  = len;
    l->line = 1;
    l->col  = 1;
    return l;
}

void vbs_lexer_destroy(vbs_lexer_t *l) { free(l); }

/* -------------------------------------------------------------------------- */
/* Internal helpers                                                            */
/* -------------------------------------------------------------------------- */

static inline char peek(vbs_lexer_t *l)
{
    return (l->pos < l->len) ? l->src[l->pos] : '\0';
}

static inline char peek2(vbs_lexer_t *l)
{
    return (l->pos + 1 < l->len) ? l->src[l->pos + 1] : '\0';
}

static inline char advance(vbs_lexer_t *l)
{
    char c = l->src[l->pos++];
    if (c == '\n') { l->line++; l->col = 1; }
    else            { l->col++; }
    return c;
}

static void skip_whitespace_and_comments(vbs_lexer_t *l)
{
    for (;;) {
        /* Whitespace */
        while (l->pos < l->len && isspace((unsigned char)peek(l)))
            advance(l);

        /* Line comment */
        if (peek(l) == '/' && peek2(l) == '/') {
            while (l->pos < l->len && peek(l) != '\n') advance(l);
            continue;
        }

        /* Block comment */
        if (peek(l) == '/' && peek2(l) == '*') {
            advance(l); advance(l);
            while (l->pos + 1 < l->len) {
                if (peek(l) == '*' && peek2(l) == '/') {
                    advance(l); advance(l); break;
                }
                advance(l);
            }
            continue;
        }
        break;
    }
}

static vbs_token_t make_tok(vbs_lexer_t *l, vbs_tok_kind_t k,
                             const char *start, size_t len)
{
    vbs_token_t t;
    t.kind  = k;
    t.start = start;
    t.len   = len;
    t.line  = l->line;
    t.col   = l->col;
    return t;
}

/* -------------------------------------------------------------------------- */
/* Number literals                                                             */
/* -------------------------------------------------------------------------- */

static vbs_token_t lex_number(vbs_lexer_t *l)
{
    const char *start = l->src + l->pos;
    int         is_float = 0;

    while (l->pos < l->len && isdigit((unsigned char)peek(l))) advance(l);
    if (peek(l) == '.') { is_float = 1; advance(l);
        while (l->pos < l->len && isdigit((unsigned char)peek(l))) advance(l);
    }
    if (peek(l) == 'e' || peek(l) == 'E') {
        is_float = 1; advance(l);
        if (peek(l) == '+' || peek(l) == '-') advance(l);
        while (l->pos < l->len && isdigit((unsigned char)peek(l))) advance(l);
    }
    if (peek(l) == 'f' || peek(l) == 'F') { is_float = 1; advance(l); }

    size_t len = (size_t)(l->src + l->pos - start);
    return make_tok(l, is_float ? TOK_FLOAT_LIT : TOK_INT_LIT, start, len);
}

/* -------------------------------------------------------------------------- */
/* Identifier / keyword                                                        */
/* -------------------------------------------------------------------------- */

static vbs_token_t lex_ident(vbs_lexer_t *l)
{
    const char *start = l->src + l->pos;
    while (l->pos < l->len &&
           (isalnum((unsigned char)peek(l)) || peek(l) == '_'))
        advance(l);

    size_t len = (size_t)(l->src + l->pos - start);

    /* Keyword lookup */
    for (int i = 0; kKeywords[i].word; i++) {
        if (strlen(kKeywords[i].word) == len &&
            memcmp(kKeywords[i].word, start, len) == 0)
            return make_tok(l, kKeywords[i].kind, start, len);
    }
    return make_tok(l, TOK_IDENT, start, len);
}

/* -------------------------------------------------------------------------- */
/* String literal                                                              */
/* -------------------------------------------------------------------------- */

static vbs_token_t lex_string(vbs_lexer_t *l)
{
    const char *start = l->src + l->pos;
    advance(l); /* opening " */
    while (l->pos < l->len && peek(l) != '"') {
        if (peek(l) == '\\') advance(l);
        advance(l);
    }
    if (l->pos < l->len) advance(l); /* closing " */
    return make_tok(l, TOK_STRING_LIT, start, (size_t)(l->src + l->pos - start));
}

/* -------------------------------------------------------------------------- */
/* Public: vbs_lexer_next                                                      */
/* -------------------------------------------------------------------------- */

vbs_token_t vbs_lexer_next(vbs_lexer_t *l)
{
    skip_whitespace_and_comments(l);
    if (l->pos >= l->len)
        return make_tok(l, TOK_EOF, l->src + l->pos, 0);

    char c = peek(l);

    if (isdigit((unsigned char)c)) return lex_number(l);
    if (isalpha((unsigned char)c) || c == '_') return lex_ident(l);
    if (c == '"') return lex_string(l);

    const char *start = l->src + l->pos;
    advance(l);
    char n = peek(l);

#define DUO(a, b, k)  if (c == (a) && n == (b)) { advance(l); \
        return make_tok(l, k, start, 2); }

    DUO('=','=', TOK_EQ_EQ)
    DUO('!','=', TOK_BANG_EQ)
    DUO('<','=', TOK_LT_EQ)
    DUO('>','=', TOK_GT_EQ)
    DUO('&','&', TOK_AMP_AMP)
    DUO('|','|', TOK_PIPE_PIPE)
    DUO('<','<', TOK_SHL)
    DUO('>','>', TOK_SHR)
    DUO('+','=', TOK_PLUS_EQ)
    DUO('-','=', TOK_MINUS_EQ)
    DUO('*','=', TOK_STAR_EQ)
    DUO('/','=', TOK_SLASH_EQ)
    DUO('+','+', TOK_PLUS_PLUS)
    DUO('-','-', TOK_MINUS_MINUS)
    DUO('-','>', TOK_ARROW)
#undef DUO

    /* Single-char operators and punctuation */
    vbs_tok_kind_t k;
    switch (c) {
        case '(': k = TOK_LPAREN;    break;
        case ')': k = TOK_RPAREN;    break;
        case '{': k = TOK_LBRACE;    break;
        case '}': k = TOK_RBRACE;    break;
        case '[': k = TOK_LBRACKET;  break;
        case ']': k = TOK_RBRACKET;  break;
        case ',': k = TOK_COMMA;     break;
        case ';': k = TOK_SEMI;      break;
        case '.': k = TOK_DOT;       break;
        case '+': k = TOK_PLUS;      break;
        case '-': k = TOK_MINUS;     break;
        case '*': k = TOK_STAR;      break;
        case '/': k = TOK_SLASH;     break;
        case '%': k = TOK_PERCENT;   break;
        case '=': k = TOK_EQ;        break;
        case '<': k = TOK_LT;        break;
        case '>': k = TOK_GT;        break;
        case '!': k = TOK_BANG;      break;
        case '&': k = TOK_AMP;       break;
        case '|': k = TOK_PIPE;      break;
        case '^': k = TOK_CARET;     break;
        case '~': k = TOK_TILDE;     break;
        default:  k = TOK_UNKNOWN;   break;
    }
    return make_tok(l, k, start, 1);
}

vbs_token_t vbs_lexer_peek(vbs_lexer_t *l)
{
    size_t save_pos  = l->pos;
    int    save_line = l->line;
    int    save_col  = l->col;
    vbs_token_t t = vbs_lexer_next(l);
    l->pos  = save_pos;
    l->line = save_line;
    l->col  = save_col;
    return t;
}