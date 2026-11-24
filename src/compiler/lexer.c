/* lexer.c
 * Verum-Bespoke-Singularity embedded C11 compiler — lexer / tokenizer.
 *
 * Tokenizes the C11 subset the LLM is permitted to generate:
 *   - Types:       void, int, float, double, char, size_t, const, static
 *   - Control:     if, else, for, while, return
 *   - Operators:   + - * / % = == != < > <= >= && || ! & | ^ ~ << >>
 *   - Literals:    integer, float, string
 *   - Punctuation: ( ) { } [ ] , ; .
 *
 * No preprocessor. No #include. No macros. The LLM is the optimizer.
 */

#include "../../include/vbs_compiler.h"
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>

/* Keyword table: word -> token kind, in declaration order. */
typedef struct {
    const char *word;
    vbs_tok_kind_t kind;
} keyword_entry_t;

static const keyword_entry_t k_keywords[] = {
    { "void",   TOK_VOID   },
    { "int",    TOK_INT    },
    { "float",  TOK_FLOAT  },
    { "double", TOK_DOUBLE },
    { "char",   TOK_CHAR   },
    { "size_t", TOK_SIZE_T },
    { "const",  TOK_CONST  },
    { "static", TOK_STATIC },
    { "if",     TOK_IF     },
    { "else",   TOK_ELSE   },
    { "for",    TOK_FOR    },
    { "while",  TOK_WHILE  },
    { "return", TOK_RETURN },
    { NULL,     TOK_EOF    },
};

/* Two-character operator table: first char, second char, kind. */
typedef struct {
    char first, second;
    vbs_tok_kind_t kind;
} duo_entry_t;

static const duo_entry_t k_duos[] = {
    { '=', '=', TOK_EQ_EQ       },
    { '!', '=', TOK_BANG_EQ     },
    { '<', '=', TOK_LT_EQ       },
    { '>', '=', TOK_GT_EQ       },
    { '&', '&', TOK_AMP_AMP     },
    { '|', '|', TOK_PIPE_PIPE   },
    { '<', '<', TOK_SHL         },
    { '>', '>', TOK_SHR         },
    { '+', '=', TOK_PLUS_EQ     },
    { '-', '=', TOK_MINUS_EQ    },
    { '*', '=', TOK_STAR_EQ     },
    { '/', '=', TOK_SLASH_EQ    },
    { '+', '+', TOK_PLUS_PLUS   },
    { '-', '-', TOK_MINUS_MINUS },
    { '-', '>', TOK_ARROW       },
};

/* Lexer state: a cursor over the source plus line/column tracking. */
struct vbs_lexer {
    const char *src;
    size_t      pos;
    size_t      len;
    int         line;
    int         col;
};

vbs_lexer_t *vbs_lexer_create(const char *src, size_t len) {
    vbs_lexer_t *l = calloc(1, sizeof(*l));
    if (!l) return NULL;
    l->src = src;
    l->len = len;
    l->line = 1;
    l->col = 1;
    return l;
}

void vbs_lexer_destroy(vbs_lexer_t *l) {
    free(l);
}

/* Look at the current / next character without consuming ('\0' at EOF). */
static inline char cur_char(vbs_lexer_t *l) {
    return (l->pos < l->len) ? l->src[l->pos] : '\0';
}
static inline char next_char(vbs_lexer_t *l) {
    return (l->pos + 1 < l->len) ? l->src[l->pos + 1] : '\0';
}

/* Consume one character, tracking line and column. */
static inline char take_char(vbs_lexer_t *l) {
    char c = l->src[l->pos++];
    if (c == '\n') {
        l->line++;
        l->col = 1;
    } else {
        l->col++;
    }
    return c;
}

static void skip_ws_comments(vbs_lexer_t *l) {
    for (;;) {
        while (l->pos < l->len && isspace((unsigned char)cur_char(l)))
            take_char(l);

        if (cur_char(l) == '/' && next_char(l) == '/') {
            while (l->pos < l->len && cur_char(l) != '\n') take_char(l);
            continue;
        }

        if (cur_char(l) == '/' && next_char(l) == '*') {
            take_char(l);
            take_char(l);
            while (l->pos + 1 < l->len) {
                if (cur_char(l) == '*' && next_char(l) == '/') {
                    take_char(l);
                    take_char(l);
                    break;
                }
                take_char(l);
            }
            continue;
        }
        break;
    }
}

static vbs_token_t make_token(vbs_lexer_t *l, vbs_tok_kind_t kind,
                              const char *start, size_t len) {
    vbs_token_t t;
    t.kind = kind;
    t.start = start;
    t.len = len;
    t.line = l->line;
    t.col = l->col;
    return t;
}

/* Number literal: digits, optional fraction, optional exponent,
 * optional f/F suffix. Any of the last three makes it a float. */
static vbs_token_t lex_number(vbs_lexer_t *l) {
    const char *start = l->src + l->pos;
    int is_float = 0;

    while (l->pos < l->len && isdigit((unsigned char)cur_char(l))) take_char(l);
    if (cur_char(l) == '.') {
        is_float = 1;
        take_char(l);
        while (l->pos < l->len && isdigit((unsigned char)cur_char(l))) take_char(l);
    }
    if (cur_char(l) == 'e' || cur_char(l) == 'E') {
        is_float = 1;
        take_char(l);
        if (cur_char(l) == '+' || cur_char(l) == '-') take_char(l);
        while (l->pos < l->len && isdigit((unsigned char)cur_char(l))) take_char(l);
    }
    if (cur_char(l) == 'f' || cur_char(l) == 'F') {
        is_float = 1;
        take_char(l);
    }

    size_t len = (size_t)(l->src + l->pos - start);
    return make_token(l, is_float ? TOK_FLOAT_LIT : TOK_INT_LIT, start, len);
}

/* Identifier or keyword: letters, digits, underscores; keywords
 * are matched by exact length + bytes against the table. */
static vbs_token_t lex_ident(vbs_lexer_t *l) {
    const char *start = l->src + l->pos;
    while (l->pos < l->len &&
           (isalnum((unsigned char)cur_char(l)) || cur_char(l) == '_'))
        take_char(l);

    size_t len = (size_t)(l->src + l->pos - start);
    for (int i = 0; k_keywords[i].word; i++) {
        if (strlen(k_keywords[i].word) == len &&
            memcmp(k_keywords[i].word, start, len) == 0)
            return make_token(l, k_keywords[i].kind, start, len);
    }
    return make_token(l, TOK_IDENT, start, len);
}

/* String literal: from the opening quote to the closing quote;
 * a backslash escapes the next character (including a quote). */
static vbs_token_t lex_string(vbs_lexer_t *l) {
    const char *start = l->src + l->pos;
    take_char(l); /* opening " */
    while (l->pos < l->len && cur_char(l) != '"') {
        if (cur_char(l) == '\\') take_char(l);
        take_char(l);
    }
    if (l->pos < l->len) take_char(l); /* closing " */
    return make_token(l, TOK_STRING_LIT, start,
                      (size_t)(l->src + l->pos - start));
}

/* Single-character operators and punctuation. */
static vbs_tok_kind_t single_char_kind(char c) {
    switch (c) {
        case '(': return TOK_LPAREN;
        case ')': return TOK_RPAREN;
        case '{': return TOK_LBRACE;
        case '}': return TOK_RBRACE;
        case '[': return TOK_LBRACKET;
        case ']': return TOK_RBRACKET;
        case ',': return TOK_COMMA;
        case ';': return TOK_SEMI;
        case '.': return TOK_DOT;
        case '+': return TOK_PLUS;
        case '-': return TOK_MINUS;
        case '*': return TOK_STAR;
        case '/': return TOK_SLASH;
        case '%': return TOK_PERCENT;
        case '=': return TOK_EQ;
        case '<': return TOK_LT;
        case '>': return TOK_GT;
        case '!': return TOK_BANG;
        case '&': return TOK_AMP;
        case '|': return TOK_PIPE;
        case '^': return TOK_CARET;
        case '~': return TOK_TILDE;
        default:  return TOK_UNKNOWN;
    }
}

vbs_token_t vbs_lexer_next(vbs_lexer_t *l) {
    skip_ws_comments(l);
    if (l->pos >= l->len)
        return make_token(l, TOK_EOF, l->src + l->pos, 0);

    char c = cur_char(l);

    if (isdigit((unsigned char)c)) return lex_number(l);
    if (isalpha((unsigned char)c) || c == '_') return lex_ident(l);
    if (c == '"') return lex_string(l);

    const char *start = l->src + l->pos;
    take_char(l);
    char n = cur_char(l);

    /* Two-character operators take priority over single chars. */
    for (size_t i = 0; i < sizeof(k_duos) / sizeof(k_duos[0]); i++) {
        if (c == k_duos[i].first && n == k_duos[i].second) {
            take_char(l);
            return make_token(l, k_duos[i].kind, start, 2);
        }
    }
    return make_token(l, single_char_kind(c), start, 1);
}

vbs_token_t vbs_lexer_peek(vbs_lexer_t *l) {
    size_t save_pos = l->pos;
    int save_line = l->line;
    int save_col = l->col;
    vbs_token_t t = vbs_lexer_next(l);
    l->pos = save_pos;
    l->line = save_line;
    l->col = save_col;
    return t;
}
