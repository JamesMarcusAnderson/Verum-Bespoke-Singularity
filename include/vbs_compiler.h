/* =============================================================================
 * vbs_compiler.h
 * Verum-Bespoke-Singularity — embedded C11 compiler master header
 *
 * Single include for the entire compiler pipeline:
 *   lexer → parser → typecheck → emit (x86_64 / aarch64 / ptx / msl / spirv / gcn)
 *
 * No external dependencies — pure C11, stdint, stddef, stdarg only.
 * =========================================================================== */

#ifndef VBS_COMPILER_H
#define VBS_COMPILER_H

#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------- */
/* Limits                                                                      */
/* -------------------------------------------------------------------------- */

#define AST_MAX_STMTS   256
#define AST_MAX_PARAMS  16
#define AST_MAX_ARGS    16

/* -------------------------------------------------------------------------- */
/* Token kinds                                                                 */
/* -------------------------------------------------------------------------- */

typedef enum {
    /* Sentinel */
    TOK_EOF = 0,
    TOK_UNKNOWN,

    /* Type keywords */
    TOK_VOID,
    TOK_INT,
    TOK_FLOAT,
    TOK_DOUBLE,
    TOK_CHAR,
    TOK_SIZE_T,
    TOK_CONST,
    TOK_STATIC,

    /* Control keywords */
    TOK_IF,
    TOK_ELSE,
    TOK_FOR,
    TOK_WHILE,
    TOK_RETURN,

    /* Literals */
    TOK_IDENT,
    TOK_INT_LIT,
    TOK_FLOAT_LIT,
    TOK_STRING_LIT,

    /* Punctuation */
    TOK_LPAREN,       /*  (  */
    TOK_RPAREN,       /*  )  */
    TOK_LBRACE,       /*  {  */
    TOK_RBRACE,       /*  }  */
    TOK_LBRACKET,     /*  [  */
    TOK_RBRACKET,     /*  ]  */
    TOK_COMMA,        /*  ,  */
    TOK_SEMI,         /*  ;  */
    TOK_DOT,          /*  .  */
    TOK_ARROW,        /* ->  */

    /* Arithmetic */
    TOK_PLUS,         /*  +  */
    TOK_MINUS,        /*  -  */
    TOK_STAR,         /*  *  */
    TOK_SLASH,        /*  /  */
    TOK_PERCENT,      /*  %  */

    /* Assignment */
    TOK_EQ,           /*  =  */
    TOK_PLUS_EQ,      /* +=  */
    TOK_MINUS_EQ,     /* -=  */
    TOK_STAR_EQ,      /* *=  */
    TOK_SLASH_EQ,     /* /=  */

    /* Increment / Decrement */
    TOK_PLUS_PLUS,    /* ++  */
    TOK_MINUS_MINUS,  /* --  */

    /* Comparison */
    TOK_EQ_EQ,        /* ==  */
    TOK_BANG_EQ,      /* !=  */
    TOK_LT,           /*  <  */
    TOK_GT,           /*  >  */
    TOK_LT_EQ,        /* <=  */
    TOK_GT_EQ,        /* >=  */

    /* Logical */
    TOK_AMP_AMP,      /* &&  */
    TOK_PIPE_PIPE,    /* ||  */
    TOK_BANG,         /*  !  */

    /* Bitwise */
    TOK_AMP,          /*  &  */
    TOK_PIPE,         /*  |  */
    TOK_CARET,        /*  ^  */
    TOK_TILDE,        /*  ~  */
    TOK_SHL,          /* <<  */
    TOK_SHR,          /* >>  */
} vbs_tok_kind_t;

/* -------------------------------------------------------------------------- */
/* Token                                                                       */
/* -------------------------------------------------------------------------- */

typedef struct {
    vbs_tok_kind_t  kind;
    const char     *start;   /* pointer into original source — not NUL-terminated */
    size_t          len;
    int             line;
    int             col;
} vbs_token_t;

/* -------------------------------------------------------------------------- */
/* Lexer (opaque)                                                              */
/* -------------------------------------------------------------------------- */

typedef struct vbs_lexer vbs_lexer_t;

vbs_lexer_t *vbs_lexer_create  (const char *src, size_t len);
void         vbs_lexer_destroy (vbs_lexer_t *l);
vbs_token_t  vbs_lexer_next    (vbs_lexer_t *l);
vbs_token_t  vbs_lexer_peek    (vbs_lexer_t *l);

/* -------------------------------------------------------------------------- */
/* AST — types                                                                 */
/* -------------------------------------------------------------------------- */

typedef enum {
    AST_TY_VOID = 0,
    AST_TY_INT,
    AST_TY_FLOAT,
    AST_TY_DOUBLE,
    AST_TY_CHAR,
    AST_TY_SIZE_T,
} ast_type_kind_t;

typedef struct {
    ast_type_kind_t kind;
    int             ptr_depth;   /* 0 = value, 1 = *, 2 = **, … */
    int             is_const;
} ast_type_t;

/* -------------------------------------------------------------------------- */
/* AST — node kinds                                                            */
/* -------------------------------------------------------------------------- */

typedef enum {
    /* Literals */
    AST_INT_LIT = 0,
    AST_FLOAT_LIT,

    /* References */
    AST_VAR_REF,
    AST_SUBSCRIPT,      /* array[index]       */

    /* Expressions */
    AST_BINOP,
    AST_UNARY,
    AST_ASSIGN,
    AST_CALL,

    /* Statements */
    AST_DECL,
    AST_RETURN,
    AST_BLOCK,
    AST_IF,
    AST_FOR,
    AST_WHILE,
    AST_EXPR_STMT,
} ast_node_kind_t;

/* -------------------------------------------------------------------------- */
/* AST — node                                                                  */
/* -------------------------------------------------------------------------- */

typedef struct ast_node ast_node_t;

struct ast_node {
    ast_node_kind_t kind;
    ast_type_t      resolved_type;   /* filled by typecheck.c */

    union {
        /* AST_INT_LIT */
        long long       int_val;

        /* AST_FLOAT_LIT */
        float           float_val;

        /* AST_VAR_REF */
        struct { char *name; } var;

        /* AST_BINOP, AST_ASSIGN, AST_SUBSCRIPT */
        struct {
            int          op;     /* character code e.g. '+', '-', '*', '/' */
            ast_node_t  *lhs;
            ast_node_t  *rhs;
        } binop;

        /* AST_UNARY */
        struct {
            int          op;
            ast_node_t  *operand;
        } unary;

        /* AST_CALL */
        struct {
            char        *name;
            int          argc;
            ast_node_t  *args[AST_MAX_ARGS];
        } call;

        /* AST_DECL */
        struct {
            ast_type_t   type;
            char        *name;
            ast_node_t  *init;   /* NULL if no initialiser */
        } decl;

        /* AST_RETURN */
        struct {
            ast_node_t  *value;  /* NULL for bare return */
        } ret;

        /* AST_BLOCK */
        struct {
            int          n;
            ast_node_t  *stmts[AST_MAX_STMTS];
        } block;

        /* AST_IF */
        struct {
            ast_node_t  *cond;
            ast_node_t  *then_body;
            ast_node_t  *else_body;  /* NULL if no else */
        } if_s;

        /* AST_FOR */
        struct {
            ast_node_t  *init;   /* NULL if empty */
            ast_node_t  *cond;   /* NULL if empty */
            ast_node_t  *post;   /* NULL if empty */
            ast_node_t  *body;
        } for_s;

        /* AST_WHILE */
        struct {
            ast_node_t  *cond;
            ast_node_t  *body;
        } while_s;

        /* AST_EXPR_STMT */
        struct {
            ast_node_t  *e;
        } expr_stmt;
    } u;
};

/* -------------------------------------------------------------------------- */
/* AST — function                                                              */
/* -------------------------------------------------------------------------- */

typedef struct {
    char        *name;
    ast_type_t   return_type;
    int          param_count;
    struct {
        ast_type_t  type;
        char       *name;
    } params[AST_MAX_PARAMS];
    ast_node_t  *body;   /* always AST_BLOCK */
} ast_function_t;

void ast_function_free(ast_function_t *fn);

/* -------------------------------------------------------------------------- */
/* Parser (opaque)                                                             */
/* -------------------------------------------------------------------------- */

typedef struct vbs_parser vbs_parser_t;

vbs_parser_t   *vbs_parser_create   (vbs_lexer_t *l);
void            vbs_parser_destroy  (vbs_parser_t *p);
ast_function_t *vbs_parse_function  (vbs_parser_t *p);

/* -------------------------------------------------------------------------- */
/* Typechecker                                                                 */
/* -------------------------------------------------------------------------- */

int vbs_typecheck(ast_function_t *fn, char *err_out, size_t err_sz);

/* -------------------------------------------------------------------------- */
/* Driver — full pipeline entry points                                         */
/* -------------------------------------------------------------------------- */

/* Host native code — returns mmap'd fn pointer, caller must vbs_jit_free() */
int vbs_compile(const char   *src,      size_t  src_len,
                void        **out_fn,   size_t *out_size,
                char         *err_out,  size_t  err_sz);

/* GPU IR text / binary paths */
int vbs_compile_to_ptx   (const char     *src,  size_t  src_len,
                           char           *ptx_out,  size_t  ptx_sz,
                           char           *err_out,  size_t  err_sz);

int vbs_compile_to_msl   (const char     *src,  size_t  src_len,
                           char           *msl_out,  size_t  msl_sz,
                           char           *err_out,  size_t  err_sz);

int vbs_compile_to_spirv (const char     *src,  size_t  src_len,
                           uint32_t       *spirv_out, size_t *out_words,
                           char           *err_out,   size_t  err_sz);

int vbs_compile_to_gcn   (const char     *src,  size_t  src_len,
                           uint32_t       *gcn_out,   size_t *out_words,
                           char           *err_out,   size_t  err_sz);

/* -------------------------------------------------------------------------- */
/* Validators                                                                  */
/* -------------------------------------------------------------------------- */

int ptx_validate    (const char     *ptx,   size_t len,
                     char           *err,   size_t err_sz);

int spirv_validate  (const uint32_t *words, size_t word_count,
                     char           *err,   size_t err_sz);

int gcn_validate    (const uint32_t *words, size_t word_count,
                     char           *err,   size_t err_sz);

#ifdef __cplusplus
}
#endif

#endif /* VBS_COMPILER_H */