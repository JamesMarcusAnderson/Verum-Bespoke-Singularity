#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tokenizer_t tokenizer_t;

// Loads a HuggingFace tokenizer.json file into an in-memory BPE table
tokenizer_t *tokenizer_load(const char *path);

// Encodes UTF-8 text into an array of token IDs; returns number of tokens written
size_t tokenizer_encode(tokenizer_t *tok, const char *text, uint32_t *out_tokens, size_t max_tokens);

// Decodes an array of token IDs into a heap-allocated UTF-8 string (caller frees)
char *tokenizer_decode(tokenizer_t *tok, const uint32_t *tokens, size_t n_tokens);

// Returns the detected End-Of-Sequence token ID (-1 if not present)
int tokenizer_eos_id(const tokenizer_t *tok);

// Releases all allocated vocabulary, hash tables, and merge trees
void tokenizer_free(tokenizer_t *tok);

#ifdef __cplusplus
}
#endif

#endif // TOKENIZER_H
