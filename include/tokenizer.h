/* tokenizer.h — BPE tokenizer: loads a tokenizer.json file
 * (HuggingFace format — the file format's own name, kept so the
 * loader's input is unambiguous) into an in-memory BPE table and
 * encodes/decodes UTF-8 text.
 */
#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tokenizer_t tokenizer_t;

/* Load a tokenizer.json file into an in-memory BPE table.
 * Returns NULL on failure. */
tokenizer_t *tokenizer_load(const char *path);

/* Encode UTF-8 text into an array of token IDs; returns the
 * number of tokens written. */
size_t tokenizer_encode(tokenizer_t *tok, const char *text, uint32_t *out_tokens, size_t max_tokens);

/* Decode an array of token IDs into a heap-allocated UTF-8 string
 * (caller frees). */
char *tokenizer_decode(tokenizer_t *tok, const uint32_t *tokens, size_t n_tokens);

/* The detected End-Of-Sequence token ID (-1 if not present). */
int tokenizer_eos_id(const tokenizer_t *tok);

/* Release all allocated vocabulary, hash tables, and merge trees. */
void tokenizer_free(tokenizer_t *tok);

#ifdef __cplusplus
}
#endif

#endif /* TOKENIZER_H */
