#ifndef VBS_LEARN_H
#define VBS_LEARN_H
#include <stddef.h>
#include <stdbool.h>
/* Self-learning, memory layer — NOT weight fine-tuning.
 * Verum has no backprop / optimizer, so it cannot gradient-train GGUF
 * weights. What this does, honestly:
 *  1. --learn stores prompt -> corrected answer in a local TSV file.
 *  2. --generate --memory <file> recalls the closest stored correction
 *     by word overlap and prepends it to the prompt, so a corrected
 *     answer is reused instead of re-generating the same mistake.
 *  3. Recall is scored, printed, and only used if overlap > 0.
 * File format, one per line: prompt<TAB>correct  (tabs/newlines -> space)
 */
bool vbs_learn_append(const char *path, const char *prompt, const char *correct);
double vbs_learn_recall(const char *path, const char *prompt, char *out_correct, size_t out_cap);
#endif
