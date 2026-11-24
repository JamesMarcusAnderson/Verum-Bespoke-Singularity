#ifndef VBS_LEARN_H
#define VBS_LEARN_H

#include <stddef.h>
#include <stdbool.h>

/* Correction memory — a recall layer, NOT weight fine-tuning.
 *
 * The engine has no backprop through the transformer, so it cannot
 * gradient-train model weights. What this layer does, honestly:
 *   1. --learn appends a prompt -> corrected answer pair to a local
 *      TSV file.
 *   2. --generate --memory <file> recalls the stored correction
 *      whose prompt overlaps most with the current prompt and
 *      prepends it, so a corrected answer is reused instead of
 *      re-generating the same mistake.
 *   3. Recall is scored and printed, and only used when the
 *      overlap is above zero.
 *
 * File format, one pair per line: prompt<TAB>correct. Tabs and
 * newlines inside either side are replaced with spaces on write. */
bool vbs_learn_append(const char *path, const char *prompt,
                      const char *correct);
double vbs_learn_recall(const char *path, const char *prompt,
                        char *out_correct, size_t out_cap);

#endif /* VBS_LEARN_H */
