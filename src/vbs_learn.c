/* vbs_learn.c — correction memory.
 *
 * Stores prompt -> corrected-answer pairs in a TSV file and recalls
 * the best-matching correction for a new prompt by word overlap:
 * the fraction of the prompt's words that also appear in the stored
 * prompt. Words are maximal runs of alphanumeric characters, matched
 * case-insensitively. Ties keep the first (earliest) stored pair,
 * because recall only replaces the best score on a strict improvement.
 */

#include "vbs_learn.h"
#include "vbs_port.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* Flatten tabs and newlines to spaces so a value fits on one TSV line. */
static void flatten(char *s) {
    for (; *s; s++) {
        if (*s == '\t' || *s == '\n' || *s == '\r') *s = ' ';
    }
}

/* Advance *cursor past separators and return the next word's start,
 * or NULL when no word remains. *len_out gets the word's length. */
static const char *next_word(const char **cursor, size_t *len_out) {
    const char *p = *cursor;
    while (*p && !isalnum((unsigned char)*p)) p++;
    const char *start = p;
    while (*p && isalnum((unsigned char)*p)) p++;
    *cursor = p;
    *len_out = (size_t)(p - start);
    return p == start ? NULL : start;
}

/* True when `word` (length n) appears as a whole word in `text`. */
static bool contains_word(const char *text, const char *word, size_t n) {
    const char *cursor = text;
    size_t len = 0;
    const char *w;
    while ((w = next_word(&cursor, &len)) != NULL) {
        if (len == n && vbs_strncasecmp(w, word, n) == 0) return true;
    }
    return false;
}

/* Fraction of a's words that appear in b (0 when a has no words). */
static double word_overlap(const char *a, const char *b) {
    int matched = 0;
    int total = 0;
    const char *cursor = a;
    size_t len = 0;
    const char *w;
    while ((w = next_word(&cursor, &len)) != NULL) {
        total++;
        if (contains_word(b, w, len)) matched++;
    }
    return total ? (double)matched / total : 0.0;
}

bool vbs_learn_append(const char *path, const char *prompt,
                      const char *correct) {
    FILE *f = fopen(path, "a");
    if (!f) return false;
    char *p = vbs_strdup(prompt);
    char *c = vbs_strdup(correct);
    if (!p || !c) {
        free(p);
        free(c);
        fclose(f);
        return false;
    }
    flatten(p);
    flatten(c);
    fprintf(f, "%s\t%s\n", p, c);
    free(p);
    free(c);
    fclose(f);
    return true;
}

double vbs_learn_recall(const char *path, const char *prompt,
                        char *out_correct, size_t out_cap) {
    FILE *f = fopen(path, "r");
    if (!f) return 0.0;
    char line[8192];
    double best = 0.0;
    out_correct[0] = '\0';
    while (fgets(line, sizeof(line), f)) {
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = '\0';
        char *corr = tab + 1;
        corr[strcspn(corr, "\r\n")] = '\0';
        double score = word_overlap(prompt, line);
        if (score > best) {
            best = score;
            snprintf(out_correct, out_cap, "%s", corr);
        }
    }
    fclose(f);
    return best;
}
