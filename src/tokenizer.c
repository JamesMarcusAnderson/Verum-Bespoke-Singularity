/* tokenizer.c — BPE tokenizer for tokenizer.json files
 * (HuggingFace format): a minimal JSON reader, a vocabulary map,
 * a merge table, the GPT-2 byte-level encoder, and the GPT-2
 * pre-tokenizer segmentation rules.
 */
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <ctype.h>

/* --- Minimal JSON reader over a memory buffer --- */

typedef struct {
    const char *p, *end;
} json_reader_t;

static void json_skip_ws(json_reader_t *r) {
    while (r->p < r->end &&
           (*r->p == ' ' || *r->p == '\t' || *r->p == '\n' || *r->p == '\r'))
        r->p++;
}

/* Consume `c` after whitespace; 1 if it was there. */
static int json_expect(json_reader_t *r, char c) {
    json_skip_ws(r);
    if (r->p < r->end && *r->p == c) {
        r->p++;
        return 1;
    }
    return 0;
}

/* Encode one code point as UTF-8; returns the byte count. */
static int utf8_encode_cp(uint32_t cp, char *out) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

static int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* Read up to 4 hex digits (a \u escape); stops at the first
 * non-hex character. */
static uint32_t json_hex4(json_reader_t *r) {
    uint32_t v = 0;
    for (int i = 0; i < 4 && r->p < r->end; i++, r->p++) {
        int d = hex_digit(*r->p);
        if (d < 0) break;
        v = v * 16 + (uint32_t)d;
    }
    return v;
}

/* Read a JSON string, decoding escapes into UTF-8. Returns a
 * malloc'd NUL-terminated buffer (length in *out_len), or NULL. */
static char *json_string(json_reader_t *r, size_t *out_len) {
    if (!json_expect(r, '"')) return NULL;
    size_t cap = 64, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;
    while (r->p < r->end && *r->p != '"') {
        if (len + 8 > cap) {
            cap *= 2;
            char *nb = (char *)realloc(buf, cap);
            if (!nb) {
                free(buf);
                return NULL;
            }
            buf = nb;
        }
        char c = *r->p++;
        if (c == '\\' && r->p < r->end) {
            char e = *r->p++;
            switch (e) {
                case '"':  buf[len++] = '"';  break;
                case '\\': buf[len++] = '\\'; break;
                case '/':  buf[len++] = '/';  break;
                case 'b':  buf[len++] = '\b'; break;
                case 'f':  buf[len++] = '\f'; break;
                case 'n':  buf[len++] = '\n'; break;
                case 'r':  buf[len++] = '\r'; break;
                case 't':  buf[len++] = '\t'; break;
                case 'u': {
                    if (r->end - r->p < 4) {
                        free(buf);
                        return NULL;
                    }
                    uint32_t cp = json_hex4(r);
                    if (cp >= 0xD800 && cp <= 0xDBFF &&
                        r->end - r->p >= 6 && r->p[0] == '\\' && r->p[1] == 'u') {
                        r->p += 2;
                        uint32_t lo = json_hex4(r);
                        if (lo >= 0xDC00 && lo <= 0xDFFF)
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    len += (size_t)utf8_encode_cp(cp, buf + len);
                } break;
                default: buf[len++] = e; break;
            }
        } else {
            buf[len++] = c;
        }
    }
    if (r->p >= r->end) {
        free(buf);
        return NULL;
    }
    r->p++;
    buf[len] = 0;
    if (out_len) *out_len = len;
    return buf;
}

static void json_skip_value(json_reader_t *r);

static void json_skip_array(json_reader_t *r) {
    if (!json_expect(r, '[')) return;
    while (r->p < r->end && *r->p != ']') {
        json_skip_value(r);
        json_skip_ws(r);
        if (r->p < r->end && *r->p == ',') r->p++;
    }
    json_expect(r, ']');
}

static void json_skip_object(json_reader_t *r) {
    if (!json_expect(r, '{')) return;
    while (r->p < r->end && *r->p != '}') {
        size_t kl;
        char *k = json_string(r, &kl);
        free(k);
        json_expect(r, ':');
        json_skip_value(r);
        json_skip_ws(r);
        if (r->p < r->end && *r->p == ',') r->p++;
    }
    json_expect(r, '}');
}

static void json_skip_value(json_reader_t *r) {
    json_skip_ws(r);
    if (r->p >= r->end) return;
    switch (*r->p) {
        case '"': {
            size_t l;
            char *s = json_string(r, &l);
            free(s);
            break;
        }
        case '[': json_skip_array(r); break;
        case '{': json_skip_object(r); break;
        default:
            while (r->p < r->end && !isspace((unsigned char)*r->p) &&
                   !strchr(",]}", *r->p))
                r->p++;
    }
}

/* --- Hash maps (open addressing, linear probing) --- */

static uint64_t fnv1a_hash(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static size_t pow2_at_least(size_t n) {
    size_t c = 64;
    while (c < n) c <<= 1;
    return c;
}

/* String -> id map (vocabulary lookup). */
typedef struct {
    char *key;
    size_t len;
    int id;
} str_slot_t;

typedef struct {
    str_slot_t *slots;
    size_t cap;
} str_map_t;

static void str_map_init(str_map_t *m, size_t expected) {
    m->cap = pow2_at_least(expected * 2 + 16);
    m->slots = (str_slot_t *)calloc(m->cap, sizeof(str_slot_t));
    for (size_t i = 0; i < m->cap; i++) m->slots[i].id = -1;
}

/* First definition wins: a duplicate key is ignored. */
static void str_map_put(str_map_t *m, const char *key, size_t len, int id) {
    size_t idx = (size_t)(fnv1a_hash(key, len) & (m->cap - 1));
    for (size_t probe = 0; probe < m->cap; probe++) {
        if (m->slots[idx].key == NULL) break;
        if (m->slots[idx].len == len && memcmp(m->slots[idx].key, key, len) == 0)
            return;
        idx = (idx + 1) & (m->cap - 1);
    }
    if (m->slots[idx].key != NULL) return;
    char *copy = (char *)malloc(len + 1);
    if (!copy) return;
    memcpy(copy, key, len);
    copy[len] = 0;
    m->slots[idx].key = copy;
    m->slots[idx].len = len;
    m->slots[idx].id = id;
}

static int str_map_get(const str_map_t *m, const char *key, size_t len) {
    size_t idx = (size_t)(fnv1a_hash(key, len) & (m->cap - 1));
    while (m->slots[idx].key != NULL) {
        if (m->slots[idx].len == len && memcmp(m->slots[idx].key, key, len) == 0)
            return m->slots[idx].id;
        idx = (idx + 1) & (m->cap - 1);
    }
    return -1;
}

/* (token, token) -> merged token map, keyed by rank. */
typedef struct {
    uint64_t key;
    int rank;
    uint32_t merged;
} merge_slot_t;

typedef struct {
    merge_slot_t *slots;
    size_t cap;
} merge_map_t;

#define MERGE_EMPTY UINT64_MAX

static void merge_map_init(merge_map_t *m, size_t expected) {
    m->cap = pow2_at_least(expected * 2 + 16);
    m->slots = (merge_slot_t *)calloc(m->cap, sizeof(merge_slot_t));
    for (size_t i = 0; i < m->cap; i++) m->slots[i].key = MERGE_EMPTY;
}

static void merge_map_put(merge_map_t *m, uint32_t a, uint32_t b, int rank, uint32_t merged) {
    uint64_t k = ((uint64_t)a << 32) | (uint64_t)b;
    size_t idx = (size_t)((k * 11400714819323198485ULL) & (m->cap - 1));
    for (size_t probe = 0; probe < m->cap; probe++) {
        if (m->slots[idx].key == MERGE_EMPTY) break;
        idx = (idx + 1) & (m->cap - 1);
    }
    if (m->slots[idx].key != MERGE_EMPTY) return;
    m->slots[idx].key = k;
    m->slots[idx].rank = rank;
    m->slots[idx].merged = merged;
}

static int merge_map_get(const merge_map_t *m, uint32_t a, uint32_t b, uint32_t *merged) {
    uint64_t k = ((uint64_t)a << 32) | (uint64_t)b;
    size_t idx = (size_t)((k * 11400714819323198485ULL) & (m->cap - 1));
    while (m->slots[idx].key != MERGE_EMPTY) {
        if (m->slots[idx].key == k) {
            if (merged) *merged = m->slots[idx].merged;
            return m->slots[idx].rank;
        }
        idx = (idx + 1) & (m->cap - 1);
    }
    return -1;
}

/* --- Vocabulary --- */

typedef struct {
    char *text;
    size_t len;
    int byte_value;
} vocab_entry_t;

struct tokenizer_t {
    vocab_entry_t *vocab;
    int vocab_size, vocab_cap;
    str_map_t text2id;
    merge_map_t merges;
    int byte_to_token[256];
    int eos_token_id;
};

static void vocab_put(tokenizer_t *tok, int id, const char *text, size_t len) {
    if (id < 0 || id >= tok->vocab_cap) return;
    free(tok->vocab[id].text);
    tok->vocab[id].text = (char *)malloc(len + 1);
    if (!tok->vocab[id].text) return;
    memcpy(tok->vocab[id].text, text, len);
    tok->vocab[id].text[len] = 0;
    tok->vocab[id].len = len;
    tok->vocab[id].byte_value = -1;
    if (id >= tok->vocab_size) tok->vocab_size = id + 1;
    str_map_put(&tok->text2id, text, len, id);
}

static int parse_vocab_object(json_reader_t *r, tokenizer_t *tok) {
    if (!json_expect(r, '{')) return 0;
    while (r->p < r->end && *r->p != '}') {
        size_t tl;
        char *text = json_string(r, &tl);
        if (!text) break;
        json_expect(r, ':');
        json_skip_ws(r);
        long long id = strtoll(r->p, (char **)&r->p, 10);
        vocab_put(tok, (int)id, text, tl);
        free(text);
        json_skip_ws(r);
        if (r->p < r->end && *r->p == ',') r->p++;
    }
    json_expect(r, '}');
    return 1;
}

static int parse_merges_array(json_reader_t *r, tokenizer_t *tok) {
    if (!json_expect(r, '[')) return 0;
    int rank = 0;
    while (r->p < r->end && *r->p != ']') {
        size_t ml;
        char *m = json_string(r, &ml);
        if (!m) break;
        const char *sp = (const char *)memchr(m, ' ', ml);
        if (sp && sp > m && sp < m + ml - 1) {
            size_t la = (size_t)(sp - m), lb = ml - la - 1;
            int id_a = str_map_get(&tok->text2id, m, la);
            int id_b = str_map_get(&tok->text2id, sp + 1, lb);
            char *mt = (char *)malloc(la + lb + 1);
            if (!mt) {
                free(m);
                break;
            }
            memcpy(mt, m, la);
            memcpy(mt + la, sp + 1, lb);
            mt[la + lb] = 0;
            int id_m = str_map_get(&tok->text2id, mt, la + lb);
            free(mt);
            if (id_a >= 0 && id_b >= 0 && id_m >= 0)
                merge_map_put(&tok->merges, (uint32_t)id_a, (uint32_t)id_b,
                              rank, (uint32_t)id_m);
            rank++;
        }
        free(m);
        json_skip_ws(r);
        if (r->p < r->end && *r->p == ',') r->p++;
    }
    json_expect(r, ']');
    return 1;
}

/* The GPT-2 byte encoder maps bytes to code points: printable
 * bytes keep their own code point; the rest are assigned 256, 257,
 * ... in byte order. */
static int byte_is_printable(int b) {
    return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
}

tokenizer_t *tokenizer_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long flen = ftell(f);
    if (flen <= 0) {
        fclose(f);
        return NULL;
    }
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)flen + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)flen, f) != (size_t)flen) {
        fclose(f);
        free(buf);
        return NULL;
    }
    buf[flen] = 0;
    fclose(f);

    tokenizer_t *tok = (tokenizer_t *)calloc(1, sizeof(tokenizer_t));
    if (!tok) {
        free(buf);
        return NULL;
    }
    tok->vocab_cap = 200000;
    tok->vocab = (vocab_entry_t *)calloc((size_t)tok->vocab_cap, sizeof(vocab_entry_t));
    if (!tok->vocab) {
        free(buf);
        free(tok);
        return NULL;
    }
    for (int i = 0; i < tok->vocab_cap; i++) tok->vocab[i].byte_value = -1;
    str_map_init(&tok->text2id, (size_t)tok->vocab_cap);
    merge_map_init(&tok->merges, 160000);
    tok->eos_token_id = -1;

    json_reader_t r = { buf, buf + flen };
    json_expect(&r, '{');
    while (r.p < r.end && *r.p != '}') {
        size_t kl;
        char *key = json_string(&r, &kl);
        if (!key) break;
        json_expect(&r, ':');
        if (strcmp(key, "model") == 0) {
            json_expect(&r, '{');
            while (r.p < r.end && *r.p != '}') {
                size_t ikl;
                char *ik = json_string(&r, &ikl);
                if (!ik) break;
                json_expect(&r, ':');
                if (strcmp(ik, "vocab") == 0) parse_vocab_object(&r, tok);
                else if (strcmp(ik, "merges") == 0) parse_merges_array(&r, tok);
                else json_skip_value(&r);
                free(ik);
                json_skip_ws(&r);
                if (r.p < r.end && *r.p == ',') r.p++;
            }
            json_expect(&r, '}');
        } else {
            json_skip_value(&r);
        }
        free(key);
        json_skip_ws(&r);
        if (r.p < r.end && *r.p == ',') r.p++;
    }
    free(buf);

    /* Build the byte -> token table from the byte encoder. */
    for (int i = 0; i < 256; i++) tok->byte_to_token[i] = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        uint32_t cp = byte_is_printable(b) ? (uint32_t)b : (uint32_t)(256 + n++);
        char u8[4];
        int l = utf8_encode_cp(cp, u8);
        int id = str_map_get(&tok->text2id, u8, (size_t)l);
        tok->byte_to_token[b] = id;
        if (id >= 0 && id < tok->vocab_size) tok->vocab[id].byte_value = b;
    }

    /* Chat models stop on <|im_end|>, not <|endoftext|> (which is
     * often the padding / unknown token); plain GPT-2-style
     * tokenizers have no <|im_end|>, so they fall back to
     * <|endoftext|> as before. */
    int eos = str_map_get(&tok->text2id, "<|im_end|>", 10);
    if (eos < 0) eos = str_map_get(&tok->text2id, "<|endoftext|>", 13);
    tok->eos_token_id = eos;
    return tok;
}

/* --- GPT-2 pre-tokenizer segmentation --- */

static size_t utf8_len_of(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static int is_ascii_letter(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

/* Length of the pre-tokenizer segment starting at p. */
static size_t segment_len_at(const char *p) {
    unsigned char c = (unsigned char)*p;
    size_t n = 0;

    if (c == '\'') {
        /* GPT-2 pre-tokenizer contractions, longest first ('tis ->
         * 't + is, matching the reference regex, which has no word
         * boundary). */
        static const char *suf[] = { "'re", "'ve", "'ll", "'s", "'t", "'m", "'d" };
        for (size_t i = 0; i < sizeof(suf) / sizeof(suf[0]); i++) {
            size_t l = strlen(suf[i]);
            if (strncmp(p, suf[i], l) == 0) return l;
        }
    }
    if (c == ' ' && p[1] != '\0' && p[1] != ' ' && p[1] != '\t' &&
        p[1] != '\r' && p[1] != '\n') {
        /* The GPT-2 pre-tokenizer regex attaches one leading space
         * to the following run (" Hello" is one piece, "ĠHello"),
         * so a lone space belongs to the next segment, not its
         * own. A run of 2+ spaces stays a whitespace segment, as
         * in the reference. */
        return 1 + segment_len_at(p + 1);
    }
    if (c == '\r' || c == '\n') {
        while (p[n] == '\r' || p[n] == '\n') n++;
        return n;
    }
    if (c == ' ' || c == '\t' || c == '\v' || c == '\f') {
        while (p[n] == ' ' || p[n] == '\t' || p[n] == '\v' || p[n] == '\f') n++;
        return n;
    }
    if (c >= '0' && c <= '9') {
        while (p[n] >= '0' && p[n] <= '9' && n < 3) n++;
        return n;
    }
    /* The reference pre-tokenizer matches Unicode letters
     * (\p{L}+), so a word continues through multi-byte characters
     * ("naïve" is one piece, not "na" + "ï" + "ve"). Treating
     * every multi-byte sequence as a letter is a safe
     * approximation here: BPE can only merge pairs that exist in
     * the merge table, so a stray punctuation mark inside a
     * segment still tokenizes on its own. */
    if (is_ascii_letter(c) || c >= 0x80) {
        while (1) {
            unsigned char d = (unsigned char)p[n];
            if (is_ascii_letter(d)) n++;
            else if (d >= 0x80) n += utf8_len_of(d);
            else break;
        }
        return n;
    }
    /* Punctuation run. */
    while (p[n] >= 0x21 && p[n] <= 0x7E && !is_ascii_letter((unsigned char)p[n]) &&
           !(p[n] >= '0' && p[n] <= '9')) {
        /* Stop before a "<|" so the special-token scan in the
         * encoder sees it at a segment boundary ("?<|im_end|>"
         * must not glue "?<" to the special token). */
        if (p[n] == '<' && p[n + 1] == '|') break;
        n++;
    }
    return n > 0 ? n : 1;
}

/* Apply BPE merges to a token sequence in place: repeatedly find
 * the lowest-rank adjacent pair and merge every occurrence of it
 * (left to right, non-overlapping) until no pair merges. */
static void bpe_apply(const tokenizer_t *tok, uint32_t *t, int *n_inout) {
    int n = *n_inout;
    while (n >= 2) {
        int best = -1, best_rank = INT_MAX;
        uint32_t ba = 0, bb = 0, bm = 0;
        for (int i = 0; i < n - 1; i++) {
            uint32_t merged;
            int rank = merge_map_get(&tok->merges, t[i], t[i + 1], &merged);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best = i;
                ba = t[i];
                bb = t[i + 1];
                bm = merged;
            }
        }
        if (best < 0) break;
        int w = 0;
        for (int i = 0; i < n; i++) {
            if (i < n - 1 && t[i] == ba && t[i + 1] == bb) {
                t[w++] = bm;
                i++;
            } else {
                t[w++] = t[i];
            }
        }
        n = w;
    }
    *n_inout = n;
}

size_t tokenizer_encode(tokenizer_t *tok, const char *text, uint32_t *out_tokens, size_t max_tokens) {
    if (!tok || !text || !out_tokens || max_tokens == 0) return 0;
    size_t written = 0;
    const char *p = text;
    while (*p && written < max_tokens) {
        /* Added special tokens (<|im_start|>, <|im_end|>, ...) are
         * single vocab entries, unreachable through BPE merges;
         * match them literally before segmenting. An unmatched
         * "<|...|>" falls through to normal segmentation. */
        if (p[0] == '<' && p[1] == '|') {
            const char *end = strstr(p, "|>");
            if (end) {
                size_t slen = (size_t)(end - p) + 2;
                int sid = str_map_get(&tok->text2id, p, slen);
                if (sid >= 0) {
                    out_tokens[written++] = (uint32_t)sid;
                    p += slen;
                    continue;
                }
            }
        }
        size_t seg = segment_len_at(p);
        if (seg == 0) seg = 1;
        uint32_t *tmp = (uint32_t *)malloc(seg * sizeof(uint32_t));
        if (!tmp) return written;
        int n = 0;
        for (size_t i = 0; i < seg && p[i]; i++) {
            int tid = tok->byte_to_token[(unsigned char)p[i]];
            tmp[n++] = (tid >= 0) ? (uint32_t)tid : 0;
        }
        if (n >= 2) bpe_apply(tok, tmp, &n);
        for (int i = 0; i < n && written < max_tokens; i++) out_tokens[written++] = tmp[i];
        free(tmp);
        p += seg;
    }
    return written;
}

/* Inverse of the GPT-2 byte-level encoder used in
 * tokenizer_load: every code point in a vocab string stands for
 * one original byte (U+0120 "Ġ" is a space, U+010A "Ċ" is a
 * newline, ...). Decoding must run the whole piece through this
 * map, not just single-byte tokens, or multi-character pieces
 * come out with literal "Ġ" characters instead of spaces. */
static int cp_to_byte(uint32_t cp) {
    if (cp < 256) {
        return byte_is_printable((int)cp) ? (int)cp : -1;
    }
    /* Non-printable bytes were assigned 256.. in byte order;
     * invert that. */
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (!byte_is_printable(b)) {
            if (cp == (uint32_t)(256 + n)) return b;
            n++;
        }
    }
    return -1;
}

static uint32_t utf8_decode_cp(const unsigned char *s, size_t len, size_t *used) {
    unsigned char c = s[0];
    if (c < 0x80) {
        *used = 1;
        return c;
    }
    if ((c & 0xE0) == 0xC0 && len >= 2) {
        *used = 2;
        return ((uint32_t)(c & 0x1F) << 6) | (uint32_t)(s[1] & 0x3F);
    }
    if ((c & 0xF0) == 0xE0 && len >= 3) {
        *used = 3;
        return ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) |
               (uint32_t)(s[2] & 0x3F);
    }
    if ((c & 0xF8) == 0xF0 && len >= 4) {
        *used = 4;
        return ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12) |
               ((uint32_t)(s[2] & 0x3F) << 6) | (uint32_t)(s[3] & 0x3F);
    }
    *used = 1;
    return c;
}

char *tokenizer_decode(tokenizer_t *tok, const uint32_t *tokens, size_t n_tokens) {
    if (!tok) {
        char *e = (char *)malloc(1);
        if (e) e[0] = 0;
        return e;
    }
    size_t total = 0;
    for (size_t i = 0; i < n_tokens; i++) {
        uint32_t id = tokens[i];
        if (id < (uint32_t)tok->vocab_size && tok->vocab[id].text) {
            /* Decoded output is never longer than the vocab text:
             * every code point maps to at most one byte. */
            total += tok->vocab[id].len;
        }
    }
    char *r = (char *)malloc(total + 1);
    if (!r) return NULL;
    char *o = r;
    for (size_t i = 0; i < n_tokens; i++) {
        uint32_t id = tokens[i];
        if (id >= (uint32_t)tok->vocab_size || !tok->vocab[id].text) continue;
        const unsigned char *text = (const unsigned char *)tok->vocab[id].text;
        size_t len = tok->vocab[id].len, pos = 0;
        while (pos < len) {
            size_t used = 0;
            uint32_t cp = utf8_decode_cp(text + pos, len - pos, &used);
            int b = cp_to_byte(cp);
            if (b >= 0) {
                *o++ = (char)b;
            } else {
                /* Not a byte-level code point: keep the original bytes. */
                memcpy(o, text + pos, used);
                o += used;
            }
            pos += used;
        }
    }
    *o = 0;
    return r;
}

int tokenizer_eos_id(const tokenizer_t *tok) {
    return tok ? tok->eos_token_id : -1;
}

void tokenizer_free(tokenizer_t *tok) {
    if (!tok) return;
    for (int i = 0; i < tok->vocab_size; i++) free(tok->vocab[i].text);
    free(tok->vocab);
    if (tok->text2id.slots) {
        for (size_t i = 0; i < tok->text2id.cap; i++) free(tok->text2id.slots[i].key);
        free(tok->text2id.slots);
    }
    free(tok->merges.slots);
    free(tok);
}
