/* emit_buf.h — shared output buffer for the text emitters.
 *
 * Single canonical copy of the fixed-capacity text buffer the PTX
 * and MSL emitters both used to carry verbatim. A write that would
 * overflow the buffer is silently dropped and the buffer stays
 * NUL-terminated; callers size the buffer for the largest expected
 * output.
 */

#ifndef VBS_EMIT_BUF_H
#define VBS_EMIT_BUF_H

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    char *buf;
    size_t pos;
    size_t cap;
} vbs_textbuf_t;

static inline void vbs_textbuf_init(vbs_textbuf_t *b, char *buf, size_t cap) {
    b->buf = buf;
    b->pos = 0;
    b->cap = cap;
}

/* Append a string, dropping it whole if it would overflow (the
 * buffer keeps room for the terminator). */
static inline void vbs_textbuf_write(vbs_textbuf_t *b, const char *s) {
    size_t n = strlen(s);
    if (b->pos + n + 1 < b->cap) {
        memcpy(b->buf + b->pos, s, n);
        b->pos += n;
        b->buf[b->pos] = '\0';
    }
}

/* Append formatted text (via a 512-byte scratch buffer). */
static inline void vbs_textbuf_fmt(vbs_textbuf_t *b, const char *fmt, ...) {
    char tmp[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    vbs_textbuf_write(b, tmp);
}

#endif /* VBS_EMIT_BUF_H */
