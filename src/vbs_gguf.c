#include "vbs_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    M_U8 = 0, M_I8 = 1, M_U16 = 2, M_I16 = 3, M_U32 = 4, M_I32 = 5,
    M_F32 = 6, M_BOOL = 7, M_STRING = 8, M_ARRAY = 9, M_U64 = 10, M_I64 = 11, M_F64 = 12
};

bool vbs_get_geometry(vbs_ggml_type_t t, vbs_block_geometry_t *g) {
    vbs_block_geometry_t x = {0, 0};
    switch (t) {
        case VBS_GGML_TYPE_F32:     x = (vbs_block_geometry_t){1, 4}; break;
        case VBS_GGML_TYPE_F16:
        case VBS_GGML_TYPE_BF16:    x = (vbs_block_geometry_t){1, 2}; break;
        case VBS_GGML_TYPE_I8:      x = (vbs_block_geometry_t){1, 1}; break;
        case VBS_GGML_TYPE_I16:     x = (vbs_block_geometry_t){1, 2}; break;
        case VBS_GGML_TYPE_I32:     x = (vbs_block_geometry_t){1, 4}; break;
        case VBS_GGML_TYPE_I64:
        case VBS_GGML_TYPE_F64:     x = (vbs_block_geometry_t){1, 8}; break;
        case VBS_GGML_TYPE_Q4_0:    x = (vbs_block_geometry_t){32, 18}; break;
        case VBS_GGML_TYPE_Q4_1:    x = (vbs_block_geometry_t){32, 20}; break;
        case VBS_GGML_TYPE_Q5_0:    x = (vbs_block_geometry_t){32, 22}; break;
        case VBS_GGML_TYPE_Q5_1:    x = (vbs_block_geometry_t){32, 24}; break;
        case VBS_GGML_TYPE_Q8_0:    x = (vbs_block_geometry_t){32, 34}; break;
        case VBS_GGML_TYPE_Q8_1:    x = (vbs_block_geometry_t){32, 40}; break;
        case VBS_GGML_TYPE_Q2_K:    x = (vbs_block_geometry_t){256, 84}; break;
        case VBS_GGML_TYPE_Q3_K:    x = (vbs_block_geometry_t){256, 110}; break;
        case VBS_GGML_TYPE_Q4_K:    x = (vbs_block_geometry_t){256, 144}; break;
        case VBS_GGML_TYPE_Q5_K:    x = (vbs_block_geometry_t){256, 176}; break;
        case VBS_GGML_TYPE_Q6_K:    x = (vbs_block_geometry_t){256, 210}; break;
        case VBS_GGML_TYPE_Q8_K:    x = (vbs_block_geometry_t){256, 292}; break;
        case VBS_GGML_TYPE_LNS8:    x = (vbs_block_geometry_t){32, 36}; break;
        default: return false;
    }
    if (g) *g = x;
    return true;
}

typedef struct { const uint8_t *base; uint64_t size; uint64_t off; } cursor_t;

static inline bool cur_read(cursor_t *c, void *dst, uint64_t n) {
    if (n > c->size || c->off > c->size - n) return false;
    if (dst) memcpy(dst, c->base + c->off, (size_t)n);
    c->off += n;
    return true;
}
static inline bool cur_u32(cursor_t *c, uint32_t *v) { return cur_read(c, v, 4); }
static inline bool cur_u64(cursor_t *c, uint64_t *v) { return cur_read(c, v, 8); }

static char *cur_str(cursor_t *c) {
    uint64_t n = 0;
    if (!cur_u64(c, &n) || n > c->size - c->off) return NULL;
    char *s = (char *)malloc((size_t)n + 1);
    if (!s) return NULL;
    if (!cur_read(c, s, n)) { free(s); return NULL; }
    s[n] = '\0';
    return s;
}

static bool skip_val(cursor_t *c, uint32_t t);
static bool skip_arr(cursor_t *c) {
    uint32_t sub; uint64_t n;
    if (!cur_u32(c, &sub) || !cur_u64(c, &n)) return false;
    for (uint64_t i = 0; i < n; i++) if (!skip_val(c, sub)) return false;
    return true;
}
static bool skip_val(cursor_t *c, uint32_t t) {
    uint64_t n = 0;
    switch (t) {
        case M_U8: case M_I8: case M_BOOL: n = 1; break;
        case M_U16: case M_I16: n = 2; break;
        case M_U32: case M_I32: case M_F32: n = 4; break;
        case M_U64: case M_I64: case M_F64: n = 8; break;
        case M_STRING: if (!cur_u64(c, &n)) return false; break;
        case M_ARRAY: return skip_arr(c);
        default: return false;
    }
    return cur_read(c, NULL, n);
}

vbs_gguf_model_t *vbs_gguf_open(const char *path, char *err_buf, size_t err_cap) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        snprintf(err_buf, err_cap, "Cannot open GGUF file: %s", path);
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) || st.st_size < 24) {
        snprintf(err_buf, err_cap, "Invalid file size for GGUF: %s", path);
        close(fd);
        return NULL;
    }
    uint64_t file_size = (uint64_t)st.st_size;
    void *map = mmap(NULL, (size_t)file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        snprintf(err_buf, err_cap, "mmap failed for %s", path);
        close(fd);
        return NULL;
    }

    cursor_t c = {(const uint8_t *)map, file_size, 0};
    char magic[4];
    uint32_t version;
    uint64_t nt = 0, nm = 0;
    if (!cur_read(&c, magic, 4) || memcmp(magic, "GGUF", 4) != 0 ||
        !cur_u32(&c, &version) || !cur_u64(&c, &nt) || !cur_u64(&c, &nm)) {
        snprintf(err_buf, err_cap, "Malformed GGUF header");
        munmap(map, (size_t)file_size);
        close(fd);
        return NULL;
    }

    vbs_gguf_model_t *m = (vbs_gguf_model_t *)calloc(1, sizeof(vbs_gguf_model_t));
    m->path = strdup(path);
    m->fd = fd;
    m->map = map;
    m->file_size = file_size;
    m->version = version;
    m->n_metadata = (size_t)nm;
    m->metadata = (vbs_gguf_metadata_t *)calloc(m->n_metadata, sizeof(vbs_gguf_metadata_t));

    for (size_t i = 0; i < m->n_metadata; i++) {
        char *k = cur_str(&c);
        uint32_t t;
        if (!k || !cur_u32(&c, &t)) {
            snprintf(err_buf, err_cap, "Malformed metadata record %zu", i);
            vbs_gguf_close(m);
            return NULL;
        }
        m->metadata[i].key = k;
        m->metadata[i].type = t;
        m->metadata[i].value_offset = c.off;
        if (!skip_val(&c, t)) {
            snprintf(err_buf, err_cap, "Failed skipping metadata value for '%s'", k);
            vbs_gguf_close(m);
            return NULL;
        }
        m->metadata[i].value_end = c.off;
    }

    uint32_t align = 32;
    vbs_gguf_get_uint32_metadata(m, "general.alignment", &align);
    if (align < 8 || (align & (align - 1)) != 0) align = 32;
    m->alignment = align;

    m->n_tensors = (size_t)nt;
    m->tensors = (vbs_tensor_t *)calloc(m->n_tensors, sizeof(vbs_tensor_t));

    for (size_t i = 0; i < m->n_tensors; i++) {
        char *n = cur_str(&c);
        uint32_t nd = 0, type = 0;
        uint64_t rel = 0;
        if (!n || !cur_u32(&c, &nd) || nd == 0 || nd > 4) {
            snprintf(err_buf, err_cap, "Malformed tensor metadata %zu", i);
            vbs_gguf_close(m);
            return NULL;
        }
        m->tensors[i].name = n;
        m->tensors[i].n_dims = nd;
        uint64_t elements = 1;
        for (uint32_t d = 0; d < nd; d++) {
            uint64_t dim;
            if (!cur_u64(&c, &dim) || dim == 0) {
                snprintf(err_buf, err_cap, "Invalid tensor dimension for '%s'", n);
                vbs_gguf_close(m);
                return NULL;
            }
            m->tensors[i].shape[d] = dim;
            elements *= dim;
        }
        if (!cur_u32(&c, &type) || !cur_u64(&c, &rel)) {
            snprintf(err_buf, err_cap, "Truncated tensor info for '%s'", n);
            vbs_gguf_close(m);
            return NULL;
        }
        vbs_block_geometry_t g;
        if (!vbs_get_geometry((vbs_ggml_type_t)type, &g) ||
            m->tensors[i].shape[0] % g.elements_per_block != 0) {
            snprintf(err_buf, err_cap, "Unsupported/misaligned tensor encoding %u: %s", type, n);
            vbs_gguf_close(m);
            return NULL;
        }
        uint64_t cols = m->tensors[i].shape[0];
        uint64_t rows = elements / cols;
        uint64_t blocks = cols / g.elements_per_block;
        uint64_t row_bytes = blocks * g.bytes_per_block;

        m->tensors[i].type = (vbs_ggml_type_t)type;
        m->tensors[i].columns = cols;
        m->tensors[i].rows = rows;
        m->tensors[i].packed_row_bytes = row_bytes;
        m->tensors[i].byte_length = rows * row_bytes;
        m->tensors[i].data_offset = rel;
    }

    uint64_t aligned_data_start = ((c.off + m->alignment - 1) / m->alignment) * m->alignment;
    for (size_t i = 0; i < m->n_tensors; i++) {
        uint64_t abs_offset = aligned_data_start + m->tensors[i].data_offset;
        if (abs_offset > m->file_size || m->tensors[i].byte_length > m->file_size - abs_offset) {
            snprintf(err_buf, err_cap, "Tensor bounds exceed file length: %s", m->tensors[i].name);
            vbs_gguf_close(m);
            return NULL;
        }
        m->tensors[i].data_offset = abs_offset;
    }

    return m;
}

void vbs_gguf_close(vbs_gguf_model_t *m) {
    if (!m) return;
    if (m->metadata) {
        for (size_t i = 0; i < m->n_metadata; i++) free(m->metadata[i].key);
        free(m->metadata);
    }
    if (m->tensors) {
        for (size_t i = 0; i < m->n_tensors; i++) free(m->tensors[i].name);
        free(m->tensors);
    }
    if (m->map && m->map != MAP_FAILED) munmap(m->map, (size_t)m->file_size);
    if (m->fd >= 0) close(m->fd);
    free(m->path);
    free(m);
}

const vbs_tensor_t *vbs_gguf_get_tensor(const vbs_gguf_model_t *model, const char *name) {
    if (!model || !name) return NULL;
    for (size_t i = 0; i < model->n_tensors; i++) {
        if (strcmp(model->tensors[i].name, name) == 0) return &model->tensors[i];
    }
    return NULL;
}

const char *vbs_gguf_get_string_metadata(const vbs_gguf_model_t *model, const char *key) {
    if (!model || !key) return NULL;
    for (size_t i = 0; i < model->n_metadata; i++) {
        if (strcmp(model->metadata[i].key, key) == 0 && model->metadata[i].type == M_STRING) {
            cursor_t c = {(const uint8_t *)model->map, model->file_size, model->metadata[i].value_offset};
            return cur_str(&c);
        }
    }
    return NULL;
}

bool vbs_gguf_get_uint32_metadata(const vbs_gguf_model_t *model, const char *key, uint32_t *out_val) {
    if (!model || !key) return false;
    for (size_t i = 0; i < model->n_metadata; i++) {
        if (strcmp(model->metadata[i].key, key) == 0) {
            const uint8_t *p = (const uint8_t *)model->map + model->metadata[i].value_offset;
            if (model->metadata[i].type == M_U32) { memcpy(out_val, p, 4); return true; }
            if (model->metadata[i].type == M_U8)  { *out_val = *p; return true; }
        }
    }
    return false;
}

bool vbs_gguf_get_float32_metadata(const vbs_gguf_model_t *model, const char *key, float *out_val) {
    if (!model || !key) return false;
    for (size_t i = 0; i < model->n_metadata; i++) {
        if (strcmp(model->metadata[i].key, key) == 0) {
            const uint8_t *p = (const uint8_t *)model->map + model->metadata[i].value_offset;
            if (model->metadata[i].type == M_F32) { memcpy(out_val, p, 4); return true; }
            if (model->metadata[i].type == M_F64) { double d; memcpy(&d, p, 8); *out_val = (float)d; return true; }
        }
    }
    return false;
}

bool vbs_gguf_copy_packed_tile(
    const vbs_gguf_model_t *model,
    const vbs_tensor_t *t,
    uint64_t rs, uint64_t rc,
    uint64_t cs, uint64_t cc,
    void *dst, size_t cap, size_t *bytes_copied,
    char *err_buf, size_t err_cap
) {
    vbs_block_geometry_t g;
    if (!dst || !vbs_get_geometry(t->type, &g) ||
        cs % g.elements_per_block != 0 || cc % g.elements_per_block != 0 ||
        rs > t->rows || rc > t->rows - rs ||
        cs > t->columns || cc > t->columns - cs) {
        snprintf(err_buf, err_cap, "Tile coordinates out of bounds or not block-aligned");
        return false;
    }
    uint64_t tile_row_bytes = (cc / g.elements_per_block) * g.bytes_per_block;
    uint64_t total_bytes = tile_row_bytes * rc;
    if (total_bytes > cap) {
        snprintf(err_buf, err_cap, "Packed tile exceeds staging buffer capacity (%llu > %zu)", total_bytes, cap);
        return false;
    }
    uint64_t col_byte_offset = (cs / g.elements_per_block) * g.bytes_per_block;
    const uint8_t *src = (const uint8_t *)model->map + t->data_offset + rs * t->packed_row_bytes + col_byte_offset;

    for (uint64_t r = 0; r < rc; r++) {
        memcpy((uint8_t *)dst + r * tile_row_bytes, src + r * t->packed_row_bytes, (size_t)tile_row_bytes);
    }
    if (bytes_copied) *bytes_copied = (size_t)total_bytes;
    return true;
}
