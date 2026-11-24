/* vbs_model.c — model file reader.
 *
 * Parses the binary model format straight from the spec: a 24-byte
 * header ("GGUF" magic, version, tensor count, metadata count),
 * metadata records (length-prefixed key, type tag, value), tensor
 * records (name, dimensions, type code, relative data offset), then
 * tensor data starting at the aligned data offset. The file is mapped
 * read-only and tensors are views into the mapping.
 *
 * The format's own vocabulary stays literal where the file demands it:
 * the "GGUF" magic, metadata keys such as "general.alignment", and the
 * numeric type codes in vbs_get_geometry. Everything else here is
 * Verum's own code.
 */

#include "vbs_model.h"
#include "vbs_port.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Metadata value kinds, as stored in the file. */
enum {
    META_U8 = 0, META_I8 = 1, META_U16 = 2, META_I16 = 3,
    META_U32 = 4, META_I32 = 5, META_F32 = 6, META_BOOL = 7,
    META_STR = 8, META_ARR = 9, META_U64 = 10, META_I64 = 11,
    META_F64 = 12
};

/* --- Block geometry -------------------------------------------------- */

/* One row per supported type: elements packed in a block, bytes the
 * block occupies. Types with no row here are refused by the loader. */
static const struct {
    vbs_model_type_t kind;
    uint32_t elems;
    uint32_t bytes;
} k_geometry[] = {
    { VBS_MODEL_TYPE_F32,  1,   4 },
    { VBS_MODEL_TYPE_F16,  1,   2 },
    { VBS_MODEL_TYPE_BF16, 1,   2 },
    { VBS_MODEL_TYPE_I8,   1,   1 },
    { VBS_MODEL_TYPE_I16,  1,   2 },
    { VBS_MODEL_TYPE_I32,  1,   4 },
    { VBS_MODEL_TYPE_I64,  1,   8 },
    { VBS_MODEL_TYPE_F64,  1,   8 },
    { VBS_MODEL_TYPE_Q4_0, 32,  18 },
    { VBS_MODEL_TYPE_Q4_1, 32,  20 },
    { VBS_MODEL_TYPE_Q5_0, 32,  22 },
    { VBS_MODEL_TYPE_Q5_1, 32,  24 },
    { VBS_MODEL_TYPE_Q8_0, 32,  34 },
    { VBS_MODEL_TYPE_Q8_1, 32,  40 },
    { VBS_MODEL_TYPE_Q2_K, 256, 84 },
    { VBS_MODEL_TYPE_Q3_K, 256, 110 },
    { VBS_MODEL_TYPE_Q4_K, 256, 144 },
    { VBS_MODEL_TYPE_Q5_K, 256, 176 },
    { VBS_MODEL_TYPE_Q6_K, 256, 210 },
    { VBS_MODEL_TYPE_Q8_K, 256, 292 },
    { VBS_MODEL_TYPE_LNS8, 32,  36 },
};

bool vbs_get_geometry(vbs_model_type_t t, vbs_block_geometry_t *g) {
    for (size_t i = 0; i < sizeof(k_geometry) / sizeof(k_geometry[0]); i++) {
        if (k_geometry[i].kind == t) {
            if (g) {
                g->elements_per_block = k_geometry[i].elems;
                g->bytes_per_block = k_geometry[i].bytes;
            }
            return true;
        }
    }
    return false;
}

/* --- Bounds-checked reader ------------------------------------------- */

/* A cursor over the mapped file. Every read checks the remaining length
 * first, so a truncated or corrupt file can never read past the end. */
typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint64_t off;
} reader_t;

static bool rd_bytes(reader_t *r, void *dst, uint64_t n) {
    if (n > r->size || r->off > r->size - n) return false;
    if (dst) memcpy(dst, r->base + r->off, (size_t)n);
    r->off += n;
    return true;
}

static bool rd_u32(reader_t *r, uint32_t *v) { return rd_bytes(r, v, 4); }
static bool rd_u64(reader_t *r, uint64_t *v) { return rd_bytes(r, v, 8); }

/* Length-prefixed string (u64 length + bytes, no NUL in the file). */
static char *rd_string(reader_t *r) {
    uint64_t n = 0;
    if (!rd_u64(r, &n) || n > r->size - r->off) return NULL;
    char *s = (char *)malloc((size_t)n + 1);
    if (!s) return NULL;
    if (!rd_bytes(r, s, n)) { free(s); return NULL; }
    s[n] = '\0';
    return s;
}

/* Skip one metadata value of the given kind without reading it. */
static bool rd_skip_value(reader_t *r, uint32_t kind);

static bool rd_skip_array(reader_t *r) {
    uint32_t elem_kind = 0;
    uint64_t count = 0;
    if (!rd_u32(r, &elem_kind) || !rd_u64(r, &count)) return false;
    for (uint64_t i = 0; i < count; i++)
        if (!rd_skip_value(r, elem_kind)) return false;
    return true;
}

static bool rd_skip_value(reader_t *r, uint32_t kind) {
    uint64_t n = 0;
    switch (kind) {
        case META_U8: case META_I8: case META_BOOL:
            n = 1;
            break;
        case META_U16: case META_I16:
            n = 2;
            break;
        case META_U32: case META_I32: case META_F32:
            n = 4;
            break;
        case META_U64: case META_I64: case META_F64:
            n = 8;
            break;
        case META_STR:
            if (!rd_u64(r, &n)) return false;
            break;
        case META_ARR:
            return rd_skip_array(r);
        default:
            return false;
    }
    return rd_bytes(r, NULL, n);
}

/* --- Open / close ---------------------------------------------------- */

static void fail_open(vbs_model_t *m, vbs_mmap_t *fm, char *err, size_t cap,
                      const char *msg) {
    if (err) snprintf(err, cap, "%s", msg);
    if (m) vbs_model_close(m);
    else if (fm) vbs_mmap_close(fm);
}

vbs_model_t *vbs_model_open(const char *path, char *err_buf, size_t err_cap) {
    if (!vbs_is_little_endian()) {
        snprintf(err_buf, err_cap, "Model file requires a little-endian host");
        return NULL;
    }

    vbs_mmap_t fm;
    int mrc = vbs_mmap_open_ro(path, &fm);
    if (mrc == VBS_MMAP_ERR_OPEN) {
        snprintf(err_buf, err_cap, "Cannot open model file: %s", path);
        return NULL;
    }
    if (mrc == VBS_MMAP_ERR_SIZE || fm.size < 24) {
        snprintf(err_buf, err_cap, "Invalid file size for model file: %s", path);
        if (mrc != VBS_MMAP_ERR_SIZE) vbs_mmap_close(&fm);
        return NULL;
    }
    if (mrc == VBS_MMAP_ERR_MAP) {
        snprintf(err_buf, err_cap, "mmap failed for %s", path);
        return NULL;
    }

    reader_t r = { (const uint8_t *)fm.ptr, fm.size, 0 };
    char magic[4];
    uint32_t version = 0;
    uint64_t n_tensors = 0, n_meta = 0;
    /* "GGUF" is the format's magic bytes — an external interface name
     * that must stay to recognize model files. */
    if (!rd_bytes(&r, magic, 4) || memcmp(magic, "GGUF", 4) != 0 ||
        !rd_u32(&r, &version) || !rd_u64(&r, &n_tensors) ||
        !rd_u64(&r, &n_meta)) {
        snprintf(err_buf, err_cap, "Malformed model header");
        vbs_mmap_close(&fm);
        return NULL;
    }

    vbs_model_t *m = (vbs_model_t *)calloc(1, sizeof(vbs_model_t));
    if (!m) {
        snprintf(err_buf, err_cap, "Out of memory");
        vbs_mmap_close(&fm);
        return NULL;
    }
    m->path = vbs_strdup(path);
    m->filemap = fm;
    m->map = fm.ptr;
    m->file_size = fm.size;
    m->version = version;

    /* Metadata records: key + type + value span. Values are not decoded
     * here; only their byte spans are recorded for lazy lookup. */
    m->n_metadata = (size_t)n_meta;
    m->metadata = (vbs_model_metadata_t *)calloc(m->n_metadata,
                                                 sizeof(vbs_model_metadata_t));
    for (size_t i = 0; i < m->n_metadata; i++) {
        char *key = rd_string(&r);
        uint32_t kind = 0;
        if (!key || !rd_u32(&r, &kind)) {
            char msg[128];
            snprintf(msg, sizeof(msg), "Malformed metadata record %zu", i);
            fail_open(m, NULL, err_buf, err_cap, msg);
            return NULL;
        }
        m->metadata[i].key = key;
        m->metadata[i].type = kind;
        m->metadata[i].value_offset = r.off;
        if (!rd_skip_value(&r, kind)) {
            char msg[256];
            snprintf(msg, sizeof(msg), "Failed skipping metadata value for '%s'", key);
            fail_open(m, NULL, err_buf, err_cap, msg);
            return NULL;
        }
        m->metadata[i].value_end = r.off;
    }

    /* Alignment for the tensor data section (default 32; a corrupt or
     * non-power-of-two value falls back to 32). */
    uint32_t align = 32;
    vbs_model_get_uint32_metadata(m, "general.alignment", &align);
    if (align < 8 || (align & (align - 1)) != 0) align = 32;
    m->alignment = align;

    /* Tensor records. */
    m->n_tensors = (size_t)n_tensors;
    m->tensors = (vbs_tensor_t *)calloc(m->n_tensors, sizeof(vbs_tensor_t));
    for (size_t i = 0; i < m->n_tensors; i++) {
        char *name = rd_string(&r);
        uint32_t n_dims = 0, type_code = 0;
        uint64_t rel_off = 0;
        if (!name || !rd_u32(&r, &n_dims) || n_dims == 0 || n_dims > 4) {
            char msg[128];
            snprintf(msg, sizeof(msg), "Malformed tensor metadata %zu", i);
            fail_open(m, NULL, err_buf, err_cap, msg);
            return NULL;
        }
        vbs_tensor_t *t = &m->tensors[i];
        t->name = name;
        t->n_dims = n_dims;
        uint64_t elements = 1;
        for (uint32_t d = 0; d < n_dims; d++) {
            uint64_t dim = 0;
            if (!rd_u64(&r, &dim) || dim == 0) {
                char msg[256];
                snprintf(msg, sizeof(msg), "Invalid tensor dimension for '%s'", name);
                fail_open(m, NULL, err_buf, err_cap, msg);
                return NULL;
            }
            t->shape[d] = dim;
            elements *= dim;
        }
        if (!rd_u32(&r, &type_code) || !rd_u64(&r, &rel_off)) {
            char msg[256];
            snprintf(msg, sizeof(msg), "Truncated tensor info for '%s'", name);
            fail_open(m, NULL, err_buf, err_cap, msg);
            return NULL;
        }
        vbs_block_geometry_t g;
        if (!vbs_get_geometry((vbs_model_type_t)type_code, &g) ||
            t->shape[0] % g.elements_per_block != 0) {
            char msg[256];
            snprintf(msg, sizeof(msg), "Unsupported/misaligned tensor encoding %u: %s",
                     type_code, name);
            fail_open(m, NULL, err_buf, err_cap, msg);
            return NULL;
        }
        uint64_t cols = t->shape[0];
        uint64_t rows = elements / cols;
        uint64_t row_bytes = (cols / g.elements_per_block) * g.bytes_per_block;
        t->type = (vbs_model_type_t)type_code;
        t->columns = cols;
        t->rows = rows;
        t->packed_row_bytes = row_bytes;
        t->byte_length = rows * row_bytes;
        t->data_offset = rel_off; /* relative for now; made absolute below */
    }

    /* Tensor data starts at the first aligned offset past the header. */
    uint64_t data_start = ((r.off + m->alignment - 1) / m->alignment) * m->alignment;
    for (size_t i = 0; i < m->n_tensors; i++) {
        vbs_tensor_t *t = &m->tensors[i];
        uint64_t abs_off = data_start + t->data_offset;
        if (abs_off > m->file_size ||
            t->byte_length > m->file_size - abs_off) {
            char msg[256];
            snprintf(msg, sizeof(msg), "Tensor bounds exceed file length: %s", t->name);
            fail_open(m, NULL, err_buf, err_cap, msg);
            return NULL;
        }
        t->data_offset = abs_off;
    }

    return m;
}

void vbs_model_close(vbs_model_t *m) {
    if (!m) return;
    if (m->metadata) {
        for (size_t i = 0; i < m->n_metadata; i++) free(m->metadata[i].key);
        free(m->metadata);
    }
    if (m->tensors) {
        for (size_t i = 0; i < m->n_tensors; i++) free(m->tensors[i].name);
        free(m->tensors);
    }
    vbs_mmap_close(&m->filemap);
    m->map = NULL;
    free(m->path);
    free(m);
}

/* --- Lookups --------------------------------------------------------- */

const vbs_tensor_t *vbs_model_get_tensor(const vbs_model_t *model, const char *name) {
    if (!model || !name) return NULL;
    for (size_t i = 0; i < model->n_tensors; i++)
        if (strcmp(model->tensors[i].name, name) == 0) return &model->tensors[i];
    return NULL;
}

static const vbs_model_metadata_t *find_metadata(const vbs_model_t *model,
                                                 const char *key) {
    if (!model || !key) return NULL;
    for (size_t i = 0; i < model->n_metadata; i++)
        if (strcmp(model->metadata[i].key, key) == 0) return &model->metadata[i];
    return NULL;
}

const char *vbs_model_get_string_metadata(const vbs_model_t *model, const char *key) {
    const vbs_model_metadata_t *md = find_metadata(model, key);
    if (!md || md->type != META_STR) return NULL;
    reader_t r = { (const uint8_t *)model->map, model->file_size, md->value_offset };
    return rd_string(&r);
}

bool vbs_model_get_uint32_metadata(const vbs_model_t *model, const char *key,
                                   uint32_t *out_val) {
    const vbs_model_metadata_t *md = find_metadata(model, key);
    if (!md) return false;
    const uint8_t *p = (const uint8_t *)model->map + md->value_offset;
    if (md->type == META_U32) { memcpy(out_val, p, 4); return true; }
    if (md->type == META_U8) { *out_val = *p; return true; }
    return false;
}

bool vbs_model_get_float32_metadata(const vbs_model_t *model, const char *key,
                                    float *out_val) {
    const vbs_model_metadata_t *md = find_metadata(model, key);
    if (!md) return false;
    const uint8_t *p = (const uint8_t *)model->map + md->value_offset;
    if (md->type == META_F32) { memcpy(out_val, p, 4); return true; }
    if (md->type == META_F64) {
        double d = 0;
        memcpy(&d, p, 8);
        *out_val = (float)d;
        return true;
    }
    return false;
}

/* --- Packed tile copy ------------------------------------------------ */

bool vbs_model_copy_packed_tile(
    const vbs_model_t *model,
    const vbs_tensor_t *t,
    uint64_t rs, uint64_t rc,
    uint64_t cs, uint64_t cc,
    void *dst, size_t cap, size_t *bytes_copied,
    char *err_buf, size_t err_cap
) {
    vbs_block_geometry_t g;
    bool bad = !dst || !vbs_get_geometry(t->type, &g) ||
               cs % g.elements_per_block != 0 ||
               cc % g.elements_per_block != 0 ||
               rs > t->rows || rc > t->rows - rs ||
               cs > t->columns || cc > t->columns - cs;
    if (bad) {
        snprintf(err_buf, err_cap, "Tile coordinates out of bounds or not block-aligned");
        return false;
    }
    uint64_t tile_row_bytes = (cc / g.elements_per_block) * g.bytes_per_block;
    uint64_t total_bytes = tile_row_bytes * rc;
    if (total_bytes > cap) {
        snprintf(err_buf, err_cap,
                 "Packed tile exceeds staging buffer capacity (%llu > %zu)",
                 (unsigned long long)total_bytes, cap);
        return false;
    }
    uint64_t col_byte_off = (cs / g.elements_per_block) * g.bytes_per_block;
    const uint8_t *src = (const uint8_t *)model->map + t->data_offset +
                         rs * t->packed_row_bytes + col_byte_off;
    uint8_t *out = (uint8_t *)dst;
    for (uint64_t r = 0; r < rc; r++)
        memcpy(out + r * tile_row_bytes,
               src + r * t->packed_row_bytes, (size_t)tile_row_bytes);
    if (bytes_copied) *bytes_copied = (size_t)total_bytes;
    return true;
}
