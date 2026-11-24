#ifndef VBS_MODEL_H
#define VBS_MODEL_H

/* vbs_model.h — model file reader (the engine's model format).
 *
 * Reads the binary model format: a header, key/value metadata, tensor
 * descriptions, then 32-byte-aligned tensor data. The file is memory
 * mapped read-only; tensors are views into the mapping, never copied.
 *
 * Kept external interface names (these must stay exactly as written
 * because they are the file format's own vocabulary, not our naming):
 *   - "GGUF" magic bytes: the format's file signature; the loader
 *     rejects files that do not start with these four bytes.
 *   - Metadata keys ("general.*", "llama.*", "tokenizer.ggml.*"):
 *     key strings stored inside model files; renaming them would make
 *     real files unreadable.
 *   - The ".gguf" filename extension: the format's file suffix.
 *   - Quantization codes (Q8_0, Q4_K, ...) and the numeric values of
 *     the VBS_MODEL_TYPE_* enum: the format's type vocabulary. The
 *     enum's spelling is ours; its numbers are the format's.
 */

#include "vbs_preamble.h"
#include "vbs_port.h"

VBS_EXTERN_C_BEGIN

/* Tensor element encoding. Values are the format's type codes. */
typedef enum {
    VBS_MODEL_TYPE_F32     = 0,
    VBS_MODEL_TYPE_F16     = 1,
    VBS_MODEL_TYPE_Q4_0    = 2,
    VBS_MODEL_TYPE_Q4_1    = 3,
    VBS_MODEL_TYPE_Q5_0    = 6,
    VBS_MODEL_TYPE_Q5_1    = 7,
    VBS_MODEL_TYPE_Q8_0    = 8,
    VBS_MODEL_TYPE_Q8_1    = 9,
    VBS_MODEL_TYPE_Q2_K    = 10,
    VBS_MODEL_TYPE_Q3_K    = 11,
    VBS_MODEL_TYPE_Q4_K    = 12,
    VBS_MODEL_TYPE_Q5_K    = 13,
    VBS_MODEL_TYPE_Q6_K    = 14,
    VBS_MODEL_TYPE_Q8_K    = 15,
    VBS_MODEL_TYPE_IQ2_XXS = 16,
    VBS_MODEL_TYPE_IQ2_XS  = 17,
    VBS_MODEL_TYPE_IQ3_XXS = 18,
    VBS_MODEL_TYPE_IQ1_S   = 19,
    VBS_MODEL_TYPE_IQ4_NL  = 20,
    VBS_MODEL_TYPE_IQ3_S   = 21,
    VBS_MODEL_TYPE_IQ2_S   = 22,
    VBS_MODEL_TYPE_IQ4_XS  = 23,
    VBS_MODEL_TYPE_I8      = 24,
    VBS_MODEL_TYPE_I16     = 25,
    VBS_MODEL_TYPE_I32     = 26,
    VBS_MODEL_TYPE_I64     = 27,
    VBS_MODEL_TYPE_F64     = 28,
    VBS_MODEL_TYPE_IQ1_M   = 29,
    VBS_MODEL_TYPE_BF16    = 30,
    VBS_MODEL_TYPE_TQ1_0   = 34,
    VBS_MODEL_TYPE_TQ2_0   = 35,
    VBS_MODEL_TYPE_MXFP4   = 39,
    VBS_MODEL_TYPE_LNS8    = 40,
    VBS_MODEL_TYPE_I8G     = 100  /* internal dequant target only, not a model file type */
} vbs_model_type_t;

/* How one block of a tensor type is laid out on disk. */
typedef struct {
    uint32_t elements_per_block;
    uint32_t bytes_per_block;
} vbs_block_geometry_t;

/* Fill *out_geom for a type. Returns false for types with no known
 * block geometry (the loader refuses those tensors). */
bool vbs_get_geometry(vbs_model_type_t type, vbs_block_geometry_t *out_geom);

typedef struct {
    char *key;
    uint32_t type;
    uint64_t value_offset;
    uint64_t value_end;
} vbs_model_metadata_t;

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t shape[4];
    vbs_model_type_t type;
    uint64_t data_offset;
    uint64_t byte_length;
    uint64_t columns;
    uint64_t rows;
    uint64_t packed_row_bytes;
} vbs_tensor_t;

typedef struct {
    char *path;
    vbs_mmap_t filemap; /* owns the file mapping (fd/HANDLEs); map mirrors it */
    void *map;          /* == filemap.ptr; kept so readers need no change */
    uint64_t file_size;
    uint32_t version;
    uint32_t alignment;

    size_t n_metadata;
    vbs_model_metadata_t *metadata;

    size_t n_tensors;
    vbs_tensor_t *tensors;
} vbs_model_t;

vbs_model_t *vbs_model_open(const char *path, char *err_buf, size_t err_cap);
void vbs_model_close(vbs_model_t *model);

const vbs_tensor_t *vbs_model_get_tensor(const vbs_model_t *model, const char *name);
const char *vbs_model_get_string_metadata(const vbs_model_t *model, const char *key);
bool vbs_model_get_uint32_metadata(const vbs_model_t *model, const char *key, uint32_t *out_val);
bool vbs_model_get_float32_metadata(const vbs_model_t *model, const char *key, float *out_val);

bool vbs_model_copy_packed_tile(
    const vbs_model_t *model,
    const vbs_tensor_t *tensor,
    uint64_t row_start, uint64_t row_count,
    uint64_t col_start, uint64_t col_count,
    void *dst, size_t capacity, size_t *bytes_copied,
    char *err_buf, size_t err_cap
);

VBS_EXTERN_C_END

#endif /* VBS_MODEL_H */
