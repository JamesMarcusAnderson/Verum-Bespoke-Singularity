#ifndef VBS_GGUF_H
#define VBS_GGUF_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VBS_GGML_TYPE_F32     = 0,
    VBS_GGML_TYPE_F16     = 1,
    VBS_GGML_TYPE_Q4_0    = 2,
    VBS_GGML_TYPE_Q4_1    = 3,
    VBS_GGML_TYPE_Q5_0    = 6,
    VBS_GGML_TYPE_Q5_1    = 7,
    VBS_GGML_TYPE_Q8_0    = 8,
    VBS_GGML_TYPE_Q8_1    = 9,
    VBS_GGML_TYPE_Q2_K    = 10,
    VBS_GGML_TYPE_Q3_K    = 11,
    VBS_GGML_TYPE_Q4_K    = 12,
    VBS_GGML_TYPE_Q5_K    = 13,
    VBS_GGML_TYPE_Q6_K    = 14,
    VBS_GGML_TYPE_Q8_K    = 15,
    VBS_GGML_TYPE_IQ2_XXS = 16,
    VBS_GGML_TYPE_IQ2_XS  = 17,
    VBS_GGML_TYPE_IQ3_XXS = 18,
    VBS_GGML_TYPE_IQ1_S   = 19,
    VBS_GGML_TYPE_IQ4_NL  = 20,
    VBS_GGML_TYPE_IQ3_S   = 21,
    VBS_GGML_TYPE_IQ2_S   = 22,
    VBS_GGML_TYPE_IQ4_XS  = 23,
    VBS_GGML_TYPE_I8      = 24,
    VBS_GGML_TYPE_I16     = 25,
    VBS_GGML_TYPE_I32     = 26,
    VBS_GGML_TYPE_I64     = 27,
    VBS_GGML_TYPE_F64     = 28,
    VBS_GGML_TYPE_IQ1_M   = 29,
    VBS_GGML_TYPE_BF16    = 30,
    VBS_GGML_TYPE_TQ1_0   = 34,
    VBS_GGML_TYPE_TQ2_0   = 35,
    VBS_GGML_TYPE_MXFP4   = 39
} vbs_ggml_type_t;

typedef struct {
    uint32_t elements_per_block;
    uint32_t bytes_per_block;
} vbs_block_geometry_t;

bool vbs_get_geometry(vbs_ggml_type_t type, vbs_block_geometry_t *out_geom);

typedef struct {
    char *key;
    uint32_t type;
    uint64_t value_offset;
    uint64_t value_end;
} vbs_gguf_metadata_t;

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t shape[4];
    vbs_ggml_type_t type;
    uint64_t data_offset;
    uint64_t byte_length;
    uint64_t columns;
    uint64_t rows;
    uint64_t packed_row_bytes;
} vbs_tensor_t;

typedef struct {
    char *path;
    int fd;
    void *map;
    uint64_t file_size;
    uint32_t version;
    uint32_t alignment;

    size_t n_metadata;
    vbs_gguf_metadata_t *metadata;

    size_t n_tensors;
    vbs_tensor_t *tensors;
} vbs_gguf_model_t;

vbs_gguf_model_t *vbs_gguf_open(const char *path, char *err_buf, size_t err_cap);
void vbs_gguf_close(vbs_gguf_model_t *model);

const vbs_tensor_t *vbs_gguf_get_tensor(const vbs_gguf_model_t *model, const char *name);
const char *vbs_gguf_get_string_metadata(const vbs_gguf_model_t *model, const char *key);
bool vbs_gguf_get_uint32_metadata(const vbs_gguf_model_t *model, const char *key, uint32_t *out_val);
bool vbs_gguf_get_float32_metadata(const vbs_gguf_model_t *model, const char *key, float *out_val);

bool vbs_gguf_copy_packed_tile(
    const vbs_gguf_model_t *model,
    const vbs_tensor_t *tensor,
    uint64_t row_start, uint64_t row_count,
    uint64_t col_start, uint64_t col_count,
    void *dst, size_t capacity, size_t *bytes_copied,
    char *err_buf, size_t err_cap
);

#ifdef __cplusplus
}
#endif

#endif // VBS_GGUF_H
