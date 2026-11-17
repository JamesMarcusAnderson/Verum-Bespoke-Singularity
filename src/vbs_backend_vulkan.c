/* vbs_backend_vulkan.c — Vulkan / SPIR-V backend: the UNIVERSAL GPU path.
 *
 * One backend that runs on AMD, NVIDIA, Intel (and Apple via MoltenVK),
 * behind the canonical OLD interface (include/vbs_backend.h). The Vulkan
 * loader is dlopen'd at runtime; there is no link-time dependency on
 * libvulkan and no Vulkan SDK is needed to build this file — the small
 * subset of the Vulkan 1.0 ABI used here is declared by hand below, and
 * every struct size is checked with _Static_assert against the values
 * measured from the real vulkan.h (see the ABI note at the end).
 *
 * Design:
 *   - init: create a Vulkan instance, pick the best physical device
 *     with a compute queue (discrete > integrated > virtual > CPU),
 *     create one logical device + one compute queue. If the loader is
 *     missing, no ICD is installed, or no device has a compute queue,
 *     init returns false and the engine falls back to the CPU backend
 *     (see vbs_backend_select.c). A CPU-class device (llvmpipe /
 *     Lavapipe) is accepted but logged as a software device: it
 *     executes real Vulkan compute, correctly, just not on a GPU.
 *   - kernels: precompiled SPIR-V blobs (kernels/vulkan_kernels.spv.h,
 *     generated from the GLSL sources in kernels/vulkan/ by
 *     kernels/build-spirv.sh; every module passed spirv-val). Each
 *     blob is re-checked with the in-repo structural validator
 *     (spirv_validate) before vkCreateShaderModule. The engine's own
 *     SPIR-V emitter (vbs_compile_to_spirv) is wired through
 *     vbs_vulkan_compile_kernel_source() for source-level kernels, but
 *     that emitter currently produces a structural shell with no body
 *     codegen, so it is NOT used for real dispatch (see vbs_vulkan.h).
 *   - buffers: one VkBuffer + dedicated HOST_VISIBLE|HOST_COHERENT
 *     memory per vbs_buffer_t, mapped persistently. host_ptr IS the
 *     mapped pointer, so the slice executor's staging writes land
 *     directly in device-visible memory and no copies are needed.
 *   - dispatch: one descriptor set per pipeline (updated per dispatch),
 *     one one-shot command buffer + fence per dispatch, synchronous
 *     (vkWaitForFences). Workgroup counts are derived from the problem
 *     size and each kernel's fixed local size; the bridge's grid_x /
 *     block_x hints are ignored (they are CUDA-isms) — documented in
 *     dispatch_primitive.
 *   - multi-device: v1 supports ONE device (the best). The ops table
 *     allows more, but per-device pipelines/pools would multiply the
 *     state; single-device is correct and sufficient for the engine,
 *     which uses device 0 as primary.
 *
 * Honest limits (see REPORT / docs):
 *   - No real GPU was available in the build VM; execution was verified
 *     on Lavapipe (llvmpipe, a conformant software Vulkan device).
 *     Real-hardware performance numbers need a real GPU.
 *   - Linear/embed types without a SPIR-V kernel (Q5_*, Q2_K, Q3_K,
 *     Q5_K, Q8_1, BF16, ...) are REFUSED with an error, not faked.
 *     The CPU backend covers those types.
 *   - HOST_VISIBLE|HOST_COHERENT memory is required; a device without
 *     such a memory type fails init (all desktop drivers provide one).
 */

#include "vbs_backend.h"
#include "vbs_vulkan.h"
#include "vbs_compiler.h"
#include "vbs_port.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../kernels/vulkan_kernels.spv.h"

/* ================================================================== */
/* Minimal Vulkan 1.0 ABI declarations (no SDK headers).              */
/* Struct sizes verified against /usr/include/vulkan/vulkan.h        */
/* (libvulkan-dev 1.3.275): see _Static_asserts below.                */
/* ================================================================== */

typedef uint32_t vbs_vk_flags_t;
typedef uint32_t vbs_vk_bool32_t;
typedef uint64_t vbs_vk_device_size_t;
typedef int32_t  vbs_vk_result_t;

typedef void *vbs_vk_instance_t;
typedef void *vbs_vk_physical_device_t;
typedef void *vbs_vk_device_t;
typedef void *vbs_vk_queue_t;
typedef void *vbs_vk_command_pool_t;
typedef void *vbs_vk_command_buffer_t;
typedef uint64_t vbs_vk_buffer_t;
typedef uint64_t vbs_vk_device_memory_t;
typedef uint64_t vbs_vk_shader_module_t;
typedef uint64_t vbs_vk_descriptor_set_layout_t;
typedef uint64_t vbs_vk_pipeline_layout_t;
typedef uint64_t vbs_vk_pipeline_t;
typedef uint64_t vbs_vk_descriptor_pool_t;
typedef uint64_t vbs_vk_descriptor_set_t;
typedef uint64_t vbs_vk_fence_t;

#define VBS_VK_NULL_HANDLE 0ULL
#define VBS_VK_WHOLE_SIZE (~0ULL)
#define VBS_VK_API_VERSION_1_0 0x00400000u

#define VBS_VK_SUCCESS 0
#define VBS_VK_ERROR_OUT_OF_HOST_MEMORY (-1)
#define VBS_VK_ERROR_INITIALIZATION_FAILED (-3)

#define VBS_VK_STRUCTURE_TYPE_APPLICATION_INFO 0
#define VBS_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO 1
#define VBS_VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO 2
#define VBS_VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO 3
#define VBS_VK_STRUCTURE_TYPE_SUBMIT_INFO 4
#define VBS_VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO 5
#define VBS_VK_STRUCTURE_TYPE_FENCE_CREATE_INFO 8
#define VBS_VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO 12
#define VBS_VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO 16
#define VBS_VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO 18
#define VBS_VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO 29
#define VBS_VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO 30
#define VBS_VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO 32
#define VBS_VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO 33
#define VBS_VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO 34
#define VBS_VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET 35
#define VBS_VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO 39
#define VBS_VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO 40
#define VBS_VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO 42

#define VBS_VK_PHYSICAL_DEVICE_TYPE_OTHER 0
#define VBS_VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU 1
#define VBS_VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU 2
#define VBS_VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU 3
#define VBS_VK_PHYSICAL_DEVICE_TYPE_CPU 4

#define VBS_VK_QUEUE_COMPUTE_BIT 0x00000002u
#define VBS_VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 0x00000020u
#define VBS_VK_SHARING_MODE_EXCLUSIVE 0
#define VBS_VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT 0x00000002u
#define VBS_VK_MEMORY_PROPERTY_HOST_COHERENT_BIT 0x00000004u
#define VBS_VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT 0x00000001u
#define VBS_VK_DESCRIPTOR_TYPE_STORAGE_BUFFER 7
#define VBS_VK_SHADER_STAGE_COMPUTE_BIT 0x00000020u
#define VBS_VK_PIPELINE_BIND_POINT_COMPUTE 1
#define VBS_VK_COMMAND_BUFFER_LEVEL_PRIMARY 0
#define VBS_VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT 0x00000001u
#define VBS_VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT 0x00000002u

typedef struct {
    uint32_t sType;
    const void *pNext;
    const char *pApplicationName;
    uint32_t applicationVersion;
    const char *pEngineName;
    uint32_t engineVersion;
    uint32_t apiVersion;
} vbs_vk_application_info_t;

typedef struct {
    uint32_t sType;
    const void *pNext;
    vbs_vk_flags_t flags;
    const vbs_vk_application_info_t *pApplicationInfo;
    uint32_t enabledLayerCount;
    const char *const *ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char *const *ppEnabledExtensionNames;
} vbs_vk_instance_create_info_t;

typedef struct {
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint32_t vendorID;
    uint32_t deviceID;
    uint32_t deviceType;
    char deviceName[256];
    uint8_t _limits_and_sparse[548]; /* sizeof == 824, like the real struct */
} vbs_vk_physical_device_properties_t;

typedef struct {
    uint32_t queueFlags;
    uint32_t queueCount;
    uint32_t timestampValidBits;
    uint32_t minImageTransferGranularity[3];
} vbs_vk_queue_family_properties_t;

typedef struct {
    uint32_t propertyFlags;
    uint32_t heapIndex;
} vbs_vk_memory_type_t;

typedef struct {
    vbs_vk_device_size_t size;
    uint32_t flags;
} vbs_vk_memory_heap_t;

typedef struct {
    uint32_t memoryTypeCount;
    vbs_vk_memory_type_t memoryTypes[32];
    uint32_t memoryHeapCount;
    vbs_vk_memory_heap_t memoryHeaps[16];
} vbs_vk_physical_device_memory_properties_t;

typedef struct {
    uint32_t sType;
    const void *pNext;
    vbs_vk_flags_t flags;
    uint32_t queueFamilyIndex;
    uint32_t queueCount;
    const float *pQueuePriorities;
} vbs_vk_device_queue_create_info_t;

typedef struct {
    uint32_t sType;
    const void *pNext;
    vbs_vk_flags_t flags;
    uint32_t queueCreateInfoCount;
    const vbs_vk_device_queue_create_info_t *pQueueCreateInfos;
    uint32_t enabledLayerCount;
    const char *const *ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char *const *ppEnabledExtensionNames;
    const void *pEnabledFeatures;
} vbs_vk_device_create_info_t;

typedef struct {
    uint32_t sType;
    const void *pNext;
    vbs_vk_flags_t flags;
    vbs_vk_device_size_t size;
    uint32_t usage;
    uint32_t sharingMode;
    uint32_t queueFamilyIndexCount;
    const uint32_t *pQueueFamilyIndices;
} vbs_vk_buffer_create_info_t;

typedef struct {
    vbs_vk_device_size_t size;
    vbs_vk_device_size_t alignment;
    uint32_t memoryTypeBits;
} vbs_vk_memory_requirements_t;

typedef struct {
    uint32_t sType;
    const void *pNext;
    vbs_vk_device_size_t allocationSize;
    uint32_t memoryTypeIndex;
} vbs_vk_memory_allocate_info_t;

typedef struct {
    uint32_t binding;
    uint32_t descriptorType;
    uint32_t descriptorCount;
    uint32_t stageFlags;
    const void *pImmutableSamplers;
} vbs_vk_descriptor_set_layout_binding_t;

typedef struct {
    uint32_t sType;
    const void *pNext;
    vbs_vk_flags_t flags;
    uint32_t bindingCount;
    const vbs_vk_descriptor_set_layout_binding_t *pBindings;
} vbs_vk_descriptor_set_layout_create_info_t;

typedef struct {
    uint32_t stageFlags;
    uint32_t offset;
    uint32_t size;
} vbs_vk_push_constant_range_t;

typedef struct {
    uint32_t sType;
    const void *pNext;
    vbs_vk_flags_t flags;
    uint32_t setLayoutCount;
    const vbs_vk_descriptor_set_layout_t *pSetLayouts;
    uint32_t pushConstantRangeCount;
    const vbs_vk_push_constant_range_t *pPushConstantRanges;
} vbs_vk_pipeline_layout_create_info_t;

typedef struct {
    uint32_t sType;
    const void *pNext;
    vbs_vk_flags_t flags;
    size_t codeSize;
    const uint32_t *pCode;
} vbs_vk_shader_module_create_info_t;

typedef struct {
    uint32_t sType;
    const void *pNext;
    vbs_vk_flags_t flags;
    uint32_t stage;
    vbs_vk_shader_module_t module;
    const char *pName;
    const void *pSpecializationInfo;
} vbs_vk_pipeline_shader_stage_create_info_t;

typedef struct {
    uint32_t sType;
    const void *pNext;
    vbs_vk_flags_t flags;
    vbs_vk_pipeline_shader_stage_create_info_t stage;
    vbs_vk_pipeline_layout_t layout;
    vbs_vk_pipeline_t basePipelineHandle;
    int32_t basePipelineIndex;
} vbs_vk_compute_pipeline_create_info_t;

typedef struct {
    uint32_t type;
    uint32_t descriptorCount;
} vbs_vk_descriptor_pool_size_t;

typedef struct {
    uint32_t sType;
    const void *pNext;
    vbs_vk_flags_t flags;
    uint32_t maxSets;
    uint32_t poolSizeCount;
    const vbs_vk_descriptor_pool_size_t *pPoolSizes;
} vbs_vk_descriptor_pool_create_info_t;

typedef struct {
    uint32_t sType;
    const void *pNext;
    vbs_vk_descriptor_pool_t descriptorPool;
    uint32_t descriptorSetCount;
    const vbs_vk_descriptor_set_layout_t *pSetLayouts;
} vbs_vk_descriptor_set_allocate_info_t;

typedef struct {
    vbs_vk_buffer_t buffer;
    vbs_vk_device_size_t offset;
    vbs_vk_device_size_t range;
} vbs_vk_descriptor_buffer_info_t;

typedef struct {
    uint32_t sType;
    const void *pNext;
    vbs_vk_descriptor_set_t dstSet;
    uint32_t dstBinding;
    uint32_t dstArrayElement;
    uint32_t descriptorCount;
    uint32_t descriptorType;
    const void *pImageInfo;
    const vbs_vk_descriptor_buffer_info_t *pBufferInfo;
    const void *pTexelBufferView;
} vbs_vk_write_descriptor_set_t;

typedef struct {
    uint32_t sType;
    const void *pNext;
    vbs_vk_flags_t flags;
    uint32_t queueFamilyIndex;
} vbs_vk_command_pool_create_info_t;

typedef struct {
    uint32_t sType;
    const void *pNext;
    vbs_vk_command_pool_t commandPool;
    uint32_t level;
    uint32_t commandBufferCount;
} vbs_vk_command_buffer_allocate_info_t;

typedef struct {
    uint32_t sType;
    const void *pNext;
    vbs_vk_flags_t flags;
    const void *pInheritanceInfo;
} vbs_vk_command_buffer_begin_info_t;

typedef struct {
    uint32_t sType;
    const void *pNext;
    uint32_t waitSemaphoreCount;
    const void *pWaitSemaphores;
    const uint32_t *pWaitDstStageMask;
    uint32_t commandBufferCount;
    const vbs_vk_command_buffer_t *pCommandBuffers;
    uint32_t signalSemaphoreCount;
    const void *pSignalSemaphores;
} vbs_vk_submit_info_t;

typedef struct {
    uint32_t sType;
    const void *pNext;
    vbs_vk_flags_t flags;
} vbs_vk_fence_create_info_t;

/* ABI checks: sizes measured from the real vulkan.h (1.3.275). */
_Static_assert(sizeof(vbs_vk_application_info_t) == 48, "vk abi");
_Static_assert(sizeof(vbs_vk_instance_create_info_t) == 64, "vk abi");
_Static_assert(sizeof(vbs_vk_physical_device_properties_t) == 824, "vk abi");
_Static_assert(sizeof(vbs_vk_queue_family_properties_t) == 24, "vk abi");
_Static_assert(sizeof(vbs_vk_physical_device_memory_properties_t) == 520, "vk abi");
_Static_assert(sizeof(vbs_vk_device_queue_create_info_t) == 40, "vk abi");
_Static_assert(sizeof(vbs_vk_device_create_info_t) == 72, "vk abi");
_Static_assert(sizeof(vbs_vk_buffer_create_info_t) == 56, "vk abi");
_Static_assert(sizeof(vbs_vk_memory_requirements_t) == 24, "vk abi");
_Static_assert(sizeof(vbs_vk_memory_allocate_info_t) == 32, "vk abi");
_Static_assert(sizeof(vbs_vk_descriptor_set_layout_binding_t) == 24, "vk abi");
_Static_assert(sizeof(vbs_vk_descriptor_set_layout_create_info_t) == 32, "vk abi");
_Static_assert(sizeof(vbs_vk_push_constant_range_t) == 12, "vk abi");
_Static_assert(sizeof(vbs_vk_pipeline_layout_create_info_t) == 48, "vk abi");
_Static_assert(sizeof(vbs_vk_shader_module_create_info_t) == 40, "vk abi");
_Static_assert(sizeof(vbs_vk_pipeline_shader_stage_create_info_t) == 48, "vk abi");
_Static_assert(sizeof(vbs_vk_compute_pipeline_create_info_t) == 96, "vk abi");
_Static_assert(sizeof(vbs_vk_descriptor_pool_size_t) == 8, "vk abi");
_Static_assert(sizeof(vbs_vk_descriptor_pool_create_info_t) == 40, "vk abi");
_Static_assert(sizeof(vbs_vk_descriptor_set_allocate_info_t) == 40, "vk abi");
_Static_assert(sizeof(vbs_vk_descriptor_buffer_info_t) == 24, "vk abi");
_Static_assert(sizeof(vbs_vk_write_descriptor_set_t) == 64, "vk abi");
_Static_assert(sizeof(vbs_vk_command_pool_create_info_t) == 24, "vk abi");
_Static_assert(sizeof(vbs_vk_command_buffer_allocate_info_t) == 32, "vk abi");
_Static_assert(sizeof(vbs_vk_command_buffer_begin_info_t) == 32, "vk abi");
_Static_assert(sizeof(vbs_vk_submit_info_t) == 72, "vk abi");
_Static_assert(sizeof(vbs_vk_fence_create_info_t) == 24, "vk abi");

/* Function pointer types (Vulkan 1.0 prototypes). */
typedef void *(*vbs_vk_get_instance_proc_addr_fn)(vbs_vk_instance_t, const char *);
typedef void *(*vbs_vk_get_device_proc_addr_fn)(vbs_vk_device_t, const char *);
typedef vbs_vk_result_t (*vbs_vk_create_instance_fn)(const vbs_vk_instance_create_info_t *, const void *, vbs_vk_instance_t *);
typedef void (*vbs_vk_destroy_instance_fn)(vbs_vk_instance_t, const void *);
typedef vbs_vk_result_t (*vbs_vk_enumerate_physical_devices_fn)(vbs_vk_instance_t, uint32_t *, vbs_vk_physical_device_t *);
typedef void (*vbs_vk_get_physical_device_properties_fn)(vbs_vk_physical_device_t, vbs_vk_physical_device_properties_t *);
typedef void (*vbs_vk_get_physical_device_queue_family_properties_fn)(vbs_vk_physical_device_t, uint32_t *, vbs_vk_queue_family_properties_t *);
typedef void (*vbs_vk_get_physical_device_memory_properties_fn)(vbs_vk_physical_device_t, vbs_vk_physical_device_memory_properties_t *);
typedef vbs_vk_result_t (*vbs_vk_create_device_fn)(vbs_vk_physical_device_t, const vbs_vk_device_create_info_t *, const void *, vbs_vk_device_t *);
typedef void (*vbs_vk_get_device_queue_fn)(vbs_vk_device_t, uint32_t, uint32_t, vbs_vk_queue_t *);
typedef vbs_vk_result_t (*vbs_vk_create_buffer_fn)(vbs_vk_device_t, const vbs_vk_buffer_create_info_t *, const void *, vbs_vk_buffer_t *);
typedef void (*vbs_vk_get_buffer_memory_requirements_fn)(vbs_vk_device_t, vbs_vk_buffer_t, vbs_vk_memory_requirements_t *);
typedef vbs_vk_result_t (*vbs_vk_allocate_memory_fn)(vbs_vk_device_t, const vbs_vk_memory_allocate_info_t *, const void *, vbs_vk_device_memory_t *);
typedef vbs_vk_result_t (*vbs_vk_bind_buffer_memory_fn)(vbs_vk_device_t, vbs_vk_buffer_t, vbs_vk_device_memory_t, vbs_vk_device_size_t);
typedef vbs_vk_result_t (*vbs_vk_map_memory_fn)(vbs_vk_device_t, vbs_vk_device_memory_t, vbs_vk_device_size_t, vbs_vk_device_size_t, vbs_vk_flags_t, void **);
typedef void (*vbs_vk_unmap_memory_fn)(vbs_vk_device_t, vbs_vk_device_memory_t);
typedef void (*vbs_vk_free_memory_fn)(vbs_vk_device_t, vbs_vk_device_memory_t, const void *);
typedef void (*vbs_vk_destroy_buffer_fn)(vbs_vk_device_t, vbs_vk_buffer_t, const void *);
typedef vbs_vk_result_t (*vbs_vk_create_shader_module_fn)(vbs_vk_device_t, const vbs_vk_shader_module_create_info_t *, const void *, vbs_vk_shader_module_t *);
typedef vbs_vk_result_t (*vbs_vk_create_descriptor_set_layout_fn)(vbs_vk_device_t, const vbs_vk_descriptor_set_layout_create_info_t *, const void *, vbs_vk_descriptor_set_layout_t *);
typedef vbs_vk_result_t (*vbs_vk_create_pipeline_layout_fn)(vbs_vk_device_t, const vbs_vk_pipeline_layout_create_info_t *, const void *, vbs_vk_pipeline_layout_t *);
typedef vbs_vk_result_t (*vbs_vk_create_compute_pipelines_fn)(vbs_vk_device_t, vbs_vk_pipeline_t, uint32_t, const vbs_vk_compute_pipeline_create_info_t *, const void *, vbs_vk_pipeline_t *);
typedef vbs_vk_result_t (*vbs_vk_create_descriptor_pool_fn)(vbs_vk_device_t, const vbs_vk_descriptor_pool_create_info_t *, const void *, vbs_vk_descriptor_pool_t *);
typedef vbs_vk_result_t (*vbs_vk_allocate_descriptor_sets_fn)(vbs_vk_device_t, const vbs_vk_descriptor_set_allocate_info_t *, vbs_vk_descriptor_set_t *);
typedef void (*vbs_vk_update_descriptor_sets_fn)(vbs_vk_device_t, uint32_t, const vbs_vk_write_descriptor_set_t *, uint32_t, const void *);
typedef vbs_vk_result_t (*vbs_vk_create_command_pool_fn)(vbs_vk_device_t, const vbs_vk_command_pool_create_info_t *, const void *, vbs_vk_command_pool_t *);
typedef vbs_vk_result_t (*vbs_vk_allocate_command_buffers_fn)(vbs_vk_device_t, const vbs_vk_command_buffer_allocate_info_t *, vbs_vk_command_buffer_t *);
typedef vbs_vk_result_t (*vbs_vk_begin_command_buffer_fn)(vbs_vk_command_buffer_t, const vbs_vk_command_buffer_begin_info_t *);
typedef vbs_vk_result_t (*vbs_vk_end_command_buffer_fn)(vbs_vk_command_buffer_t);
typedef vbs_vk_result_t (*vbs_vk_reset_command_buffer_fn)(vbs_vk_command_buffer_t, vbs_vk_flags_t);
typedef void (*vbs_vk_cmd_bind_pipeline_fn)(vbs_vk_command_buffer_t, uint32_t, vbs_vk_pipeline_t);
typedef void (*vbs_vk_cmd_bind_descriptor_sets_fn)(vbs_vk_command_buffer_t, uint32_t, vbs_vk_pipeline_layout_t, uint32_t, uint32_t, const vbs_vk_descriptor_set_t *, uint32_t, const uint32_t *);
typedef void (*vbs_vk_cmd_push_constants_fn)(vbs_vk_command_buffer_t, vbs_vk_pipeline_layout_t, uint32_t, uint32_t, uint32_t, const void *);
typedef void (*vbs_vk_cmd_dispatch_fn)(vbs_vk_command_buffer_t, uint32_t, uint32_t, uint32_t);
typedef vbs_vk_result_t (*vbs_vk_create_fence_fn)(vbs_vk_device_t, const vbs_vk_fence_create_info_t *, const void *, vbs_vk_fence_t *);
typedef vbs_vk_result_t (*vbs_vk_reset_fences_fn)(vbs_vk_device_t, uint32_t, const vbs_vk_fence_t *);
typedef vbs_vk_result_t (*vbs_vk_wait_for_fences_fn)(vbs_vk_device_t, uint32_t, const vbs_vk_fence_t *, vbs_vk_bool32_t, uint64_t);
typedef vbs_vk_result_t (*vbs_vk_queue_submit_fn)(vbs_vk_queue_t, uint32_t, const vbs_vk_submit_info_t *, vbs_vk_fence_t);
typedef vbs_vk_result_t (*vbs_vk_queue_wait_idle_fn)(vbs_vk_queue_t);

/* ================================================================== */
/* Loader + state                                                     */
/* ================================================================== */

static struct {
    void *handle;
    vbs_vk_get_instance_proc_addr_fn vkGetInstanceProcAddr;
    vbs_vk_get_device_proc_addr_fn vkGetDeviceProcAddr;
    vbs_vk_create_instance_fn vkCreateInstance;
    vbs_vk_destroy_instance_fn vkDestroyInstance;
    vbs_vk_enumerate_physical_devices_fn vkEnumeratePhysicalDevices;
    vbs_vk_get_physical_device_properties_fn vkGetPhysicalDeviceProperties;
    vbs_vk_get_physical_device_queue_family_properties_fn vkGetPhysicalDeviceQueueFamilyProperties;
    vbs_vk_get_physical_device_memory_properties_fn vkGetPhysicalDeviceMemoryProperties;
    vbs_vk_create_device_fn vkCreateDevice;
    vbs_vk_get_device_queue_fn vkGetDeviceQueue;
    vbs_vk_create_buffer_fn vkCreateBuffer;
    vbs_vk_get_buffer_memory_requirements_fn vkGetBufferMemoryRequirements;
    vbs_vk_allocate_memory_fn vkAllocateMemory;
    vbs_vk_bind_buffer_memory_fn vkBindBufferMemory;
    vbs_vk_map_memory_fn vkMapMemory;
    vbs_vk_unmap_memory_fn vkUnmapMemory;
    vbs_vk_free_memory_fn vkFreeMemory;
    vbs_vk_destroy_buffer_fn vkDestroyBuffer;
    vbs_vk_create_shader_module_fn vkCreateShaderModule;
    vbs_vk_create_descriptor_set_layout_fn vkCreateDescriptorSetLayout;
    vbs_vk_create_pipeline_layout_fn vkCreatePipelineLayout;
    vbs_vk_create_compute_pipelines_fn vkCreateComputePipelines;
    vbs_vk_create_descriptor_pool_fn vkCreateDescriptorPool;
    vbs_vk_allocate_descriptor_sets_fn vkAllocateDescriptorSets;
    vbs_vk_update_descriptor_sets_fn vkUpdateDescriptorSets;
    vbs_vk_create_command_pool_fn vkCreateCommandPool;
    vbs_vk_allocate_command_buffers_fn vkAllocateCommandBuffers;
    vbs_vk_begin_command_buffer_fn vkBeginCommandBuffer;
    vbs_vk_end_command_buffer_fn vkEndCommandBuffer;
    vbs_vk_reset_command_buffer_fn vkResetCommandBuffer;
    vbs_vk_cmd_bind_pipeline_fn vkCmdBindPipeline;
    vbs_vk_cmd_bind_descriptor_sets_fn vkCmdBindDescriptorSets;
    vbs_vk_cmd_push_constants_fn vkCmdPushConstants;
    vbs_vk_cmd_dispatch_fn vkCmdDispatch;
    vbs_vk_create_fence_fn vkCreateFence;
    vbs_vk_reset_fences_fn vkResetFences;
    vbs_vk_wait_for_fences_fn vkWaitForFences;
    vbs_vk_queue_submit_fn vkQueueSubmit;
    vbs_vk_queue_wait_idle_fn vkQueueWaitIdle;

    vbs_vk_instance_t instance;
    vbs_vk_physical_device_t phys;
    vbs_vk_device_t device;
    vbs_vk_queue_t queue;
    uint32_t queue_family;
    uint32_t mem_type_index; /* HOST_VISIBLE | HOST_COHERENT */
    vbs_vk_command_pool_t cmd_pool;
    vbs_vk_command_buffer_t cmd;
    vbs_vk_descriptor_pool_t desc_pool;
    vbs_vk_fence_t fence;
    bool ready;
    bool probed;
    bool software_device;
} g_vk;

static char g_vk_last_error[256] = "not probed yet";

const char *vbs_vulkan_last_error(void) { return g_vk_last_error; }

static void vk_fail(const char *msg) {
    snprintf(g_vk_last_error, sizeof(g_vk_last_error), "%s", msg);
}

static vbs_device_t g_vk_devices[1];
static int g_vk_dev_count = 0;

/* One VkBuffer + dedicated host-visible memory per engine buffer. */
typedef struct {
    vbs_vk_buffer_t buf;
    vbs_vk_device_memory_t mem;
    void *mapped;
    size_t size;
} vbs_vk_buffer_wrap_t;

/* A compiled compute pipeline (linear, embed, or primitive). */
typedef struct {
    vbs_prim_pipeline_t base; /* MUST be first: dispatch casts to this */
    vbs_vk_pipeline_t pipeline;
    vbs_vk_pipeline_layout_t layout;
    vbs_vk_descriptor_set_t set;
    uint32_t n_bindings;
    vbs_ggml_type_t type;     /* weight type for linear/embed */
    const char *kernel_name;  /* for diagnostics */
} vbs_vk_pipeline_wrap_t;

/* SPIR-V kernel table: name -> precompiled blob. */
typedef struct {
    const char *name;
    const uint32_t *words;
    size_t n_words;
} vbs_spv_entry_t;

#define VBS_SPV_ENTRY(nm) { #nm, vbs_spv_##nm, vbs_spv_##nm##_words }
static const vbs_spv_entry_t g_spv_table[] = {
    VBS_SPV_ENTRY(add),
    VBS_SPV_ENTRY(silu_mul),
    VBS_SPV_ENTRY(rope),
    VBS_SPV_ENTRY(rms_norm),
    VBS_SPV_ENTRY(embed_f32),
    VBS_SPV_ENTRY(embed_f16),
    VBS_SPV_ENTRY(embed_q8_0),
    VBS_SPV_ENTRY(embed_q4_0),
    VBS_SPV_ENTRY(embed_q4_1),
    VBS_SPV_ENTRY(embed_q4_k),
    VBS_SPV_ENTRY(embed_q6_k),
    VBS_SPV_ENTRY(kv_cache_store),
    VBS_SPV_ENTRY(attention_gqa),
    VBS_SPV_ENTRY(matvec_f32),
    VBS_SPV_ENTRY(matvec_f16),
    VBS_SPV_ENTRY(matvec_q8_0),
    VBS_SPV_ENTRY(matvec_q4_0),
    VBS_SPV_ENTRY(matvec_q4_1),
    VBS_SPV_ENTRY(matvec_q4_k),
    VBS_SPV_ENTRY(matvec_q6_k),
    VBS_SPV_ENTRY(matvec_lns8),
};
#undef VBS_SPV_ENTRY

static const vbs_spv_entry_t *vk_find_spv(const char *name) {
    for (size_t i = 0; i < sizeof(g_spv_table) / sizeof(g_spv_table[0]); i++)
        if (strcmp(g_spv_table[i].name, name) == 0) return &g_spv_table[i];
    return NULL;
}

/* Kernel name + binding count per op kind / weight type. */
static const char *vk_linear_kernel(vbs_ggml_type_t type) {
    switch (type) {
        case VBS_GGML_TYPE_F32:  return "matvec_f32";
        case VBS_GGML_TYPE_F16:  return "matvec_f16";
        case VBS_GGML_TYPE_Q8_0: return "matvec_q8_0";
        case VBS_GGML_TYPE_Q4_0: return "matvec_q4_0";
        case VBS_GGML_TYPE_Q4_1: return "matvec_q4_1";
        case VBS_GGML_TYPE_Q4_K: return "matvec_q4_k";
        case VBS_GGML_TYPE_Q6_K: return "matvec_q6_k";
        case VBS_GGML_TYPE_LNS8: return "matvec_lns8";
        default: return NULL;
    }
}

static const char *vk_embed_kernel(vbs_ggml_type_t type) {
    switch (type) {
        case VBS_GGML_TYPE_F32:  return "embed_f32";
        case VBS_GGML_TYPE_F16:  return "embed_f16";
        case VBS_GGML_TYPE_Q8_0: return "embed_q8_0";
        case VBS_GGML_TYPE_Q4_0: return "embed_q4_0";
        case VBS_GGML_TYPE_Q4_1: return "embed_q4_1";
        case VBS_GGML_TYPE_Q4_K: return "embed_q4_k";
        case VBS_GGML_TYPE_Q6_K: return "embed_q6_k";
        default: return NULL;
    }
}

static const char *vk_prim_kernel(vbs_op_kind_t kind) {
    switch (kind) {
        case VBS_OP_EMBEDDING:   return "embed_f32";
        case VBS_OP_NORMALIZE:   return "rms_norm";
        case VBS_OP_ROTARY:      return "rope";
        case VBS_OP_CACHE:       return "kv_cache_store";
        case VBS_OP_ATTENTION:   return "attention_gqa";
        case VBS_OP_ACTIVATION:  return "silu_mul";
        case VBS_OP_ELEMENTWISE: return "add";
        default: return NULL;
    }
}

/* Number of storage-buffer bindings per kernel. */
static uint32_t vk_kernel_bindings(const char *name) {
    if (strcmp(name, "rope") == 0) return 1;
    if (strncmp(name, "embed_", 6) == 0) return 2;
    if (strcmp(name, "kv_cache_store") == 0) return 4;
    if (strcmp(name, "attention_gqa") == 0) return 4;
    return 3; /* add, silu_mul, rms_norm, matvec_* */
}

/* Push-constant size per kernel (bytes). */
static uint32_t vk_kernel_push_size(const char *name) {
    if (strncmp(name, "matvec_", 7) == 0) return 24; /* 6 x u32 */
    if (strcmp(name, "attention_gqa") == 0) return 24;
    if (strcmp(name, "kv_cache_store") == 0) return 16;
    if (strcmp(name, "rope") == 0) return 16;
    if (strcmp(name, "rms_norm") == 0) return 8;
    if (strncmp(name, "embed_", 6) == 0) return 8;
    return 4; /* add, silu_mul: n */
}

/* ================================================================== */
/* Loader + init                                                      */
/* ================================================================== */

/* dlopen candidates: Linux, macOS (MoltenVK), Windows. */
static const char *const k_vk_lib_names[] = {
    "libvulkan.so.1", "libvulkan.so",
    "libvulkan.dylib", "libMoltenVK.dylib",
    "vulkan-1.dll", NULL
};

static bool vk_load_loader(void) {
    for (int i = 0; k_vk_lib_names[i]; i++) {
        g_vk.handle = vbs_dlopen(k_vk_lib_names[i], 1);
        if (g_vk.handle) break;
    }
    if (!g_vk.handle) { vk_fail("Vulkan loader not found (no libvulkan)"); return false; }

    g_vk.vkGetInstanceProcAddr =
        (vbs_vk_get_instance_proc_addr_fn)vbs_dlsym(g_vk.handle, "vkGetInstanceProcAddr");
    if (!g_vk.vkGetInstanceProcAddr) { vk_fail("vkGetInstanceProcAddr missing"); return false; }

    g_vk.vkCreateInstance =
        (vbs_vk_create_instance_fn)g_vk.vkGetInstanceProcAddr(NULL, "vkCreateInstance");
    if (!g_vk.vkCreateInstance) { vk_fail("vkCreateInstance missing"); return false; }
    return true;
}

#define LOAD_INST(name)                                                        \
    g_vk.name = (__typeof__(g_vk.name))g_vk.vkGetInstanceProcAddr(g_vk.instance, #name); \
    if (!g_vk.name) { vk_fail("loader: " #name " missing"); return false; }

#define LOAD_DEV(name)                                                         \
    g_vk.name = (__typeof__(g_vk.name))g_vk.vkGetDeviceProcAddr(g_vk.device, #name); \
    if (!g_vk.name) { vk_fail("loader: " #name " missing"); return false; }

static bool vk_load_instance_fns(void) {
    LOAD_INST(vkDestroyInstance);
    LOAD_INST(vkEnumeratePhysicalDevices);
    LOAD_INST(vkGetPhysicalDeviceProperties);
    LOAD_INST(vkGetPhysicalDeviceQueueFamilyProperties);
    LOAD_INST(vkGetPhysicalDeviceMemoryProperties);
    LOAD_INST(vkCreateDevice);
    LOAD_INST(vkGetDeviceQueue);
    LOAD_INST(vkGetDeviceProcAddr);
    return true;
}

static bool vk_load_device_fns(void) {
    LOAD_DEV(vkCreateBuffer);
    LOAD_DEV(vkGetBufferMemoryRequirements);
    LOAD_DEV(vkAllocateMemory);
    LOAD_DEV(vkBindBufferMemory);
    LOAD_DEV(vkMapMemory);
    LOAD_DEV(vkUnmapMemory);
    LOAD_DEV(vkFreeMemory);
    LOAD_DEV(vkDestroyBuffer);
    LOAD_DEV(vkCreateShaderModule);
    LOAD_DEV(vkCreateDescriptorSetLayout);
    LOAD_DEV(vkCreatePipelineLayout);
    LOAD_DEV(vkCreateComputePipelines);
    LOAD_DEV(vkCreateDescriptorPool);
    LOAD_DEV(vkAllocateDescriptorSets);
    LOAD_DEV(vkUpdateDescriptorSets);
    LOAD_DEV(vkCreateCommandPool);
    LOAD_DEV(vkAllocateCommandBuffers);
    LOAD_DEV(vkBeginCommandBuffer);
    LOAD_DEV(vkEndCommandBuffer);
    LOAD_DEV(vkResetCommandBuffer);
    LOAD_DEV(vkCmdBindPipeline);
    LOAD_DEV(vkCmdBindDescriptorSets);
    LOAD_DEV(vkCmdPushConstants);
    LOAD_DEV(vkCmdDispatch);
    LOAD_DEV(vkCreateFence);
    LOAD_DEV(vkResetFences);
    LOAD_DEV(vkWaitForFences);
    LOAD_DEV(vkQueueSubmit);
    LOAD_DEV(vkQueueWaitIdle);
    return true;
}

#undef LOAD_INST
#undef LOAD_DEV

static int vk_device_score(uint32_t type) {
    switch (type) {
        case VBS_VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return 4;
        case VBS_VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
        case VBS_VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return 2;
        case VBS_VK_PHYSICAL_DEVICE_TYPE_CPU:            return 1;
        default: return 0;
    }
}

static bool vk_find_memory_type(uint32_t type_bits, uint32_t *out) {
    vbs_vk_physical_device_memory_properties_t mp;
    g_vk.vkGetPhysicalDeviceMemoryProperties(g_vk.phys, &mp);
    /* Prefer HOST_VISIBLE|HOST_COHERENT|DEVICE_LOCAL, fall back to
     * HOST_VISIBLE|HOST_COHERENT. */
    for (uint32_t pass = 0; pass < 2; pass++) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
            if (!(type_bits & (1u << i))) continue;
            uint32_t f = mp.memoryTypes[i].propertyFlags;
            uint32_t want = VBS_VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VBS_VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            if (pass == 0) want |= VBS_VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            if ((f & want) == want) { *out = i; return true; }
        }
    }
    return false;
}

static bool vulkan_init(void) {
    if (g_vk.ready) return true;
    if (g_vk.probed) return false; /* probed once, failed; don't retry */
    g_vk.probed = true;

    if (!vk_load_loader()) return false;

    vbs_vk_application_info_t app;
    memset(&app, 0, sizeof(app));
    app.sType = VBS_VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "Verum-Bespoke-Singularity";
    app.applicationVersion = 1;
    app.pEngineName = "verum-slice";
    app.engineVersion = 1;
    app.apiVersion = VBS_VK_API_VERSION_1_0;

    vbs_vk_instance_create_info_t ici;
    memset(&ici, 0, sizeof(ici));
    ici.sType = VBS_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;

    if (g_vk.vkCreateInstance(&ici, NULL, &g_vk.instance) != VBS_VK_SUCCESS) {
        vk_fail("vkCreateInstance failed (no Vulkan ICD/driver?)");
        return false;
    }
    if (!vk_load_instance_fns()) return false;

    uint32_t n_devs = 0;
    g_vk.vkEnumeratePhysicalDevices(g_vk.instance, &n_devs, NULL);
    if (n_devs == 0) { vk_fail("no Vulkan physical devices"); return false; }
    vbs_vk_physical_device_t *devs = malloc(n_devs * sizeof(*devs));
    if (!devs) { vk_fail("out of memory"); return false; }
    g_vk.vkEnumeratePhysicalDevices(g_vk.instance, &n_devs, devs);

    /* Pick the best device that has a compute queue. */
    int best_score = -1;
    vbs_vk_physical_device_properties_t best_props;
    uint32_t best_family = 0;
    for (uint32_t d = 0; d < n_devs; d++) {
        vbs_vk_physical_device_properties_t props;
        g_vk.vkGetPhysicalDeviceProperties(devs[d], &props);
        uint32_t n_fam = 0;
        g_vk.vkGetPhysicalDeviceQueueFamilyProperties(devs[d], &n_fam, NULL);
        if (n_fam == 0) continue;
        vbs_vk_queue_family_properties_t *fams = malloc(n_fam * sizeof(*fams));
        if (!fams) continue;
        g_vk.vkGetPhysicalDeviceQueueFamilyProperties(devs[d], &n_fam, fams);
        int fam_idx = -1;
        for (uint32_t f = 0; f < n_fam; f++)
            if (fams[f].queueFlags & VBS_VK_QUEUE_COMPUTE_BIT) { fam_idx = (int)f; break; }
        free(fams);
        if (fam_idx < 0) continue;
        int score = vk_device_score(props.deviceType);
        if (score > best_score) {
            best_score = score;
            g_vk.phys = devs[d];
            best_props = props;
            best_family = (uint32_t)fam_idx;
        }
    }
    free(devs);
    if (best_score < 0) { vk_fail("no Vulkan device with a compute queue"); return false; }
    g_vk.software_device = (best_props.deviceType == VBS_VK_PHYSICAL_DEVICE_TYPE_CPU);
    g_vk.queue_family = best_family;

    float prio = 1.0f;
    vbs_vk_device_queue_create_info_t qci;
    memset(&qci, 0, sizeof(qci));
    qci.sType = VBS_VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = best_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    vbs_vk_device_create_info_t dci;
    memset(&dci, 0, sizeof(dci));
    dci.sType = VBS_VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;

    if (g_vk.vkCreateDevice(g_vk.phys, &dci, NULL, &g_vk.device) != VBS_VK_SUCCESS) {
        vk_fail("vkCreateDevice failed");
        return false;
    }
    g_vk.vkGetDeviceQueue(g_vk.device, best_family, 0, &g_vk.queue);
    if (!vk_load_device_fns()) return false;

    /* Memory type: probe with a dummy 1-byte buffer's requirements. */
    {
        vbs_vk_buffer_create_info_t bci;
        memset(&bci, 0, sizeof(bci));
        bci.sType = VBS_VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = 1;
        bci.usage = VBS_VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VBS_VK_SHARING_MODE_EXCLUSIVE;
        vbs_vk_buffer_t tmp;
        if (g_vk.vkCreateBuffer(g_vk.device, &bci, NULL, &tmp) != VBS_VK_SUCCESS) {
            vk_fail("vkCreateBuffer failed (probe)");
            return false;
        }
        vbs_vk_memory_requirements_t req;
        g_vk.vkGetBufferMemoryRequirements(g_vk.device, tmp, &req);
        g_vk.vkDestroyBuffer(g_vk.device, tmp, NULL);
        if (!vk_find_memory_type(req.memoryTypeBits, &g_vk.mem_type_index)) {
            vk_fail("no HOST_VISIBLE|HOST_COHERENT memory type");
            return false;
        }
    }

    /* Command pool + one primary command buffer + fence. */
    vbs_vk_command_pool_create_info_t cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType = VBS_VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VBS_VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = best_family;
    if (g_vk.vkCreateCommandPool(g_vk.device, &cpci, NULL, &g_vk.cmd_pool) != VBS_VK_SUCCESS) {
        vk_fail("vkCreateCommandPool failed");
        return false;
    }
    vbs_vk_command_buffer_allocate_info_t cbai;
    memset(&cbai, 0, sizeof(cbai));
    cbai.sType = VBS_VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = g_vk.cmd_pool;
    cbai.level = VBS_VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (g_vk.vkAllocateCommandBuffers(g_vk.device, &cbai, &g_vk.cmd) != VBS_VK_SUCCESS) {
        vk_fail("vkAllocateCommandBuffers failed");
        return false;
    }
    vbs_vk_fence_create_info_t fci;
    memset(&fci, 0, sizeof(fci));
    fci.sType = VBS_VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (g_vk.vkCreateFence(g_vk.device, &fci, NULL, &g_vk.fence) != VBS_VK_SUCCESS) {
        vk_fail("vkCreateFence failed");
        return false;
    }

    /* Descriptor pool: 64 sets x up to 8 descriptors. */
    vbs_vk_descriptor_pool_size_t psize = { VBS_VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512 };
    vbs_vk_descriptor_pool_create_info_t dpci;
    memset(&dpci, 0, sizeof(dpci));
    dpci.sType = VBS_VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 64;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &psize;
    if (g_vk.vkCreateDescriptorPool(g_vk.device, &dpci, NULL, &g_vk.desc_pool) != VBS_VK_SUCCESS) {
        vk_fail("vkCreateDescriptorPool failed");
        return false;
    }

    /* Publish the one device. */
    memset(&g_vk_devices[0], 0, sizeof(g_vk_devices[0]));
    snprintf(g_vk_devices[0].name, sizeof(g_vk_devices[0].name), "Vulkan %.100s%s",
             best_props.deviceName, g_vk.software_device ? " [software]" : "");
    g_vk_devices[0].type = VBS_BACKEND_VULKAN;
    /* Nominal 1 GiB caps: the real bound is host RAM via HOST_VISIBLE
     * memory; precise heap accounting needs per-heap enumeration and
     * is a v2 improvement, not a correctness issue: alloc failures are
     * reported honestly at alloc time. */
    g_vk_devices[0].max_working_set_bytes = 1ull << 30;
    g_vk_devices[0].max_buffer_bytes = 1ull << 30;
    g_vk_devices[0].internal_handle = NULL;
    g_vk_dev_count = 1;
    g_vk.ready = true;
    vk_fail("ok");
    return true;
}

/* ================================================================== */
/* Buffers                                                            */
/* ================================================================== */

static size_t vulkan_get_device_count(void) {
    if (!vulkan_init()) return 0;
    return (size_t)g_vk_dev_count;
}

static vbs_device_t *vulkan_get_device(size_t index) {
    if (!vulkan_init()) return NULL;
    return index == 0 ? &g_vk_devices[0] : NULL;
}

static vbs_buffer_t *vulkan_alloc_buffer(vbs_device_t *dev, size_t bytes, bool shared) {
    (void)shared; /* all Vulkan memory here is host-visible by design */
    if (!g_vk.ready && !vulkan_init()) return NULL;
    if (bytes == 0) bytes = 1;

    vbs_vk_buffer_wrap_t *w = calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->size = bytes;

    vbs_vk_buffer_create_info_t bci;
    memset(&bci, 0, sizeof(bci));
    bci.sType = VBS_VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = VBS_VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VBS_VK_SHARING_MODE_EXCLUSIVE;
    if (g_vk.vkCreateBuffer(g_vk.device, &bci, NULL, &w->buf) != VBS_VK_SUCCESS) {
        free(w);
        return NULL;
    }
    vbs_vk_memory_requirements_t req;
    g_vk.vkGetBufferMemoryRequirements(g_vk.device, w->buf, &req);
    vbs_vk_memory_allocate_info_t mai;
    memset(&mai, 0, sizeof(mai));
    mai.sType = VBS_VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = g_vk.mem_type_index;
    if (g_vk.vkAllocateMemory(g_vk.device, &mai, NULL, &w->mem) != VBS_VK_SUCCESS) {
        g_vk.vkDestroyBuffer(g_vk.device, w->buf, NULL);
        free(w);
        return NULL;
    }
    if (g_vk.vkBindBufferMemory(g_vk.device, w->buf, w->mem, 0) != VBS_VK_SUCCESS) {
        g_vk.vkFreeMemory(g_vk.device, w->mem, NULL);
        g_vk.vkDestroyBuffer(g_vk.device, w->buf, NULL);
        free(w);
        return NULL;
    }
    if (g_vk.vkMapMemory(g_vk.device, w->mem, 0, VBS_VK_WHOLE_SIZE, 0, &w->mapped) != VBS_VK_SUCCESS) {
        g_vk.vkFreeMemory(g_vk.device, w->mem, NULL);
        g_vk.vkDestroyBuffer(g_vk.device, w->buf, NULL);
        free(w);
        return NULL;
    }

    vbs_buffer_t *b = calloc(1, sizeof(*b));
    if (!b) {
        g_vk.vkUnmapMemory(g_vk.device, w->mem);
        g_vk.vkFreeMemory(g_vk.device, w->mem, NULL);
        g_vk.vkDestroyBuffer(g_vk.device, w->buf, NULL);
        free(w);
        return NULL;
    }
    b->internal_handle = w;
    b->host_ptr = w->mapped; /* staging writes go straight to device memory */
    b->size = bytes;
    b->device = dev;
    return b;
}

static void vulkan_free_buffer(vbs_buffer_t *b) {
    if (!b) return;
    vbs_vk_buffer_wrap_t *w = b->internal_handle;
    if (w) {
        if (w->mapped) g_vk.vkUnmapMemory(g_vk.device, w->mem);
        if (w->buf) g_vk.vkDestroyBuffer(g_vk.device, w->buf, NULL);
        if (w->mem) g_vk.vkFreeMemory(g_vk.device, w->mem, NULL);
        free(w);
    }
    free(b);
}

static bool vulkan_copy_to_device(vbs_buffer_t *dst, const void *src, size_t bytes) {
    if (!dst || !dst->host_ptr || !src) return false;
    if (bytes > dst->size) return false;
    memcpy(dst->host_ptr, src, bytes); /* HOST_COHERENT: visible immediately */
    return true;
}

static bool vulkan_copy_to_host(void *dst, const vbs_buffer_t *src, size_t bytes) {
    if (!dst || !src || !src->host_ptr) return false;
    if (bytes > src->size) return false;
    memcpy(dst, src->host_ptr, bytes);
    return true;
}

static void vulkan_synchronize_device(vbs_device_t *dev) {
    (void)dev;
    if (g_vk.ready) g_vk.vkQueueWaitIdle(g_vk.queue);
}

/* ================================================================== */
/* Pipelines                                                          */
/* ================================================================== */

static void *vk_make_pipeline(const char *kernel_name, vbs_op_kind_t kind,
                              vbs_ggml_type_t type, char *err, size_t err_size) {
    const vbs_spv_entry_t *e = vk_find_spv(kernel_name);
    if (!e) {
        if (err) snprintf(err, err_size, "no SPIR-V kernel '%s'", kernel_name);
        return NULL;
    }
    /* Structural validation before the driver ever sees the module. */
    if (spirv_validate(e->words, e->n_words, err, err_size) != 0) return NULL;

    vbs_vk_shader_module_create_info_t smci;
    memset(&smci, 0, sizeof(smci));
    smci.sType = VBS_VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = e->n_words * 4;
    smci.pCode = e->words;
    vbs_vk_shader_module_t module = VBS_VK_NULL_HANDLE;
    if (g_vk.vkCreateShaderModule(g_vk.device, &smci, NULL, &module) != VBS_VK_SUCCESS) {
        if (err) snprintf(err, err_size, "vkCreateShaderModule failed (%s)", kernel_name);
        return NULL;
    }

    uint32_t n_bind = vk_kernel_bindings(kernel_name);
    vbs_vk_descriptor_set_layout_binding_t binds[8];
    for (uint32_t i = 0; i < n_bind; i++) {
        binds[i].binding = i;
        binds[i].descriptorType = VBS_VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VBS_VK_SHADER_STAGE_COMPUTE_BIT;
        binds[i].pImmutableSamplers = NULL;
    }
    vbs_vk_descriptor_set_layout_create_info_t dslci;
    memset(&dslci, 0, sizeof(dslci));
    dslci.sType = VBS_VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = n_bind;
    dslci.pBindings = binds;
    vbs_vk_descriptor_set_layout_t layout_set = VBS_VK_NULL_HANDLE;
    if (g_vk.vkCreateDescriptorSetLayout(g_vk.device, &dslci, NULL, &layout_set) != VBS_VK_SUCCESS) {
        if (err) snprintf(err, err_size, "vkCreateDescriptorSetLayout failed (%s)", kernel_name);
        return NULL;
    }
    vbs_vk_push_constant_range_t pcr = { VBS_VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                         vk_kernel_push_size(kernel_name) };
    vbs_vk_pipeline_layout_create_info_t plci;
    memset(&plci, 0, sizeof(plci));
    plci.sType = VBS_VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &layout_set;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    vbs_vk_pipeline_layout_t layout = VBS_VK_NULL_HANDLE;
    if (g_vk.vkCreatePipelineLayout(g_vk.device, &plci, NULL, &layout) != VBS_VK_SUCCESS) {
        if (err) snprintf(err, err_size, "vkCreatePipelineLayout failed (%s)", kernel_name);
        return NULL;
    }
    vbs_vk_pipeline_shader_stage_create_info_t stage;
    memset(&stage, 0, sizeof(stage));
    stage.sType = VBS_VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VBS_VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";
    vbs_vk_compute_pipeline_create_info_t cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType = VBS_VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = stage;
    cpci.layout = layout;
    vbs_vk_pipeline_t pipeline = VBS_VK_NULL_HANDLE;
    if (g_vk.vkCreateComputePipelines(g_vk.device, VBS_VK_NULL_HANDLE, 1, &cpci, NULL, &pipeline)
        != VBS_VK_SUCCESS) {
        if (err) snprintf(err, err_size, "vkCreateComputePipelines failed (%s)", kernel_name);
        return NULL;
    }
    vbs_vk_descriptor_set_allocate_info_t dsai;
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType = VBS_VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = g_vk.desc_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &layout_set;
    vbs_vk_descriptor_set_t set = VBS_VK_NULL_HANDLE;
    if (g_vk.vkAllocateDescriptorSets(g_vk.device, &dsai, &set) != VBS_VK_SUCCESS) {
        if (err) snprintf(err, err_size, "vkAllocateDescriptorSets failed (%s)", kernel_name);
        return NULL;
    }

    vbs_vk_pipeline_wrap_t *w = calloc(1, sizeof(*w));
    if (!w) { if (err) snprintf(err, err_size, "out of memory"); return NULL; }
    w->pipeline = pipeline;
    w->layout = layout;
    w->set = set;
    w->base.native = w;
    w->base.kind = kind;
    w->n_bindings = n_bind;
    w->type = type;
    w->kernel_name = kernel_name;
    return w;
}

/* Run one dispatch synchronously. bufs[i] binds to binding i. */
static bool vk_run(vbs_vk_pipeline_wrap_t *w, vbs_buffer_t **bufs,
                   const void *push, uint32_t push_size,
                   uint32_t gx, uint32_t gy, uint32_t gz) {
    vbs_vk_descriptor_buffer_info_t infos[8];
    vbs_vk_write_descriptor_set_t writes[8];
    for (uint32_t i = 0; i < w->n_bindings; i++) {
        vbs_vk_buffer_wrap_t *bw = bufs[i]->internal_handle;
        infos[i].buffer = bw->buf;
        infos[i].offset = 0;
        infos[i].range = VBS_VK_WHOLE_SIZE;
        memset(&writes[i], 0, sizeof(writes[i]));
        writes[i].sType = VBS_VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = w->set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VBS_VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    g_vk.vkUpdateDescriptorSets(g_vk.device, w->n_bindings, writes, 0, NULL);

    g_vk.vkResetCommandBuffer(g_vk.cmd, 0);
    vbs_vk_command_buffer_begin_info_t cbbi;
    memset(&cbbi, 0, sizeof(cbbi));
    cbbi.sType = VBS_VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VBS_VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (g_vk.vkBeginCommandBuffer(g_vk.cmd, &cbbi) != VBS_VK_SUCCESS) return false;
    g_vk.vkCmdBindPipeline(g_vk.cmd, VBS_VK_PIPELINE_BIND_POINT_COMPUTE, w->pipeline);
    g_vk.vkCmdBindDescriptorSets(g_vk.cmd, VBS_VK_PIPELINE_BIND_POINT_COMPUTE,
                                 w->layout, 0, 1, &w->set, 0, NULL);
    if (push && push_size)
        g_vk.vkCmdPushConstants(g_vk.cmd, w->layout, VBS_VK_SHADER_STAGE_COMPUTE_BIT,
                                0, push_size, push);
    g_vk.vkCmdDispatch(g_vk.cmd, gx, gy, gz);
    if (g_vk.vkEndCommandBuffer(g_vk.cmd) != VBS_VK_SUCCESS) return false;

    g_vk.vkResetFences(g_vk.device, 1, &g_vk.fence);
    vbs_vk_submit_info_t si;
    memset(&si, 0, sizeof(si));
    si.sType = VBS_VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_vk.cmd;
    if (g_vk.vkQueueSubmit(g_vk.queue, 1, &si, g_vk.fence) != VBS_VK_SUCCESS) return false;
    if (g_vk.vkWaitForFences(g_vk.device, 1, &g_vk.fence, 1, UINT64_MAX) != VBS_VK_SUCCESS)
        return false;
    return true;
}

static void *vulkan_compile_linear_pipeline(vbs_device_t *dev, vbs_ggml_type_t type,
                                            char *err, size_t err_size) {
    (void)dev;
    if (!g_vk.ready && !vulkan_init()) {
        if (err) snprintf(err, err_size, "Vulkan unavailable: %s", g_vk_last_error);
        return NULL;
    }
    const char *k = vk_linear_kernel(type);
    if (!k) {
        if (err) snprintf(err, err_size, "Vulkan: no matvec kernel for type %d", (int)type);
        return NULL;
    }
    return vk_make_pipeline(k, VBS_OP_LINEAR, type, err, err_size);
}

static void *vulkan_compile_embed_pipeline(vbs_device_t *dev, vbs_ggml_type_t type,
                                           char *err, size_t err_size) {
    (void)dev;
    if (!g_vk.ready && !vulkan_init()) {
        if (err) snprintf(err, err_size, "Vulkan unavailable: %s", g_vk_last_error);
        return NULL;
    }
    const char *k = vk_embed_kernel(type);
    if (!k) {
        if (err) snprintf(err, err_size, "Vulkan: no embed kernel for type %d", (int)type);
        return NULL;
    }
    return vk_make_pipeline(k, VBS_OP_EMBEDDING, type, err, err_size);
}

static void *vulkan_compile_primitive_pipeline(vbs_device_t *dev, vbs_op_kind_t kind,
                                               char *err, size_t err_size) {
    (void)dev;
    if (!g_vk.ready && !vulkan_init()) {
        if (err) snprintf(err, err_size, "Vulkan unavailable: %s", g_vk_last_error);
        return NULL;
    }
    if (kind == VBS_OP_LINEAR) {
        if (err) snprintf(err, err_size, "Vulkan: use compile_linear_pipeline for LINEAR");
        return NULL;
    }
    const char *k = vk_prim_kernel(kind);
    if (!k) {
        if (err) snprintf(err, err_size, "Vulkan: no kernel for op kind %d", (int)kind);
        return NULL;
    }
    return vk_make_pipeline(k, kind, VBS_GGML_TYPE_F32, err, err_size);
}

/* NOTE: the ops table has no free_pipeline slot — pipelines are
 * process-lifetime objects in the engine, so Vulkan handles are
 * intentionally not destroyed (same as CUDA/Metal). */

/* ================================================================== */
/* Dispatch                                                           */
/* ================================================================== */

/* matvec push constants: cols, rows, col_off, row_off, accumulate, row_base */
typedef struct { uint32_t cols, rows, col_off, row_off, accumulate, row_base; } vbs_mv_pc_t;

static bool vulkan_dispatch_linear_tile(vbs_device_t *dev, void *pipeline,
                                        vbs_buffer_t *priv,
                                        vbs_buffer_t *tile, size_t tile_bytes,
                                        vbs_buffer_t *x, vbs_buffer_t *y,
                                        uint32_t cols, uint32_t rows,
                                        uint32_t col_off, uint32_t row_off, bool accumulate) {
    (void)dev;
    vbs_vk_pipeline_wrap_t *w = pipeline;
    if (!w || !w->pipeline) return false;
    if (rows == 0) return true;

    /* The matvec kernels index weights by (row_base + row) * stride, so
     * the weight buffer must hold the FULL tile rows starting at row 0.
     * The slice executor's convention (tile staging buffer) and the
     * bridge's convention (priv IS the resident weight tensor) both
     * arrive here with the weight data in `tile`/`priv` respectively;
     * unify: weights = tile->host_ptr already staged (convention B) or
     * priv (convention A, tile == NULL). */
    vbs_buffer_t *weights = tile ? tile : priv;
    if (!weights || !x || !y) return false;
    (void)tile_bytes;

    /* Chunk rows so no dispatch exceeds the Vulkan minimum
     * maxComputeWorkGroupCount[0] = 65535. */
    uint32_t done = 0;
    while (done < rows) {
        uint32_t chunk = rows - done;
        if (chunk > 65535) chunk = 65535;
        vbs_mv_pc_t pc = { cols, chunk, col_off, row_off,
                           accumulate ? 1u : 0u, done };
        vbs_buffer_t *bufs[3] = { weights, x, y };
        if (!vk_run(w, bufs, &pc, sizeof(pc), chunk, 1, 1)) return false;
        done += chunk;
    }
    return true;
}

/* --- primitive arg helpers (bridge argument order) --- */

/* Wrap a raw internal handle (from a vbs_arg_t's .value) in a temporary
 * vbs_buffer_t so vk_run can read internal_handle. The wrapper is
 * stack-allocated by the caller. */
static vbs_buffer_t wrap_buf(void *handle) {
    vbs_buffer_t b;
    memset(&b, 0, sizeof(b));
    b.internal_handle = handle;
    return b;
}

static bool vulkan_dispatch_primitive(vbs_device_t *dev, void *pipeline,
                                      const vbs_arg_t *args, size_t n_args,
                                      size_t grid_x, size_t block_x) {
    (void)dev;
    vbs_vk_pipeline_wrap_t *w = pipeline;
    if (!w || !w->pipeline) return false;
    /* NOTE: grid_x / block_x are CUDA launch hints from the bridge and
     * are intentionally IGNORED: workgroup counts are derived from the
     * problem size and each kernel's fixed local size. */
    (void)grid_x; (void)block_x;

    switch (w->base.kind) {
    case VBS_OP_EMBEDDING: {
        /* args: B w, S token, B out, S dim */
        if (n_args < 4) return false;
        vbs_buffer_t wb = wrap_buf((void *)args[0].value);
        vbs_buffer_t ob = wrap_buf((void *)args[2].value);
        uint32_t token = *(const uint32_t *)args[1].value;
        uint32_t dim = *(const uint32_t *)args[3].value;
        if (dim == 0) return true;
        uint32_t pc[2] = { token, dim };
        vbs_buffer_t *bufs[2] = { &wb, &ob };
        /* embed local_size 256 */
        return vk_run(w, bufs, pc, sizeof(pc), (dim + 255) / 256, 1, 1);
    }
    case VBS_OP_NORMALIZE: {
        /* args: B in, B w, B out, S dim, S eps */
        if (n_args < 5) return false;
        vbs_buffer_t ib = wrap_buf((void *)args[0].value);
        vbs_buffer_t wb = wrap_buf((void *)args[1].value);
        vbs_buffer_t ob = wrap_buf((void *)args[2].value);
        uint32_t dim = *(const uint32_t *)args[3].value;
        float eps = *(const float *)args[4].value;
        if (dim == 0) return true;
        struct { uint32_t dim; float eps; } pc = { dim, eps };
        vbs_buffer_t *bufs[3] = { &ib, &wb, &ob };
        /* rms_norm local_size 256; one invocation per element. */
        return vk_run(w, bufs, &pc, sizeof(pc), (dim + 255) / 256, 1, 1);
    }
    case VBS_OP_ROTARY: {
        /* args: B x, S pos, S head_dim, S n_heads, S theta */
        if (n_args < 5) return false;
        vbs_buffer_t xb = wrap_buf((void *)args[0].value);
        uint32_t pos = *(const uint32_t *)args[1].value;
        uint32_t head_dim = *(const uint32_t *)args[2].value;
        uint32_t n_heads = *(const uint32_t *)args[3].value;
        float theta = *(const float *)args[4].value;
        if (head_dim == 0 || n_heads == 0) return true;
        struct { uint32_t pos, head_dim, n_heads; float theta; } pc =
            { pos, head_dim, n_heads, theta };
        vbs_buffer_t *bufs[1] = { &xb };
        /* rope local_size 256, one invocation per element pair */
        uint32_t n = head_dim * n_heads;
        return vk_run(w, bufs, &pc, sizeof(pc), (n / 2 + 255) / 256, 1, 1);
    }
    case VBS_OP_CACHE: {
        /* args: B k, B v, B kc, B vc, S pos, S layer, S max_ctx, S kv_dim */
        if (n_args < 8) return false;
        vbs_buffer_t kb = wrap_buf((void *)args[0].value);
        vbs_buffer_t vb = wrap_buf((void *)args[1].value);
        vbs_buffer_t kcb = wrap_buf((void *)args[2].value);
        vbs_buffer_t vcb = wrap_buf((void *)args[3].value);
        uint32_t pos = *(const uint32_t *)args[4].value;
        uint32_t layer = *(const uint32_t *)args[5].value;
        uint32_t max_ctx = *(const uint32_t *)args[6].value;
        uint32_t kv_dim = *(const uint32_t *)args[7].value;
        if (kv_dim == 0) return true;
        if (kv_dim % 2 != 0) return false; /* kernel packs f16 pairs */
        /* 32-bit index guard: (layer*max_ctx+pos)*kv_dim must fit u32. */
        uint64_t idx = ((uint64_t)layer * max_ctx + pos) * kv_dim;
        if (idx + kv_dim > 0xFFFFFFFFull) return false;
        uint32_t pc[4] = { pos, layer, max_ctx, kv_dim };
        vbs_buffer_t *bufs[4] = { &kb, &vb, &kcb, &vcb };
        /* one invocation per f16 pair, local_size 256 */
        return vk_run(w, bufs, pc, sizeof(pc), (kv_dim / 2 + 255) / 256, 1, 1);
    }
    case VBS_OP_ATTENTION: {
        /* args: B q, B kc, B vc, B out, S seq_len, S layer, S max_ctx,
         *       S n_q_heads, S n_kv_heads, S head_dim, S scale */
        if (n_args < 11) return false;
        vbs_buffer_t qb = wrap_buf((void *)args[0].value);
        vbs_buffer_t kcb = wrap_buf((void *)args[1].value);
        vbs_buffer_t vcb = wrap_buf((void *)args[2].value);
        vbs_buffer_t ob = wrap_buf((void *)args[3].value);
        uint32_t seq_len = *(const uint32_t *)args[4].value;
        uint32_t layer = *(const uint32_t *)args[5].value;
        uint32_t max_ctx = *(const uint32_t *)args[6].value;
        uint32_t nqh = *(const uint32_t *)args[7].value;
        uint32_t nkh = *(const uint32_t *)args[8].value;
        uint32_t hd = *(const uint32_t *)args[9].value;
        float scale = *(const float *)args[10].value;
        if (nqh == 0 || hd == 0) return true;
        if (hd > 512) return false; /* kernel's local acc array cap */
        if (nkh == 0 || nqh % nkh != 0) return false;
        /* 32-bit cache-index guard: layer*max_ctx*nkh*hd must fit u32. */
        uint64_t off = (uint64_t)layer * max_ctx * nkh * hd;
        if (off + (uint64_t)seq_len * nkh * hd > 0xFFFFFFFFull) return false;
        /* Kernel push layout: seq_len, n_q_heads, n_kv_heads, head_dim,
         * scale, cache_off — the layer slice is folded into cache_off
         * (f16 elements), exactly as the CUDA backend folds it into its
         * cache pointers. */
        struct { uint32_t seq_len, n_q_heads, n_kv_heads, head_dim;
                 float scale; uint32_t cache_off; } pc =
            { seq_len, nqh, nkh, hd, scale, (uint32_t)off };
        vbs_buffer_t *bufs[4] = { &qb, &kcb, &vcb, &ob };
        /* local_size 1: one invocation per Q head */
        return vk_run(w, bufs, &pc, sizeof(pc), nqh, 1, 1);
    }
    case VBS_OP_ACTIVATION: {
        /* args: B gate, B up, B out, S n */
        if (n_args < 4) return false;
        vbs_buffer_t gb = wrap_buf((void *)args[0].value);
        vbs_buffer_t ub = wrap_buf((void *)args[1].value);
        vbs_buffer_t ob = wrap_buf((void *)args[2].value);
        uint32_t n = *(const uint32_t *)args[3].value;
        if (n == 0) return true;
        uint32_t pc = n;
        vbs_buffer_t *bufs[3] = { &gb, &ub, &ob };
        return vk_run(w, bufs, &pc, sizeof(pc), (n + 255) / 256, 1, 1);
    }
    case VBS_OP_ELEMENTWISE: {
        /* args: B a, B b, B out, S n */
        if (n_args < 4) return false;
        vbs_buffer_t ab = wrap_buf((void *)args[0].value);
        vbs_buffer_t bb = wrap_buf((void *)args[1].value);
        vbs_buffer_t ob = wrap_buf((void *)args[2].value);
        uint32_t n = *(const uint32_t *)args[3].value;
        if (n == 0) return true;
        uint32_t pc = n;
        vbs_buffer_t *bufs[3] = { &ab, &bb, &ob };
        return vk_run(w, bufs, &pc, sizeof(pc), (n + 255) / 256, 1, 1);
    }
    default:
        return false;
    }
}

/* ================================================================== */
/* Emitter wiring (source-level SPIR-V)                               */
/* ================================================================== */

/* Compile a source kernel to SPIR-V through the in-repo emitter and
 * validate the result. See the header comment in vbs_vulkan.h: the
 * emitter currently produces a structural shell, so this is a
 * plumbing / test hook, NOT a dispatch path. Returns 0 on success. */
int vbs_vulkan_compile_kernel_source(const char *source, size_t source_len,
                                     uint32_t *out_words, size_t *out_words_used,
                                     char *err, size_t err_size) {
    if (!source || !out_words || !out_words_used) {
        if (err) snprintf(err, err_size, "null argument");
        return -1;
    }
    /* *out_words_used is capacity in, words used out (emitter needs >= 64). */
    if (vbs_compile_to_spirv(source, source_len, out_words, out_words_used,
                             err, err_size) != 0)
        return -1;
    if (spirv_validate(out_words, *out_words_used, err, err_size) != 0) return -1;
    return 0;
}

/* ================================================================== */
/* Ops table                                                          */
/* ================================================================== */

const vbs_backend_ops_t g_vbs_vulkan_backend = {
    .init = vulkan_init,
    .get_device_count = vulkan_get_device_count,
    .get_device = vulkan_get_device,
    .alloc_buffer = vulkan_alloc_buffer,
    .free_buffer = vulkan_free_buffer,
    .copy_to_device = vulkan_copy_to_device,
    .copy_to_host = vulkan_copy_to_host,
    .compile_linear_pipeline = vulkan_compile_linear_pipeline,
    .compile_embed_pipeline = vulkan_compile_embed_pipeline,
    .compile_primitive_pipeline = vulkan_compile_primitive_pipeline,
    .dispatch_linear_tile = vulkan_dispatch_linear_tile,
    .dispatch_primitive = vulkan_dispatch_primitive,
    .synchronize_device = vulkan_synchronize_device,
};
