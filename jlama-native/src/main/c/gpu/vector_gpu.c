#include "vector_gpu.h"
#include "webgpu.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UNUSED(x) (void)x;

#define RM 8
#define RN 8
#define RN_M1 64

// Info for quantization
#define Q8_BLOCK_SIZE 32
#define Q4_BLOCK_SIZE 32

typedef struct {
    WGPUBuffer input_buffer;
    WGPUBuffer input2_buffer;
    WGPUBuffer params_buffer;
    WGPUBuffer result_buffer;
    WGPUBuffer result_staging_buffer;
    WGPUBuffer empty_buffer;
} Scratch;

// Parameters struct matching WGSL
typedef struct {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
} Params;

static void log_callback(WGPULoggingType level, struct WGPUStringView message, void * userdata) {
  UNUSED(userdata)
  char *level_str;
  switch (level) {
  case WGPULoggingType_Error:
    level_str = "error";
    break;
  case WGPULoggingType_Warning:
    level_str = "warn";
    break;
  case WGPULoggingType_Info:
    level_str = "info";
    break;
  case WGPULoggingType_Verbose:
    level_str = "verbose";
    break;
  default:
    level_str = "unknown_level";
  }
  fprintf(stderr, "[gpu] [%s] %s\n", level_str, message.data);
}

static void handle_request_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter, struct WGPUStringView message, void * userdata) {
  UNUSED(status)
  UNUSED(message)
  *(WGPUAdapter *)userdata = adapter;
}

static void handle_request_device(WGPURequestDeviceStatus status, WGPUDevice device, struct WGPUStringView message, void * userdata) {
  UNUSED(status)
  UNUSED(message)
  *(WGPUDevice *)userdata = device;
}


typedef struct {
    WGPUBuffer* R_staging_buffer;
    size_t* R_size;
    bool* mappingComplete;
    float const ** buf;
} QCallbackContext;

// Callback function that will be called when mapping is complete
void buffer_map_callback(WGPUMapAsyncStatus status, WGPUStringView message, void *userdata1, void *userdata2) {
    QCallbackContext* cc = (QCallbackContext*)userdata1;

    if (status == WGPUMapAsyncStatus_Success) {
        void const *buf = wgpuBufferGetConstMappedRange(*cc->R_staging_buffer, 0, *cc->R_size);
        assert(cc->R_staging_buffer != NULL);
        assert(cc->R_size != NULL);
        assert(buf != NULL);

        *cc->buf = (float const *)buf;
    } else {
        fprintf(stderr, "Buffer mapping failed with status: %d\n", status);
        exit(status);
    }

    // Signal that mapping is complete
    *cc->mappingComplete = true;
}

void work_done_callback(WGPUQueueWorkDoneStatus status, void *userdata1, void *userdata2) {
    UNUSED(userdata2)

    QCallbackContext* cc = (QCallbackContext*)userdata1;
    if (status == WGPUQueueWorkDoneStatus_Success) {
        WGPUBufferMapCallbackInfo mapCallbackInfo = {
                        .mode = WGPUCallbackMode_AllowSpontaneous,
                        .callback = buffer_map_callback,
                        .userdata1 = cc};

        wgpuBufferMapAsync(*cc->R_staging_buffer, WGPUMapMode_Read, 0, *cc->R_size, mapCallbackInfo);
    } else {
        fprintf(stderr, "Work done failed with status: %d\n", status);
        exit(4);
    }
}

static void shader_compilation_callback(WGPUCompilationInfoRequestStatus status, struct WGPUCompilationInfo const* compilationInfo, void* userdata1, void* userdata2)
{
    if (status != WGPUCompilationInfoRequestStatus_Success)
    {
        fprintf(stderr, "Shader compilation failed with status: %d\n", status);
        for(int i = 0; i < compilationInfo->messageCount; i++)
        {
            const WGPUCompilationMessage* message = &compilationInfo->messages[i];
            char *level;
            switch(message->type)
            {
                case WGPUCompilationMessageType_Warning:
                    level = "warning";
                    break;

                case WGPUCompilationMessageType_Info:
                    level = "info";
                    break;

                default:
                case WGPUCompilationMessageType_Error:
                    level = "error";
                    break;
            }
            fprintf(stderr, "[%s] %llu: shader compilation error: %s\n", level, message->lineNum, message->message.data);
        }
    }

    bool *compileComplete = (bool *)userdata1;
    *compileComplete = true;
}

WGPUInstance instance = NULL;
WGPUDevice device = NULL;
WGPUQueue queue = NULL;

WGPUBuffer tensor_lookup[8192];
int64_t tensor_lookup_idx = 0;

Scratch scratch_lookup[8192];
int64_t scratch_lookup_idx = 0;

WGPUShaderModule shader_lookup[1024];
int64_t shader_lookup_idx = 0;

WGPUComputePipeline shader_pipeline_lookup[1024];
int64_t shader_pipeline_lookup_idx = 0;

WGPUBindGroupLayout bind_group_layout;

static volatile bool has_errored = false;

static void on_device_error(WGPUDevice const * device, WGPUErrorType type, struct WGPUStringView message, void* userdata1, void* userdata2)
{
    fprintf(stderr, "Device error: %.*s\n", (int)message.length, message.data);
    has_errored = true;
}

void static on_lost_error(WGPUDevice const * device, WGPUDeviceLostReason reason, struct WGPUStringView message, void* userdata1, void* userdata2)
{
    fprintf(stderr, "Device lost: %.*s\n", (int)message.length, message.data);
    exit(8);
}

void init_gpu(int64_t *results) {

    WGPUDawnTogglesDescriptor toggles = {};
    toggles.chain.sType = WGPUSType_DawnTogglesDescriptor;
    toggles.chain.next = NULL;

#if defined(_WIN32)
    toggles.enabledToggleCount = 1;
    toggles.enabledToggles = (const char* const[]){"use_dxc", "skip_validation"};
#else
    toggles.enabledToggleCount = 9;
    toggles.enabledToggles = (const char* const[]){"allow_unsafe_apis", "timestamp_quantization", "skip_validation", "disable_robustness", "disallow_spirv", "disable_lazy_clear_for_mapped_at_creation_buffer", "disable_workgroup_init", "use_tint_ir", "use_dxc"};
#endif
    toggles.disabledToggleCount = 0;

    WGPUInstanceDescriptor instanceDesc = {
        .nextInChain = (WGPUChainedStruct*) &toggles,
    };

    instance = wgpuCreateInstance((const WGPUInstanceDescriptor*) &instanceDesc);
    assert(instance != NULL);

    // Configure adapter request options for high performance
    WGPURequestAdapterOptions adapterOpts = {
        .nextInChain = (WGPUChainedStruct*) &toggles,
        .featureLevel = WGPUFeatureLevel_Core,
        .powerPreference = WGPUPowerPreference_HighPerformance,  // Request high-performance GPU
        .forceFallbackAdapter = false                            // Don't fall back to software
    };

    WGPUAdapter adapter;

    WGPURequestAdapterCallbackInfo adapterCallbackInfo = {
          .nextInChain = NULL,
          .mode = WGPUCallbackMode_AllowSpontaneous,
          .callback = (WGPURequestAdapterCallback)handle_request_adapter,
          .userdata1 = &adapter
    };

    wgpuInstanceRequestAdapter(instance, &adapterOpts, adapterCallbackInfo);
    wgpuInstanceProcessEvents(instance);
    if (adapter == NULL) {
        results[0] = -1;
        results[1] = -1;
        results[2] = -1;
        return;
    }

    // Adapter limits (memory-related insights)
    struct WGPULimits limits = {};
    struct WGPULimits requiredLimits = {};
    if (wgpuAdapterGetLimits(adapter, &limits)) {
       results[0] = limits.maxBufferSize;
       results[1] = limits.maxBindGroups;
       results[2] = sizeof(Params);

       //MAX out the limits because why not?
       requiredLimits = limits;

    } else {
       results[0] = -1;
       results[1] = -1;
       results[2] = -1;
       return;
    }

    WGPUDeviceDescriptor deviceDesc = {
        .nextInChain = (WGPUChainedStruct*) &toggles,
        .uncapturedErrorCallbackInfo.callback = &on_device_error,
        .deviceLostCallbackInfo.callback = &on_lost_error,
        .deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous,
        .requiredFeatureCount = 1, //Disable at 0
        .requiredFeatures = (const WGPUFeatureName[]) {WGPUFeatureName_DawnNative},
        .requiredLimits = &requiredLimits,
        .label       = "default device",
    };

    WGPURequestDeviceCallbackInfo deviceCallbackInfo = {
          .mode = WGPUCallbackMode_AllowSpontaneous,
          .callback = (WGPURequestDeviceCallback)handle_request_device,
          .userdata1 = &device
    };

    wgpuAdapterRequestDevice(adapter, &deviceDesc, deviceCallbackInfo);
    wgpuInstanceProcessEvents(instance);
    assert(device != NULL);

    // Set logging callback
    WGPULoggingCallbackInfo loggingCallbackInfo = {
          .callback = (WGPULoggingCallback)log_callback,
          .userdata1 = NULL
    };

    wgpuDeviceSetLoggingCallback(device, loggingCallbackInfo);
    wgpuInstanceProcessEvents(instance);


    queue = wgpuDeviceGetQueue(device);
    assert(queue != NULL);


    tensor_lookup_idx = 0;
    scratch_lookup_idx = 0;
    shader_lookup_idx = 0;
    shader_pipeline_lookup_idx = 0;


    // Create bind group layout for pipeline creation
    WGPUBindGroupLayoutEntry layout_entries[6] = {
                    { .binding = 0, .visibility = WGPUShaderStage_Compute, .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage }},
                    { .binding = 1, .visibility = WGPUShaderStage_Compute, .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage }},
                    { .binding = 2, .visibility = WGPUShaderStage_Compute, .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage }},
                    { .binding = 3, .visibility = WGPUShaderStage_Compute, .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage }},
                    { .binding = 4, .visibility = WGPUShaderStage_Compute, .buffer = { .type = WGPUBufferBindingType_Storage }},
                    { .binding = 5, .visibility = WGPUShaderStage_Compute, .buffer = { .type = WGPUBufferBindingType_Uniform }},
               };

    bind_group_layout = wgpuDeviceCreateBindGroupLayout(device, &(WGPUBindGroupLayoutDescriptor){
            .entryCount = 6,
            .entries = layout_entries,
    });
}

WGPUComputePipeline init_pipeline(WGPUShaderModule shader_module) {
    assert(shader_module != NULL);
    assert(device != NULL);

    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
            .bindGroupLayoutCount = 1,
            .bindGroupLayouts = &bind_group_layout, //cached
        };

    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &pipeline_layout_desc);

    WGPUComputeState compute_state = {};
    compute_state.module = shader_module;
    compute_state.entryPoint.data = "main";
    compute_state.entryPoint.length = 4;

    // Create compute pipeline with the pipeline layout
    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(device, &(WGPUComputePipelineDescriptor){
            .layout = pipeline_layout,  // Include the pipeline layout
            .compute = compute_state,
    });

    return pipeline;
}

void on_error_scope(WGPUPopErrorScopeStatus status,  WGPUErrorType type,  struct WGPUStringView message,  void *userdata1, void *userdata2) {
    bool *oom_flag = (bool*)userdata1;
    if (type == WGPUErrorType_OutOfMemory) {
        // Mark that this allocation failed due to OOM
        *oom_flag = true;
    }
}


WGPUBuffer create_buffer(WGPUDevice device, void* data, size_t size, WGPUBufferUsage usage) {
    bool sawOOM = false;

    wgpuDevicePushErrorScope(device, WGPUErrorFilter_OutOfMemory);

    WGPUBufferDescriptor desc = {
        .usage = usage,
        .size = size,
        .mappedAtCreation = true,
        .label = "weights"
    };
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &desc);
    WGPUPopErrorScopeCallbackInfo popCallbackInfo = {
          .mode = WGPUCallbackMode_AllowSpontaneous,
          .callback = on_error_scope,
          .userdata1 = &sawOOM
    };
    wgpuDevicePopErrorScope(device, popCallbackInfo);
    wgpuInstanceProcessEvents(instance);
    if (sawOOM) {
        if (buffer != NULL) {
            wgpuBufferDestroy(buffer);
        }
        fprintf(stderr, "Failed to allocate buffer of size %zu\n", size);
        return NULL;
    }

    void* mappedMemory = wgpuBufferGetMappedRange(buffer, 0, size);
    memcpy(mappedMemory, data, size);
    wgpuBufferUnmap(buffer);

    return buffer;
}

WGPUBuffer create_working_buffer(WGPUDevice device, char const *label, size_t size, WGPUBufferUsage usage) {
    WGPUBufferDescriptor desc = {
        .usage = usage,
        .size = size,
        .mappedAtCreation = false,
        .label = label
    };
    return wgpuDeviceCreateBuffer(device, &desc);
}

int64_t register_tensor(const char *data, int size) {
    WGPUBuffer buffer = create_buffer(device, (void *)data, size, WGPUBufferUsage_Storage);

    if (buffer == NULL) {
        return -1;
    }

    tensor_lookup[tensor_lookup_idx] = buffer;
    int64_t id = tensor_lookup_idx;
    tensor_lookup_idx++;

    return id;
}

int64_t register_scratch_buffers( int params_size, int input_size, int result_size) {
    WGPUBuffer input_buffer = create_working_buffer(device, "input", input_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer input2_buffer = create_working_buffer(device, "input2", (input_size/Q8_BLOCK_SIZE) * sizeof(float), WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer params_buffer = create_working_buffer(device, "params", params_size, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst );
    WGPUBuffer result_buffer = create_working_buffer(device, "result", result_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc );
    WGPUBuffer result_staging_buffer = create_working_buffer(device, "staging", result_size, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    WGPUBuffer empty_buffer = create_working_buffer(device, "empty", 8, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);

    Scratch s = {input_buffer, input2_buffer, params_buffer, result_buffer, result_staging_buffer, empty_buffer};
    scratch_lookup[scratch_lookup_idx] = s;
    int64_t id = scratch_lookup_idx;
    scratch_lookup_idx++;

    return id;
}

WGPUShaderModule create_shader_module(WGPUDevice device, const char* shader_code) {

    // Explicitly initialize the chained WGSL descriptor.
    WGPUShaderModuleWGSLDescriptor wgslDesc;
    wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL; // Expected to be non-zero (e.g. 2)
    wgslDesc.chain.next  = NULL;
    wgslDesc.code.data   = shader_code;
    wgslDesc.code.length = strlen(shader_code);

    WGPUShaderModuleDescriptor shaderDesc = {
        .label       = "shader",
        .nextInChain = (WGPUChainedStruct *)&wgslDesc,
    };

    WGPUShaderModule shader_module = wgpuDeviceCreateShaderModule(device, &shaderDesc);

    bool compileComplete = false;
    WGPUCompilationInfoCallbackInfo compilationCallbackInfo = {
          .mode = WGPUCallbackMode_AllowSpontaneous,
          .callback = shader_compilation_callback,
          .userdata1 = &compileComplete
    };

    wgpuShaderModuleGetCompilationInfo(shader_module, compilationCallbackInfo);

    wgpuInstanceProcessEvents(instance);
    while (!compileComplete) {
       wgpuInstanceProcessEvents(instance);
    }

    wgpuInstanceProcessEvents(instance);
    wgpuDeviceTick(device);

    assert(shader_module != NULL);
    return shader_module;
}

int64_t register_shader(const char *data, int size) {

    assert(shader_lookup_idx < 1024);

    WGPUShaderModule shader = create_shader_module(device, data);

    if (has_errored) {
        return -1;
    }

    shader_lookup[shader_lookup_idx] = shader;
    int64_t id = shader_lookup_idx;

    shader_pipeline_lookup[shader_pipeline_lookup_idx] = init_pipeline(shader);

    if (has_errored) {
        return -1;
    }

    shader_lookup_idx++;
    shader_pipeline_lookup_idx++;

    assert(shader_lookup_idx == shader_pipeline_lookup_idx);

    return id;
}

void gpu_gemm(int64_t scratch_id, int64_t shader, const void *a, const void *a2, int aoffset, int alimit, int64_t bid, int64_t bid2, int boffset, int blimit, float *r, int roffset, int rlimit, int m, int n0, int n, int k, int lda, int ldb, int ldc, int m1_optimized) {

    WGPUShaderModule shader_module = shader_lookup[shader];
    assert(shader_module);

    Scratch s = scratch_lookup[scratch_id];

    size_t A_size = alimit - aoffset;
    WGPUBuffer A_buffer = s.input_buffer;
    assert(A_buffer);

    WGPUBuffer A2_buffer = a2 == NULL ? s.empty_buffer : s.input2_buffer;
    // For Q8 the offsets are not in bytes, but in 4 byte chunks
    // So to get to the offset & size of scales array we multiply by 4 to get the number of floats quantized
    size_t A2_size = a2 == NULL ? 8 : (A_size * 4)/Q8_BLOCK_SIZE; // use bind size 8 in case of zero due to min bind size in windows
    size_t A2_offset = a2 == NULL ? 0 : (aoffset * 4)/Q8_BLOCK_SIZE;

    size_t B_size = blimit - boffset;
    WGPUBuffer B_buffer = tensor_lookup[bid];
    assert(B_buffer);

    WGPUBuffer B2_buffer = bid2 == -1 ? s.empty_buffer : tensor_lookup[bid2];
    // For Q4 the offsets are not in bytes, but in 4 byte chunks
    // So to get to the offset & size of scales array we double to get to bytes (Q4 -> u8) and multiply by 4 to the number of floats quantized
    size_t B2_size = bid2 == -1 ? 8 : (B_size * 2 * 4)/Q4_BLOCK_SIZE; // use bind size 8 in case of zero due to min bind size in windows
    size_t B2_offset = bid2 == -1 ? 0 : (boffset * 2 * 4)/Q4_BLOCK_SIZE;

    size_t R_size = rlimit;
    WGPUBuffer R_buffer = s.result_buffer;
    WGPUBuffer R_staging_buffer = s.result_staging_buffer;
    assert(R_buffer);
    assert(R_staging_buffer);

    // Create uniform buffer for parameters
    Params params = {m, n + n0, k, lda, ldb, ldc};
    size_t params_size = sizeof(Params);

    WGPUBuffer params_buffer = s.params_buffer;
    assert(params_buffer);

    // Copy data to GPU (since A is a float array, we need to divide the offset by 4)
    wgpuQueueWriteBuffer(queue, A_buffer, 0, a + aoffset, A_size);
    if (a2 != NULL) {
        wgpuQueueWriteBuffer(queue, A2_buffer, 0, a2 + A2_offset, A2_size);
    }
    wgpuQueueWriteBuffer(queue, params_buffer, 0, &params, params_size);

    // Create bind group
    WGPUBindGroupEntry bind_group_entries[6] = {
        { .binding = 0, .buffer = A_buffer, .offset = 0, .size = A_size },
        { .binding = 1, .buffer = A2_buffer, .offset = 0, .size = A2_size },
        { .binding = 2, .buffer = B_buffer, .offset = boffset, .size = B_size },
        { .binding = 3, .buffer = B2_buffer, .offset = B2_offset, .size = B2_size },
        { .binding = 4, .buffer = R_buffer, .offset = 0, .size = R_size },
        { .binding = 5, .buffer = params_buffer, .offset = 0, .size = params_size },
    };
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
        .layout = bind_group_layout,
        .entryCount = 6,
        .entries = bind_group_entries,
    });

    WGPUComputePipeline pipeline = shader_pipeline_lookup[shader];
    assert(pipeline);

    // Execute the compute shader
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &(const WGPUCommandEncoderDescriptor){
                                                                            .label = "command_encoder",
                                                                        });
    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(encoder, &(const WGPUComputePassDescriptor){
                                                                                         .label = "compute_pass",
                                                                                       });
    wgpuComputePassEncoderSetPipeline(compute_pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, bind_group, 0, NULL);

     // Calculate workgroup counts (adjust based on your shader's workgroup size)
    uint32_t workgroup_count_x = (n + RN - 1) / RN;
    uint32_t workgroup_count_y = (m + RM - 1) / RM;

    if (params.m == 1 && m1_optimized > 0) {
        // Bind the M=1 optimized pipeline
        workgroup_count_x = (n + RN_M1 - 1) / RN_M1;
        workgroup_count_y = 1;
    }

    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, workgroup_count_x, workgroup_count_y, 1);
    wgpuComputePassEncoderEnd(compute_pass);
    wgpuComputePassEncoderRelease(compute_pass);

    wgpuCommandEncoderCopyBufferToBuffer(encoder, R_buffer, 0, R_staging_buffer, 0, R_size);

    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(encoder, &(const WGPUCommandBufferDescriptor){
                                                                                 .label = "command_buffer",
                                                                         });

    // Submit the command buffer
    wgpuQueueSubmit(queue, 1, &command_buffer);

    // Wait for work to complete
    bool mappingComplete = false;
    float const *buf = NULL;
    QCallbackContext context = {.mappingComplete = &mappingComplete, .R_size = &R_size, .R_staging_buffer = &R_staging_buffer, .buf = &buf};

    WGPUQueueWorkDoneCallbackInfo workDoneCallbackInfo = {
          .mode = WGPUCallbackMode_AllowSpontaneous,
          .callback = work_done_callback,
          .userdata1 = &context
    };

    wgpuQueueOnSubmittedWorkDone(queue, workDoneCallbackInfo);


    //Wait for response
    while (!mappingComplete) {
       wgpuInstanceProcessEvents(instance);
    }

    // Copy the result back to the host
    for (int rm = 0; rm < m; ++rm) {
        for (int rn = n0, rn2 = 0; rn < params.n; ++rn, ++rn2) {
            int idx = (rm * ldc) + rn - roffset;
            int idx2 = (rm * ldc) + (n0 + rn2);

            r[idx] = buf[idx2];
        }
    }

    wgpuBufferUnmap(R_staging_buffer);
    wgpuCommandBufferRelease(command_buffer);
    wgpuCommandEncoderRelease(encoder); // release encoder after it's finished

    wgpuBindGroupRelease(bind_group);
}


