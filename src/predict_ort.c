#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <klib/kstring.h>
#include <onnxruntime_c_api.h>

#include "predict.h"
#include "logging/log.h"
#include "utils.h"

struct Model {
    const OrtApi *api;
    OrtEnv *env;
    OrtMemoryInfo *mem_info;
    OrtAllocator *allocator;
    OrtSession *session[NUM_SPLICEAI_MODELS];
    char *input_name[NUM_SPLICEAI_MODELS];
    char *output_name[NUM_SPLICEAI_MODELS];
    int max_chunk_len;
};

typedef enum { ORT_EP_AUTO, ORT_EP_CUDA, ORT_EP_CPU } OrtEpMode;

// Logs and releases an OrtStatus*. Returns EXIT_SUCCESS if status was NULL (no error).
static int ort_check(const OrtApi *api, OrtStatus *status, const char *what) {
    if (status == NULL) return EXIT_SUCCESS;
    log_error("%s: %s", what, api->GetErrorMessage(status));
    api->ReleaseStatus(status);
    return EXIT_FAILURE;
}

static OrtEpMode get_ep_mode(void) {
    const char *v = getenv("CPLICEAI_ORT_EP");
    if (v == NULL || v[0] == '\0' || strcmp(v, "auto") == 0) return ORT_EP_AUTO;
    if (strcmp(v, "cuda") == 0) return ORT_EP_CUDA;
    if (strcmp(v, "cpu") == 0) return ORT_EP_CPU;
    log_warn("Unrecognized CPLICEAI_ORT_EP=%s, defaulting to auto", v);
    return ORT_EP_AUTO;
}

static OrtLoggingLevel get_log_severity(void) {
    const char *v = getenv("CPLICEAI_ORT_LOG_SEVERITY");
    int level = v ? atoi(v) : ORT_LOGGING_LEVEL_WARNING;
    if (level < ORT_LOGGING_LEVEL_VERBOSE) level = ORT_LOGGING_LEVEL_VERBOSE;
    if (level > ORT_LOGGING_LEVEL_FATAL) level = ORT_LOGGING_LEVEL_FATAL;
    return (OrtLoggingLevel) level;
}

// Default of 250000 is the largest input length confirmed to run reliably on the CUDA execution
// provider without exhausting GPU memory on real hardware; sequences longer than this are split
// into overlapping windows by predict() (see run_models_sum() / the chunking branch below).
#define DEFAULT_MAX_CHUNK_LEN 250000

static int get_max_chunk_len(void) {
    const char *v = getenv("CPLICEAI_ORT_MAX_CHUNK_LEN");
    int len = (v && v[0] != '\0') ? atoi(v) : DEFAULT_MAX_CHUNK_LEN;
    if (len <= CONTEXT_SIZE) {
        log_fatal("CPLICEAI_ORT_MAX_CHUNK_LEN=%d must be greater than CONTEXT_SIZE (%d)", len, CONTEXT_SIZE);
        exit(EXIT_FAILURE);
    }
    return len;
}

// Logs status at WARN (not ERROR) and releases it. Used for the CUDA-availability probe below,
// where failure is an expected, routinely-hit path in `auto` mode (e.g. every run against a
// CPU-only ORT build) rather than a genuine error -- ort_check()'s ERROR-level logging would be
// misleading there. Returns EXIT_SUCCESS if status was NULL (no error).
static int ort_probe(const OrtApi *api, OrtStatus *status, const char *what) {
    if (status == NULL) return EXIT_SUCCESS;
    log_warn("%s: %s", what, api->GetErrorMessage(status));
    api->ReleaseStatus(status);
    return EXIT_FAILURE;
}

// Appends the CUDA execution provider via the V2 provider-options API. These entry points are
// OrtApi struct members present in every ORT build (CPU or GPU) -- unlike the older
// OrtSessionOptionsAppendExecutionProvider_CUDA() free-function symbol, which only exists in GPU
// builds and would force a compile/link-time fork. Here the CPU/GPU difference is a runtime
// status check: on a CPU-only ORT build this fails immediately (even at CreateCUDAProviderOptions
// -- the CPU-only build has no CUDA provider implementation registered at all), which the caller
// treats as "unavailable" rather than a hard error unless CPLICEAI_ORT_EP=cuda was explicitly
// requested.
static int append_cuda_ep(const OrtApi *api, OrtSessionOptions *session_options) {
    OrtCUDAProviderOptionsV2 *cuda_options = NULL;
    if (ort_probe(api, api->CreateCUDAProviderOptions(&cuda_options), "CUDA execution provider unavailable") != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    char device_id_str[16];
    const char *device_id_env = getenv("CPLICEAI_ORT_DEVICE_ID");
    snprintf(device_id_str, sizeof(device_id_str), "%d", device_id_env ? atoi(device_id_env) : 0);

    const char *algo_search = getenv("CPLICEAI_ORT_CUDNN_CONV_ALGO_SEARCH");
    if (algo_search == NULL || algo_search[0] == '\0') algo_search = "HEURISTIC";

    // HEURISTIC (not ORT's default EXHAUSTIVE) is deliberate: every call site here feeds a
    // different sequence length with no batching, so exhaustive per-shape cuDNN autotuning would
    // re-benchmark every convolution layer on every single call.
    const char *keys[] = {"device_id", "cudnn_conv_algo_search", "arena_extend_strategy"};
    const char *values[] = {device_id_str, algo_search, "kSameAsRequested"};

    if (ort_probe(api, api->UpdateCUDAProviderOptions(cuda_options, keys, values, 3), "CUDA execution provider unavailable") != EXIT_SUCCESS) {
        api->ReleaseCUDAProviderOptions(cuda_options);
        return EXIT_FAILURE;
    }

    OrtStatus *status = api->SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, cuda_options);
    api->ReleaseCUDAProviderOptions(cuda_options);
    if (ort_probe(api, status, "CUDA execution provider unavailable") != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

static int configure_session_options(const OrtApi *api, OrtSessionOptions *so, OrtEpMode ep_mode, int *cuda_engaged) {
    if (ort_check(api, api->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL), "SetSessionGraphOptimizationLevel") != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    const char *threads_env = getenv("CPLICEAI_ORT_INTRA_OP_THREADS");
    int threads = threads_env ? atoi(threads_env) : 0;
    if (threads > 0) {
        if (ort_check(api, api->SetIntraOpNumThreads(so, threads), "SetIntraOpNumThreads") != EXIT_SUCCESS) {
            return EXIT_FAILURE;
        }
        if (threads == 1) {
            // Pinning intra-op threads alone does not make output bit-reproducible: ORT's
            // default ORT_PARALLEL execution mode still schedules independent graph branches
            // (this model has several, from its residual/skip connections) on separate
            // inter-op threads, and floating-point summation order across those threads isn't
            // fixed run-to-run. Requesting exactly 1 thread is treated as "I want fully
            // deterministic single-threaded execution" (used by the test suite) and also pins
            // inter-op threads + sequential execution mode; confirmed empirically to produce
            // bit-identical output across repeated runs only when all three are set together.
            if (ort_check(api, api->SetInterOpNumThreads(so, 1), "SetInterOpNumThreads") != EXIT_SUCCESS) {
                return EXIT_FAILURE;
            }
            if (ort_check(api, api->SetSessionExecutionMode(so, ORT_SEQUENTIAL), "SetSessionExecutionMode") != EXIT_SUCCESS) {
                return EXIT_FAILURE;
            }
        }
    }

    if (ep_mode != ORT_EP_CPU) {
        if (append_cuda_ep(api, so) == EXIT_SUCCESS) {
            *cuda_engaged = 1;
        } else if (ep_mode == ORT_EP_CUDA) {
            log_error("CPLICEAI_ORT_EP=cuda was requested but the CUDA execution provider could not be enabled");
            return EXIT_FAILURE;
        } else {
            log_warn("Falling back to CPU execution provider");
        }
    }

    return EXIT_SUCCESS;
}

static void log_active_providers(const OrtApi *api, int cuda_engaged) {
    char **providers = NULL;
    int n_providers = 0;
    if (ort_check(api, api->GetAvailableProviders(&providers, &n_providers), "GetAvailableProviders") != EXIT_SUCCESS) {
        return;
    }

    kstring_t list = {0};
    for (int i = 0; i < n_providers; i++) {
        if (i > 0) kputs(", ", &list);
        kputs(providers[i], &list);
    }
    log_info("onnxruntime providers available: [%s] | active: %s",
             list.s ? list.s : "", cuda_engaged ? "CUDAExecutionProvider" : "CPUExecutionProvider");
    free(list.s);

    ort_check(api, api->ReleaseAvailableProviders(providers, n_providers), "ReleaseAvailableProviders");
}

Model *load_models(const char *models_dir) {
    Model *m = calloc(1, sizeof(Model));
    if (m == NULL) {
        log_fatal("Failed to allocate %zu bytes for models", sizeof(Model));
        exit(EXIT_FAILURE);
    }

    m->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (m->api == NULL) {
        log_fatal("Failed to get ONNX Runtime API for ORT_API_VERSION=%d", ORT_API_VERSION);
        exit(EXIT_FAILURE);
    }

    if (ort_check(m->api, m->api->CreateEnv(get_log_severity(), "cpliceai", &m->env), "CreateEnv") != EXIT_SUCCESS) {
        exit(EXIT_FAILURE);
    }

    if (ort_check(m->api, m->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &m->mem_info), "CreateCpuMemoryInfo") != EXIT_SUCCESS) {
        exit(EXIT_FAILURE);
    }

    if (ort_check(m->api, m->api->GetAllocatorWithDefaultOptions(&m->allocator), "GetAllocatorWithDefaultOptions") != EXIT_SUCCESS) {
        exit(EXIT_FAILURE);
    }

    OrtEpMode ep_mode = get_ep_mode();
    int cuda_engaged = 0;
    m->max_chunk_len = get_max_chunk_len();

    for (int i = 0; i < NUM_SPLICEAI_MODELS; i++) {
        kstring_t model_path = {0};
        kputs(models_dir, &model_path);
        kputc('/', &model_path);
        kputs(SPLICEAI_MODEL_PREFIX, &model_path);
        kputl(i + 1, &model_path);
        kputs(".onnx", &model_path);

        OrtSessionOptions *so = NULL;
        if (ort_check(m->api, m->api->CreateSessionOptions(&so), "CreateSessionOptions") != EXIT_SUCCESS) {
            exit(EXIT_FAILURE);
        }
        if (configure_session_options(m->api, so, ep_mode, &cuda_engaged) != EXIT_SUCCESS) {
            exit(EXIT_FAILURE);
        }

        if (ort_check(m->api, m->api->CreateSession(m->env, model_path.s, so, &m->session[i]), "CreateSession") != EXIT_SUCCESS) {
            log_error("Failed to load model: %s", model_path.s);
            exit(EXIT_FAILURE);
        }
        m->api->ReleaseSessionOptions(so);
        free(model_path.s);

        if (ort_check(m->api, m->api->SessionGetInputName(m->session[i], 0, m->allocator, &m->input_name[i]), "SessionGetInputName") != EXIT_SUCCESS) {
            exit(EXIT_FAILURE);
        }
        if (ort_check(m->api, m->api->SessionGetOutputName(m->session[i], 0, m->allocator, &m->output_name[i]), "SessionGetOutputName") != EXIT_SUCCESS) {
            exit(EXIT_FAILURE);
        }
    }

    log_active_providers(m->api, cuda_engaged);

    return m;
}

void destroy_models(Model *m) {
    for (int i = 0; i < NUM_SPLICEAI_MODELS; i++) {
        if (m->input_name[i]) ort_check(m->api, m->api->AllocatorFree(m->allocator, m->input_name[i]), "AllocatorFree");
        if (m->output_name[i]) ort_check(m->api, m->api->AllocatorFree(m->allocator, m->output_name[i]), "AllocatorFree");
        if (m->session[i]) m->api->ReleaseSession(m->session[i]);
    }
    if (m->mem_info) m->api->ReleaseMemoryInfo(m->mem_info);
    if (m->env) m->api->ReleaseEnv(m->env);
    free(m);
}

// Runs all NUM_SPLICEAI_MODELS models over a single [1, window_seq_len, 4] input window and
// accumulates (sums, does not average) their outputs into dest_sum[0 .. (window_seq_len -
// CONTEXT_SIZE) * NUM_SCORES). dest_sum must be pre-zeroed by the caller -- predict() calls this
// once per chunk when chunking, so each chunk's contribution simply adds into its own slice of
// the shared output buffer.
static int run_models_sum(Model *m, float *window_data, int window_seq_len, float *dest_sum) {
    int64_t input_dims[] = {1, window_seq_len, 4};

    OrtValue *input_tensor = NULL;
    if (ort_check(m->api, m->api->CreateTensorWithDataAsOrtValue(
            m->mem_info, window_data, (size_t) window_seq_len * ENCODING_SIZE * sizeof(float),
            input_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor),
            "CreateTensorWithDataAsOrtValue") != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    const OrtValue *inputs[] = {input_tensor};

    int window_output_len = (window_seq_len - CONTEXT_SIZE) * NUM_SCORES;

    for (int i = 0; i < NUM_SPLICEAI_MODELS; i++) {
        const char *input_names[] = {m->input_name[i]};
        const char *output_names[] = {m->output_name[i]};
        OrtValue *output_tensor = NULL;

        if (ort_check(m->api, m->api->Run(m->session[i], NULL, input_names, inputs, 1, output_names, 1, &output_tensor), "Run") != EXIT_SUCCESS) {
            m->api->ReleaseValue(input_tensor);
            return EXIT_FAILURE;
        }

        OrtTensorTypeAndShapeInfo *shape_info = NULL;
        int shape_ok = ort_check(m->api, m->api->GetTensorTypeAndShape(output_tensor, &shape_info), "GetTensorTypeAndShape") == EXIT_SUCCESS;
        size_t elem_count = 0;
        if (shape_ok) {
            shape_ok = ort_check(m->api, m->api->GetTensorShapeElementCount(shape_info, &elem_count), "GetTensorShapeElementCount") == EXIT_SUCCESS;
            m->api->ReleaseTensorTypeAndShapeInfo(shape_info);
        }
        if (!shape_ok || (int) elem_count != window_output_len) {
            log_error("Model %d: unexpected output element count %zu, expected %d", i + 1, elem_count, window_output_len);
            m->api->ReleaseValue(output_tensor);
            m->api->ReleaseValue(input_tensor);
            return EXIT_FAILURE;
        }

        float *output_data = NULL;
        if (ort_check(m->api, m->api->GetTensorMutableData(output_tensor, (void **) &output_data), "GetTensorMutableData") != EXIT_SUCCESS) {
            m->api->ReleaseValue(output_tensor);
            m->api->ReleaseValue(input_tensor);
            return EXIT_FAILURE;
        }
        for (int j = 0; j < window_output_len; j++) {
            dest_sum[j] += output_data[j];
        }

        m->api->ReleaseValue(output_tensor);
    }
    m->api->ReleaseValue(input_tensor);

    return EXIT_SUCCESS;
}

// Splits a single (num_data == 1) sequence longer than m->max_chunk_len into overlapping windows
// and stitches the results back together. Safe because the model's receptive field is bounded by
// CONTEXT_SIZE/BOUNDARY_SIZE (fully convolutional, no cross-position state) -- each output base is
// produced by exactly one window, using real flanking sequence already present in `data` (the
// caller pads only the true ends of the sequence; internal window boundaries see actual bases).
static int predict_chunked(Model *m, int seq_len, float *data, int *num_out, float *out[]) {
    int gene_len = seq_len - CONTEXT_SIZE;
    int output_len = gene_len * NUM_SCORES;
    int chunk_gene_len = m->max_chunk_len - CONTEXT_SIZE;

    float *outputs = calloc(output_len, sizeof(float));
    if (outputs == NULL) {
        log_fatal("Failed to allocate %zu bytes for outputs", (size_t) output_len * sizeof(float));
        exit(EXIT_FAILURE);
    }

    for (int start = 0; start < gene_len; start += chunk_gene_len) {
        int len = chunk_gene_len < (gene_len - start) ? chunk_gene_len : (gene_len - start);
        float *window_data = data + (size_t) start * ENCODING_SIZE;
        float *dest = outputs + (size_t) start * NUM_SCORES;

        if (run_models_sum(m, window_data, len + CONTEXT_SIZE, dest) != EXIT_SUCCESS) {
            free(outputs);
            return EXIT_FAILURE;
        }
    }

    for (int i = 0; i < output_len; i++) outputs[i] /= NUM_SPLICEAI_MODELS;

    *num_out = output_len;
    *out = outputs;

    return EXIT_SUCCESS;
}

int predict(Model *m, int data_size, int num_data, float *data, int *num_out, float *out[]) {
    int seq_len = data_size / ENCODING_SIZE;

    if (num_data == 1 && seq_len > m->max_chunk_len) {
        return predict_chunked(m, seq_len, data, num_out, out);
    }

    if (num_data == 1) {
        int output_len = (seq_len - CONTEXT_SIZE) * NUM_SCORES;
        float *outputs = calloc(output_len, sizeof(float));
        if (outputs == NULL) {
            log_fatal("Failed to allocate %zu bytes for outputs", (size_t) output_len * sizeof(float));
            exit(EXIT_FAILURE);
        }

        if (run_models_sum(m, data, seq_len, outputs) != EXIT_SUCCESS) {
            free(outputs);
            return EXIT_FAILURE;
        }

        for (int i = 0; i < output_len; i++) outputs[i] /= NUM_SPLICEAI_MODELS;

        *num_out = output_len;
        *out = outputs;

        return EXIT_SUCCESS;
    }

    // num_data != 1 is never exercised by any current call site (every caller predicts one
    // sequence at a time), kept as-is rather than folding into run_models_sum's batch-of-1
    // assumption.
    int64_t input_dims[] = {num_data, seq_len, 4};

    OrtValue *input_tensor = NULL;
    if (ort_check(m->api, m->api->CreateTensorWithDataAsOrtValue(
            m->mem_info, data, (size_t) num_data * data_size * sizeof(float),
            input_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor),
            "CreateTensorWithDataAsOrtValue") != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    const OrtValue *inputs[] = {input_tensor};

    int output_len = (seq_len - CONTEXT_SIZE) * NUM_SCORES;
    float *outputs = calloc(output_len, sizeof(float));
    if (outputs == NULL) {
        log_fatal("Failed to allocate %zu bytes for outputs", (size_t) output_len * sizeof(float));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < NUM_SPLICEAI_MODELS; i++) {
        const char *input_names[] = {m->input_name[i]};
        const char *output_names[] = {m->output_name[i]};
        OrtValue *output_tensor = NULL;

        if (ort_check(m->api, m->api->Run(m->session[i], NULL, input_names, inputs, 1, output_names, 1, &output_tensor), "Run") != EXIT_SUCCESS) {
            free(outputs);
            m->api->ReleaseValue(input_tensor);
            return EXIT_FAILURE;
        }

        OrtTensorTypeAndShapeInfo *shape_info = NULL;
        int shape_ok = ort_check(m->api, m->api->GetTensorTypeAndShape(output_tensor, &shape_info), "GetTensorTypeAndShape") == EXIT_SUCCESS;
        size_t elem_count = 0;
        if (shape_ok) {
            shape_ok = ort_check(m->api, m->api->GetTensorShapeElementCount(shape_info, &elem_count), "GetTensorShapeElementCount") == EXIT_SUCCESS;
            m->api->ReleaseTensorTypeAndShapeInfo(shape_info);
        }
        if (!shape_ok || (int) elem_count != output_len) {
            log_error("Model %d: unexpected output element count %zu, expected %d", i + 1, elem_count, output_len);
            m->api->ReleaseValue(output_tensor);
            free(outputs);
            m->api->ReleaseValue(input_tensor);
            return EXIT_FAILURE;
        }

        float *output_data = NULL;
        if (ort_check(m->api, m->api->GetTensorMutableData(output_tensor, (void **) &output_data), "GetTensorMutableData") != EXIT_SUCCESS) {
            m->api->ReleaseValue(output_tensor);
            free(outputs);
            m->api->ReleaseValue(input_tensor);
            return EXIT_FAILURE;
        }
        for (int j = 0; j < output_len; j++) {
            outputs[j] += output_data[j];
        }

        m->api->ReleaseValue(output_tensor);
    }
    m->api->ReleaseValue(input_tensor);

    for (int i = 0; i < output_len; i++) outputs[i] /= NUM_SPLICEAI_MODELS;

    *num_out = output_len;
    *out = outputs;

    return EXIT_SUCCESS;
}
