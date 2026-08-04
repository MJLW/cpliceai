#include <stdio.h>
#include <klib/kstring.h>
#include <stdlib.h>
#include <string.h>

#include "predict.h"
#include "logging/log.h"
#include "utils.h"

void deallocator(void *data, size_t a, void *b) {
    free(data);
}

void noop_deallocator(void *data, size_t a, void *b) {}

// Check the status and print an error message if any
int check_status(TF_Status *status, const char *msg) {
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error: %s: %s\n", msg, TF_Message(status));
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

static inline Model load_model(const char *path) {
    TF_Status *status = TF_NewStatus();
    TF_Graph *graph = TF_NewGraph();
    TF_SessionOptions *sess_opts = TF_NewSessionOptions();
    TF_Buffer *run_opts = NULL;

    const char *tags = SPLICEAI_TAGS;
    TF_Session *session = TF_LoadSessionFromSavedModel(sess_opts, run_opts, path, &tags, 1, graph, NULL, status);
    check_status(status, "Loading model");

    return (Model) { status, graph, sess_opts, run_opts, session };
}

Model *load_models(const char *models_dir) {
    Model *models = malloc(NUM_SPLICEAI_MODELS * sizeof(Model));
    if (models == NULL) {
        log_fatal("Failed to allocate %zu bytes for models", NUM_SPLICEAI_MODELS * sizeof(Model));
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < NUM_SPLICEAI_MODELS; i++) {
        kstring_t model_path = {0};
        kputs(models_dir, &model_path);
        kputc('/', &model_path);
        kputs(SPLICEAI_MODEL_PREFIX, &model_path);
        kputl(i+1, &model_path);
        models[i] = load_model(model_path.s);
        free(model_path.s);
    }
    return models;
}

void destroy_models(Model *models) {
    for (int i = 0; i < NUM_SPLICEAI_MODELS; i++) {
        TF_DeleteGraph(models[i].graph);
        TF_DeleteSessionOptions(models[i].sess_opts);
        TF_DeleteBuffer(models[i].run_opts);
        TF_CloseSession(models[i].session, models[i].status);
        TF_DeleteSession(models[i].session, models[i].status);
        TF_DeleteStatus(models[i].status);
    }
    free(models);
}

int predict(Model *models, int data_size, int num_data, float *data, int *num_out, float *out[]) {
    // Define the input dimensions
    int64_t input_dims[] = {num_data, data_size / ENCODING_SIZE, 4};

    TF_Tensor *input_tensor = TF_NewTensor(TF_FLOAT, input_dims, 3, data, num_data * data_size * sizeof(float), &noop_deallocator, 0);
    if (!input_tensor) {
        log_error("Failed to create input tensor");
        return EXIT_FAILURE;
    }

    // int chunk_len = ((data_size / ENCODING_SIZE) - CONTEXT_SIZE) * NUM_SCORES;
    // float *chunked_outputs[num_data];
    // for (int i = 0; i < num_data; i++) chunked_outputs[i] = out + i * chunk_len;

    int output_len = ((data_size / ENCODING_SIZE) - CONTEXT_SIZE) * NUM_SCORES;
    float *outputs = calloc(output_len, sizeof(float));

    for (int i = 0; i < NUM_SPLICEAI_MODELS; i++) {
        Model model = models[i];
        // Find input and output operations by name
        TF_Operation* input_op = TF_GraphOperationByName(model.graph, "serving_default_input_1");
        if (input_op == NULL) {
            log_error("Failed to find input operation");
            return EXIT_FAILURE;
        }

        TF_Operation* output_op = TF_GraphOperationByName(model.graph, "StatefulPartitionedCall");
        if (output_op == NULL) {
            log_error("Failed to find output operation");
            return EXIT_FAILURE;
        }

        // Prepare the output tensor array
        TF_Tensor *output_tensor;

        // Prepare input/output operations and tensors
        TF_Output input_opout = {input_op, 0};
        TF_Output output_opout = {output_op, 0};

        // Run the session
        TF_SessionRun(model.session, model.run_opts,
                      &input_opout, &input_tensor, 1, // Input tensors and count
                      &output_opout, &output_tensor, 1, // Output tensors and count
                      NULL, 0, // Target operations, target operations count
                      NULL, // Run metadata
                      model.status);
        // check_status(model.status, "Running model");
        if (TF_GetCode(model.status) != TF_OK) {
            log_error("Error running the model: %s", TF_Message(model.status));
            return EXIT_FAILURE;
        }

        // Process the output data
        const float* output_data = (float*)TF_TensorData(output_tensor);
        for (int j = 0; j < output_len; j++) {
            outputs[j] += (float) output_data[j];
        }

        TF_DeleteTensor(output_tensor);
    }
    TF_DeleteTensor(input_tensor);

    for (int i = 0; i < output_len; i++) outputs[i] /= NUM_SPLICEAI_MODELS;

    *num_out = output_len;
    *out = outputs;

    return EXIT_SUCCESS;
}

int predict_padded_sequence(Model *models, const char *seq, int seq_len, char strand, float **predictions, int *num_predictions) {
    float *encoding = malloc(seq_len * ENCODING_SIZE * sizeof(float));
    if (encoding == NULL) {
        log_fatal("Failed to allocate %zu bytes for sequence encoding", seq_len * ENCODING_SIZE * sizeof(float));
        exit(EXIT_FAILURE);
    }
    memset(encoding, 0, seq_len * ENCODING_SIZE * sizeof(float));
    int encoding_len = one_hot_encode(seq, seq_len, encoding);

    if (strand == '-') reverse_encoding(encoding, encoding_len);

    int ret = predict(models, encoding_len, 1, encoding, num_predictions, predictions);
    free(encoding);
    if (ret != EXIT_SUCCESS) return ret;

    if (strand == '-') reverse_prediction(*predictions, *num_predictions, NUM_SCORES);

    return EXIT_SUCCESS;
}

