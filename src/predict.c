#include <stdio.h>
#include <klib/kstring.h>

#include "predict.h"

#define SPLICEAI_MODEL_PREFIX "spliceai"
#define NUM_SPLICEAI_MODELS 5
#define SPLICEAI_TAGS "serve"

#define CONTEXT_SIZE 10000
#define BOUNDARY_SIZE 5000

#define ENCODING_SIZE 4
#define BASE_A 'A'
#define BASE_A_ENC 0
#define BASE_C 'C'
#define BASE_C_ENC 1
#define BASE_G 'G'
#define BASE_G_ENC 2
#define BASE_T 'T'
#define BASE_T_ENC 3

void deallocator(void* data, size_t a, void* b) { }

// Check the status and print an error message if any
int check_status(TF_Status* status, const char* msg) {
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error: %s: %s\n", msg, TF_Message(status));
        return 1;
    }
    return 0;
}

static inline Model load_model(const char *path) {
    TF_Status* status = TF_NewStatus();
    TF_Graph* graph = TF_NewGraph();
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    TF_Buffer* run_opts = NULL;

    const char* tags = SPLICEAI_TAGS;
    TF_Session* session = TF_LoadSessionFromSavedModel(sess_opts, run_opts, path, &tags, 1, graph, NULL, status);
    check_status(status, "Loading model");

    return (Model) { status, graph, sess_opts, run_opts, session };
}

Model *load_models(const char *models_dir) {
    Model *models = malloc(NUM_SPLICEAI_MODELS * sizeof(Model));
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

int predict(Model *models, int data_size, int num_data, float **data, int *num_out, float *out[]) {
    // Define the input dimensions
    int64_t input_dims[] = {num_data, data_size / ENCODING_SIZE, 4};

    float *input_data = malloc(num_data * data_size * sizeof(float));
    for (int i = 0; i < num_data; i++) {
        for (int j = 0; j < data_size; j++) {
            input_data[i * data_size + j] = data[i][j];
        }
    }
    // Create the input tensor
    TF_Tensor *input_tensors[num_data];
    for (int i = 0; i < 1; i++) {
        input_tensors[i] = TF_NewTensor(TF_FLOAT, input_dims, 3, input_data, num_data * data_size * sizeof(float), &deallocator, 0);
        if (!input_tensors[i]) {
            fprintf(stderr, "Failed to create input tensor\n");
            return 1;
        }
    }

    int output_len = ((data_size / ENCODING_SIZE) - CONTEXT_SIZE)* 3;
    float *outputs = calloc(output_len, sizeof(float));

    for (int i = 0; i < NUM_SPLICEAI_MODELS; i++) {
        Model model = models[i];
        // Find input and output operations by name
        TF_Operation* input_op = TF_GraphOperationByName(model.graph, "serving_default_input_1");
        if (input_op == NULL) {
            fprintf(stderr, "Failed to find input operation\n");
            return 1;
        }

        TF_Operation* output_op = TF_GraphOperationByName(model.graph, "StatefulPartitionedCall");
        if (output_op == NULL) {
            fprintf(stderr, "Failed to find output operation\n");
            return 1;
        }

        // Prepare the output tensor array
        TF_Tensor* output_tensors[1];

        // Prepare input/output operations and tensors
        TF_Output input_opout = {input_op, 0};
        TF_Output output_opout = {output_op, 0};

        // Run the session
        TF_SessionRun(model.session, model.run_opts,
                      &input_opout, input_tensors, 1, // Input tensors and count
                      &output_opout, output_tensors, 1, // Output tensors and count
                      NULL, 0, // Target operations, target operations count
                      NULL, // Run metadata
                      model.status);
        // check_status(model.status, "Running model");
        if (TF_GetCode(model.status) != TF_OK) {
            fprintf(stderr, "Error running the model: %s\n", TF_Message(model.status));
            return 1;
        }

        // Process the output data
        float* output_data = (float*)TF_TensorData(output_tensors[0]);
        for (int k = 0; k < output_len; k++) outputs[k] += (float) output_data[k];

        TF_DeleteTensor(output_tensors[0]);
    }
    TF_DeleteTensor(input_tensors[0]);
    free(input_data);

    for (int i = 0; i < output_len; i++) outputs[i] /= NUM_SPLICEAI_MODELS;

    *num_out = output_len;
    *out = outputs;

    return 0;
}

