#ifndef PREDICT_H
#define PREDICT_H

#include <tensorflow/c/c_api.h>
#include <tensorflow/c/tf_buffer.h>
#include <tensorflow/c/tf_datatype.h>
#include <tensorflow/c/tf_status.h>
#include <tensorflow/c/tf_tensor.h>


#define SPLICEAI_MODEL_PREFIX "spliceai"
#define NUM_SPLICEAI_MODELS 5
#define SPLICEAI_TAGS "serve"

#define CONTEXT_SIZE 10000
#define BOUNDARY_SIZE 5000

typedef struct {
    TF_Status *status;
    TF_Graph *graph;
    TF_SessionOptions *sess_opts;
    TF_Buffer *run_opts;
    TF_Session *session;
} Model;


int check_status(TF_Status *status, const char *msg);

Model *load_models(const char *models_dir);

void destroy_models(Model *models);

int predict(Model *models, int data_size, int num_data, float *data, int *num_out, float *out[]);

/*
 * predict_padded_sequence - Run the shared one-hot-encode -> predict -> reverse-if-negative-strand
 * pipeline over a single already-padded sequence.
 *
 * Parameters:
 *   models          - loaded SpliceAI models.
 *   seq             - padded sequence to encode and predict over.
 *   seq_len         - length of seq.
 *   strand          - '+' or '-'; predictions are reversed to match the padded sequence's orientation when '-'.
 *   predictions     - out-param, set to a newly allocated predictions array (caller frees).
 *   num_predictions - out-param, set to the length of *predictions.
 *
 * Returns EXIT_SUCCESS on success, EXIT_FAILURE if prediction fails.
 */
int predict_padded_sequence(Model *models, const char *seq, int seq_len, char strand, float **predictions, int *num_predictions);

#endif

