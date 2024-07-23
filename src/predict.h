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


extern int check_status(TF_Status* status, const char* msg);

extern Model *load_models(const char *models_dir);

extern void destroy_models(Model *models);

extern int predict(Model *models, int data_size, int num_data, float **data, int *num_out, float *out[]);

