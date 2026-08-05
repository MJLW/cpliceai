#include <stdlib.h>
#include <string.h>

#include "predict.h"
#include "logging/log.h"
#include "utils.h"

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
