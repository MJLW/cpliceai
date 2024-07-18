#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <klib/kstring.h>
#include <klib/kvec.h>
#include <klib/khash.h>

#include <tensorflow/c/c_api.h>
#include <htslib/hts.h>
#include <htslib/vcf.h>
#include <htslib/faidx.h>
#include <htslib/tbx.h>
#include <htslib/regidx.h>
#include <tensorflow/c/tf_buffer.h>
#include <tensorflow/c/tf_datatype.h>
#include <tensorflow/c/tf_status.h>
#include <tensorflow/c/tf_tensor.h>

#include "logging/log.h"

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

#define NUM_SCORES 3
#define ACCEPTOR_POS 1
#define DONOR_POS 2

#define SPLICEAI_TAG "SpliceAI"
#define SPLICEAI_DESC "##INFO=<ID=SpliceAI,Number=.,Type=String,Description=\"SpliceAIv1.3.1 variant annotation. These include delta scores (DS) and delta positions (DP) for acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL\">"

typedef struct {
    TF_Status *status;
    TF_Graph *graph;
    TF_SessionOptions *sess_opts;
    TF_Buffer *run_opts;
    TF_Session *session;
} Model;

typedef struct {
    char *alt;
    char *gene;
    float ag;
    float al;
    float dg;
    float dl;
    int ag_idx;
    int al_idx;
    int dg_idx;
    int dl_idx;
} Score;

typedef struct {
    bcf1_t *v;
    int a_idx; // Allele index
    int a_len; // Allele length
    kstring_t seq;
    double *preds;
} Allele;

typedef kvec_t(Allele *) Batch;
KHASH_MAP_INIT_INT(ihash, Batch *);


int get_direction(char *s, int len) {
    char *ss = strdup(s);
    char *token = strtok(ss, "\t");
    while (token != NULL) {
        if (*token == '-') { free(ss); return -1; }
        if (*token == '+') { free(ss); return 1; }

        token = strtok(NULL, "\t");
    }

    free(ss);
    return 0;
}

char *get_name(char *s, int len) {
    char *ss = strdup(s);
    char *token = strtok(ss, "\t");
    for(int i = 0; token != NULL; i++) {
        if (i == 3) { char *res = strdup(token); free(ss); return res; }

        token = strtok(NULL, "\t");
    }

    free(ss);
    return "";
}

void reverse_encoding(float enc[], int len) {
    float tmp;
    for (int i = 0, j = len - 1; i < j; i++, j--) {
        tmp = enc[i];
        enc[i] = enc[j];
        enc[j] = tmp;
    }
}

void reverse_prediction(float preds[], int len, int size) {
    int num_preds = len / size;
    float tmp;
    for (int i = 0; i < num_preds / 2; i++) {
        for (int j = 0; j < size; j++) {
            tmp = preds[i * size + j];
            preds[i * size + j] = preds[(num_preds - 1 - i) * size + j];
            preds[(num_preds - 1 - i) * size + j] = tmp;
        }
    }
}

void deallocator(void* data, size_t a, void* b) {
    // free(data);
}

// Check the status and print an error message if any
int check_status(TF_Status* status, const char* msg) {
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error: %s: %s\n", msg, TF_Message(status));
        return 1;
    }
    return 0;
}

Model load_model(const char *path) {
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

int open_annotations(const char *path, htsFile **bed, tbx_t **tbx) {
    *bed = hts_open(path, "r");
    if (*bed == NULL) {
        log_error("Failed to open BED file: %s", path);
        return -1;
    }

    *tbx = tbx_index_load(path);
    if (*tbx == NULL) {
        log_error("Failed to open tabix index for file: %s", path);
        hts_close(*bed);
        return -1;
    }

    return 0;
}

int open_input_vcf(const char *path, htsFile **vcf, bcf_hdr_t **hdr) {
    *vcf = bcf_open(path, "r");
    if (*vcf == NULL) {
        log_error("Failed to open VCF file: %s", path);
        return -1;
    }

    *hdr = bcf_hdr_read(*vcf);
    if (*hdr == NULL) {
        log_error("Failed to read header from VCF file: %s", path);
        bcf_close(*vcf);
        return -1;
    }

    return 0;
}


int prepare_output_vcf(const char *path, bcf_hdr_t *hdr, htsFile **out) {
    *out = bcf_open(path, "w");
    if (*out == NULL) {
        log_error("Failed to open vcf output file: %s", path);
        return -1;
    }

    if (bcf_hdr_append(hdr, SPLICEAI_DESC) != 0) {
        log_error("Failed to append description for tag %s to vcf header.", SPLICEAI_TAG);
        bcf_close(*out);
        return -1;
    }

    if (bcf_hdr_write(*out, hdr) != 0) {
        log_error("Failed to write to vcf file: %s", path);
        bcf_close(*out);
        return -1;
    }

    return 0;
}

reg_t find_transcript_boundary(const int position, const int start, const int end, const int width) {
    int distance_from_start = width/2 + (start - position);
    int distance_from_end = width/2 - (end - (position+1)); // End is open, so +1
    return (reg_t) { distance_from_start > 0 ? distance_from_start : 0, distance_from_end > 0 ? distance_from_end : 0 };
}

char *pad_sequence(const char *seq, const reg_t boundary, const int width) {
    char *padded_seq = malloc(width + 1);

    int c = 0;
    for (; c < boundary.start; c++) padded_seq[c] = 'N';
    for (; c < width - boundary.end; c++) padded_seq[c] = seq[c];
    for (; c < width; c++) padded_seq[c] = 'N';
    padded_seq[width] = '\0';

    return padded_seq;
}

char *replace_variant(const char *seq, const int len, const int rlen, const char *alt, const int alen) {
    int alt_seq_len = len - rlen + alen;
    char *alt_seq = malloc(alt_seq_len + 1);

    int c = 0;
    for (; c < len/2; c++) alt_seq[c] = seq[c];
    for (int ai = 0; ai < alen; ai++, c++) alt_seq[c] = alt[ai];
    for (int ri = len/2 + rlen; ri < len; c++, ri++) alt_seq[c] = seq[ri];
    alt_seq[alt_seq_len] = '\0';

    return alt_seq;
}

int one_hot_encode(const char *sequence, const int len, float *encoding_out[]) {
    int enc_len = len * ENCODING_SIZE;
    float *encoding = calloc(enc_len, sizeof(float));
    for (int i = 0; i < enc_len; i+=ENCODING_SIZE, sequence++) {
        switch (*sequence) {
            case BASE_A:
                encoding[i + BASE_A_ENC] = 1.0f;
                break;
            case BASE_C:
                encoding[i + BASE_C_ENC] = 1.0f;
                break;
            case BASE_G:
                encoding[i + BASE_G_ENC] = 1.0f;
                break;
            case BASE_T:
                encoding[i + BASE_T_ENC] = 1.0f;
                break;
        }
    } 

    *encoding_out = encoding;
    return enc_len;
}

// int resize_alt_to_ref2(float **predictions_alt, int num_alts, int num_refs) {
//     if (num_refs > num_alts) {
//         int diff = num_refs - num_alts;
//         *predictions_alt = realloc(*predictions_alt, num_refs * sizeof(float));
//         for (int i = num_refs-1; i > num_alts; i--) (*predictions_alt)[i] = (*predictions_alt)[i-diff];
//         for (int i = num_alts;
//     }
//
//
//     return -1;
// }

// NOTE: Requires refactor, it is too messy
int resize_alt_to_ref(float **predictions_alt, int alt_len, int ref_len, int num_predictions_ref, int cov) {

    if (ref_len > alt_len && alt_len == 1) {
        int del_len = ref_len - alt_len;
        *predictions_alt = realloc(*predictions_alt, sizeof(float) * num_predictions_ref);
        for (int i = num_predictions_ref-1; i >= cov/2*3+ref_len*3; i--) (*predictions_alt)[i] = (*predictions_alt)[i-del_len*3];
        for (int i = cov/2*3+alt_len*3; i < cov/2*3+ref_len*3; i++) (*predictions_alt)[i] = 0.0;

        return 0;
    } else if (alt_len > ref_len && ref_len == 1) {
        int in_len = alt_len - ref_len;
        float *best_scores = calloc(3, sizeof(float));
        for (int i = cov/2*3; i < cov/2*3 + alt_len*3;) {
            for (int j = 0; j < 3; j++, i++) {
                if ((*predictions_alt)[i] > best_scores[j]) best_scores[j] = (*predictions_alt)[i];
            }
        }
        (*predictions_alt)[cov/2*3] = best_scores[0];
        (*predictions_alt)[cov/2*3+1] = best_scores[1];
        (*predictions_alt)[cov/2*3+2] = best_scores[2];
        free(best_scores);

        for (int i = cov/2*3+ref_len*3; i < num_predictions_ref; i++) (*predictions_alt)[i] = (*predictions_alt)[i+in_len*3];
        *predictions_alt = realloc(*predictions_alt, sizeof(float) * num_predictions_ref);

        return 0;
    }

    return -1;
}

Score calculate_delta_scores(char *allele, char *gene_symbol, float *predictions_ref, float *predictions_alt, int len, int offset) {
    float ag_best = 0.0, al_best = 0.0, dg_best = 0.0, dl_best = 0.0;
    int ag_idx = 0, al_idx = 0, dg_idx = 0, dl_idx = 0;

    for (int p = 0; p < len; p += NUM_SCORES) {
        float ag = predictions_alt[p + ACCEPTOR_POS] - predictions_ref[p + ACCEPTOR_POS];
        float al = predictions_ref[p + ACCEPTOR_POS] - predictions_alt[p + ACCEPTOR_POS];
        float dg = predictions_alt[p + DONOR_POS] - predictions_ref[p + DONOR_POS];
        float dl = predictions_ref[p + DONOR_POS] - predictions_alt[p + DONOR_POS];

        if (ag > ag_best) { ag_best = ag; ag_idx = (p / NUM_SCORES); }
        if (al > al_best) { al_best = al; al_idx = (p / NUM_SCORES); }
        if (dg > dg_best) { dg_best = dg; dg_idx = (p / NUM_SCORES); }
        if (dl > dl_best) { dl_best = dl; dl_idx = (p / NUM_SCORES); }
    }

    ag_idx = ag_idx-offset;
    al_idx = al_idx-offset;
    dg_idx = dg_idx-offset;
    dl_idx = dl_idx-offset;

    return (Score) { allele, gene_symbol, ag_best, al_best, dg_best, dl_best, ag_idx, al_idx, dg_idx, dl_idx };
}

int main(int argc, char *argv[]) {
    // Set tensorflow logging to WARN and above
    setenv("TF_CPP_MIN_LOG_LEVEL", "1", 1);

    // Future CLI arguments
    
    if (argc != 7) { printf("Usage: ./cpliceai <distance> <models_dir> <ann.bed> <ref.fa> <in.vcf> <out.vcf> \n"); return -1; }
    // int distance = 50;
    // const char *annotations = "data/grch37.bed.gz";
    // const char *reference = "data/hs_ref_GRCh37.p5_all_contigs.fa";
    // const char *vcf = "data/test.vcf";
    // const char *output_vcf = "data/output.vcf";
    int distance = atoi(argv[1]);
    const char *model_dir = argv[2];
    const char *annotations = argv[3];
    const char *reference = argv[4];
    const char *vcf = argv[5];
    const char *output_vcf = argv[6];

    // int batch_size = 700;

    const char *spliceai_tag = SPLICEAI_TAG; // Info tag

    // Load SpliceAI models
    Model *models = load_models(model_dir);

    // Open BED/fasta/vcf files
    htsFile *bed; tbx_t *tbx; 
    if (open_annotations(annotations, &bed, &tbx) < 0) return -1; // Load annotations for transcript regions

    faidx_t *fa_in;
    if ((fa_in = fai_load(reference)) < 0) return -1; // Load reference fasta for sequence lookup

    htsFile *vcf_in; bcf_hdr_t *hdr;
    if (open_input_vcf(vcf, &vcf_in, &hdr) < 0) return -1; // Load input vcf

    htsFile *vcf_out;
    if (prepare_output_vcf(output_vcf, hdr, &vcf_out) < 0) return -1; // Prepare output vcf

    // Sequence size
    int cov = 2 * distance + 1;
    int width = CONTEXT_SIZE + cov;
    int half_width = width / 2;

    bcf1_t *v = bcf_init();
    hts_itr_t *itr;
    // int batch_count = 0;
    // kvec_t(int) ref_idxs;
    // kv_init(ref_idxs);
    // Batch batch;
    // kv_init(batch);

    while (bcf_read(vcf_in, hdr, v) >= 0) {
        if (!v) continue;

        // if (batch_count < batch_size) {
        //     bcf_unpack(v, BCF_UN_INFO);
        //     bcf1_t *vc = bcf_dup(v);
        //
        //     int slen;
        //     char *s = faidx_fetch_seq(fa_in, bcf_hdr_id2name(hdr, v->rid), v->pos - half_width, v->pos + half_width, &slen);
        //     s[slen] = '\0';
        //     kstring_t seq = {slen, slen+1, s};
        //     int ref_len = strlen(v->d.allele[0]);
        //
        //     Allele *ref_allele = malloc(sizeof(Allele));
        //     ref_allele->v = vc;
        //     ref_allele->a_idx = 0;
        //     ref_allele->a_len = ref_len;
        //     ref_allele->seq = seq;
        //     kv_push(Allele *, batch, ref_allele);
        //
        //     for (int i = 1; i < v->n_allele; i++) {
        //         char *alt = v->d.allele[i];
        //         int alt_len = strlen(alt);
        //         int alt_seq_len = seq.l - ref_len + alt_len;
        //         kstring_t alt_seq = {alt_seq_len, alt_seq_len + 1, replace_variant(seq.s, seq.l, ref_len, alt, alt_len)};
        //
        //         Allele *alt_allele = malloc(sizeof(Allele));
        //         alt_allele->v = vc;
        //         alt_allele->a_idx = i;
        //         alt_allele->a_len = alt_len;
        //         alt_allele->seq = alt_seq;
        //         kv_push(Allele *, batch, alt_allele);
        //     }
        //
        //     if (++batch_count < batch_size) continue;
        // }
        //
        // // Initialize hash table
        // khint_t k;
        // khash_t(ihash) *groups = kh_init(ihash);
        // kvec_t(int) keys;
        // kv_init(keys);
        // // Group by allele size
        // for (int i = 0; i < batch.n; i++) {
        //     Allele *a = kv_A(batch, i);
        //
        //     k = kh_get_ihash(groups, a->a_len);
        //     if (k != kh_end(groups)) {
        //         Batch *group = kh_value(groups, k);
        //         kv_push(Allele *, *group, a);
        //         continue;
        //     }
        //
        //     Batch *group = malloc(sizeof(Batch));
        //     kv_init(*group);
        //     kv_push(Allele *, *group, a);
        //     int ret;
        //     k = kh_put(ihash, groups, a->a_len, &ret);
        //     kh_val(groups, k) = group;
        //
        //     kv_push(int, keys, a->a_len);
        // }
        //
        //
        // for (int i = 0; i < keys.n; i++) {
        //     k = kh_get_ihash(groups, kv_A(keys, i));
        //     Batch *group = kh_value(groups, k);
        //     int seq_len = kv_A(*group, 0)->seq.l;
        //
        //     kvec_t(float *) encodings;
        //     kv_init(encodings);
        //     for (int j = 0; j < group->n; j++) {
        //         kstring_t str = {0};
        //         Allele *a = kv_A(*group, j);
        //         itr = tbx_itr_queryi(tbx, a->v->rid, a->v->pos, a->v->pos+1);
        //
        //         while(tbx_itr_next(bed, tbx, itr, &str) >= 0) {
        //             reg_t boundary = find_transcript_boundary(a->v->pos, itr->curr_beg, itr->curr_end, width); // WARN: Use of 'width' is likely causing a logical bug
        //             char *padded_seq = pad_sequence(a->seq.s, boundary, a->seq.l);
        //             float *ohe; one_hot_encode(a->seq.s, a->seq.l, &ohe);
        //
        //             int strand = get_direction(str.s, str.l);
        //             if (strand == -1) reverse_encoding(ohe, a->seq.l * ENCODING_SIZE);
        //
        //             kv_push(float *, encodings, ohe);
        //         }
        //     }
        //
        //     // float *input = malloc(encodings.n * seq_len * sizeof(float));
        //     // for (int j = 0; j < encodings.n; j++) {
        //     //     float *encoding = kv_A(encodings, j);
        //     //     // double *predictions; int num_predictions;
        //     //     // predict(models, seq_len * ENCODING_SIZE, 1, &encoding, &num_predictions, &predictions);
        //     //     for (int k = 0; k < seq_len; k++) input[j * seq_len + k] = encoding[k]; 
        //     // }
        //
        //     float *predictions; int num_predictions;
        //     predict(models, seq_len * ENCODING_SIZE, encodings.n, encodings.a, &num_predictions, &predictions);
        //
        //     // float avg = 0.0;
        //     // printf("Completed batch predictions!\n");
        //     // for (int j = 0; j < num_predictions; j++) {
        //     //     // printf("Prediction: %f\n", predictions[j]);
        //     //     avg += predictions[j];
        //     // }
        //     // printf("Average for batch: %f\n", avg / num_predictions);
        //
        // }
        //
        // kv_destroy(batch);
        // kv_init(batch);



        bcf_unpack(v, BCF_UN_INFO);
        itr = tbx_itr_queryi(tbx, v->rid, v->pos, v->pos+1);

        kstring_t info_str = {0};
        int ref_len = strlen(v->d.allele[0]);

        int slen;
        char *seq = faidx_fetch_seq(fa_in, bcf_hdr_id2name(hdr, v->rid), v->pos - half_width, v->pos + half_width, &slen);
        seq[slen] = '\0';

        for (int i = 1; i < v->n_allele; i++) {
            if (slen != width) { log_warn("Skipping record (near chromosome end) at %s:%d", bcf_hdr_id2name(hdr, v->rid), v->pos); continue; } 
            if (ref_len > distance) { log_warn("Skipping record (ref too long) at %s:%d", bcf_hdr_id2name(hdr, v->rid), v->pos); continue; } 
            if ('.' == v->d.allele[i][0] || // Deletion
                '*' == v->d.allele[i][0] || // Missing
                '<' == v->d.allele[i][0] // <ID> string
            ) continue;

            kstring_t str = {0};
            while (tbx_itr_next(bed, tbx, itr, &str) >= 0) {
                // Replace/pad bases beyond transcript boundary with N
                reg_t boundary = find_transcript_boundary(v->pos, itr->curr_beg, itr->curr_end, width);

                // Pad ref and alt using transcript boundary
                int alt_len = strlen(v->d.allele[i]);
                char *padded_ref = pad_sequence(seq, boundary, width);
                char *padded_alt = replace_variant(padded_ref, width, ref_len, v->d.allele[i], alt_len);
                int padded_alt_len = width - ref_len + alt_len;

                // Encode data for ref and alt
                float *ohe_ref; one_hot_encode(padded_ref, width, &ohe_ref);
                float *ohe_alt; one_hot_encode(padded_alt, padded_alt_len, &ohe_alt);
                free(padded_ref); free(padded_alt);

                // Reversing the encoding creates the complementary strand in opposite direction
                int strand = get_direction(str.s, str.l);
                if (strand == -1) { reverse_encoding(ohe_ref, width * ENCODING_SIZE); reverse_encoding(ohe_alt, padded_alt_len * ENCODING_SIZE); }

                // Predict
                float *predictions_ref; int num_predictions_ref;
                predict(models, width * ENCODING_SIZE, 1, &ohe_ref, &num_predictions_ref, &predictions_ref);
                float *predictions_alt; int num_predictions_alt;
                predict(models, padded_alt_len * ENCODING_SIZE, 1, &ohe_alt, &num_predictions_alt, &predictions_alt);
                free(ohe_ref); free(ohe_alt);

                // Reverse prediction order for negative strands
                if (strand == -1) { reverse_prediction(predictions_ref, num_predictions_ref, 3); reverse_prediction(predictions_alt, num_predictions_alt, 3); }

                if (info_str.l > 0) kputc(',', &info_str);

                char *gene = get_name(str.s, str.l);

                // Resizes the alt predictions to match the length of ref predictions
                if ((ref_len != 1 || ref_len != 1) && resize_alt_to_ref(&predictions_alt, alt_len, ref_len, num_predictions_ref, cov) != 0) { 
                    log_warn("Problem in vcf at pos %s:%d", bcf_hdr_id2name(hdr, v->rid), v->pos);
                    char tmp[4096];
                    sprintf(tmp, "%s|%s|.|.|.|.|.|.|.|.", v->d.allele[i], gene);
                    kputs(tmp, &info_str);

                    free(predictions_ref); free(predictions_alt);
                    free(gene);
                    continue;
                }

                Score score = calculate_delta_scores(v->d.allele[i], gene, predictions_ref, predictions_alt, num_predictions_ref, cov/2);
                free(predictions_ref); free(predictions_alt);

                char tmp[4096];
                sprintf(tmp, "%s|%s|%.2f|%.2f|%.2f|%.2f|%d|%d|%d|%d", score.alt, score.gene, score.ag, score.al, score.dg, score.dl, score.ag_idx, score.al_idx, score.dg_idx, score.dl_idx);
                kputs(tmp, &info_str);

                free(gene); // Duplicated from bed str
            }
            free(str.s);
        }

        if (info_str.l > 0) {
            bcf_update_info_string(hdr, v, spliceai_tag, info_str.s);
            free(info_str.s);
        }

        bcf_write(vcf_out, hdr, v);
        free(seq);
    }

    bcf_destroy(v);
    hts_itr_destroy(itr);

    hts_close(bed); tbx_destroy(tbx);
    fai_destroy(fa_in);
    hts_close(vcf_in);
    hts_close(vcf_out);
    bcf_hdr_destroy(hdr);

    destroy_models(models);

    return 0;
}

