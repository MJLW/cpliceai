#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <klib/kstring.h>
#include <klib/kvec.h>

#include <tensorflow/c/c_api.h>
#include <htslib/hts.h>
#include <htslib/vcf.h>
#include <htslib/faidx.h>

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

#define ACCEPTOR_POS 1
#define DONOR_POS 2

typedef struct {
    TF_Status *status;
    TF_Graph *graph;
    TF_SessionOptions *sess_opts;
    TF_Buffer *run_opts;
    TF_Session *session;
} Model;

typedef struct {
    int start;
    int stop;
} Exon;

typedef struct {
    char *name;
    char *chr;
    char strand;
    int start;
    int stop;
    int num_exons;
    Exon *exons;
} Transcript;

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

void reverse_encoding(float *enc, int len) {
    char tmp;
    for (int i = 0, j = len - 1; i < j; i++, j--) {
        tmp = enc[i];
        enc[i] = enc[j];
        enc[j] = tmp;
    }
}


int parse_exon_fields(char *s, int *num_out, int *out[]) {
    char *es = s;
    int len = strlen(es) - 1; // Fields end in ,\0

    kvec_t(int) fields;
    kv_init(fields);
    int flen = 0;
    while (flen < len) {
        char *field = es + flen;

        while (es[flen] != ',') flen++;
        es[flen++] = '\0';

        kv_push(int, fields, atoi(field));
    }

    *num_out = fields.n;
    *out = fields.a;

    return 0;
}

#define LINE_LENGTH 8192
int parse_transcripts(char *fp_path, int *num_out, Transcript *out[]) {
    FILE *fp = fopen(fp_path, "r");
    char *line = malloc(LINE_LENGTH * sizeof(char));

    kvec_t(Transcript) transcripts;
    kv_init(transcripts);

    fgets(line, LINE_LENGTH, fp);
    int num_line = 1;

    while (fgets(line, LINE_LENGTH, fp) != NULL) {
        line[strcspn(line, "\n")] = '\0';
        int end = strlen(line);
        int rlen = 0;

        Transcript t = {0};

        int *starts; int nstarts;
        int *stops; int nstops;

        int slen = 0;
        for (int i = 0; rlen < end; i++) {
            char *field = line + rlen;

            while (line[rlen] != '\0' && line[rlen] != '\t') rlen++;
            line[rlen++] = '\0';

            switch (i) {
                case 0:
                    t.name = strdup(field);
                    break;
                case 1:
                    kstring_t s = {0};
                    kputs("chr", &s);
                    kputs(field, &s);
                    t.chr = s.s;
                    break;
                case 2:
                    t.strand = *field;
                    break;
                case 3:
                    t.start = atoi(field);
                    break;
                case 4:
                    t.stop = atoi(field);
                    break;
                case 5:
                    parse_exon_fields(field, &nstarts, &starts);
                    break;
                case 6:
                    parse_exon_fields(field, &nstops, &stops);
                    break;
            }
        }
        if (nstarts != nstops) { fprintf(stderr, "Encountered error in line %d.\n", num_line); return 1; }

        t.num_exons = nstarts;
        t.exons = malloc(nstarts * sizeof(Exon));
        for (int i = 0; i < nstarts; i++) t.exons[i] = (Exon) {starts[i], stops[i]};
        free(starts); free(stops);

        kv_push(Transcript, transcripts, t);

        num_line++;
    }
    free(line);
    fclose(fp);

    *num_out = transcripts.n;
    *out = transcripts.a;

    return 0;
}


void NoOpDeallocator(void* data, size_t a, void* b) {}

// Check the status and print an error message if any
int CheckStatus(TF_Status* status, const char* msg) {
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
    CheckStatus(status, "Loading model");

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

int predict(Model *models, int data_len, float *data, int *num_out, double *out[]) {
    // Define the input dimensions
    int64_t input_dims[] = {1, data_len / ENCODING_SIZE, 4};

    // Create the input tensor
    TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, input_dims, 3, data, data_len * sizeof(float), &NoOpDeallocator, 0);
    if (!input_tensor) {
        fprintf(stderr, "Failed to create input tensor\n");
        return 1;
    }

    int output_len = ((data_len / ENCODING_SIZE) - CONTEXT_SIZE) * 3;
    double *outputs = calloc(output_len, sizeof(double));

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
        TF_SessionRun(model.session, NULL,
                      &input_opout, &input_tensor, 1, // Input tensors and count
                      &output_opout, output_tensors, 1, // Output tensors and count
                      NULL, 0, // Target operations, target operations count
                      NULL, // Run metadata
                      model.status);
        CheckStatus(model.status, "Running model");

        // Process the output data
        float* output_data = (float*)TF_TensorData(output_tensors[0]);
        // int output_size = TF_TensorByteSize(output_tensors[0]) / sizeof(float);
        for (int i = 0; i < output_len; i++) outputs[i] += (double) output_data[i];
    }

    for (int i = 0; i < output_len; i++) outputs[i] /= NUM_SPLICEAI_MODELS;

    *num_out = output_len;
    *out = outputs;

    return 0;
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

int main() {
    int distance = 50;
    char *output_vcf = "data/output.vcf";

    // Parse transcript and exon boundaries from SpliceAI annotation
    char *annotations = "data/grch37.txt";
    int num_transcripts;
    Transcript *transcripts;
    parse_transcripts(annotations, &num_transcripts, &transcripts);

    // Load SpliceAI models
    Model *models = load_models("models");

    // Load reference fasta for sequence lookup
    char *reference = "data/hs_ref_GRCh37.p5_all_contigs.fa";
    faidx_t *fa_in = fai_load(reference);

    // Open input vcf for reading
    char *vcf = "data/test.vcf";
    htsFile *vcf_in = bcf_open(vcf, "r");
    bcf_hdr_t *hdr = bcf_hdr_read(vcf_in);

    htsFile *vcf_out =  bcf_open(output_vcf, "w");
    const char *spliceai_tag = "SpliceAI";
    const char *spliceai_desc = "##INFO=<ID=SpliceAI,Number=.,Type=String,Description=\"SpliceAIv1.3.1 variant annotation. These include delta scores (DS) and delta positions (DP) for acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL\">";
    bcf_hdr_append(hdr, spliceai_desc);
    bcf_hdr_write(vcf_out, hdr);

    bcf1_t *v = bcf_init();

    int cov = 2 * distance + 1;
    int width = CONTEXT_SIZE + cov;

    int tidx = 0;
    int num_overlaps = 0;
    log_info("Starting predictions.");
    int count = 0;
    while (bcf_read(vcf_in, hdr, v) >= 0) {
        if (!v) continue;

        // Find transcripts overlapping current variant
        while (tidx < num_transcripts && (v->rid > bcf_hdr_name2id(hdr, transcripts[tidx].chr) || v->pos > transcripts[tidx].stop)) tidx++;

        // Count the number of overlapping transcripts
        num_overlaps = 0;
        while (
            tidx + num_overlaps < num_transcripts && 
            v->rid == bcf_hdr_name2id(hdr, transcripts[tidx + num_overlaps].chr) && 
            v->pos >= transcripts[tidx + num_overlaps].start &&
            v->pos <= transcripts[tidx + num_overlaps].stop
        ) num_overlaps++;

        kstring_t info_str = {0};
        if (num_overlaps == 0) {
            kputc('.', &info_str);
            bcf_update_info_string(hdr, v, spliceai_tag, info_str.s);
            free(info_str.s);
            bcf_write(vcf_out, hdr, v);
            continue;
        }

        int half_width = width / 2;
        int slen;
        char *seq = faidx_fetch_seq(fa_in, transcripts[tidx].chr, v->pos - half_width, v->pos + half_width, &slen);
        seq[slen] = '\0';

        bcf_unpack(v, BCF_UN_ALL);

        int ref_len = strlen(v->d.allele[0]);
        for (int i = 1; i < v->n_allele; i++) {
            if ('.' == v->d.allele[i][0] || // Deletion
                '*' == v->d.allele[i][0] || // Missing
                '<' == v->d.allele[i][0] // <ID> string
            ) continue;

            for (int j = 0; j < num_overlaps; j++) {
                // Replace/pad bases beyond transcript boundary with N
                int start_distance = half_width + (transcripts[tidx + j].start - v->pos);
                int stop_distance = half_width - (transcripts[tidx + j].stop - v->pos) + 1;

                int pad_start = start_distance > 0 ? start_distance : 0;
                int pad_stop = stop_distance > 0 ? stop_distance : 0;

                // Create padded ref
                char *padded_ref = malloc(width + 1);
                int c = 0;
                for (; c < pad_start; c++) padded_ref[c] = 'N';
                for (; c < width - pad_stop; c++) padded_ref[c] = seq[c];
                for (; c < width; c++) padded_ref[c] = 'N';
                padded_ref[width] = '\0';

                // Create padded alt
                int alt_len = strlen(v->d.allele[i]);
                int padded_alt_len = width - ref_len + alt_len;
                char *padded_alt = malloc(padded_alt_len + 1);
                for (c = 0; c < half_width; c++) padded_alt[c] = padded_ref[c];
                for (int ai = 0; ai < alt_len; ai++, c++) padded_alt[c] = v->d.allele[i][ai];
                for (int ri = half_width + ref_len; ri < width; c++, ri++) padded_alt[c] = padded_ref[ri];
                padded_alt[padded_alt_len] = '\0';

                // Predict
                float *ohe_ref;
                one_hot_encode(padded_ref, width, &ohe_ref);

                float *ohe_alt;
                one_hot_encode(padded_alt, padded_alt_len, &ohe_alt);

                // Reversing the encoding creates the complementary strand in opposite direction
                if ('-' == transcripts[tidx + j].strand) { reverse_encoding(ohe_ref, width * ENCODING_SIZE); reverse_encoding(ohe_alt, padded_alt_len * ENCODING_SIZE); }

                double *predictions_ref;
                int num_predictions_ref;
                predict(models, width * ENCODING_SIZE, ohe_ref, &num_predictions_ref, &predictions_ref);

                double *predictions_alt;
                int num_predictions_alt;
                predict(models, padded_alt_len * ENCODING_SIZE, ohe_alt, &num_predictions_alt, &predictions_alt);

                // printf("#ref_preds: %d, #alt_preds: %d\n", num_predictions_ref, num_predictions_alt);
                // if (ref_len != alt_len) continue;
                // Resizes the alt predictions to match the length of ref predictions
                // WARN: This will break/cause logical bugs if neither ref_len nor alt_len are of size 1
                if (ref_len > alt_len) {
                    int del_len = ref_len - alt_len;
                    predictions_alt = reallocarray(predictions_alt, sizeof(double), num_predictions_ref);
                    for (int i = num_predictions_ref-1; i >= cov/2*3+ref_len*3; i--) predictions_alt[i] = predictions_alt[i-del_len*3];
                    for (int i = cov/2*3+alt_len*3; i < cov/2*3+ref_len*3; i++) predictions_alt[i] = 0.0;
                } else if (alt_len > ref_len) {
                    int in_len = alt_len - ref_len;
                    float *best_scores = calloc(3, sizeof(float));
                    for (int i = cov/2*3; i < cov/2*3 + alt_len*3;) {
                        for (int j = 0; j < 3; j++, i++) {
                            if (predictions_alt[i] > best_scores[j]) best_scores[j] = predictions_alt[i];
                        }
                    }
                    predictions_alt[cov/2*3] = best_scores[0];
                    predictions_alt[cov/2*3+1] = best_scores[1];
                    predictions_alt[cov/2*3+2] = best_scores[2];
                    free(best_scores);

                    for (int i = cov/2*3+ref_len*3; i < num_predictions_ref; i++) predictions_alt[i] = predictions_alt[i+in_len*3];
                    predictions_alt = reallocarray(predictions_alt, sizeof(double), num_predictions_ref);
                }

                float ag_best = 0.0;
                int ag_idx = 0;
                float al_best = 0.0;
                int al_idx = 0;
                float dg_best = 0.0;
                int dg_idx = 0;
                float dl_best = 0.0;
                int dl_idx = 0;

                for (int p = 0; p < num_predictions_ref; p += 3) {
                    float ag = predictions_alt[p + ACCEPTOR_POS] - predictions_ref[p + ACCEPTOR_POS];
                    float al = predictions_ref[p + ACCEPTOR_POS] - predictions_alt[p + ACCEPTOR_POS];
                    float dg = predictions_alt[p + DONOR_POS] - predictions_ref[p + DONOR_POS];
                    float dl = predictions_ref[p + DONOR_POS] - predictions_alt[p + DONOR_POS];

                    if (ag > ag_best) { ag_best = ag; ag_idx = (p / 3); }
                    if (al > al_best) { al_best = al; al_idx = (p / 3); }
                    if (dg > dg_best) { dg_best = dg; dg_idx = (p / 3); }
                    if (dl > dl_best) { dl_best = dl; dl_idx = (p / 3); }
                }

                ag_idx = ag_idx-cov/2;
                al_idx = al_idx-cov/2;
                dg_idx = dg_idx-cov/2;
                dl_idx = dl_idx-cov/2;
                if ('-' == transcripts[tidx + j].strand) {
                    ag_idx *= -1;
                    al_idx *= -1;
                    dg_idx *= -1;
                    dl_idx *= -1;
                }

                if (info_str.l > 0) kputc(',', &info_str);
                Score score = (Score) { v->d.allele[i], transcripts[tidx + j].name, ag_best, al_best, dg_best, dl_best, ag_idx, al_idx, dg_idx, dl_idx };

                char tmp[4096];  
                sprintf(tmp, "%s|%s|%.2f|%.2f|%.2f|%.2f|%d|%d|%d|%d", score.alt, score.gene, score.ag, score.al, score.dg, score.dl, score.ag_idx, score.al_idx, score.dg_idx, score.dl_idx);
                kputs(tmp, &info_str);
            }
        }

        bcf_update_info_string(hdr, v, spliceai_tag, info_str.s);
        free(info_str.s);
        bcf_write(vcf_out, hdr, v);

        count++;
    }
    log_info("N of predictions: %d\n", count);

    bcf_destroy(v);
    for (int i = 0; i < num_transcripts; i++) {
        free(transcripts[i].name);
        free(transcripts[i].chr);
        free(transcripts[i].exons);
    }
    free(transcripts);

    fai_destroy(fa_in);
    bcf_hdr_destroy(hdr);
    hts_close(vcf_in);

    hts_close(vcf_out);

    // FILE *fasta = fopen("test.fasta", "r");
    // char input_sequence[1000000];
    // fgets(input_sequence, 1000000, fasta);
    // input_sequence[strcspn(input_sequence, "\n")] = 0;

    // char *input_sequence = "CGATCTGACGTGGGTGTCATCGCATTATCGATATTGCAT";
    // int input_len = strlen(input_sequence);
    // char *padded_sequence = malloc(CONTEXT_SIZE + input_len);
    // for (int i = 0; i < CONTEXT_SIZE / 2; i++) padded_sequence[i] = 'N';
    // for (int i = 0; i < input_len; i++) padded_sequence[CONTEXT_SIZE / 2 + i] = input_sequence[i];
    // for (int i = 0; i < CONTEXT_SIZE / 2; i++) padded_sequence[CONTEXT_SIZE / 2 + input_len + i] = 'N';
    //
    // Model *models = load_models("models");
    //
    // float *input_data;
    // int enc_len = one_hot_encode(padded_sequence, strlen(padded_sequence), &input_data);
    //
    // // Process the output data
    // double* output_data;
    // int output_size;
    // predict(models, enc_len, input_data, &output_size, &output_data);
    //
    // printf("Acceptor Probability: ");
    // for (int i = 0; i < output_size; i += 3) printf("%f ", output_data[i+1]); 
    // printf("\nDonor Probability: ");
    // for (int i = 0; i < output_size; i += 3) printf("%f ", output_data[i+2]);
    // printf("\n");
    //
    // // Calculate mean probabilities for acceptor and donor
    // double acceptor_prob = 0.0f;
    // double donor_prob = 0.0f;
    //
    // for (int i = 0; i < output_size; i += 3) {
    //     acceptor_prob += output_data[i + 1];
    //     donor_prob += output_data[i + 2];
    // }
    //
    // acceptor_prob /= ((double) output_size / 3);
    // donor_prob /= ((double) output_size / 3);
    //
    // printf("Acceptor Probability: %f\n", acceptor_prob);
    // printf("Donor Probability: %f\n", donor_prob);
    //
    // // Cleanup
    // // TF_DeleteTensor(input_tensor);
    // // TF_DeleteTensor(output_tensors[0]);
    // for (int i = 0; i < NUM_SPLICEAI_MODELS; i++) {
    //     TF_DeleteGraph(models[i].graph);
    //     TF_DeleteSession(models[i].session, models[i].status);
    //     TF_DeleteSessionOptions(models[i].sess_opts);
    //     TF_DeleteStatus(models[i].status);
    // }
    // free(models);
    // free(input_data);

    return 0;
}

