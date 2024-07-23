#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <klib/kstring.h>
#include <klib/kvec.h>
#include <klib/khash.h>

#include <htslib/hts.h>
#include <htslib/vcf.h>
#include <htslib/faidx.h>
#include <htslib/tbx.h>
#include <htslib/regidx.h>

#include "logging/log.h"
#include "predict.h"
#include "utils.h"


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

// NOTE: Requires refactor, it is too messy
int resize_alt_to_ref(float **predictions_alt, int alt_len, int ref_len, int num_predictions_ref, int cov) {

    if (ref_len > alt_len) {
        int del_len = ref_len - alt_len;
        *predictions_alt = realloc(*predictions_alt, sizeof(float) * num_predictions_ref);
        for (int i = num_predictions_ref-1; i >= cov/2*3+ref_len*3; i--) (*predictions_alt)[i] = (*predictions_alt)[i-del_len*3];
        for (int i = cov/2*3+alt_len*3; i < cov/2*3+ref_len*3; i++) (*predictions_alt)[i] = 0.0;

        return 0;
    } else if (alt_len > ref_len) {
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

void predict2(Model *models, char *seq, int seq_len, int strand, float *predictions[], int *num_predictions) {
    float *ohe; one_hot_encode(seq, seq_len, &ohe);

    if (strand == -1) reverse_encoding(ohe, seq_len * ENCODING_SIZE);

    predict(models, seq_len * ENCODING_SIZE, 1, &ohe, num_predictions, predictions);
    free(ohe);

    if (strand == -1) reverse_prediction(*predictions, *num_predictions, 3);
}

int main(int argc, char *argv[]) {
    // Set tensorflow logging to WARN and above
    setenv("TF_CPP_MIN_LOG_LEVEL", "1", 1);

    // Parse CLI arguments
    if (argc != 7) { printf("Usage: ./cpliceai <distance> <models_dir> <ann.bed> <ref.fa> <in.vcf> <out.vcf> \n"); return -1; }
    // int distance = 50;
    // const char *annotations = "data/grch37.bed.gz";
    // const char *reference = "data/hs_ref_GRCh37.p5_all_contigs.fa";
    // const char *vcf = "data/test.vcf";
    // const char *output_vcf = "data/output.vcf";
    const int distance = atoi(argv[1]);
    const char *model_dir = argv[2];
    const char *annotations = argv[3];
    const char *reference = argv[4];
    const char *vcf = argv[5];
    const char *output_vcf = argv[6];

    // int batch_size = 700;

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
    const int cov = 2 * distance + 1;
    const int width = CONTEXT_SIZE + cov;
    const int half_width = width / 2;

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

                int strand = get_direction(str.s, str.l);

                float *predictions_ref, *predictions_alt; int num_predictions_ref, num_predictions_alt;
                predict2(models, padded_ref, width, strand, &predictions_ref, &num_predictions_ref);
                predict2(models, padded_alt, padded_alt_len, strand, &predictions_alt, &num_predictions_alt);
                free(padded_ref); free(padded_alt);

                if (info_str.l > 0) kputc(',', &info_str);
                char *gene = get_name(str.s, str.l);

                // Resizes the alt predictions to match the length of ref predictions
                if ((ref_len != 1 || alt_len != 1) && resize_alt_to_ref(&predictions_alt, alt_len, ref_len, num_predictions_ref, cov) != 0) { 
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
            bcf_update_info_string(hdr, v, SPLICEAI_TAG, info_str.s);
            free(info_str.s);
        }

        bcf_write(vcf_out, hdr, v);
        free(seq);
    }

    bcf_destroy(v);
    hts_itr_destroy(itr);

    hts_close(bed); tbx_destroy(tbx);
    fai_destroy(fa_in);
    hts_close(vcf_in); hts_close(vcf_out);
    bcf_hdr_destroy(hdr);

    destroy_models(models);

    return 0;
}

