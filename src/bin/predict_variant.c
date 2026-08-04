#include <htslib/kstring.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <klib/kstring.h>
#include <klib/kvec.h>

#include <htslib/hts.h>
#include <htslib/vcf.h>
#include <htslib/faidx.h>
#include <htslib/tbx.h>
#include <htslib/regidx.h>

#include "../logging/log.h"
#include "../predict.h"
#include "../utils.h"
#include "../reference.h"
#include "../gene_reference.h"
#include "../gene_regions.h"

#define VEP_CSQ "CSQ"

#define REQUIRED_ARGS \
    REQUIRED_STRING_ARG(variants, "variants", "VCF file containing variants (split multiallelics) to predict for using SpliceAI") \
    REQUIRED_STRING_ARG(reference_bin, "reference_scores", "Binary file containing reference scores") \
    REQUIRED_STRING_ARG(model_dir, "model_dir", "Directory containing SpliceAI models") \
    REQUIRED_STRING_ARG(fasta, "fasta", "Human reference fasta") \
    REQUIRED_STRING_ARG(regions, "regions", "Gene region structure parsed from GFF with gff_to_bed.py") \
    REQUIRED_STRING_ARG(output, "output", "Output VCF file with SpliceAI annotations")

#define OPTIONAL_ARGS \
    OPTIONAL_INT_ARG(window_size, 500, "--window-size", "window size", "Window size for the SpliceAI predictions") \
    OPTIONAL_STRING_ARG(splice_output, "\0", "--splice-output", "file", "Output TSV with sparse splice predictions per variant")

#define BOOLEAN_ARGS \
    BOOLEAN_ARG(help, "-h", "Show help")

#include <easyargs.h>

typedef struct {
    char *consequence;
    char *gene;
} ConsequenceAnnotation;

typedef struct {
    ConsequenceAnnotation *annotations;
    kstring_t field;
} Consequences;

int open_input_vcf(const char *path, htsFile **vcf, bcf_hdr_t **hdr) {
    *vcf = bcf_open(path, "r");
    if (*vcf == NULL) {
        log_error("Failed to open VCF file: %s", path);
        return EXIT_FAILURE;
    }

    *hdr = bcf_hdr_read(*vcf);
    if (*hdr == NULL) {
        log_error("Failed to read header from VCF file: %s", path);
        bcf_close(*vcf);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


int prepare_output_vcf(const char *path, bcf_hdr_t *hdr, htsFile **out) {
    *out = bcf_open(path, "w");
    if (*out == NULL) {
        log_error("Failed to open vcf output file: %s", path);
        return EXIT_FAILURE;
    }

    if (bcf_hdr_append(hdr, SPLICEAI_DESC) != 0) {
        log_error("Failed to append description for tag %s to vcf header.", SPLICEAI_TAG);
        bcf_close(*out);
        return EXIT_FAILURE;
    }

    if (bcf_hdr_write(*out, hdr) != 0) {
        log_error("Failed to write to vcf file: %s", path);
        bcf_close(*out);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int gene_regions_build_regidx(const char *path, regidx_t **idx) {
    FILE *fp = open_file_or_log(path, "r");
    if (fp == NULL) return EXIT_FAILURE;

    regidx_t *gene_index = regidx_init(NULL, NULL, NULL, sizeof(Gene), NULL);

    Gene gene;
    while (read_gene_region(fp, &gene) == 0) {
        hts_pos_t beg = gene.tx_start, end = gene.tx_end;
        char *chr = gene.chrom, *chr_end = chr + strlen(chr) + 1;
        regidx_push(gene_index, chr, chr_end, beg, end, &gene);
    }

    *idx = gene_index;

    return EXIT_SUCCESS;
}

int score_allele_for_gene(Model *models, faidx_t *fa, const Reference *ref, GeneReference *gene_reference, int distance, int cov, const bcf_hdr_t *hdr, bcf1_t *v, int allele_idx, Gene gene, int ref_len, kstring_t *info_str) {
    int alt_len = strlen(v->d.allele[allele_idx]);

    if (ref_len > distance || alt_len > distance) {
        log_warn("Oversized indel found. Skipping prediction for at %s:%d:%s", bcf_hdr_id2name(hdr, v->rid), v->pos, gene.name);
        kputc('.', info_str);
        return EXIT_SUCCESS;
    }

    if (strncmp(gene.name, gene_reference->name, FIELD_MAX_LEN) != 0 && gene_reference_update(gene.chrom, gene.name, fa, ref, gene_reference) == EXIT_FAILURE) {
        log_warn("Failed to find reference for gene %s. Skipping prediction for %s:%li:%s...", gene.name, bcf_hdr_id2name(hdr, v->rid), v->pos + 1, gene.name);
        kputc('.', info_str);
        return EXIT_SUCCESS;
    }

    int num_ref_predictions = cov * NUM_SCORES;
    float *ref_predictions;
    if (gene_reference_get_score_window(v->pos, distance, gene_reference, &ref_predictions) == EXIT_FAILURE) {
        // Memory issue, already logged by function
        return EXIT_FAILURE;
    }

    // Predict alt
    int num_alt_predictions;
    float *alt_predictions;

    // The model trims BOUNDARY_SIZE (half of CONTEXT_SIZE) of context from each side of its
    // input to produce predictions for the remaining "core" region, so the window fetched here
    // must reserve BOUNDARY_SIZE (not the full CONTEXT_SIZE) of margin on each side of the
    // cov-wide region of interest for padded_seq's width (CONTEXT_SIZE + cov) to hold it.
    hts_pos_t gene_pos = v->pos - gene.tx_start;
    hts_pos_t start = gene_pos - (BOUNDARY_SIZE + distance);
    hts_pos_t start_offset = 0;
    if (start < 0) {
        start_offset = -start;
        start = 0;
    }

    hts_pos_t gene_length = gene.tx_end - gene.tx_start;
    hts_pos_t end = gene_pos + (BOUNDARY_SIZE + distance + 1);
    if (end > gene_length) {
        end = gene_length;
    }

    const int width = CONTEXT_SIZE + cov;
    char padded_seq[width];
    memset(padded_seq, 'N', width);
    memcpy(padded_seq + start_offset, gene_reference->seq.s + start, end - start);

    if (predict_padded_sequence(models, padded_seq, width, gene_reference->strand, &alt_predictions, &num_alt_predictions) != EXIT_SUCCESS) {
        free(ref_predictions);
        return EXIT_FAILURE;
    }

    // Align
    if (alt_len != ref_len) {
        align_predictions_alt_to_ref(distance, cov, ref_len, alt_len, &alt_predictions);
    }

    // Form scores
    Score score = calculate_delta_scores(v->d.allele[allele_idx], gene.name, ref_predictions, alt_predictions, num_ref_predictions, distance);
    free(ref_predictions);
    free(alt_predictions);

    const char *score_fmt = "%s|%s|%.2f|%.2f|%.2f|%.2f|%d|%d|%d|%d";
    int tmp_len = snprintf(NULL, 0, score_fmt, score.alt, score.gene, score.ag, score.al, score.dg, score.dl, score.ag_idx, score.al_idx, score.dg_idx, score.dl_idx);
    char *tmp = malloc(tmp_len + 1);
    if (tmp == NULL) {
        log_fatal("Failed to allocate %d bytes for score string", tmp_len + 1);
        exit(EXIT_FAILURE);
    }
    snprintf(tmp, tmp_len + 1, score_fmt, score.alt, score.gene, score.ag, score.al, score.dg, score.dl, score.ag_idx, score.al_idx, score.dg_idx, score.dl_idx);
    kputs(tmp, info_str);
    free(tmp);

    return EXIT_SUCCESS;
}

int process_variant_record(Model *models, faidx_t *fa, const Reference *ref, GeneReference *gene_reference, int distance, int cov, regidx_t *gene_index, regitr_t *itr, bcf_hdr_t *hdr, bcf1_t *v, htsFile *vcf_out, const char *annotated_variants) {
    bcf_unpack(v, BCF_UN_STR);
    int ref_len = strlen(v->d.allele[0]);
    // INFO: Could skip here if ref_len > distance, but this logic is kept with alt_len > distance for consistency in output

    if (!regidx_overlap(gene_index, bcf_hdr_id2name(hdr, v->rid), v->pos, v->pos+1, itr)) {
        if (bcf_write(vcf_out, hdr, v) != EXIT_SUCCESS) log_error("Writing failed for file: %s", annotated_variants);
        return EXIT_SUCCESS;
    }

    kstring_t info_str = {0};
    for (int i = 1; i < v->n_allele; i++) {
        if ('.' == v->d.allele[i][0] || // Deletion
            '*' == v->d.allele[i][0] || // Missing
            '<' == v->d.allele[i][0] // <ID> string
        ) {
            log_warn("Unsupported alternate allele found: %s. Skipping prediction(s) for %s:%li", v->d.allele[i], bcf_hdr_id2name(hdr, v->rid), v->pos+1);
            if (info_str.l > 0) kputc(',', &info_str);
            kputc('.', &info_str);
            continue;
        }

        while (regitr_overlap(itr)) {
            if (info_str.l > 0) kputc(',', &info_str);

            Gene gene = regitr_payload(itr, Gene);

            if (score_allele_for_gene(models, fa, ref, gene_reference, distance, cov, hdr, v, i, gene, ref_len, &info_str) != EXIT_SUCCESS) {
                free(info_str.s);
                return EXIT_FAILURE;
            }
        }
    }

    if (info_str.l > 0) {
        bcf_update_info_string(hdr, v, SPLICEAI_TAG, info_str.s);
        free(info_str.s);
    }

    if (bcf_write(vcf_out, hdr, v) != EXIT_SUCCESS) {
        log_error("Writing failed for file: %s", annotated_variants);
    }

    return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
    setenv("TF_CPP_MIN_LOG_LEVEL", "2", 1);
    setenv("NVIDIA_TF32_OVERRIDE", "1", 1);
    setenv("TF_CUDNN_USE_AUTOTUNE", "0", 1);

    args_t args = make_default_args();
    if (!parse_args(argc, argv, &args) || args.help) {
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    const char *variants = args.variants;
    const char *reference_bin = args.reference_bin;
    const char *model_dir = args.model_dir;
    const char *fasta = args.fasta;
    const char *gene_regions = args.regions;
    const char *annotated_variants = args.output;

    const int distance = args.window_size;
    const char *prediction_output = args.splice_output;
    const bool produce_splice_output = strncmp(prediction_output, "\0", 1) == 0;

    // Load SpliceAI models
    Model *models = load_models(model_dir);

    Reference ref;
    if (reference_read(reference_bin, &ref) != EXIT_SUCCESS) {
        log_error("Failed to read reference scores binary: %s", reference_bin);
        return EXIT_FAILURE;
    }

    faidx_t *fa_in;
    if ((fa_in = fai_load(fasta)) == NULL) return EXIT_FAILURE; // Load reference fasta for sequence lookup

    htsFile *vcf_in; bcf_hdr_t *hdr;
    if (open_input_vcf(variants, &vcf_in, &hdr) != EXIT_SUCCESS) return EXIT_FAILURE; // Load input vcf


    regidx_t *gene_index = NULL;
    gene_regions_build_regidx(gene_regions, &gene_index);
    if (gene_index == NULL) {
        return EXIT_FAILURE;
    }

    htsFile *vcf_out;
    if (prepare_output_vcf(annotated_variants, hdr, &vcf_out) != EXIT_SUCCESS) return EXIT_FAILURE; // Prepare output vcf

    // Sequence size
    const int cov = 2 * distance + 1;

    // Loop initialisations
    regitr_t *itr = regitr_init(gene_index);
    bcf1_t *v = bcf_init();
    GeneReference gene_reference;
    gene_reference_init(&gene_reference);
    while (bcf_read(vcf_in, hdr, v) >= 0) {
        if (!v) continue; // TODO: Can this even happen given bcf_init? Also, need to handle the non critical reading errors.

        if (process_variant_record(models, fa_in, &ref, &gene_reference, distance, cov, gene_index, itr, hdr, v, vcf_out, annotated_variants) != EXIT_SUCCESS) {
            return EXIT_FAILURE;
        }
    }

    bcf_destroy(v);
    regitr_destroy(itr);
    fai_destroy(fa_in);
    hts_close(vcf_in); hts_close(vcf_out);
    bcf_hdr_destroy(hdr);

    destroy_models(models);

    return EXIT_SUCCESS;
}

