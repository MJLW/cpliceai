#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <klib/kstring.h>
#include <klib/kvec.h>
#include <klib/khash.h>
#include <htslib/hts.h>
#include <htslib/faidx.h>
#include <htslib/tbx.h>
#include <htslib/regidx.h>

#include "../logging/log.h"
#include "../predict.h"
#include "../reference.h"
#include "../gene_regions.h"
#include "../utils.h"


#define REQUIRED_ARGS \
    REQUIRED_STRING_ARG(model_dir, "model_dir", "Directory containing SpliceAI models") \
    REQUIRED_STRING_ARG(fasta, "fasta", "Human reference fasta") \
    REQUIRED_STRING_ARG(regions, "regions", "Gene region structure parsed from GFF with gff_to_bed.py") \
    REQUIRED_STRING_ARG(output, "output", "Binary output for reference")

#define BOOLEAN_ARGS \
    BOOLEAN_ARG(help, "-h", "Show help")

#include <easyargs.h>


int init_ref_from_gene_region_scan(FILE *gene_regions, Reference *ref) {
    Gene gene = { 0 };
    int ret = read_gene_region(gene_regions, &gene);

    uint32_t n_contigs = 1;
    uint32_t contig_bytes = strnlen(gene.chrom, FIELD_MAX_LEN) + 1;
    char *current_contig = strdup(gene.chrom);

    uint32_t n_regions = 1;
    uint32_t region_bytes = strnlen(gene.name, FIELD_MAX_LEN) + 1;

    uint64_t region_size = gene.tx_end - gene.tx_start;
    uint64_t n_chunks = (region_size / CHUNK_SIZE) + ((region_size % CHUNK_SIZE) > 0);

    while ((ret = read_gene_region(gene_regions, &gene)) == 0) {
        if (strncmp(current_contig, gene.chrom, FIELD_MAX_LEN) != 0) {
            n_contigs++;
            contig_bytes += strnlen(gene.chrom, FIELD_MAX_LEN) + 1;

            free(current_contig);
            current_contig = strdup(gene.chrom);
        }

        n_regions++;
        region_bytes += strnlen(gene.name, FIELD_MAX_LEN) + 1;

        region_size = gene.tx_end - gene.tx_start;
        n_chunks += (region_size / CHUNK_SIZE) + ((region_size % CHUNK_SIZE) > 0);
    }

    free(current_contig);
    rewind(gene_regions);

    reference_alloc(ref, n_contigs, contig_bytes, n_regions, region_bytes, n_chunks, n_chunks);

    return EXIT_SUCCESS;
}

int process_gene_region(Model *models, faidx_t *fa, const Gene *gene, Reference *ref) {
    int slen;
    char *seq = faidx_fetch_seq(fa, gene->chrom, (int) gene->tx_start, (int) gene->tx_end, &slen);
    seq[slen] = '\0';

    // Add BOUNDAR_SIZE'd padding to the gene sequence, so that each position of the gene gets a prediction
    int padded_slen = slen + CONTEXT_SIZE;
    char *padded_seq = malloc(padded_slen);
    if (padded_seq == NULL) {
        log_fatal("Failed to allocate %d bytes for padded sequence", padded_slen);
        exit(EXIT_FAILURE);
    }
    memset(padded_seq, 'N', BOUNDARY_SIZE); // Prepend with 5000 Ns
    memcpy(padded_seq + BOUNDARY_SIZE, seq, slen);
    memset(padded_seq + (padded_slen - BOUNDARY_SIZE), 'N', BOUNDARY_SIZE); // Append with 5000 Ns
    free(seq);

    int num_gene_predictions;
    float *gene_predictions;
    if (predict_padded_sequence(models, padded_seq, padded_slen, gene->strand, &gene_predictions, &num_gene_predictions) != EXIT_SUCCESS) {
        free(padded_seq);
        return EXIT_FAILURE;
    }
    free(padded_seq);

    log_info("%s\t%li\t%li\t%s\t%c\t%i", gene->chrom, gene->tx_start, gene->tx_end, gene->name, gene->strand, slen);

    uint64_t size = gene->tx_end - gene->tx_start;

    // Turn predictions into chunks and PositionScores
    for (uint64_t i = 0; i < size; i += CHUNK_SIZE) {
        Chunk chunk = { .n_scores = 0, .scores_start = ref->n_scores };
        reference_add_chunk(chunk, ref);

        // Iterate over scores in chunk
        for (uint64_t j = i; j < i + CHUNK_SIZE && j < size; j++) {
            size_t prediction_index = j * NUM_SCORES;
            if (gene_predictions[prediction_index + ACCEPTOR_POS] < ZERO_EPSILON && gene_predictions[prediction_index + DONOR_POS] < ZERO_EPSILON) continue;

            PositionScore score = { .pos = j, .acceptor = gene_predictions[prediction_index + ACCEPTOR_POS], .donor = gene_predictions[prediction_index + DONOR_POS] };
            reference_add_score(score, ref);
        }
    }

    free(gene_predictions);

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

    const char *model_dir = args.model_dir;
    const char *fasta = args.fasta;
    const char *gene_regions = args.regions;
    const char *output_path = args.output;

    Model *models = load_models(model_dir);

    // Load annotations for transcript regions
    FILE *gene_regions_in = open_file_or_log(gene_regions, "r");
    if (gene_regions_in == NULL) return EXIT_FAILURE;

    Reference ref;
    init_ref_from_gene_region_scan(gene_regions_in, &ref);

    // Load reference fasta for sequence lookup
    faidx_t *fa_in;
    if ((fa_in = fai_load(fasta)) == NULL) {
        log_error("Failed to read fasta: %s", fasta);
        return EXIT_FAILURE; // Load reference fasta for sequence lookup
    }

    // Loop over all regions
    Gene gene = { 0 };
    int ret;
    char *current_region = NULL;

    while ((ret = read_gene_region(gene_regions_in, &gene)) == 0) {
        if (current_region == NULL || strncmp(gene.chrom, current_region, FIELD_MAX_LEN) != 0) {
            Contig contig = { .n_regions = 0, .region_start = ref.n_genes, .name_start = ref.contig_names_len};
            reference_add_contig(contig, gene.chrom, &ref);

            if (current_region != NULL) free(current_region);
            current_region = strndup(gene.chrom, FIELD_MAX_LEN);
        }

        Region region = { .n_chunks = 0, .chunk_size = CHUNK_SIZE, .size = gene.tx_end - gene.tx_start, .strand = gene.strand, .chunk_start = ref.n_chunks, .name_start = ref.region_names_len };
        reference_add_region(region, gene.name, gene.tx_start, gene.tx_end, &ref);

        if (process_gene_region(models, fa_in, &gene, &ref) != EXIT_SUCCESS) {
            return EXIT_FAILURE;
        }
    }

    free(current_region);

    log_info("Writing out binarized reference to: %s", output_path);
    reference_write(output_path, &ref);
    reference_free(&ref);

    fclose(gene_regions_in);
    fai_destroy(fa_in);
    destroy_models(models);

    return EXIT_SUCCESS;
}

