#include <stdlib.h>
#include <stdio.h>

#include <htslib/faidx.h>

#include "predict.h"
#include "reference.h"


#define REQUIRED_ARGS \
    REQUIRED_STRING_ARG(input_vcf, "vcf", "VCF file to predict for") \
    REQUIRED_STRING_ARG(reference_bin, "reference_scores", "Binary file containing reference scores") \
    REQUIRED_STRING_ARG(model_dir, "model_dir", "Directory containing SpliceAI models") \
    REQUIRED_STRING_ARG(fasta, "fasta", "Human reference fasta") \
    REQUIRED_STRING_ARG(regions, "regions", "Gene region structure parsed from GFF with gff_to_bed.py") \
    REQUIRED_STRING_ARG(output, "output", "TSV of splice sites found, where REF>0.2|ALT>0.2.")

#define BOOLEAN_ARGS \
    BOOLEAN_ARG(help, "-h", "Show help")

#include <easyargs.h>



int main(int argc, char *argv[]) {
    setenv("TF_CPP_MIN_LOG_LEVEL", "1", 1);

    // Parse arguments
    args_t args = make_default_args();
    if (!parse_args(argc, argv, &args) || args.help) {
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    // Load SpliceAI tensorflow models
    Model *models = load_models(args.model_dir);

    // Load annotations for transcript regions
    FILE *gene_regions_in;
    gene_regions_in = fopen(args.regions, "r");
    // TODO: Parse gene regions into a HashMap
    if (gene_regions_in == NULL) return EXIT_FAILURE;

    Reference ref;
    reference_read(args.reference_bin, &ref);

    // Load reference fasta for sequence lookup
    faidx_t *fa_in;
    if ((fa_in = fai_load(args.fasta)) < 0) return EXIT_FAILURE; // Load reference fasta for sequence lookup


    // Loop over VCF variants
    // Group variants to a gene
    // For each gene, get reference scores
    // For variants on gene, predict splice sites given variant
    // Encode changes made by each variants sparsely by offsets from median score
    // 




}

