#ifndef GENE_REGIONS_H
#define GENE_REGIONS_H

#include <stdint.h>
#include <stdio.h>

#define GENE_NAME_MAX  256
#define GENE_CHROM_MAX 256

#define POSITIVE_STRAND '+'
#define NEGATIVE_STRAND '-'

typedef struct {
    char    name[GENE_NAME_MAX];
    char    chrom[GENE_CHROM_MAX];
    char    strand;
    int64_t tx_start;
    int64_t tx_end;
} Gene;

/*
 * read_gene_region - Parse the next line containing a tab-delimited gene region into a Gene struct. Lines beginning with '#' are skipped.
 *
 * Parameters:
 *   fp        - File pointer for the input file.
 *   gene      - gene pointer to populate
 *
 * Returns EXIT_SUCCESS on success, -1 on file end, EXIT_FAILURE on parsing failure.
 */
extern int read_gene_region(FILE *fp, Gene *gene);


#endif /* GENE_REGIONS_H */
