#ifndef GENE_REGIONS_H
#define GENE_REGIONS_H

#include <stdint.h>
#include <stdio.h>

#define FIELD_MAX_LEN 256

#define POSITIVE_STRAND '+'
#define NEGATIVE_STRAND '-'

typedef struct {
    char    name[FIELD_MAX_LEN];
    char    chrom[FIELD_MAX_LEN];
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
int read_gene_region(FILE *fp, Gene *gene);

/*
 * next_tsv_field - Consume the next tab-delimited field from *cursor, in place (mutates the
 * underlying buffer by writing a '\0' at the delimiter, like strtok). Advances *cursor past the
 * field so the next call returns the following field.
 *
 * Parameters:
 *   cursor - pointer to the current position in a mutable, NUL-terminated buffer.
 *
 * Returns a pointer to the NUL-terminated field, or NULL if no field remains.
 */
char *next_tsv_field(char **cursor);


#endif /* GENE_REGIONS_H */
