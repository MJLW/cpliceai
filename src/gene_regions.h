#ifndef GENE_REGIONS_H
#define GENE_REGIONS_H

#include <stdint.h>
#include <stdio.h>

#include <htslib/hts.h>
#include <htslib/regidx.h>

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

/* gene_region_reader_next returns this, rather than EXIT_FAILURE, at a clean end of input. */
#define GENE_REGION_EOF (-1)

/*
 * Reader over a gene regions file:
 *
 *     NAME	CHROM	STRAND	TX_START	TX_END	[EXON_START	EXON_END]
 *
 * Plain or compressed (gzip/BGZF) - it is opened through htslib, which decompresses
 * transparently. A header line is optional and detected by content: if TX_START on the first
 * record-shaped line does not parse as a coordinate, that line is a header. Lines beginning
 * with '#' are also skipped, for files that mark their header that way.
 */
typedef struct GeneRegionReader GeneRegionReader;

int gene_region_reader_open(const char *path, GeneRegionReader **reader);

/*
 * Returns EXIT_SUCCESS, GENE_REGION_EOF at end of input, or EXIT_FAILURE (having logged) on a
 * malformed line.
 */
int gene_region_reader_next(GeneRegionReader *reader, Gene *gene);

/*
 * gene_region_reader_digest - Fingerprint of every gene read so far.
 *
 * Accumulated over the parsed fields rather than the raw bytes, so it identifies the gene set
 * itself and is unaffected by compression, a header being present or absent, or whitespace.
 * Call after reading to the end. Compared against the value stored in a reference scores file
 * to catch a regions file that does not match the one it was built from.
 */
uint64_t gene_region_reader_digest(const GeneRegionReader *reader);

void gene_region_reader_close(GeneRegionReader *reader);

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

/*
 * A growable list of genes, as returned by gene_regions_overlapping. Reuse one across
 * records rather than allocating per record; gene_regions_overlapping resets it each call.
 */
typedef struct {
    Gene *genes;
    size_t n, m;
} GeneList;

void gene_list_init(GeneList *list);

void gene_list_destroy(GeneList *list);

/*
 * gene_regions_build_regidx - Build an interval index of Gene payloads from a regions file.
 *
 * Parameters:
 *   path   - Gene region structure parsed from GFF with gff_to_bed.py.
 *   idx    - receives the index, which the caller must regidx_destroy.
 *   digest - receives the gene set's fingerprint (see gene_region_reader_digest); may be NULL.
 *
 * Returns EXIT_SUCCESS on success, EXIT_FAILURE (having logged) otherwise.
 */
int gene_regions_build_regidx(const char *path, regidx_t **idx, uint64_t *digest);

/*
 * gene_regions_containing - Collect every gene that fully contains a variant.
 *
 * A variant occupies [pos, pos + ref_len) and must lie entirely inside a gene's
 * [tx_start, tx_end) to be scored against it. Merely overlapping is not enough: an indel
 * anchored inside a gene but running past its end has no reference sequence to be compared
 * against beyond the boundary.
 *
 * The matches are copied into *out rather than left behind an iterator, so that callers can
 * walk them repeatedly - once per alternate allele, say - without re-querying.
 *
 * Parameters:
 *   idx     - index built by gene_regions_build_regidx.
 *   itr     - scratch iterator, from regitr_init(idx).
 *   chrom   - contig name.
 *   pos     - 0-based start of the variant.
 *   ref_len - length of the REF allele, i.e. how many reference bases it spans.
 *   out     - receives the matches; reset on every call.
 *
 * Returns the number of containing genes.
 */
size_t gene_regions_containing(regidx_t *idx, regitr_t *itr, const char *chrom,
                               hts_pos_t pos, int ref_len, GeneList *out);


#endif /* GENE_REGIONS_H */
