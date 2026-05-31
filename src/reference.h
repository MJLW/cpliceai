#ifndef REFERENCE_H
#define REFERENCE_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>

#define REF_MAGIC   0x52454653434F5245ULL  // "REFSCORE"
#define REF_VERSION 1

#define CHUNK_SIZE 256
#define ZERO_EPSILON 1e-3

typedef struct {
    uint32_t pos;
    float acceptor;
    float donor;
} PositionScore;

typedef struct {
    uint16_t n_scores;
    uint32_t scores_start;
} Chunk;

typedef struct {
    uint16_t n_chunks;
    uint32_t chunk_start;
    uint32_t name_start;

    uint32_t size;
    uint16_t chunk_size;
} Region;

typedef struct {
    uint32_t n_regions;
    uint32_t region_start;
    uint32_t name_start;
} Contig;

typedef struct {
    size_t block_size;
    char *block;

    size_t n_contigs, m_contigs;
    Contig *contigs;

    size_t contig_names_len, contig_names_cap;
    char *contig_names;

    size_t n_genes, m_genes;
    Region *genes;

    uint64_t *gene_starts;
    uint64_t *gene_ends;

    size_t region_names_len, region_names_cap;
    char *region_names;

    size_t n_chunks, m_chunks;
    Chunk *chunks;

    size_t n_scores, m_scores;
    PositionScore *scores;
} Reference;

typedef struct {
    uint64_t magic;
    uint64_t version;
    uint64_t file_size;
    // counts
    uint64_t n_contigs;
    uint64_t contig_names_len;
    uint64_t n_genes;
    uint64_t n_chunks;
    uint64_t n_scores;
    uint64_t m_scores;
    uint64_t block_size;
    uint64_t region_names_len;
    // offsets from start of file
    uint64_t off_contigs;
    uint64_t off_contig_names;
    uint64_t off_genes;
    uint64_t off_gene_starts;
    uint64_t off_gene_ends;
    uint64_t off_region_names;
    uint64_t off_chunks;
    uint64_t off_scores;
    uint64_t off_block;
} FileHeader;

int reference_alloc(
    Reference *ref,
    size_t m_contigs,
    size_t contig_names_cap,
    size_t m_genes,
    size_t region_names_cap,
    size_t m_chunks,
    size_t m_scores
);

/*
 * MANIPULATIONS
 */
void reference_free(Reference *ref);

void reference_add_score(const PositionScore score, Reference *ref);

void reference_add_chunk(const Chunk chunk, Reference *ref);

void reference_add_region(const Region region, const char *name, Reference *ref);

void reference_add_contig(const Contig contig, const char *name, Reference *ref);


/*
* IO
*/
int reference_write(const char *path, const Reference *ref);

int reference_read(const char *path, Reference *ref);

void reference_unmap(char *base, size_t size);

#endif
