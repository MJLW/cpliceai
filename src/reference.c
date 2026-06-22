#include "reference.h"
#include "logging/log.h"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


void reference_free(Reference *ref) {
    if (ref->block != NULL) free(ref->block);
}
 
static inline size_t align_up(size_t offset, size_t align) {
    return (offset + align - 1) & ~(align - 1);
}

int reference_alloc(
    Reference *ref,
    size_t m_contigs,
    size_t contig_names_cap,
    size_t m_genes,
    size_t region_names_cap,
    size_t m_chunks,
    size_t m_scores
) {
    size_t off_contigs = 0;
    size_t off_contig_names = align_up(off_contigs + m_contigs * sizeof(Contig), _Alignof(char));
    size_t off_genes = align_up(off_contig_names + contig_names_cap, _Alignof(Region));
    size_t off_gene_starts = align_up(off_genes + m_genes * sizeof(Region), _Alignof(uint64_t));
    size_t off_gene_ends = align_up(off_gene_starts + m_genes * sizeof(uint64_t), _Alignof(uint64_t));
    size_t off_region_names = align_up(off_gene_ends + m_genes * sizeof(uint64_t), _Alignof(char));
    size_t off_chunks = align_up(off_region_names + region_names_cap, _Alignof(Chunk));
    size_t off_scores = align_up(off_chunks + m_chunks * sizeof(Chunk), _Alignof(PositionScore));
    size_t total = off_scores + m_scores * sizeof(PositionScore);

    char *block = malloc(total);
    if (block == NULL) return EXIT_FAILURE;

    ref->block = block;
    ref->block_size = total;
    ref->contigs = (Contig *) (block + off_contigs);
    ref->contig_names = (char *) (block + off_contig_names);
    ref->genes = (Region *) (block + off_genes);
    ref->gene_starts = (uint64_t *) (block + off_gene_starts);
    ref->gene_ends = (uint64_t *) (block + off_gene_ends);
    ref->region_names = (char *) (block + off_region_names);
    ref->chunks = (Chunk *) (block + off_chunks);
    ref->scores = (PositionScore *) (block + off_scores);

    ref->n_contigs = 0;
    ref->m_contigs = m_contigs;
    ref->contig_names_len = 0;
    ref->contig_names_cap = contig_names_cap;
    ref->n_genes = 0;
    ref->m_genes = m_genes;
    ref->region_names_len = 0;
    ref->region_names_cap = region_names_cap;
    ref->n_chunks = 0;
    ref->m_chunks = m_chunks;
    ref->n_scores = 0;
    ref->m_scores = m_scores;

    return EXIT_SUCCESS;
}

int reference_resize(Reference *ref) {
    size_t new_block_size = ref->block_size + ref->m_scores * sizeof(PositionScore);

    const char *old_block_ptr = ref->block;
    ref->block = realloc(ref->block, new_block_size);
    if (ref->block == NULL) {
        return EXIT_FAILURE;
    }

    ref->block_size = new_block_size;
    ref->m_scores += ref->m_scores;

    if (ref->block == old_block_ptr) {
        return EXIT_SUCCESS;
    }
 
    // Realloc has moved the block! Move interior pointers to new allocation
    const ptrdiff_t ptr_offset = (ref->block - old_block_ptr);
    ref->contigs = (Contig *) ((char *) ref->contigs + ptr_offset);
    ref->contig_names = ref->contig_names + ptr_offset;
    ref->genes = (Region *) ((char *) ref->genes + ptr_offset);
    ref->gene_starts = (uint64_t *) ((char *) ref->gene_starts + ptr_offset);
    ref->gene_ends = (uint64_t *) ((char *) ref->gene_ends + ptr_offset);
    ref->region_names = ref->region_names + ptr_offset;
    ref->chunks = (Chunk *) ((char *) ref->chunks + ptr_offset);
    ref->scores = (PositionScore *) ((char *) ref->scores + ptr_offset);

    return EXIT_SUCCESS;
}

void reference_add_score(const PositionScore score, Reference *ref) {
    if (ref->n_scores == ref->m_scores) {
        reference_resize(ref);
    }

    ref->scores[ref->n_scores++] = score;
    ref->chunks[ref->n_chunks-1].n_scores++;
}

void reference_add_chunk(const Chunk chunk, Reference *ref) {
    ref->chunks[ref->n_chunks++] = chunk;
    ref->genes[ref->n_genes-1].n_chunks++;
}

void reference_add_region(const Region region, const char *name, const int64_t start, const int64_t end, Reference *ref) {
    ref->genes[ref->n_genes] = region;
    ref->gene_starts[ref->n_genes] = start;
    ref->gene_ends[ref->n_genes] = end;
    ref->n_genes++;
    ref->contigs[ref->n_contigs-1].n_regions++;

    size_t len = strlen(name) + 1;
    memcpy(ref->region_names + ref->region_names_len, name, len);
    ref->region_names_len += len;
}

void reference_add_contig(const Contig contig, const char *name, Reference *ref) {
    ref->contigs[ref->n_contigs++] = contig;

    size_t len = strlen(name) + 1;
    memcpy(ref->contig_names + ref->contig_names_len, name, len);
    ref->contig_names_len += len;
}

int reference_write(const char *path, const Reference *ref) {
    // --- Calculate offsets ---
    size_t off = sizeof(FileHeader);

    size_t off_block = off;

    size_t off_contigs = align_up(off, _Alignof(Contig));
    off = off_contigs + ref->m_contigs * sizeof(Contig);

    size_t off_contig_names = off;  // char[], no alignment needed
    off += ref->contig_names_cap;

    size_t off_genes = align_up(off, _Alignof(Region));
    off = off_genes + ref->m_genes * sizeof(Region);

    size_t off_gene_starts = align_up(off, _Alignof(uint64_t));
    off = off_gene_starts + ref->m_genes * sizeof(uint64_t);

    size_t off_gene_ends = align_up(off, _Alignof(uint64_t));
    off = off_gene_ends + ref->m_genes * sizeof(uint64_t);

    size_t off_region_names = off;
    off += ref->region_names_cap;

    size_t off_chunks = align_up(off, _Alignof(Chunk));
    off = off_chunks + ref->m_chunks * sizeof(Chunk);

    size_t off_scores = align_up(off, _Alignof(PositionScore));
    off = off_scores + ref->m_scores * sizeof(PositionScore);

    size_t file_size = off;

    // --- Open and size the file ---
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0 ) {
        log_error("Failed to open: %s", path);
        return EXIT_FAILURE;
    }
    if (ftruncate(fd, file_size) < 0) {
        log_error("Failed to truncate file.");
        close(fd);
        return EXIT_FAILURE;
    }

    char *base = mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (base == MAP_FAILED) {
        log_error("Failed to open MMAP sync to: %s", path);
        return EXIT_FAILURE;
    }

    // Write header
    FileHeader hdr = {
        .magic           = REF_MAGIC,
        .version         = REF_VERSION,
        .file_size       = file_size,
        .n_contigs       = ref->n_contigs,
        .contig_names_len= ref->contig_names_len,
        .n_genes         = ref->n_genes,
        .n_chunks        = ref->n_chunks,
        .n_scores        = ref->n_scores,
        .m_scores        = ref->m_scores,
        .block_size      = ref->block_size,
        .region_names_len= ref->region_names_len,
        .off_contigs     = off_contigs,
        .off_contig_names= off_contig_names,
        .off_genes       = off_genes,
        .off_gene_starts = off_gene_starts,
        .off_gene_ends   = off_gene_ends,
        .off_region_names= off_region_names,
        .off_chunks      = off_chunks,
        .off_scores      = off_scores,
        .off_block       = off_block
    };

    memcpy(base, &hdr, sizeof(hdr));
    memcpy(base + off_block, ref->block, ref->block_size);

    // Clear mmap
    munmap(base, file_size);

    return 0;
}

int reference_read(const char *path, Reference *ref) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return -1; }

    FileHeader hdr;
    if (read(fd, &hdr, sizeof(hdr)) != sizeof(hdr)) { close(fd); return -1; }

    if (hdr.magic != REF_MAGIC || hdr.version != REF_VERSION) {
        fprintf(stderr, "Bad magic or version\n");
        close(fd);
        return -1;
    }

    char *base = mmap(NULL, hdr.file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) { perror("mmap"); return -1; }

    // Hint to the kernel: we'll scan sequentially
    madvise(base, hdr.file_size, MADV_SEQUENTIAL);

    // Parse structure from header
    ref->n_contigs        = hdr.n_contigs;
    ref->m_contigs        = hdr.n_contigs;
    ref->contig_names_len = hdr.contig_names_len;
    ref->contig_names_cap = hdr.contig_names_len;
    ref->n_genes          = hdr.n_genes;
    ref->m_genes          = hdr.n_genes;
    ref->n_chunks         = hdr.n_chunks;
    ref->m_chunks         = hdr.n_chunks;
    ref->n_scores         = hdr.n_scores;
    ref->m_scores         = hdr.m_scores;
    ref->block_size       = hdr.block_size;
    ref->region_names_len = hdr.region_names_len;
    ref->region_names_cap = hdr.region_names_len;

    // Populate structure
    ref->block         = (char *)          (base + hdr.off_block);
    ref->contigs       = (Contig *)        (base + hdr.off_contigs);
    ref->contig_names  = (char *)          (base + hdr.off_contig_names);
    ref->genes         = (Region *)        (base + hdr.off_genes);
    ref->gene_starts   = (uint64_t *)      (base + hdr.off_gene_starts);
    ref->gene_ends     = (uint64_t *)      (base + hdr.off_gene_ends);
    ref->region_names  = (char *)          (base + hdr.off_region_names);
    ref->chunks        = (Chunk *)         (base + hdr.off_chunks);
    ref->scores        = (PositionScore *) (base + hdr.off_scores);

    return 0;
}

void reference_unmap(char *base, size_t size) {
    munmap(base, size);
}
