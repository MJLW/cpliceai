#include "gene_regions.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "logging/log.h"
#include "utils.h"

#define MAX_LINE  8192

char *next_tsv_field(char **cursor) {
    if (*cursor == NULL) return NULL;

    char *field = *cursor;
    char *tab = strchr(field, '\t');
    if (tab != NULL) {
        *tab = '\0';
        *cursor = tab + 1;
    } else {
        *cursor = NULL;
    }

    return field;
}

/* parse_line's verdict when a coordinate is unparseable on a line that could be a header. */
#define PARSE_NOT_A_RECORD (-2)

/* strtoll with an end check: atoll would silently turn "TX_START" into 0. */
static int parse_coordinate(const char *tok, int64_t *out) {
    char *end;
    long long value = strtoll(tok, &end, 10);
    if (end == tok || end[0] != '\0' || value < 0) return EXIT_FAILURE;
    *out = (int64_t) value;
    return EXIT_SUCCESS;
}

/*
 * Parse one gene region line: NAME CHROM STRAND TX_START TX_END [EXON_START EXON_END].
 *
 * Mutates line in place (next_tsv_field is strtok-style).
 *
 * Returns PARSE_NOT_A_RECORD, without logging, when a coordinate does not parse - the caller
 * decides whether that means "this is the header" or "this line is broken".
 */
static int parse_line(char *buf, Gene *gene) {
    char *cursor = buf;
    char *tok;

    tok = next_tsv_field(&cursor);
    if (!tok) return EXIT_FAILURE;
    strncpy(gene->name, tok, FIELD_MAX_LEN - 1);
    gene->name[FIELD_MAX_LEN - 1] = '\0';

    tok = next_tsv_field(&cursor);
    if (!tok) return EXIT_FAILURE;
    strncpy(gene->chrom, tok, FIELD_MAX_LEN - 1);
    gene->chrom[FIELD_MAX_LEN - 1] = '\0';

    tok = next_tsv_field(&cursor);
    if (!tok) return EXIT_FAILURE;
    gene->strand = tok[0];

    /* TX_START/TX_END are already 0-based half-open (BED), as produced by gff_to_bed.py:
       TX_START is the 1-based start minus one, and TX_END is the 1-based end unchanged, which
       under 0-based indexing is the exclusive end. Neither needs converting. */
    tok = next_tsv_field(&cursor);
    if (!tok) return EXIT_FAILURE;
    if (parse_coordinate(tok, &gene->tx_start) != EXIT_SUCCESS) return PARSE_NOT_A_RECORD;

    tok = next_tsv_field(&cursor);
    if (!tok) return EXIT_FAILURE;
    if (parse_coordinate(tok, &gene->tx_end) != EXIT_SUCCESS) return PARSE_NOT_A_RECORD;

    return EXIT_SUCCESS;
}

struct GeneRegionReader {
    char *path;
    htsFile *fp;
    kstring_t line;
    uint64_t line_no;
    bool checked_first_line; /* gates header detection to exactly one line */
    uint64_t digest;
};

int gene_region_reader_open(const char *path, GeneRegionReader **reader) {
    GeneRegionReader *r = calloc(1, sizeof(GeneRegionReader));
    if (r == NULL) {
        log_fatal("Failed to allocate %zu bytes for gene region reader", sizeof(GeneRegionReader));
        exit(EXIT_FAILURE);
    }

    r->fp = hts_open(path, "r");
    if (r->fp == NULL) {
        log_error("Failed to open gene regions file: %s", path);
        free(r);
        return EXIT_FAILURE;
    }

    r->path = strdup(path);
    r->digest = DIGEST_SEED;
    *reader = r;

    return EXIT_SUCCESS;
}

uint64_t gene_region_reader_digest(const GeneRegionReader *reader) {
    return reader->digest;
}

int gene_region_reader_next(GeneRegionReader *reader, Gene *gene) {
    /* hts_getline documents its delimiter argument as unused but required to be '\n'. */
    while (hts_getline(reader->fp, '\n', &reader->line) >= 0) {
        reader->line_no++;

        const char *s = reader->line.s;
        if (reader->line.l == 0 || s[0] == '#' || s[0] == '\r') continue;

        /* hts_getline strips '\n' but leaves a '\r' from CRLF files behind. */
        if (reader->line.s[reader->line.l - 1] == '\r') {
            reader->line.s[--reader->line.l] = '\0';
        }

        /* Exactly one line is ever a header candidate. */
        const bool may_be_header = !reader->checked_first_line;
        reader->checked_first_line = true;

        int ret = parse_line(reader->line.s, gene);

        if (ret == PARSE_NOT_A_RECORD) {
            if (may_be_header) {
                log_info("Skipping presumed header line in %s", reader->path);
                continue;
            }
            log_error("%s:%lu: could not parse TX_START/TX_END as coordinates.",
                      reader->path, reader->line_no);
            return EXIT_FAILURE;
        }

        if (ret != EXIT_SUCCESS) {
            log_error("%s:%lu: failed to parse gene region.", reader->path, reader->line_no);
            return EXIT_FAILURE;
        }

        // Fingerprint the parsed gene rather than the line, so the digest is unaffected by
        // formatting, compression or a header being present.
        reader->digest = digest_update_str(reader->digest, gene->name);
        reader->digest = digest_update_str(reader->digest, gene->chrom);
        reader->digest = digest_update(reader->digest, &gene->strand, sizeof(gene->strand));
        reader->digest = digest_update_u64(reader->digest, (uint64_t) gene->tx_start);
        reader->digest = digest_update_u64(reader->digest, (uint64_t) gene->tx_end);

        return EXIT_SUCCESS;
    }

    return GENE_REGION_EOF;
}

void gene_region_reader_close(GeneRegionReader *reader) {
    if (reader == NULL) return;

    if (reader->fp != NULL) hts_close(reader->fp);
    free(reader->line.s);
    free(reader->path);
    free(reader);
}

void gene_list_init(GeneList *list) {
    list->genes = NULL;
    list->n = 0;
    list->m = 0;
}

void gene_list_destroy(GeneList *list) {
    free(list->genes);
    gene_list_init(list);
}

static void gene_list_push(GeneList *list, const Gene *gene) {
    if (list->n == list->m) {
        size_t m = list->m ? list->m * 2 : 8;
        Gene *genes = realloc(list->genes, m * sizeof(Gene));
        if (genes == NULL) {
            log_fatal("Failed to allocate %zu bytes for gene list", m * sizeof(Gene));
            exit(EXIT_FAILURE);
        }
        list->genes = genes;
        list->m = m;
    }

    list->genes[list->n++] = *gene;
}

int gene_regions_build_regidx(const char *path, regidx_t **idx, uint64_t *digest) {
    GeneRegionReader *reader;
    if (gene_region_reader_open(path, &reader) != EXIT_SUCCESS) return EXIT_FAILURE;

    regidx_t *gene_index = regidx_init(NULL, NULL, NULL, sizeof(Gene), NULL);
    if (gene_index == NULL) {
        log_error("Failed to allocate an interval index for regions file: %s", path);
        gene_region_reader_close(reader);
        return EXIT_FAILURE;
    }

    Gene gene;
    int ret;
    while ((ret = gene_region_reader_next(reader, &gene)) == EXIT_SUCCESS) {
        hts_pos_t beg = gene.tx_start, end = gene.tx_end;
        // htslib wants the first and last character of the name, not one past the end.
        char *chr = gene.chrom, *chr_end = chr + strlen(chr) - 1;
        regidx_push(gene_index, chr, chr_end, beg, end, &gene);
    }

    if (digest != NULL) *digest = gene_region_reader_digest(reader);
    gene_region_reader_close(reader);

    if (ret == EXIT_FAILURE) {
        regidx_destroy(gene_index);
        return EXIT_FAILURE;
    }

    *idx = gene_index;

    return EXIT_SUCCESS;
}

size_t gene_regions_containing(regidx_t *idx, regitr_t *itr, const char *chrom,
                               hts_pos_t pos, int ref_len, GeneList *out) {
    out->n = 0;

    // Querying the anchor base alone suffices: a gene containing the whole variant necessarily
    // contains pos, so this yields a superset of the matches.
    if (!regidx_overlap(idx, chrom, pos, pos + 1, itr)) return 0;

    while (regitr_overlap(itr)) {
        Gene gene = regitr_payload(itr, Gene);

        if (gene.tx_start <= pos && pos + ref_len <= gene.tx_end) {
            gene_list_push(out, &gene);
        } else {
            log_warn("Variant at %s:%"PRIhts_pos" spans %d bases and is not fully contained in "
                     "%s (%li-%li). Skipping prediction for this gene.",
                     chrom, pos + 1, ref_len, gene.name, gene.tx_start, gene.tx_end);
        }
    }

    return out->n;
}
