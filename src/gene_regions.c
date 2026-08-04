#include "gene_regions.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "logging/log.h"

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

static int parse_line(const char *line, Gene *gene) {
    char buf[MAX_LINE];
    strncpy(buf, line, MAX_LINE - 1);
    buf[MAX_LINE - 1] = '\0';
    buf[strcspn(buf, "\r\n")] = '\0';

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

    tok = next_tsv_field(&cursor);
    if (!tok) return EXIT_FAILURE;
    gene->tx_start = (int64_t)atoll(tok) - 1;

    tok = next_tsv_field(&cursor);
    if (!tok) return EXIT_FAILURE;
    gene->tx_end = (int64_t)atoll(tok) - 1;

    return EXIT_SUCCESS;
}

int read_gene_region(FILE *fp, Gene *gene) {
    char line[MAX_LINE];

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;

        if (parse_line(line, gene) != EXIT_SUCCESS) {
            log_error("failed to parse line: %s", line);
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }

    return EXIT_FAILURE;
}
