#include "gene_regions.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "logging/log.h"

#define MAX_LINE  8192
#define MAX_FIELD 256

static int parse_line(const char *line, Gene *gene)
{
    char buf[MAX_LINE];
    strncpy(buf, line, MAX_LINE - 1);
    buf[MAX_LINE - 1] = '\0';
    buf[strcspn(buf, "\r\n")] = '\0';

    const char *delim = "\t";
    char *tok;

    tok = strtok(buf, delim);
    if (!tok) return EXIT_FAILURE;
    strncpy(gene->name, tok, MAX_FIELD - 1);

    tok = strtok(NULL, delim);
    if (!tok) return EXIT_FAILURE;
    strncpy(gene->chrom, tok, MAX_FIELD - 1);

    tok = strtok(NULL, delim);
    if (!tok) return EXIT_FAILURE;
    gene->strand = tok[0];

    tok = strtok(NULL, delim);
    if (!tok) return EXIT_FAILURE;
    gene->tx_start = (int64_t)atoll(tok);

    tok = strtok(NULL, delim);
    if (!tok) return EXIT_FAILURE;
    gene->tx_end = (int64_t)atoll(tok);

    return 0;
}

int read_gene_region(FILE *fp, Gene *gene)
{
    char line[MAX_LINE];

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
            continue;

        if (parse_line(line, gene) != EXIT_SUCCESS) {
            log_error("failed to parse line: %s", line);
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }

    return EXIT_FAILURE;
}
