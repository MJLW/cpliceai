#include "variant_input.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include <htslib/kstring.h>
#include <klib/kvec.h>

#include "gene_regions.h" /* next_tsv_field */
#include "logging/log.h"

typedef kvec_t(char *) AltVec;

struct VariantReader {
    VariantFormat format; /* always VCF or TSV; AUTO is resolved at open time */
    char *path;
    htsFile *fp;

    /* VCF */
    bcf_hdr_t *hdr;
    bcf1_t *rec;

    /* TSV */
    kstring_t line;
    AltVec alts;
    uint64_t line_no;
    bool checked_first_line; /* gates header detection to exactly one line */
};

int variant_input_format_parse(const char *s, VariantFormat *fmt) {
    if (strcmp(s, "vcf") == 0)  { *fmt = VARIANT_FORMAT_VCF;  return EXIT_SUCCESS; }
    if (strcmp(s, "tsv") == 0)  { *fmt = VARIANT_FORMAT_TSV;  return EXIT_SUCCESS; }
    if (strcmp(s, "auto") == 0) { *fmt = VARIANT_FORMAT_AUTO; return EXIT_SUCCESS; }

    log_error("Unrecognised input format '%s'. Expected one of: vcf, tsv, auto.", s);
    return EXIT_FAILURE;
}

const char *variant_format_name(VariantFormat fmt) {
    switch (fmt) {
        case VARIANT_FORMAT_VCF:  return "vcf";
        case VARIANT_FORMAT_TSV:  return "tsv";
        case VARIANT_FORMAT_AUTO: return "auto";
    }
    return "unknown";
}

/*
 * Resolve VARIANT_FORMAT_AUTO from an already-opened file. htslib has done the magic-number
 * and BGZF sniffing for us, so we only have to decide which of its verdicts count as VCF.
 *
 * The test is deliberately one-sided: only htslib's own vcf/bcf verdicts are treated as VCF,
 * and every other verdict (text_format, bed, unknown_format, ...) falls through to TSV. A
 * plain TSV can therefore never be handed to the VCF parser. The cost is that a VCF missing
 * its "##fileformat=" line is detected as TSV and fails as a TSV parse error - that is what
 * --input-format vcf is for.
 */
static VariantFormat detect_format(const htsFile *fp) {
    const htsFormat *fmt = hts_get_format((htsFile *) fp);
    if (fmt != NULL && (fmt->format == vcf || fmt->format == bcf)) return VARIANT_FORMAT_VCF;
    return VARIANT_FORMAT_TSV;
}

int variant_reader_open(const char *path, VariantFormat fmt, VariantReader **reader) {
    VariantReader *r = calloc(1, sizeof(VariantReader));
    if (r == NULL) {
        log_fatal("Failed to allocate %zu bytes for variant reader", sizeof(VariantReader));
        exit(EXIT_FAILURE);
    }

    r->fp = hts_open(path, "r");
    if (r->fp == NULL) {
        log_error("Failed to open variants file: %s", path);
        free(r);
        return EXIT_FAILURE;
    }

    r->path = strdup(path);
    r->format = (fmt == VARIANT_FORMAT_AUTO) ? detect_format(r->fp) : fmt;

    if (fmt == VARIANT_FORMAT_AUTO) {
        log_info("Detected %s input: %s", variant_format_name(r->format), path);
    }

    if (r->format == VARIANT_FORMAT_VCF) {
        r->hdr = bcf_hdr_read(r->fp);
        if (r->hdr == NULL) {
            log_error("Failed to read header from VCF file: %s", path);
            variant_reader_close(r);
            return EXIT_FAILURE;
        }
        r->rec = bcf_init();
        if (r->rec == NULL) {
            log_fatal("Failed to allocate a VCF record");
            exit(EXIT_FAILURE);
        }
    } else {
        kv_init(r->alts);
    }

    *reader = r;

    return EXIT_SUCCESS;
}

VariantFormat variant_reader_format(const VariantReader *reader) {
    return reader->format;
}

bcf_hdr_t *variant_reader_hdr(const VariantReader *reader) {
    return reader->hdr;
}

static int variant_reader_next_vcf(VariantReader *reader, VariantRecord *record) {
    int ret = bcf_read(reader->fp, reader->hdr, reader->rec);
    if (ret == -1) return VARIANT_READER_EOF;
    if (ret < 0) {
        log_error("Failed to read record from VCF file: %s", reader->path);
        return EXIT_FAILURE;
    }

    bcf1_t *v = reader->rec;
    bcf_unpack(v, BCF_UN_STR);

    record->chrom = bcf_hdr_id2name(reader->hdr, v->rid);
    record->pos   = v->pos;
    record->ref   = v->d.allele[0];
    record->n_alt = v->n_allele - 1;
    record->alt   = v->d.allele + 1;
    record->bcf   = v;

    return EXIT_SUCCESS;
}

/*
 * Split a mutable ALT field on ',' in place, collecting the pieces into reader->alts. A TSV
 * row is allowed to be multiallelic ("G\tA,T") so that it can express everything the
 * equivalent VCF record can.
 */
static int split_alts(VariantReader *reader, char *field) {
    kv_size(reader->alts) = 0;

    char *cursor = field;
    while (cursor != NULL) {
        char *comma = strchr(cursor, ',');
        if (comma != NULL) *comma = '\0';

        if (cursor[0] == '\0') {
            log_error("%s:%lu: empty alternate allele.", reader->path, reader->line_no);
            return EXIT_FAILURE;
        }

        kv_push(char *, reader->alts, cursor);
        cursor = (comma != NULL) ? comma + 1 : NULL;
    }

    return EXIT_SUCCESS;
}

/* parse_tsv_line's verdict when POS is unparseable on a line that could still be a header. */
#define PARSE_NOT_A_RECORD (-2)

/*
 * Parse one TSV data line: CHROM POS REF ALT.
 *
 * Fields beyond the fourth are read past and ignored. POS is 1-based on disk and 0-based in
 * the record, matching bcf1_t::pos.
 *
 * Returns PARSE_NOT_A_RECORD, without logging an error, when POS does not parse as a positive
 * integer. Only the caller knows whether that means "this is the header" or "this row is
 * broken", so the message is left to it.
 */
static int parse_tsv_line(VariantReader *reader, VariantRecord *record) {
    char *cursor = reader->line.s;

    char *chrom = next_tsv_field(&cursor);
    if (chrom == NULL || chrom[0] == '\0') {
        log_error("%s:%lu: missing CHROM field.", reader->path, reader->line_no);
        return EXIT_FAILURE;
    }

    char *pos = next_tsv_field(&cursor);
    if (pos == NULL || pos[0] == '\0') return PARSE_NOT_A_RECORD;

    char *pos_end;
    unsigned long long pos_value = strtoull(pos, &pos_end, 10);
    if (pos_end[0] != '\0' || pos_value == 0) return PARSE_NOT_A_RECORD;

    char *ref = next_tsv_field(&cursor);
    if (ref == NULL || ref[0] == '\0') {
        log_error("%s:%lu: missing REF field.", reader->path, reader->line_no);
        return EXIT_FAILURE;
    }

    char *alt = next_tsv_field(&cursor);
    if (alt == NULL || alt[0] == '\0') {
        log_error("%s:%lu: missing ALT field.", reader->path, reader->line_no);
        return EXIT_FAILURE;
    }

    if (split_alts(reader, alt) != EXIT_SUCCESS) return EXIT_FAILURE;

    record->chrom = chrom;
    record->pos   = (hts_pos_t) pos_value - 1;
    record->ref   = ref;
    record->n_alt = kv_size(reader->alts);
    record->alt   = reader->alts.a;
    record->bcf   = NULL;

    return EXIT_SUCCESS;
}

static int variant_reader_next_tsv(VariantReader *reader, VariantRecord *record) {
    /* hts_getline documents its delimiter argument as unused but required to be '\n'. */
    while (hts_getline(reader->fp, '\n', &reader->line) >= 0) {
        reader->line_no++;

        /* Blank lines and '#' comments carry no record. The header needs no '#' (see below),
           but one is still accepted for files that carry it. */
        const char *s = reader->line.s;
        if (reader->line.l == 0 || s[0] == '#' || s[0] == '\r') continue;

        /* hts_getline strips '\n' but leaves a '\r' from CRLF files behind. */
        if (reader->line.l > 0 && reader->line.s[reader->line.l - 1] == '\r') {
            reader->line.s[--reader->line.l] = '\0';
        }

        /* Exactly one line is ever a header candidate: the first that is neither blank nor a
           '#' comment. Any later unparseable POS is a malformed row, not a second header. */
        const bool may_be_header = !reader->checked_first_line;
        reader->checked_first_line = true;

        int ret = parse_tsv_line(reader, record);

        if (ret == PARSE_NOT_A_RECORD) {
            /* POS is POS by definition, so a first line whose POS is not a position is not a
               valid record under any reading and skipping it loses nothing. */
            if (may_be_header) {
                log_info("Skipping presumed header line in %s: %s", reader->path, reader->line.s);
                continue;
            }
            log_error("%s:%lu: could not parse POS as a 1-based position.",
                      reader->path, reader->line_no);
            return EXIT_FAILURE;
        }

        return ret;
    }

    return VARIANT_READER_EOF;
}

int variant_reader_next(VariantReader *reader, VariantRecord *record) {
    if (reader->format == VARIANT_FORMAT_VCF) return variant_reader_next_vcf(reader, record);
    return variant_reader_next_tsv(reader, record);
}

void variant_reader_close(VariantReader *reader) {
    if (reader == NULL) return;

    if (reader->rec != NULL) bcf_destroy(reader->rec);
    if (reader->hdr != NULL) bcf_hdr_destroy(reader->hdr);
    if (reader->fp != NULL) hts_close(reader->fp);

    free(reader->line.s);
    kv_destroy(reader->alts);
    free(reader->path);
    free(reader);
}
