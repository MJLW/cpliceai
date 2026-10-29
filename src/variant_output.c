#include "variant_output.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "logging/log.h"
#include "utils.h"

struct VariantWriter {
    VariantFormat format;
    char *path;
    bool local_only;

    /* VCF: borrowed from the reader, never owned here. */
    htsFile *vcf;
    bcf_hdr_t *hdr;

    /* TSV */
    FILE *tsv;

    kstring_t buf;
};

static int variant_writer_open_vcf(VariantWriter *w, bcf_hdr_t *in_hdr) {
    w->hdr = in_hdr;

    w->vcf = bcf_open(w->path, "w");
    if (w->vcf == NULL) {
        log_error("Failed to open vcf output file: %s", w->path);
        return EXIT_FAILURE;
    }

    const char *descriptions[] = { SPLICEAI_DESC, SPLICEAI_HAP_DESC, SPLICEAI_TOT_DESC };
    const char *tags[] = { SPLICEAI_TAG, SPLICEAI_HAP_TAG, SPLICEAI_TOT_TAG };
    const size_t n_tags = w->local_only ? 1 : sizeof(tags) / sizeof(tags[0]);
    for (size_t i = 0; i < n_tags; i++) {
        if (bcf_hdr_append(w->hdr, descriptions[i]) != 0) {
            log_error("Failed to append description for tag %s to vcf header.", tags[i]);
            return EXIT_FAILURE;
        }
    }

    if (bcf_hdr_write(w->vcf, w->hdr) != 0) {
        log_error("Failed to write to vcf file: %s", w->path);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

static int variant_writer_open_tsv(VariantWriter *w) {
    w->tsv = open_file_or_log(w->path, "w");
    if (w->tsv == NULL) return EXIT_FAILURE;

    /* The first five columns are exactly the input schema, so this output is valid input. */
    if (w->local_only) {
        fprintf(w->tsv, "CHROM\tPOS\tREF\tALT\tGT\t%s\n", SPLICEAI_TAG);
    } else {
        fprintf(w->tsv, "CHROM\tPOS\tREF\tALT\tGT\t%s\t%s\t%s\n",
                SPLICEAI_TAG, SPLICEAI_HAP_TAG, SPLICEAI_TOT_TAG);
    }

    return EXIT_SUCCESS;
}

int variant_writer_open(const char *path, const VariantReader *reader, bool local_only,
                        VariantWriter **writer) {
    VariantWriter *w = calloc(1, sizeof(VariantWriter));
    if (w == NULL) {
        log_fatal("Failed to allocate %zu bytes for variant writer", sizeof(VariantWriter));
        exit(EXIT_FAILURE);
    }

    w->format = variant_reader_format(reader);
    w->path = strdup(path);
    w->local_only = local_only;

    int ret = (w->format == VARIANT_FORMAT_VCF)
                  ? variant_writer_open_vcf(w, variant_reader_hdr(reader))
                  : variant_writer_open_tsv(w);
    if (ret != EXIT_SUCCESS) {
        variant_writer_close(w);
        return EXIT_FAILURE;
    }

    *writer = w;

    return EXIT_SUCCESS;
}

/*
 * Join the per-allele annotations into the single comma-separated string both formats use.
 * Alleles that were skipped are written as '.', matching the SpliceAI INFO convention.
 *
 * Returns true when at least one allele produced an annotation, i.e. when the record should
 * be annotated at all. A record that overlapped no gene produces none and is written
 * through untouched.
 */
static bool join_annotations(const HapRecord *record, const kstring_t *annotations,
                             kstring_t *out) {
    out->l = 0;
    if (annotations == NULL) return false;

    bool any = false;
    for (int i = 0; i < record->n_alt; i++) {
        if (i > 0) kputc(',', out);

        if (annotations[i].l > 0) {
            kputsn(annotations[i].s, annotations[i].l, out);
            any = true;
        } else {
            kputc('.', out);
        }
    }

    return any;
}

/*
 * Render the genotype back into VCF notation for the TSV's GT column, so an annotated file
 * carries the phasing it was scored with and can be fed back in unchanged. A record that
 * arrived without a genotype gets '.', which reads back as no genotype.
 */
static void format_gt(const HapRecord *record, kstring_t *out) {
    out->l = 0;

    if (record->ploidy == 0) {
        kputc('.', out);
        return;
    }

    for (int copy = 0; copy < record->ploidy; copy++) {
        if (copy > 0) kputc(record->phased ? '|' : '/', out);

        if (record->gt[copy] == GT_ALLELE_MISSING) kputc('.', out);
        else kputw(record->gt[copy], out);
    }
}

static int variant_writer_write_vcf(VariantWriter *w, const HapRecord *record,
                                    const kstring_t *spliceai, const kstring_t *spliceai_hap,
                                    const kstring_t *spliceai_tot) {
    /* VCF output only ever follows VCF input, so there is always a record to pass through. */
    bcf1_t *v = record->bcf;

    const kstring_t *fields[] = { spliceai, spliceai_hap, spliceai_tot };
    const char *tags[] = { SPLICEAI_TAG, SPLICEAI_HAP_TAG, SPLICEAI_TOT_TAG };
    const size_t n_tags = w->local_only ? 1 : sizeof(tags) / sizeof(tags[0]);
    for (size_t i = 0; i < n_tags; i++) {
        if (join_annotations(record, fields[i], &w->buf)) {
            bcf_update_info_string(w->hdr, v, tags[i], w->buf.s);
        }
    }

    if (bcf_write(w->vcf, w->hdr, v) != 0) {
        log_error("Writing failed for file: %s", w->path);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

static int variant_writer_write_tsv(VariantWriter *w, const HapRecord *record,
                                    const kstring_t *spliceai, const kstring_t *spliceai_hap,
                                    const kstring_t *spliceai_tot) {
    fprintf(w->tsv, "%s\t%" PRIhts_pos "\t%s\t", record->chrom, record->pos + 1, record->ref);

    for (int i = 0; i < record->n_alt; i++) {
        if (i > 0) fputc(',', w->tsv);
        fputs(record->alt[i], w->tsv);
    }

    format_gt(record, &w->buf);
    fprintf(w->tsv, "\t%s", w->buf.s);

    const kstring_t *fields[] = { spliceai, spliceai_hap, spliceai_tot };
    const size_t n_fields = w->local_only ? 1 : sizeof(fields) / sizeof(fields[0]);
    for (size_t i = 0; i < n_fields; i++) {
        const bool annotated = join_annotations(record, fields[i], &w->buf);
        fprintf(w->tsv, "\t%s", annotated ? w->buf.s : ".");
    }
    fputc('\n', w->tsv);

    return EXIT_SUCCESS;
}

int variant_writer_write(VariantWriter *writer, const HapRecord *record,
                         const kstring_t *spliceai, const kstring_t *spliceai_hap,
                         const kstring_t *spliceai_tot) {
    if (writer->format == VARIANT_FORMAT_VCF) {
        return variant_writer_write_vcf(writer, record, spliceai, spliceai_hap, spliceai_tot);
    }
    return variant_writer_write_tsv(writer, record, spliceai, spliceai_hap, spliceai_tot);
}

void variant_writer_close(VariantWriter *writer) {
    if (writer == NULL) return;

    if (writer->vcf != NULL) hts_close(writer->vcf);
    if (writer->tsv != NULL) fclose(writer->tsv);

    free(writer->buf.s);
    free(writer->path);
    free(writer);
}
