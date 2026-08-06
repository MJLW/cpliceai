#include "variant_output.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "logging/log.h"
#include "utils.h"

struct VariantWriter {
    VariantFormat format;
    char *path;

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

    if (bcf_hdr_append(w->hdr, SPLICEAI_DESC) != 0) {
        log_error("Failed to append description for tag %s to vcf header.", SPLICEAI_TAG);
        return EXIT_FAILURE;
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

    /* The first four columns are exactly the input schema, so this output is valid input. */
    fprintf(w->tsv, "CHROM\tPOS\tREF\tALT\t%s\n", SPLICEAI_TAG);

    return EXIT_SUCCESS;
}

int variant_writer_open(const char *path, const VariantReader *reader, VariantWriter **writer) {
    VariantWriter *w = calloc(1, sizeof(VariantWriter));
    if (w == NULL) {
        log_fatal("Failed to allocate %zu bytes for variant writer", sizeof(VariantWriter));
        exit(EXIT_FAILURE);
    }

    w->format = variant_reader_format(reader);
    w->path = strdup(path);

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
static bool join_annotations(const VariantRecord *record, const kstring_t *annotations,
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

static int variant_writer_write_vcf(VariantWriter *w, const VariantRecord *record,
                                    const kstring_t *annotations) {
    /* VCF output only ever follows VCF input, so there is always a record to pass through. */
    bcf1_t *v = record->bcf;

    if (join_annotations(record, annotations, &w->buf)) {
        bcf_update_info_string(w->hdr, v, SPLICEAI_TAG, w->buf.s);
    }

    if (bcf_write(w->vcf, w->hdr, v) != 0) {
        log_error("Writing failed for file: %s", w->path);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

static int variant_writer_write_tsv(VariantWriter *w, const VariantRecord *record,
                                    const kstring_t *annotations) {
    fprintf(w->tsv, "%s\t%" PRIhts_pos "\t%s\t", record->chrom, record->pos + 1, record->ref);

    for (int i = 0; i < record->n_alt; i++) {
        if (i > 0) fputc(',', w->tsv);
        fputs(record->alt[i], w->tsv);
    }

    const bool annotated = join_annotations(record, annotations, &w->buf);
    fprintf(w->tsv, "\t%s\n", annotated ? w->buf.s : ".");

    return EXIT_SUCCESS;
}

int variant_writer_write(VariantWriter *writer, const VariantRecord *record,
                         const kstring_t *annotations) {
    if (writer->format == VARIANT_FORMAT_VCF) {
        return variant_writer_write_vcf(writer, record, annotations);
    }
    return variant_writer_write_tsv(writer, record, annotations);
}

void variant_writer_close(VariantWriter *writer) {
    if (writer == NULL) return;

    if (writer->vcf != NULL) hts_close(writer->vcf);
    if (writer->tsv != NULL) fclose(writer->tsv);

    free(writer->buf.s);
    free(writer->path);
    free(writer);
}
