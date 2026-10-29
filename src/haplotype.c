#include "haplotype.h"

#include <stdlib.h>
#include <string.h>

#include <klib/kvec.h>

#include "logging/log.h"

/* Below this many evicted slots, compacting the deque costs more than the slots waste. */
#define COMPACT_THRESHOLD 1024

struct HapBuffer {
    VariantReader *reader;
    bool include_unphased;
    bool local_only;
    hts_pos_t span;

    /* Deque of buffered records: live entries are rec[head .. n). */
    HapRecord **rec;
    size_t head, n, m;

    size_t cur;    /* index of the record hap_buffer_next last returned */
    bool started;
    bool eof;

    /*
     * Contigs already finished with. The buffer only ever holds one contig's worth of records,
     * so a contig coming back after another one has started would silently score variants
     * against a haplotype missing everything already evicted.
     */
    kvec_t(char *) seen_contigs;
};

void seq_edit_list_init(SeqEditList *list) {
    list->edits = NULL;
    list->n = list->m = 0;
}

void seq_edit_list_destroy(SeqEditList *list) {
    free(list->edits);
    list->edits = NULL;
    list->n = list->m = 0;
}

static void seq_edit_list_push(SeqEditList *list, const SeqEdit edit) {
    if (list->n == list->m) {
        const size_t m = list->m == 0 ? 8 : list->m * 2;
        SeqEdit *grown = realloc(list->edits, m * sizeof(SeqEdit));
        if (grown == NULL) {
            log_fatal("Failed to allocate %zu bytes for haplotype edits", m * sizeof(SeqEdit));
            exit(EXIT_FAILURE);
        }
        list->edits = grown;
        list->m = m;
    }
    list->edits[list->n++] = edit;
}

void seq_edit_list_push_sorted(SeqEditList *list, const SeqEdit edit) {
    seq_edit_list_push(list, edit);

    size_t i = list->n - 1;
    while (i > 0 && list->edits[i - 1].gene_pos > edit.gene_pos) {
        list->edits[i] = list->edits[i - 1];
        i--;
    }
    list->edits[i] = edit;
}

static void *xmalloc(size_t size) {
    void *p = malloc(size);
    if (p == NULL) {
        log_fatal("Failed to allocate %zu bytes", size);
        exit(EXIT_FAILURE);
    }
    return p;
}

static char *xstrdup(const char *s) {
    char *copy = strdup(s);
    if (copy == NULL) {
        log_fatal("Failed to duplicate a %zu byte string", strlen(s) + 1);
        exit(EXIT_FAILURE);
    }
    return copy;
}

static void hap_record_free(HapRecord *record) {
    if (record == NULL) return;

    for (int i = 0; i < record->n_alt; i++) free(record->alt[i]);
    free(record->alt);
    free(record->hap_mask);
    free(record->ref);
    free(record->chrom);
    if (record->bcf != NULL) bcf_destroy(record->bcf);
    free(record);
}

/*
 * Work out which copies carry each alternate allele.
 *
 * The mask is what makes a variant part of a background; whether it is scored at all is a
 * separate question, since a genotype can name an allele the sample does not have.
 */
static void assign_haplotypes(const VariantRecord *src, HapRecord *dst, bool include_unphased) {
    for (int i = 0; i < dst->n_alt; i++) dst->hap_mask[i] = 0;
    dst->drop = false;

    /* No genotype: nothing to place on a copy, and every allele scored on its own. */
    if (src->ploidy == 0) return;

    const bool heterozygous = src->ploidy == 2 && src->gt[0] != src->gt[1];
    if (heterozygous && !src->phased) {
        if (!include_unphased) {
            dst->drop = true;
            return;
        }

        /* Which copy each allele is on is exactly what is missing, so put them on both. */
        for (int copy = 0; copy < src->ploidy; copy++) {
            const int allele = src->gt[copy];
            if (allele > 0 && allele <= dst->n_alt) dst->hap_mask[allele - 1] = HAP_1 | HAP_2;
        }
        return;
    }

    /*
     * A homozygous call needs no phasing to be placed: both copies carry the same allele
     * whichever way round they are written.
     */
    for (int copy = 0; copy < src->ploidy; copy++) {
        const int allele = src->gt[copy];
        if (allele > 0 && allele <= dst->n_alt) dst->hap_mask[allele - 1] |= HAP_MASK(copy);
    }
}

static HapRecord *hap_record_from(const VariantRecord *src, bool include_unphased, bool local_only) {
    HapRecord *dst = xmalloc(sizeof(HapRecord));

    dst->chrom = xstrdup(src->chrom);
    dst->pos = src->pos;
    dst->ref = xstrdup(src->ref);
    dst->n_alt = src->n_alt;
    dst->alt = src->n_alt > 0 ? xmalloc(src->n_alt * sizeof(char *)) : NULL;
    dst->hap_mask = src->n_alt > 0 ? xmalloc(src->n_alt * sizeof(int)) : NULL;
    for (int i = 0; i < src->n_alt; i++) dst->alt[i] = xstrdup(src->alt[i]);

    /* The genotype itself always round-trips to output, whether or not it drives scoring. */
    dst->gt[0] = src->gt[0];
    dst->gt[1] = src->gt[1];
    dst->ploidy = src->ploidy;
    dst->phased = src->phased;

    dst->bcf = src->bcf != NULL ? bcf_dup(src->bcf) : NULL;

    if (local_only) {
        /* Scored exactly as if no genotype were present: every allele on its own, nothing ever
           dropped or split across copies. */
        dst->has_gt = false;
        for (int i = 0; i < dst->n_alt; i++) dst->hap_mask[i] = 0;
        dst->drop = false;
    } else {
        dst->has_gt = src->ploidy > 0;
        assign_haplotypes(src, dst, include_unphased);
    }

    return dst;
}

static bool contig_seen(const HapBuffer *buffer, const char *chrom) {
    for (size_t i = 0; i < kv_size(buffer->seen_contigs); i++) {
        if (strcmp(kv_A(buffer->seen_contigs, i), chrom) == 0) return true;
    }
    return false;
}

/*
 * Read one more record onto the back of the deque.
 *
 * Returns EXIT_SUCCESS, VARIANT_READER_EOF when the input is exhausted, or EXIT_FAILURE.
 */
static int buffer_fill_one(HapBuffer *buffer) {
    if (buffer->eof) return VARIANT_READER_EOF;

    VariantRecord record;
    const int status = variant_reader_next(buffer->reader, &record);
    if (status == VARIANT_READER_EOF) {
        buffer->eof = true;
        return VARIANT_READER_EOF;
    }
    if (status != EXIT_SUCCESS) return EXIT_FAILURE;

    if (buffer->n > buffer->head) {
        const HapRecord *last = buffer->rec[buffer->n - 1];
        if (strcmp(last->chrom, record.chrom) == 0) {
            if (record.pos < last->pos) {
                log_error("%s:%"PRIhts_pos" follows %"PRIhts_pos" on the same contig. Haplotypes "
                          "are assembled from a sliding window, so the input must be sorted by "
                          "position: sort it first, e.g. bcftools sort.",
                          record.chrom, record.pos + 1, last->pos + 1);
                return EXIT_FAILURE;
            }
        } else {
            if (contig_seen(buffer, record.chrom)) {
                log_error("%s reappears after another contig. Haplotypes are assembled from a "
                          "sliding window, so each contig's variants must be contiguous: sort "
                          "the input first, e.g. bcftools sort.", record.chrom);
                return EXIT_FAILURE;
            }
            kv_push(char *, buffer->seen_contigs, xstrdup(last->chrom));
        }
    }

    if (buffer->n == buffer->m) {
        const size_t m = buffer->m == 0 ? 64 : buffer->m * 2;
        HapRecord **grown = realloc(buffer->rec, m * sizeof(HapRecord *));
        if (grown == NULL) {
            log_fatal("Failed to allocate %zu bytes for the variant buffer", m * sizeof(HapRecord *));
            exit(EXIT_FAILURE);
        }
        buffer->rec = grown;
        buffer->m = m;
    }

    buffer->rec[buffer->n++] = hap_record_from(&record, buffer->include_unphased, buffer->local_only);

    return EXIT_SUCCESS;
}

int hap_buffer_open(VariantReader *reader, bool include_unphased, bool local_only, hts_pos_t span,
                    HapBuffer **buffer) {
    HapBuffer *b = calloc(1, sizeof(HapBuffer));
    if (b == NULL) {
        log_fatal("Failed to allocate %zu bytes for the variant buffer", sizeof(HapBuffer));
        exit(EXIT_FAILURE);
    }

    b->reader = reader;
    b->include_unphased = include_unphased;
    b->local_only = local_only;
    b->span = span;
    kv_init(b->seen_contigs);

    *buffer = b;

    return EXIT_SUCCESS;
}

int hap_buffer_next(HapBuffer *buffer, const HapRecord **record) {
    if (!buffer->started) {
        buffer->cur = buffer->head;
        buffer->started = true;
    } else {
        buffer->cur++;
    }

    if (buffer->cur >= buffer->n) {
        const int status = buffer_fill_one(buffer);
        if (status != EXIT_SUCCESS) return status;
    }

    const HapRecord *current = buffer->rec[buffer->cur];

    /* Read forward until the next variant is too far away to reach this one's window. */
    while (!buffer->eof) {
        const HapRecord *last = buffer->rec[buffer->n - 1];
        if (strcmp(last->chrom, current->chrom) != 0) break;
        if (last->pos > current->pos + buffer->span) break;

        const int status = buffer_fill_one(buffer);
        if (status == EXIT_FAILURE) return EXIT_FAILURE;
    }

    /* Drop what is now too far behind, but never the record being returned. */
    while (buffer->head < buffer->cur) {
        const HapRecord *first = buffer->rec[buffer->head];
        if (strcmp(first->chrom, current->chrom) == 0 && first->pos >= current->pos - buffer->span) break;

        hap_record_free(buffer->rec[buffer->head]);
        buffer->head++;
    }

    if (buffer->head > COMPACT_THRESHOLD) {
        const size_t live = buffer->n - buffer->head;
        memmove(buffer->rec, buffer->rec + buffer->head, live * sizeof(HapRecord *));
        buffer->cur -= buffer->head;
        buffer->n = live;
        buffer->head = 0;
    }

    *record = buffer->rec[buffer->cur];

    return EXIT_SUCCESS;
}

void hap_edits_collect(const HapBuffer *buffer, int hap, const char *chrom,
                       hts_pos_t gene_start, hts_pos_t lo, hts_pos_t hi,
                       const HapRecord *exclude, int exclude_alt, SeqEditList *out) {
    out->n = 0;

    /*
     * The REF span left free for the excluded allele. A caller building a background in order to
     * put that allele back into it needs the space to still be there: two edits cannot both
     * replace the same base, and the variant being scored is the one that has to win.
     */
    const hts_pos_t hole_lo = exclude != NULL ? exclude->pos : 0;
    const hts_pos_t hole_hi = exclude != NULL ? exclude->pos + (hts_pos_t) strlen(exclude->ref) : 0;

    for (size_t i = buffer->head; i < buffer->n; i++) {
        const HapRecord *record = buffer->rec[i];

        if (record->drop) continue;
        if (strcmp(record->chrom, chrom) != 0) continue;

        const int ref_len = (int) strlen(record->ref);
        if (record->pos < lo || record->pos + ref_len > hi) continue;

        if (exclude != NULL && record->pos < hole_hi && hole_lo < record->pos + ref_len) {
            if (record != exclude) {
                log_warn("%s:%"PRIhts_pos" overlaps %s:%"PRIhts_pos" on haplotype %d. Leaving it "
                         "out of that variant's background.",
                         record->chrom, record->pos + 1, exclude->chrom, exclude->pos + 1,
                         hap == HAP_1 ? 1 : 2);
            }
            continue;
        }

        for (int a = 0; a < record->n_alt; a++) {
            if ((record->hap_mask[a] & hap) == 0) continue;
            if (record == exclude && a == exclude_alt) continue;

            /*
             * Two alleles replacing the same base cannot both be applied. This is normal for a
             * multiallelic record's own alleles when a genotype puts two of them on one copy,
             * and a data error otherwise; either way only the first can be honoured.
             */
            if (out->n > 0) {
                const SeqEdit *last = &out->edits[out->n - 1];
                if (record->pos - gene_start < last->gene_pos + last->ref_len) {
                    log_warn("%s:%"PRIhts_pos":%s overlaps an earlier variant on haplotype %d. "
                             "Leaving it out of that haplotype.",
                             record->chrom, record->pos + 1, record->alt[a],
                             hap == HAP_1 ? 1 : 2);
                    continue;
                }
            }

            seq_edit_list_push(out, (SeqEdit) {
                .gene_pos = record->pos - gene_start,
                .ref_len  = ref_len,
                .alt      = record->alt[a],
                .alt_len  = (int) strlen(record->alt[a]),
            });
        }
    }
}

void hap_buffer_close(HapBuffer *buffer) {
    if (buffer == NULL) return;

    for (size_t i = buffer->head; i < buffer->n; i++) hap_record_free(buffer->rec[i]);
    free(buffer->rec);

    for (size_t i = 0; i < kv_size(buffer->seen_contigs); i++) free(kv_A(buffer->seen_contigs, i));
    kv_destroy(buffer->seen_contigs);

    free(buffer);
}
