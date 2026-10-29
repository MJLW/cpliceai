#ifndef VARIANT_INPUT_H
#define VARIANT_INPUT_H

#include <stdbool.h>

#include <htslib/hts.h>
#include <htslib/vcf.h>

/*
 * Unified variant reader. Both VCF/BCF and the plain TSV form
 *
 *     CHROM	POS	REF	ALT	[GT]
 *
 * are read into the same VariantRecord, so callers never branch on the input format.
 * The TSV carries no gene column: genes are resolved downstream by interval overlap
 * against the gene regions file, exactly as they are for VCF records.
 *
 * GT is what makes haplotype scoring possible: it says which of the sample's two copies each
 * alternate allele sits on, so variants sharing a copy can be applied to one sequence together.
 * It is optional in both formats - a record without one is scored on its own, as every record
 * was before haplotypes existed.
 */

/* VariantRecord::gt entry for a missing allele ('.'). */
#define GT_ALLELE_MISSING (-1)

typedef enum {
    VARIANT_FORMAT_VCF,
    VARIANT_FORMAT_TSV,
    VARIANT_FORMAT_AUTO, /* resolve by detection */
} VariantFormat;

/* variant_reader_next returns this, rather than EXIT_FAILURE, on a clean end of input. */
#define VARIANT_READER_EOF (-1)

/*
 * One input record, possibly carrying several ALT alleles. All pointers are borrowed from
 * the reader and are invalidated by the next variant_reader_next() call.
 */
typedef struct {
    const char *chrom;
    hts_pos_t   pos;   /* 0-based, matching bcf1_t::pos */
    const char *ref;
    int         n_alt; /* 0 when the record has no ALT allele */
    char      **alt;
    bcf1_t     *bcf;   /* the underlying record for VCF input, NULL for TSV */

    /*
     * Genotype. ploidy is 0 when the record carries none, which is the ordinary case for a
     * sites-only VCF or a four-column TSV; gt and phased are then meaningless.
     *
     * gt holds an allele index per copy: 0 is REF, n is alt[n - 1], and GT_ALLELE_MISSING is
     * '.'. phased says the copies are ordered, i.e. gt[0] and gt[1] name specific molecules
     * rather than just a pair of alleles.
     */
    int         gt[2];
    int         ploidy;
    bool        phased;
} VariantRecord;

typedef struct VariantReader VariantReader;

/*
 * variant_input_format_parse - Parse a --input-format value: "vcf", "tsv" or "auto".
 *
 * Returns EXIT_SUCCESS, or EXIT_FAILURE (having logged) on an unrecognised value.
 */
int variant_input_format_parse(const char *s, VariantFormat *fmt);

/* Human-readable name, for log messages and help text. */
const char *variant_format_name(VariantFormat fmt);

/*
 * variant_reader_open - Open path for reading.
 *
 * fmt may be VARIANT_FORMAT_AUTO, in which case the format is resolved from the file itself
 * (see variant_reader_format). Opening does not read any records.
 *
 * A VCF carrying more than one sample is rejected here. Genotypes are read from a single
 * sample, and picking one silently would be a wrong answer wearing a right one's clothes.
 *
 * Returns EXIT_SUCCESS on success, EXIT_FAILURE (having logged) otherwise.
 */
int variant_reader_open(const char *path, VariantFormat fmt, VariantReader **reader);

/*
 * variant_reader_next - Read the next record into *record.
 *
 * Returns EXIT_SUCCESS, VARIANT_READER_EOF at clean end of input, or EXIT_FAILURE (having
 * logged) on a malformed record. A malformed record is never reported as end of input, so
 * callers can distinguish "done" from "gave up partway".
 */
int variant_reader_next(VariantReader *reader, VariantRecord *record);

/* The resolved input format: never VARIANT_FORMAT_AUTO. */
VariantFormat variant_reader_format(const VariantReader *reader);

/* The input VCF header, or NULL for TSV input. Owned by the reader. */
bcf_hdr_t *variant_reader_hdr(const VariantReader *reader);

void variant_reader_close(VariantReader *reader);

#endif /* VARIANT_INPUT_H */
