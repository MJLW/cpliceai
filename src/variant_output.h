#ifndef VARIANT_OUTPUT_H
#define VARIANT_OUTPUT_H

#include <htslib/kstring.h>
#include <htslib/vcf.h>

#include "variant_input.h"

/*
 * Writer for annotated variants. The output format always mirrors the input format: a VCF in
 * is the same VCF back with an INFO/SpliceAI annotation added, and a TSV in is the same
 * columns back with a SpliceAI column appended:
 *
 *     CHROM	POS	REF	ALT	SpliceAI
 *
 * Converting between the two is deliberately not supported. VCF -> TSV would have to discard
 * ID, QUAL, FILTER, existing INFO, FORMAT and every genotype column, and TSV -> VCF would
 * have to invent a header the input never carried.
 *
 * Annotations are handed over per ALT allele; each writer joins them the way its format
 * expects. Both emit one output record per input record, so a multiallelic input row stays
 * one row and TSV output can be fed straight back in as TSV input.
 */

typedef struct VariantWriter VariantWriter;

/*
 * variant_writer_open - Open path for writing annotated variants.
 *
 * The format and, for VCF, the header to pass through are both taken from reader, so the
 * output cannot drift from the input.
 *
 * Returns EXIT_SUCCESS on success, EXIT_FAILURE (having logged) otherwise.
 */
int variant_writer_open(const char *path, const VariantReader *reader, VariantWriter **writer);

/*
 * variant_writer_write - Write one annotated record.
 *
 * annotations holds record->n_alt entries, one per ALT allele, each already holding that
 * allele's (possibly comma-joined, one per overlapping gene) annotation. Empty entries are
 * written as '.'; if every entry is empty the record is written without a SpliceAI
 * annotation at all. annotations may be NULL when there is nothing to report.
 *
 * Returns EXIT_SUCCESS on success, EXIT_FAILURE (having logged) otherwise.
 */
int variant_writer_write(VariantWriter *writer, const VariantRecord *record,
                         const kstring_t *annotations);

void variant_writer_close(VariantWriter *writer);

#endif /* VARIANT_OUTPUT_H */
