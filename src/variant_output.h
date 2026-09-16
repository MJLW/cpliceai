#ifndef VARIANT_OUTPUT_H
#define VARIANT_OUTPUT_H

#include <htslib/kstring.h>
#include <htslib/vcf.h>

#include "haplotype.h"
#include "variant_input.h"

/*
 * Writer for annotated variants. The output format always mirrors the input format: a VCF in
 * is the same VCF back with INFO/SpliceAI, INFO/SpliceAI_HAP and INFO/SpliceAI_TOT added, and
 * a TSV in is the same columns back with those three appended:
 *
 *     CHROM	POS	REF	ALT	GT	SpliceAI	SpliceAI_HAP	SpliceAI_TOT
 *
 * Converting between the two is deliberately not supported. VCF -> TSV would have to discard
 * ID, QUAL, FILTER, existing INFO, FORMAT and every genotype column, and TSV -> VCF would
 * have to invent a header the input never carried. GT is carried through in the TSV's fifth
 * column so that an annotated file is still valid input, genotypes and all.
 *
 * Annotations are handed over per ALT allele; each writer joins them the way its format
 * expects. One output record is written per record handed over - but not per record read: a
 * variant whose genotype is heterozygous and unphased has no haplotype to be scored on, and
 * the caller drops it rather than writing it out, unless --include-unphased says otherwise.
 */

typedef struct VariantWriter VariantWriter;

/*
 * variant_writer_open - Open path for writing annotated variants.
 *
 * The format and, for VCF, the header to pass through are both taken from reader, so the
 * output cannot drift from the input.
 *
 * local_only leaves SpliceAI_HAP and SpliceAI_TOT out of the output entirely - no INFO lines,
 * no TSV columns - rather than writing them out equal to SpliceAI, since --local guarantees
 * they always would be. Whatever is passed to variant_writer_write for those two is then
 * ignored.
 *
 * Returns EXIT_SUCCESS on success, EXIT_FAILURE (having logged) otherwise.
 */
int variant_writer_open(const char *path, const VariantReader *reader, bool local_only,
                        VariantWriter **writer);

/*
 * variant_writer_write - Write one annotated record.
 *
 * Each of the three annotation arrays holds record->n_alt entries, one per ALT allele, each
 * already holding that allele's annotation - comma-joined over the overlapping genes, and for
 * the two haplotype fields over the copies the allele sits on as well. Empty entries are
 * written as '.'; if every entry of a field is empty, that field is left off the record
 * entirely. Any of the arrays may be NULL when there is nothing to report.
 *
 * Returns EXIT_SUCCESS on success, EXIT_FAILURE (having logged) otherwise.
 */
int variant_writer_write(VariantWriter *writer, const HapRecord *record,
                         const kstring_t *spliceai, const kstring_t *spliceai_hap,
                         const kstring_t *spliceai_tot);

void variant_writer_close(VariantWriter *writer);

#endif /* VARIANT_OUTPUT_H */
