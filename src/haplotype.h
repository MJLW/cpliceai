#ifndef HAPLOTYPE_H
#define HAPLOTYPE_H

#include <stdbool.h>

#include <htslib/hts.h>
#include <htslib/vcf.h>

#include "utils.h"
#include "variant_input.h"

/*
 * Haplotype assembly: turning a stream of variants into the two sequences a sample actually
 * carries.
 *
 * A variant read on its own says what one substitution does to the reference. A variant read
 * alongside the others sharing its copy of the chromosome says what it does to the molecule it
 * is really on, which is not the same thing when a neighbour has already moved a splice site.
 * Scoring the second needs variants grouped by the copy they sit on, and the reader hands them
 * over one at a time - so this module buffers them.
 *
 * The buffer is a sliding window, not the whole file: only variants within `span` bases of the
 * one being scored can reach it, since the model's receptive field is bounded. That does mean
 * the input has to be sorted by position, which variant_reader does not require and this does.
 *
 * Phase sets (FORMAT/PS) are deliberately not consulted. Every phased variant in a gene is
 * treated as belonging to one pair of haplotypes. Where a gene spans two phase blocks their
 * orientations are independent, and this will pair them arbitrarily; see the README.
 */

/*
 * Which copy a variant sits on, as a bit mask. A variant may be on both.
 *
 * Callers iterate the copies as `for (int h = 0; h < HAP_COUNT; h++)`, taking HAP_MASK(h) as
 * the mask and h + 1 as the label the output reports.
 */
#define HAP_COUNT 2
#define HAP_MASK(h) (1 << (h))
#define HAP_1 HAP_MASK(0)
#define HAP_2 HAP_MASK(1)

/*
 * A buffered variant, owning its own copies of everything: the reader reuses its storage from
 * one record to the next, and a haplotype outlives the record that contributed to it.
 */
typedef struct {
    char      *chrom;
    hts_pos_t  pos; /* 0-based */
    char      *ref;
    int        n_alt;
    char     **alt;

    /*
     * Per alternate allele, which copies carry it: 0, HAP_1, HAP_2 or both.
     *
     * With no genotype every mask is 0 and every allele is still scored - against an empty
     * background, which is what a variant considered in isolation has always been scored
     * against. With a genotype, a zero mask means the sample does not carry that allele, and it
     * is not scored at all.
     */
    int       *hap_mask;
    bool       has_gt;

    /* Rendered back for output, so a TSV keeps the GT column it arrived with. */
    int        gt[2];
    int        ploidy;
    bool       phased;

    /*
     * An unphased heterozygous call: the alleles are known, the copy each sits on is not.
     * Dropped entirely unless --include-unphased was passed.
     */
    bool       drop;

    bcf1_t    *bcf; /* owned duplicate for VCF passthrough, NULL for TSV input */
} HapRecord;

/*
 * A haplotype in the form the sequence builders want: the edits to apply to one gene's
 * reference sequence, sorted by position and non-overlapping.
 *
 * SeqEdit::alt points into the buffered records, so a list must not outlive the buffer or
 * survive a call to hap_buffer_next.
 */
typedef struct {
    SeqEdit *edits;
    size_t   n, m;
} SeqEditList;

void seq_edit_list_init(SeqEditList *list);

/*
 * seq_edit_list_push_sorted - Insert one edit, keeping the list ordered by position.
 *
 * Turning a HAP_REF background into the HAP_ALT haplotype is exactly this: putting the variant
 * under consideration back among the ones it is co-phased with.
 */
void seq_edit_list_push_sorted(SeqEditList *list, const SeqEdit edit);

void seq_edit_list_destroy(SeqEditList *list);

typedef struct HapBuffer HapBuffer;

/*
 * hap_buffer_open - Wrap a variant reader in a sliding buffer.
 *
 * span is how far either side of the variant being scored other variants can still matter -
 * BOUNDARY_SIZE + window_radius for a windowed score, the longest gene for a whole-gene one.
 * Nothing further away is retained.
 *
 * local_only makes every record behave as though it carried no genotype at all, whatever GT it
 * actually has: every allele is scored alone against the reference genome (never dropped for
 * unphased heterozygosity, never split across copies), exactly as it would be with no GT column
 * or FORMAT/GT present. The genotype itself is untouched otherwise - HapRecord::gt/ploidy/phased
 * still reflect what was read, so it still round-trips to output - only its effect on scoring is
 * suppressed. include_unphased is ignored when local_only is set, since nothing is ever dropped.
 *
 * Returns EXIT_SUCCESS on success, EXIT_FAILURE (having logged) otherwise.
 */
int hap_buffer_open(VariantReader *reader, bool include_unphased, bool local_only,
                    hts_pos_t span, HapBuffer **buffer);

/*
 * hap_buffer_next - Advance to the next variant to be scored.
 *
 * On return every variant within span bases of *record is buffered, so the haplotypes it
 * belongs to can be assembled. Anything further behind has been evicted, and *record is only
 * valid until the next call.
 *
 * Returns EXIT_SUCCESS, VARIANT_READER_EOF at clean end of input, or EXIT_FAILURE (having
 * logged) on a malformed or out-of-order record.
 */
int hap_buffer_next(HapBuffer *buffer, const HapRecord **record);

/*
 * hap_edits_collect - Assemble one haplotype over one gene.
 *
 * Gathers every buffered alternate allele carried by copy `hap` whose reference span lies
 * inside [lo, hi), as edits in gene coordinates. Alleles overlapping one already collected are
 * skipped with a warning: two edits cannot both replace the same base.
 *
 * Passing exclude/exclude_alt leaves that one allele out, which is the difference between a
 * haplotype and the same haplotype without the variant under consideration - HAP_ALT and
 * HAP_REF. Pass NULL/0 to collect everything.
 *
 * Parameters:
 *   buffer      - the buffer, positioned by hap_buffer_next.
 *   hap         - HAP_1 or HAP_2.
 *   chrom       - contig the gene is on.
 *   gene_start  - gene's 0-based start, which edit positions are relative to.
 *   lo, hi      - contig coordinates bounding which variants are close enough to matter.
 *   exclude     - record whose allele to leave out, or NULL.
 *   exclude_alt - index of that record's allele.
 *   out         - receives the edits; reset on every call.
 */
void hap_edits_collect(const HapBuffer *buffer, int hap, const char *chrom,
                       hts_pos_t gene_start, hts_pos_t lo, hts_pos_t hi,
                       const HapRecord *exclude, int exclude_alt, SeqEditList *out);

void hap_buffer_close(HapBuffer *buffer);

#endif /* HAPLOTYPE_H */
