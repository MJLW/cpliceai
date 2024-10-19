#ifndef RANGE_HEADER
#define RANGE_HEADER

#include <inttypes.h>
#include <stdint.h>

typedef struct
{
    int64_t start, end;
} Range;

/*
* ranges_overlap - Do ranges overlap?
*
* @param r1: First range
* @param r2: Second range
* @return 1 if the ranges overlap, 0 otherwise
*/
int ranges_overlap(Range r1, Range r2);

/*
* intersect_ranges - Calculate an intersection of two ranges
*
* @param r1: First range
* @param r2: Second range
* @param *intersection: Pointer to a Range which will be set to the intersection
* @return 1 if *intersection is set, 0 if it is not set
*/
int intersect_ranges(Range r1, Range r2, Range *intersection);

/*
* range_len - Get the length of a range
* @param r: Range to calculate the length of
* @return length of the range
*/
int64_t range_len(Range r);

/*
* scale_range - Calculates a scaled range
*
* @param r: Original range
* @param scale: Integer to scale the range by
* @return Scaled range
*
*/
Range scale_range(Range r, int scale);

/*
* shift_range - Shift position in a range by an offset
* 
* @param *r: Range to shift
* @param offset: Offset to shift the range by
*/
void shift_range(Range *r, int offset);


#endif

