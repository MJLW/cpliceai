#include "range.h"


int ranges_overlap(Range r1, Range r2) {
    return r1.start <= r2.end && r2.start <= r1.end;
}

int intersect_ranges(Range r1, Range r2, Range *intersection) {
    if (!ranges_overlap(r1, r2)) return 0;

    if (r1.start < r2.start) {
        if (r1.end > r2.end) {
            *intersection = (Range) {r2.start, r2.end};
        } else {
            *intersection = (Range) {r2.start, r1.end};
        }
    } else {
        if (r1.end > r2.end) {
            *intersection = (Range) {r1.start, r2.end};
        } else {
            *intersection = (Range) {r1.start, r1.end};
        }
    }

    return 1;
}

int range_len(Range r) {
    return r.end - r.start;
}

Range scale_range(Range r, int scale) {
    return (Range) {r.start * scale, r.end * scale};
}

void shift_range(Range *r, int offset) {
    r->start += offset;
    r->end += offset;
}
