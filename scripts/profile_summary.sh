#!/usr/bin/env bash
# Summarises ONNX Runtime profile JSON written by CPLICEAI_ORT_PROFILE=<prefix>.
#
# Aggregates per-node timings by op type and by execution provider, so you can see where GPU
# time actually goes -- specifically whether Transpose nodes are a meaningful share (the NCHW
# layout question) and whether any node silently fell back to CPUExecutionProvider.
#
# Usage:
#   CPLICEAI_ORT_PROFILE=/tmp/prof CPLICEAI_ORT_EP=cuda ./build/cpliceai_reference ...
#   ./scripts/profile_summary.sh /tmp/prof_model*.json
#
# Note the durations here cover only what ORT measures *inside* Run(). Compare the reported
# total against the process's own wall time: a large gap means the bottleneck is outside
# inference entirely (host-side encoding, I/O, serialization), and no amount of kernel/layout
# tuning will move it.

set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <profile.json> [profile2.json ...]" >&2
    exit 1
fi

awk '
function field_num(line, key,   s) {
    if (match(line, "\"" key "\" *: *[0-9]+")) {
        s = substr(line, RSTART, RLENGTH); sub(/^.*: */, "", s); return s + 0
    }
    return 0
}
function field_str(line, key,   s) {
    if (match(line, "\"" key "\" *: *\"[^\"]*\"")) {
        s = substr(line, RSTART, RLENGTH); sub(/^.*: *"/, "", s); sub(/"$/, "", s); return s
    }
    return ""
}
/"cat" *: *"Session"/ {
    name = field_str($0, "name")
    session[name] += field_num($0, "dur")
    next
}
/"cat" *: *"Node"/ {
    op = field_str($0, "op_name")
    if (op == "") next
    d = field_num($0, "dur")
    total += d; n_nodes++
    by_op[op] += d; cnt[op]++
    by_prov[field_str($0, "provider")] += d
}
END {
    if (total == 0) { print "No Node events found -- was the profile written by a run that executed?"; exit 1 }
    printf "Total in-Run() node time: %.3f s across %d node executions\n\n", total/1e6, n_nodes

    print "By execution provider:"
    for (p in by_prov) printf "  %-28s %10.3f s  %6.2f%%\n", (p == "" ? "(unknown)" : p), by_prov[p]/1e6, 100*by_prov[p]/total
    print ""

    print "By op type (descending):"
    for (o in by_op) printf "  %-28s %10.3f s  %6.2f%%  (%d calls)\n", o, by_op[o]/1e6, 100*by_op[o]/total, cnt[o] | "sort -k2 -gr"
    close("sort -k2 -gr")
    print ""

    print "Session-level (not included in the node total above):"
    for (s in session) printf "  %-28s %10.3f s\n", s, session[s]/1e6
}
' "$@"
