#!/usr/bin/env python3
"""Summarize scripts/run_benchmark_suite.py's results.csv: median/min wall time per
(workload, config), a pivoted markdown table for eyeballing the full matrix at a glance, and the
naive-vs-haplotype overhead per config -- the actual comparison this whole suite exists for.

Usage:
    scripts/summarize_benchmarks.py [--results data/benchmarks/suite/results.csv] [--out FILE.md]

Run with the /home/dev/.venv Python (has polars); plain `python3 scripts/summarize_benchmarks.py`
already resolves there if that venv is first on PATH.
"""

import argparse
import sys

import polars as pl

WORKLOAD_ORDER = ["reference", "variant_naive", "variant_haplotype", "gene_naive", "gene_haplotype"]
CONFIG_ORDER = ["tf-cpu", "tf-gpu", "onnx-cpu-fp32", "onnx-gpu-fp32", "onnx-gpu-fp16"]


def load(results_path):
    df = pl.read_csv(results_path, infer_schema_length=None)
    ok = df.filter(pl.col("status") == "ok")
    if ok.is_empty():
        sys.exit(f"no successful ('ok') rows in {results_path} -- nothing to summarize yet")

    # Drop each (workload, config)'s first replicate when it has more than one -- cold-cache run,
    # same rationale scripts/benchmark.sh already uses.
    ok = ok.with_columns(pl.len().over("workload", "config").alias("_n"))
    ok = ok.filter((pl.col("_n") == 1) | (pl.col("replicate") > 1)).drop("_n")
    return df, ok


def summarize(ok):
    return (
        ok.group_by("workload", "config", "backend", "precision", "device")
        .agg(
            pl.len().alias("n"),
            pl.col("wall_seconds").median().alias("median_s"),
            pl.col("wall_seconds").min().alias("min_s"),
            pl.col("ort_run_seconds").median().alias("median_ort_run_s"),
        )
        .sort(["workload", "config"])
    )


def ordered(values, order):
    known = [v for v in order if v in values]
    unknown = sorted(set(values) - set(order))
    return known + unknown


def print_table(summary):
    print(summary.select("workload", "config", "n", "median_s", "min_s", "median_ort_run_s"))


def print_pivot(summary, all_df):
    workloads = ordered(summary["workload"].unique().to_list(), WORKLOAD_ORDER)
    configs = ordered(summary["config"].unique().to_list(), CONFIG_ORDER)
    skipped = set(
        all_df.filter(pl.col("status") == "skipped")
        .select("workload", "config").unique().iter_rows()
    )

    header = "| workload | " + " | ".join(configs) + " |"
    sep = "|---|" + "---|" * len(configs)
    lines = [header, sep]
    for wl in workloads:
        row = [wl]
        for cfg in configs:
            cell = summary.filter((pl.col("workload") == wl) & (pl.col("config") == cfg))
            if cell.height:
                row.append(f"{cell['median_s'][0]:.1f}s")
            elif (wl, cfg) in skipped:
                row.append("skipped")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def print_naive_vs_haplotype(summary):
    lines = []
    for prefix, label in (("variant", "predict_variant"), ("gene", "predict_gene")):
        lines.append(f"\n{label}: haplotype overhead vs naive (median wall time)")
        for cfg in ordered(summary["config"].unique().to_list(), CONFIG_ORDER):
            naive = summary.filter((pl.col("workload") == f"{prefix}_naive") & (pl.col("config") == cfg))
            hap = summary.filter((pl.col("workload") == f"{prefix}_haplotype") & (pl.col("config") == cfg))
            if naive.height and hap.height:
                n, h = naive["median_s"][0], hap["median_s"][0]
                lines.append(f"  {cfg:<16} naive={n:.1f}s  haplotype={h:.1f}s  ({100*(h-n)/n:+.1f}%, {h/n:.2f}x)")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="data/benchmarks/suite/results.csv")
    p.add_argument("--out", help="also write the pivoted markdown table to this file")
    args = p.parse_args()

    all_df, ok = load(args.results)
    summary = summarize(ok)

    print(f"# {args.results}\n")
    print_table(summary)
    print()
    pivot_md = print_pivot(summary, all_df)
    print(pivot_md)
    print(print_naive_vs_haplotype(summary))

    n_error = all_df.filter(pl.col("status") == "error").height
    n_skipped = all_df.filter(pl.col("status") == "skipped").height
    if n_error or n_skipped:
        print(f"\n({n_error} failed run(s), {n_skipped} skipped config(s) -- see status/notes columns in {args.results})")

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("# Benchmark results\n\n")
            fh.write(pivot_md + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
