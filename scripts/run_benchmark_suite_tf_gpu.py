#!/usr/bin/env python3
"""Orchestrates the CPU/GPU runtime benchmark suite: reference vs variant vs gene predictions,
naive vs haplotype, across tf/onnx/onnx_fp16 backends, each individual run sized (via a two-point
linear calibration) to land around 15 minutes on GPU.

Usage:
    scripts/run_benchmark_suite.py --calibrate-only     # size the fixtures, print the plan, stop
    scripts/run_benchmark_suite.py                       # calibrate (if not already done) + full run
    scripts/run_benchmark_suite.py --replicates 2        # fewer reps (CPU legs are the time sink)

All fixtures, the shared reference.bin, and results.csv persist under --workdir (default
data/benchmarks/suite/, already gitignored) so a `--calibrate-only` pass and the later full run don't
have to repeat work, and so a full run can be safely re-invoked after an interruption.

Disk-space handling: this suite runs on a machine with very little free disk (see docs on this in the
plan this was built from). --max-fixture-mb caps the reference FASTA size regardless of what
calibration would otherwise pick, and --min-free-gb aborts before any big write if headroom drops too
low. Pilot (calibration) fixtures are deleted immediately once no longer needed.
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
print(REPO_ROOT)
GEN_FIXTURE = REPO_ROOT / "scripts" / "gen_synthetic_fixture.py"

BASE_ENV = {
    "LD_LIBRARY_PATH": f"{os.environ.get('LD_LIBRARY_PATH', '')}:/usr/local/lib:/opt/onnxruntime/lib",
}

CONFIGS = {
    "tf-cpu":        dict(bin_dir="build-tf",  model_dir="models/tf",       backend="tf",   precision="fp32", device="cpu", env={"CUDA_VISIBLE_DEVICES": ""}),
    "tf-gpu":        dict(bin_dir="build-tf",  model_dir="models/tf",       backend="tf",   precision="fp32", device="gpu", env={}),
    "onnx-cpu-fp32": dict(bin_dir="build-ort", model_dir="models/onnx",      backend="onnx", precision="fp32", device="cpu", env={"CPLICEAI_ORT_EP": "cpu"}),
    "onnx-gpu-fp32": dict(bin_dir="build-ort", model_dir="models/onnx",      backend="onnx", precision="fp32", device="gpu", env={"CPLICEAI_ORT_EP": "cuda"}),
    "onnx-gpu-fp16": dict(bin_dir="build-ort", model_dir="models/onnx_fp16", backend="onnx", precision="fp16", device="gpu", env={"CPLICEAI_ORT_EP": "cuda"}),
}
CALIBRATION_CONFIG = "onnx-gpu-fp32"

WORKLOADS = ["reference", "variant_naive", "variant_haplotype", "gene_naive", "gene_haplotype"]

GENE_MIN_SIZE, GENE_MAX_SIZE = 5_000, 50_000  # -> mean 27.5kb/gene, "don't need to be precise"

CSV_FIELDS = ["workload", "config", "backend", "precision", "device", "replicate",
              "wall_seconds", "ort_run_seconds", "ort_calls", "fixture_size", "status", "notes"]

ORT_TIMING_RE = re.compile(r"Timing:\s*([\d.]+)s in Run\(\) across (\d+) calls")
TF_GPU_DEVICE_RE = re.compile(r"/device:GPU:\d+")


def log(msg):
    print(f"[run_benchmark_suite] {msg}", file=sys.stderr, flush=True)


def disk_guard(min_free_gb):
    free_gb = shutil.disk_usage(REPO_ROOT).free / 1e9
    if free_gb < min_free_gb:
        sys.exit(f"error: only {free_gb:.2f}GB free (< --min-free-gb={min_free_gb}) -- aborting "
                  f"before writing more data. Free up space and re-run.")
    log(f"disk check: {free_gb:.2f}GB free")


def binary_path(config_name, name):
    return REPO_ROOT / CONFIGS[config_name]["bin_dir"] / f"cpliceai_{name}"


def run(config_name, name, args, extra_env=None, timeout=3600):
    """Run a cpliceai_* binary under `config_name`'s env. Returns (returncode, output, wall_seconds)."""
    cfg = CONFIGS[config_name]
    env = {**os.environ, **BASE_ENV, **cfg["env"], **(extra_env or {})}
    cmd = [str(binary_path(config_name, name))] + [str(a) for a in args]
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=timeout, text=True)
        wall = time.perf_counter() - start
        return proc.returncode, proc.stdout, wall
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - start
        # exc.output can come back as bytes even with text=True -- when the timeout fires inside
        # Popen._communicate(), the partial buffer is captured before the text-decode step runs.
        partial = exc.output
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        return -1, (partial or "") + "\n[TIMEOUT]", wall


def parse_ort_timing(output):
    m = ORT_TIMING_RE.search(output)
    if not m:
        return None, None
    return float(m.group(1)), int(m.group(2))


def detect_tf_gpu(pilot_fasta, pilot_regions):
    """Probe whether this environment's libtensorflow actually has GPU support engaged."""
    if not binary_path("tf-gpu", "reference").exists():
        return False
    with tempfile.TemporaryDirectory() as td:
        out_bin = Path(td) / "probe.bin"
        rc, output, _ = run("tf-gpu", "reference", ["models/tf", pilot_fasta, pilot_regions, out_bin],
                             extra_env={"TF_CPP_MIN_LOG_LEVEL": "0"}, timeout=120)
    if rc != 0:
        log(f"tf-gpu probe failed to run (rc={rc}) -- treating as unavailable")
        return False
    found = bool(TF_GPU_DEVICE_RE.search(output))
    log(f"tf-gpu probe: {'GPU device detected' if found else 'no GPU device in TF log -- CPU-only libtensorflow'}")
    return found


def gen_genome(out_prefix, num_genes, seed=1):
    subprocess.run([sys.executable, str(GEN_FIXTURE), "genome", "--out-prefix", str(out_prefix),
                     "--num-genes", str(num_genes), "--min-size", str(GENE_MIN_SIZE),
                     "--max-size", str(GENE_MAX_SIZE), "--seed", str(seed)],
                    check=True, cwd=REPO_ROOT)
    fai = Path(f"{out_prefix}.fasta.fai").read_text().strip().split("\n")
    total_bp = sum(int(line.split("\t")[1]) for line in fai)
    return Path(f"{out_prefix}.fasta"), Path(f"{out_prefix}.regions.tsv"), total_bp


def gen_variants(fasta, regions, out, count, gt, seed=1):
    subprocess.run([sys.executable, str(GEN_FIXTURE), "variants", "--fasta", str(fasta),
                     "--regions", str(regions), "--out", str(out), "--count", str(count),
                     "--gt", gt, "--seed", str(seed)], check=True, cwd=REPO_ROOT)
    return Path(out)


def linear_target(x1, y1, x2, y2, target_y):
    """Two-point linear fit y = a + b*x, solved for x at y=target_y. Separates fixed overhead
    (model load, ~constant regardless of size) from the size-proportional term, the same delta
    trick scripts/benchmark.sh already uses (small vs large fixture). Never returns less than the
    larger calibration point -- extrapolating backwards below what was actually measured isn't useful."""
    if x2 == x1:
        return x2
    b = (y2 - y1) / (x2 - x1)
    if b <= 0:
        log(f"warning: non-positive slope in calibration ({y1:.1f}s@{x1} -> {y2:.1f}s@{x2}), "
            f"falling back to 4x the larger calibration point")
        return x2 * 4
    a = y1 - b * x1
    x_target = (target_y - a) / b
    return int(max(x_target, x2))


def rmtree_fixture(prefix):
    for suffix in (".fasta", ".fasta.fai", ".regions.tsv"):
        p = Path(f"{prefix}{suffix}")
        if p.exists():
            p.unlink()


def calibrate_reference(workdir, target_seconds, max_fixture_mb, min_free_gb):
    disk_guard(min_free_gb)
    log("calibrating reference workload (2-point probe on onnx-gpu-fp32)...")
    pilot_dir = workdir / "_pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)

    points = []
    for label, n in (("p1", 15), ("p2", 60)):
        prefix = pilot_dir / label
        fasta, regions, bp = gen_genome(prefix, num_genes=n)
        with tempfile.TemporaryDirectory() as td:
            out_bin = Path(td) / "ref.bin"
            rc, output, wall = run(CALIBRATION_CONFIG, "reference",
                                    ["models/onnx", fasta, regions, out_bin], timeout=1800)
        if rc != 0:
            sys.exit(f"error: reference calibration pilot ({label}, {n} genes) failed:\n{output[-2000:]}")
        log(f"  pilot {label}: {n} genes, {bp:,}bp -> {wall:.1f}s")
        points.append((bp, wall))
        rmtree_fixture(prefix)

    (x1, y1), (x2, y2) = points
    max_bp = max_fixture_mb * 1_000_000
    target_bp = linear_target(x1, y1, x2, y2, target_seconds)
    capped = target_bp > max_bp
    final_bp = min(target_bp, max_bp)
    final_num_genes = max(5, round(final_bp / ((GENE_MIN_SIZE + GENE_MAX_SIZE) / 2)))
    if capped:
        log(f"  extrapolated {target_bp/1e6:.1f}Mb exceeds --max-fixture-mb={max_fixture_mb} -- "
            f"capping to {final_bp/1e6:.1f}Mb ({final_num_genes} genes); this run will land under "
            f"{target_seconds}s, not at it")
    else:
        log(f"  extrapolated {final_bp/1e6:.1f}Mb ({final_num_genes} genes) for ~{target_seconds}s on {CALIBRATION_CONFIG}")

    disk_guard(min_free_gb)
    final_prefix = workdir / "fixture"
    fasta, regions, bp = gen_genome(final_prefix, num_genes=final_num_genes)
    shutil.rmtree(pilot_dir, ignore_errors=True)
    return {"fasta": str(fasta), "regions": str(regions), "bp": bp, "num_genes": final_num_genes, "capped": capped}


def calibrate_variant_count(binary_name, fasta, regions, reference_bin, workdir, target_seconds,
                             pilot_counts=(100, 500), slow_pilot_seconds=90):
    """pilot_counts defaults suit predict_variant's ~per-window cost; predict_gene is far more
    expensive per variant (whole-gene inference + per-variant HAP_REF recompute -- the very reason
    it gets its own calibration), so its caller passes much smaller pilot sizes. Even so, if the
    *first* (smaller) pilot alone already takes a while, skip the second, larger pilot rather than
    risk it running long -- extrapolate from the origin instead (assume ~0 fixed overhead, which
    undercounts slightly since there is always some model-load cost, but errs toward a shorter
    final run rather than a very long calibration step)."""
    log(f"calibrating {binary_name} variant count (probe on onnx-gpu-fp32, haplotype mode)...")
    c1, c2 = pilot_counts
    vf1 = gen_variants(fasta, regions, workdir / f"_cal_{binary_name}_v1.tsv", c1, "phased")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.tsv"
        rc, output, y1 = run(CALIBRATION_CONFIG, binary_name,
                              [vf1, reference_bin, "models/onnx", fasta, regions, out], timeout=1800)
    if rc != 0:
        sys.exit(f"error: {binary_name} calibration pilot (v1, {c1} variants) failed:\n{output[-2000:]}")
    log(f"  pilot v1: {c1} variants -> {y1:.1f}s")
    vf1.unlink()

    if y1 >= slow_pilot_seconds:
        final_count = max(int(c1 * target_seconds / y1), c1)
        log(f"  pilot v1 already took {y1:.1f}s (>= {slow_pilot_seconds}s) -- skipping the larger "
            f"pilot and extrapolating from this point alone")
    else:
        vf2 = gen_variants(fasta, regions, workdir / f"_cal_{binary_name}_v2.tsv", c2, "phased")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.tsv"
            rc, output, y2 = run(CALIBRATION_CONFIG, binary_name,
                                  [vf2, reference_bin, "models/onnx", fasta, regions, out], timeout=1800)
        if rc != 0:
            sys.exit(f"error: {binary_name} calibration pilot (v2, {c2} variants) failed:\n{output[-2000:]}")
        log(f"  pilot v2: {c2} variants -> {y2:.1f}s")
        vf2.unlink()
        final_count = linear_target(c1, y1, c2, y2, target_seconds)

    log(f"  extrapolated {final_count} variants for ~{target_seconds}s on {CALIBRATION_CONFIG}")
    return final_count


def calibrate(workdir, target_seconds, max_fixture_mb, min_free_gb):
    plan_path = workdir / "calibration.json"
    if plan_path.exists():
        log(f"reusing existing calibration at {plan_path} (delete it to recalibrate)")
        return json.loads(plan_path.read_text())

    workdir.mkdir(parents=True, exist_ok=True)
    ref = calibrate_reference(workdir, target_seconds, max_fixture_mb, min_free_gb)
    fasta, regions = ref["fasta"], ref["regions"]

    disk_guard(min_free_gb)
    log("building shared reference.bin (untimed prep, reused by every variant/gene run)...")
    reference_bin = workdir / "reference.bin"
    rc, output, wall = run("onnx-cpu-fp32", "reference", ["models/onnx", fasta, regions, reference_bin], timeout=3600)
    if rc != 0:
        sys.exit(f"error: building shared reference.bin failed:\n{output[-2000:]}")
    log(f"  built in {wall:.1f}s -> {reference_bin}")

    variant_count = calibrate_variant_count("predict_variant", fasta, regions, reference_bin, workdir, target_seconds)
    # predict_gene is far more expensive per variant (whole-gene inference + per-variant HAP_REF
    # recompute) -- much smaller pilots, see calibrate_variant_count's docstring.
    gene_count = calibrate_variant_count("predict_gene", fasta, regions, reference_bin, workdir,
                                          target_seconds, pilot_counts=(10, 40))

    plan = {
        "reference": ref,
        "reference_bin": str(reference_bin),
        "variant_count": variant_count,
        "gene_count": gene_count,
    }
    plan_path.write_text(json.dumps(plan, indent=2))
    log(f"calibration complete -> {plan_path}")
    return plan


def finalize_fixtures(workdir, plan):
    fasta, regions = plan["reference"]["fasta"], plan["reference"]["regions"]
    paths = {}
    for wl, count_key, gt in (
        ("variant_naive", "variant_count", "none"),
        ("variant_haplotype", "variant_count", "phased"),
        ("gene_naive", "gene_count", "none"),
        ("gene_haplotype", "gene_count", "phased"),
    ):
        out = workdir / f"{wl}.tsv"
        if not out.exists():
            gen_variants(fasta, regions, out, plan[count_key], gt)
        paths[wl] = out
    return paths


def load_done(results_path):
    """(workload, config, replicate) tuples that already have a successful/skipped row -- lets a
    re-invocation (e.g. after a crash) resume instead of redoing already-good, possibly hours-long
    runs. Only "ok"/"skipped" count as done; "error" rows are retried."""
    done = set()
    if not results_path.exists():
        return done
    with open(results_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["status"] in ("ok", "skipped"):
                done.add((row["workload"], row["config"], int(row["replicate"])))
    return done


def run_matrix(workdir, plan, variant_files, configs_to_run, replicates, results_path, tf_gpu_ok, run_timeout):
    fasta, regions, reference_bin = plan["reference"]["fasta"], plan["reference"]["regions"], plan["reference_bin"]
    is_new = not results_path.exists()
    done = load_done(results_path)
    if done:
        log(f"resuming: {len(done)} (workload, config, replicate) combo(s) already done in {results_path}, skipping those")
    with open(results_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()

        for workload in WORKLOADS:
            for config_name in configs_to_run:
                cfg = CONFIGS[config_name]
                if config_name == "tf-gpu" and not tf_gpu_ok:
                    if (workload, config_name, 0) in done:
                        continue
                    writer.writerow({"workload": workload, "config": config_name, "backend": cfg["backend"],
                                      "precision": cfg["precision"], "device": cfg["device"], "replicate": 0,
                                      "wall_seconds": "", "ort_run_seconds": "", "ort_calls": "",
                                      "fixture_size": "", "status": "skipped",
                                      "notes": "no GPU-capable libtensorflow in this environment"})
                    fh.flush()
                    continue

                for rep in range(1, replicates + 1):
                    if (workload, config_name, rep) in done:
                        log(f"{workload} / {config_name} / rep {rep}/{replicates} -- already done, skipping")
                        continue
                    log(f"{workload} / {config_name} / rep {rep}/{replicates}")
                    extra_env = {"CPLICEAI_ORT_TIMING": "1"} if cfg["backend"] == "onnx" else {}
                    with tempfile.TemporaryDirectory() as td:
                        out_path = Path(td) / "out"
                        if workload == "reference":
                            args = [cfg["model_dir"], fasta, regions, out_path]
                            fixture_size = plan["reference"]["num_genes"]
                            binary_name = "reference"
                        else:
                            binary_name = "predict_variant" if workload.startswith("variant") else "predict_gene"
                            vf = variant_files[workload]
                            args = [vf, reference_bin, cfg["model_dir"], fasta, regions, out_path]
                            fixture_size = plan["variant_count"] if "variant" in workload else plan["gene_count"]

                        rc, output, wall = run(config_name, binary_name, args, extra_env=extra_env, timeout=run_timeout)

                    status = "ok" if rc == 0 else "error"
                    ort_run_s, ort_calls = parse_ort_timing(output) if cfg["backend"] == "onnx" else (None, None)
                    notes = "" if rc == 0 else output[-500:].replace("\n", " | ")
                    if rep == 1 and replicates > 1:
                        notes = ("cold run, exclude from summary" + ("; " + notes if notes else "")).strip("; ")

                    writer.writerow({
                        "workload": workload, "config": config_name, "backend": cfg["backend"],
                        "precision": cfg["precision"], "device": cfg["device"], "replicate": rep,
                        "wall_seconds": f"{wall:.3f}", "ort_run_seconds": ort_run_s or "",
                        "ort_calls": ort_calls or "", "fixture_size": fixture_size,
                        "status": status, "notes": notes,
                    })
                    fh.flush()
                    if rc != 0:
                        log(f"  FAILED (rc={rc}): {output[-500:]}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", default=str(REPO_ROOT / "data" / "benchmarks" / "suite"))
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--target-seconds", type=int, default=900)
    p.add_argument("--max-fixture-mb", type=int, default=300)
    p.add_argument("--min-free-gb", type=float, default=1.0)
    p.add_argument("--configs", default=",".join(CONFIGS.keys()),
                    help="comma-separated subset of: " + ",".join(CONFIGS.keys()))
    p.add_argument("--calibrate-only", action="store_true")
    p.add_argument("--run-timeout", type=int, default=10800,
                    help="per-run subprocess timeout in seconds (default 10800 = 3h -- CPU legs of "
                         "the haplotype workloads can legitimately run past 3600s)")
    args = p.parse_args()

    for c in ("build-ort", "build-tf"):
        if not (REPO_ROOT / c).is_dir():
            sys.exit(f"error: {c}/ not built. See Verification steps 1-2 in the plan "
                      f"(cmake -S . -B {c} -DCPLICEAI_INFERENCE_BACKEND=... && cmake --build {c} -j)")

    workdir = Path(args.workdir)
    disk_guard(args.min_free_gb)

    plan = calibrate(workdir, args.target_seconds, args.max_fixture_mb, args.min_free_gb)
    print(json.dumps(plan, indent=2))
    if args.calibrate_only:
        return

    variant_files = finalize_fixtures(workdir, plan)

    configs_to_run = [c.strip() for c in args.configs.split(",") if c.strip()]
    tf_gpu_ok = False
    if "tf-gpu" in configs_to_run:
        with tempfile.TemporaryDirectory() as td:
            pilot_prefix = Path(td) / "tfprobe"
            gen_genome(pilot_prefix, num_genes=3, seed=99)
            tf_gpu_ok = detect_tf_gpu(f"{pilot_prefix}.fasta", f"{pilot_prefix}.regions.tsv")

    disk_guard(args.min_free_gb)
    run_matrix(workdir, plan, variant_files, configs_to_run, args.replicates,
               workdir / "results.csv", tf_gpu_ok, args.run_timeout)
    log(f"done -> {workdir / 'results.csv'}")


if __name__ == "__main__":
    main()
