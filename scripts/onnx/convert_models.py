#!/usr/bin/env python3
"""
Convert the committed SpliceAI TensorFlow SavedModels (models/tf/spliceai1..5) to ONNX
(fp32, models/onnx/) and a float16-weights variant (models/onnx_fp16/).

Usage:
    python scripts/onnx/convert_models.py

Requires the pinned packages in scripts/onnx/requirements.txt (a throwaway venv is
fine -- nothing here needs to be installed in the project's own build/runtime).
"""
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
from onnx import helper, numpy_helper, shape_inference
from onnxconverter_common import float16

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
TF_DIR = MODELS_DIR / "tf"
ONNX_DIR = MODELS_DIR / "onnx"
ONNX_FP16_DIR = MODELS_DIR / "onnx_fp16"
OPSET = 17
NUM_MODELS = 5
NUM_CHANNELS = 4
# Two different sequence lengths, to exercise the dynamic seq-len axis.
CHECK_SEQ_LENS = [15000, 40000]
MAX_ABS_DIFF_FP32 = 1e-5

# --- Dilated-conv graph surgery -----------------------------------------------------------
#
# tf2onnx's raw conversion of this model carries ~200 Transpose nodes per model (vs. an
# expected ~2), because Keras's Conv1D isn't natively represented in the traced SavedModel
# graph: TF lowers every one of the 39 conv layers to
# `[SpaceToBatchND ->] ExpandDims(x2) -> Conv2D(NHWC) -> Squeeze [-> BatchToSpaceND] -> BiasAdd`
# (24 of the 39 layers wrapped in SpaceToBatchND/BatchToSpaceND, TF's standard trick for
# emulating dilated convolution via a dilation=1 Conv2D), and ONNX's Conv has no
# channels-last mode at all, so tf2onnx has no choice but to transpose around every one.
# onnx-simplifier does not help here (tested: 0 Transpose reduction) because its passes need
# concrete shapes to prove a fusion safe, and the sequence-length axis must stay dynamic
# (every call site feeds a different length).
#
# This replaces each of the 39 conv layers with a single native ONNX `Conv` node (which
# supports dilation and "same" padding natively, no SpaceToBatchND emulation needed),
# bringing the Transpose count down to ~53 (down from ~200): one global input transpose
# (NWC->NCW), one pre-existing transpose reconciling two skip-connection branches that
# predates this surgery, and 51 more from a real, empirically-confirmed layout quirk in the
# original graph -- every dilated layer's SpaceToBatchND/BatchToSpaceND round-trip flips its
# own output to NWC (regardless of its input layout), while plain layers stay NCW throughout;
# each of the 24 dilated layers needs a matching NCW->NWC transpose on its output to keep
# everything downstream (BatchNorm/Relu/skip-Add, all untouched, all still expecting whatever
# layout the original graph gave them at that point) working unmodified.
#
# The layer table below (kernel width + dilation rate) is the SpliceAI-10k architecture,
# extracted directly from the traced TF GraphDef (not memorized/assumed) -- see
# find_true_inputs.py-style inspection in the project history. Valid for all 5 committed
# models since they share this architecture (only weights differ). This surgery pass is
# architecture-specific by construction; it is not a generic ONNX optimization and would
# need updating if the model architecture ever changes.

LAYER_KERNEL_DILATION = {}
for _n in [1, 2, 11, 20, 29, 38, 39]:
    LAYER_KERNEL_DILATION[f"conv1d_{_n}"] = (1, 1)
for _n in range(3, 11):
    LAYER_KERNEL_DILATION[f"conv1d_{_n}"] = (11, 1)
for _n in range(12, 20):
    LAYER_KERNEL_DILATION[f"conv1d_{_n}"] = (11, 4)
for _n in range(21, 29):
    LAYER_KERNEL_DILATION[f"conv1d_{_n}"] = (21, 10)
for _n in range(30, 38):
    LAYER_KERNEL_DILATION[f"conv1d_{_n}"] = (41, 25)
assert len(LAYER_KERNEL_DILATION) == 39

# True semantic input (TF tensor name, pre-"StatefulPartitionedCall/" prefix) per conv layer
# -- i.e. what feeds it *before* any ExpandDims/SpaceToBatchND scaffolding, derived by
# walking each Conv2D's input backward through known-scaffolding ops (ExpandDims,
# SpaceToBatchND) until hitting a real op (Relu/Add) or the graph input.
TF_TRUE_INPUT = {
    "conv1d_1": "input_1", "conv1d_2": "model_1/conv1d_1/BiasAdd:0",
    "conv1d_3": "model_1/activation_1/Relu:0", "conv1d_4": "model_1/activation_2/Relu:0",
    "conv1d_5": "model_1/activation_3/Relu:0", "conv1d_6": "model_1/activation_4/Relu:0",
    "conv1d_7": "model_1/activation_5/Relu:0", "conv1d_8": "model_1/activation_6/Relu:0",
    "conv1d_9": "model_1/activation_7/Relu:0", "conv1d_10": "model_1/activation_8/Relu:0",
    "conv1d_11": "model_1/add_4/add:0", "conv1d_12": "model_1/activation_9/Relu:0",
    "conv1d_13": "model_1/activation_10/Relu:0", "conv1d_14": "model_1/activation_11/Relu:0",
    "conv1d_15": "model_1/activation_12/Relu:0", "conv1d_16": "model_1/activation_13/Relu:0",
    "conv1d_17": "model_1/activation_14/Relu:0", "conv1d_18": "model_1/activation_15/Relu:0",
    "conv1d_19": "model_1/activation_16/Relu:0", "conv1d_20": "model_1/add_9/add:0",
    "conv1d_21": "model_1/activation_17/Relu:0", "conv1d_22": "model_1/activation_18/Relu:0",
    "conv1d_23": "model_1/activation_19/Relu:0", "conv1d_24": "model_1/activation_20/Relu:0",
    "conv1d_25": "model_1/activation_21/Relu:0", "conv1d_26": "model_1/activation_22/Relu:0",
    "conv1d_27": "model_1/activation_23/Relu:0", "conv1d_28": "model_1/activation_24/Relu:0",
    "conv1d_29": "model_1/add_14/add:0", "conv1d_30": "model_1/activation_25/Relu:0",
    "conv1d_31": "model_1/activation_26/Relu:0", "conv1d_32": "model_1/activation_27/Relu:0",
    "conv1d_33": "model_1/activation_28/Relu:0", "conv1d_34": "model_1/activation_29/Relu:0",
    "conv1d_35": "model_1/activation_30/Relu:0", "conv1d_36": "model_1/activation_31/Relu:0",
    "conv1d_37": "model_1/activation_32/Relu:0", "conv1d_38": "model_1/add_19/add:0",
    "conv1d_39": "model_1/cropping1d_1/strided_slice:0",
}
assert len(TF_TRUE_INPUT) == 39


def _onnx_input_name(tf_name: str) -> str:
    return tf_name if tf_name == "input_1" else f"StatefulPartitionedCall/{tf_name}"


def simplify_dilated_convs(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    initializer_by_name = {i.name: i for i in graph.initializer}

    # Detect each layer's true-input layout (NCW vs NWC) empirically via shape inference,
    # rather than assuming -- see the module docstring on the SpaceToBatchND layout flip.
    inferred = shape_inference.infer_shapes(model, check_type=True, strict_mode=False)
    value_info_by_name = {vi.name: vi for vi in inferred.graph.value_info}

    def shape_of(name):
        vi = value_info_by_name.get(name)
        if vi:
            return [d.dim_param or d.dim_value for d in vi.type.tensor_type.shape.dim]
        for i in graph.input:
            if i.name == name:
                return [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]
        return None

    needs_pre_transpose = {}
    for layer in LAYER_KERNEL_DILATION:
        name = _onnx_input_name(TF_TRUE_INPUT[layer])
        shape = shape_of(name)
        if shape is None or len(shape) != 3:
            raise SystemExit(f"simplify_dilated_convs: bad/missing shape for {layer}'s input {name}: {shape}")
        is_nwc = isinstance(shape[2], int)
        is_ncw = isinstance(shape[1], int)
        if is_nwc == is_ncw:
            raise SystemExit(f"simplify_dilated_convs: ambiguous layout for {layer}'s input {name}: {shape}")
        needs_pre_transpose[layer] = is_nwc

    def find_node(pred):
        matches = [n for n in graph.node if pred(n)]
        if len(matches) != 1:
            raise SystemExit(f"simplify_dilated_convs: expected 1 match, got {len(matches)}: {[n.name for n in matches]}")
        return matches[0]

    new_nodes = []
    new_initializers = []
    rewire_map = {}

    for layer, (kernel_w, dilation) in LAYER_KERNEL_DILATION.items():
        conv_node = find_node(
            lambda n, layer=layer: n.op_type == "Conv" and f"/{layer}/Conv1D" in n.name
            and "__" not in n.name.split(f"/{layer}/Conv1D")[-1]
        )
        bias_node = find_node(lambda n, layer=layer: n.name.endswith(f"/{layer}/BiasAdd"))

        kernel_init = initializer_by_name[conv_node.input[1]]
        kernel_arr = numpy_helper.to_array(kernel_init)
        if kernel_arr.shape[2] != 1 or kernel_arr.shape[3] != kernel_w:
            raise SystemExit(f"simplify_dilated_convs: unexpected kernel shape for {layer}: {kernel_arr.shape}")
        out_ch, in_ch = kernel_arr.shape[0], kernel_arr.shape[1]
        new_kernel_arr = kernel_arr.reshape(out_ch, in_ch, kernel_w)
        new_kernel_name = f"{layer}_native_kernel"
        new_initializers.append(numpy_helper.from_array(new_kernel_arr, name=new_kernel_name))

        bias_operand = next(i for i in bias_node.input if i in initializer_by_name)
        bias_arr = numpy_helper.to_array(initializer_by_name[bias_operand]).reshape(out_ch)
        new_bias_name = f"{layer}_native_bias"
        new_initializers.append(numpy_helper.from_array(bias_arr, name=new_bias_name))

        true_input = _onnx_input_name(TF_TRUE_INPUT[layer])
        conv_input = true_input
        if needs_pre_transpose[layer]:
            transposed_name = f"{layer}_input_ncw"
            new_nodes.append(helper.make_node(
                "Transpose", inputs=[true_input], outputs=[transposed_name],
                name=f"{layer}_pre_transpose", perm=[0, 2, 1],
            ))
            conv_input = transposed_name

        conv_output = f"{layer}_native_conv_output"
        pad_total = (kernel_w - 1) * dilation
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        new_nodes.append(helper.make_node(
            "Conv", inputs=[conv_input, new_kernel_name, new_bias_name], outputs=[conv_output],
            name=f"{layer}_native", dilations=[dilation], strides=[1], kernel_shape=[kernel_w],
            pads=[pad_left, pad_right], group=1,
        ))
        # Dilated layers' original output was NWC (SpaceToBatchND/BatchToSpaceND round-trip
        # flips layout); plain layers' original output was NCW, matching native Conv
        # directly. Empirically confirmed this correlates exactly with dilation > 1.
        if dilation > 1:
            new_output = f"{layer}_native_output"
            new_nodes.append(helper.make_node(
                "Transpose", inputs=[conv_output], outputs=[new_output],
                name=f"{layer}_post_transpose", perm=[0, 2, 1],
            ))
        else:
            new_output = conv_output
        rewire_map[bias_node.output[0]] = new_output

    # Rewire every node (new and original) whose input references an old BiasAdd output --
    # except a pre-transpose's own source input, which must keep reading the real upstream
    # tensor rather than being redirected to itself.
    pre_transpose_inputs = {n.input[0] for n in new_nodes if n.op_type == "Transpose" and n.name.endswith("_pre_transpose")}
    for n in new_nodes + list(graph.node):
        for i, inp in enumerate(n.input):
            if inp in rewire_map and inp not in pre_transpose_inputs:
                n.input[i] = rewire_map[inp]

    all_nodes = new_nodes + list(graph.node)
    all_initializers = list(graph.initializer) + new_initializers

    by_output = {}
    for n in all_nodes:
        for o in n.output:
            by_output[o] = n

    # Dead-code elimination: keep only nodes reachable backward from the graph outputs.
    keep_node_names = set()
    frontier = [by_output[o.name] for o in graph.output]
    while frontier:
        n = frontier.pop()
        if n.name in keep_node_names:
            continue
        keep_node_names.add(n.name)
        for i in n.input:
            producer = by_output.get(i)
            if producer is not None:
                frontier.append(producer)
    kept_nodes = [n for n in all_nodes if n.name in keep_node_names]

    # Topological sort (Kahn's algorithm) -- required since new_nodes' insertion order
    # doesn't necessarily respect cross-layer dependencies (e.g. conv1d_2 depends on
    # conv1d_1's new output).
    in_degree = {}
    consumers_of = defaultdict(list)
    kept_by_output = {}
    for n in kept_nodes:
        for o in n.output:
            kept_by_output[o] = n
    for n in kept_nodes:
        deps = {kept_by_output[i].name for i in n.input if i in kept_by_output}
        in_degree[n.name] = len(deps)
        for d in deps:
            consumers_of[d].append(n.name)

    by_name = {n.name: n for n in kept_nodes}
    ready = [name for name, deg in in_degree.items() if deg == 0]
    sorted_names = []
    while ready:
        name = ready.pop()
        sorted_names.append(name)
        for c in consumers_of[name]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                ready.append(c)
    if len(sorted_names) != len(kept_nodes):
        raise SystemExit(f"simplify_dilated_convs: cycle or disconnected graph after surgery ({len(sorted_names)} vs {len(kept_nodes)})")
    sorted_nodes = [by_name[n] for n in sorted_names]

    used_tensors = {i for n in sorted_nodes for i in n.input}
    pruned_initializers = [init for init in all_initializers if init.name in used_tensors]

    new_graph = helper.make_graph(sorted_nodes, graph.name, graph.input, graph.output, pruned_initializers)
    new_model = helper.make_model(new_graph, opset_imports=model.opset_import, ir_version=model.ir_version)
    onnx.checker.check_model(new_model)
    return new_model


# --- Conversion pipeline ------------------------------------------------------------------

def convert_to_onnx(saved_model_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "tf2onnx.convert",
            "--saved-model", str(saved_model_dir),
            "--tag", "serve",
            "--signature_def", "serving_default",
            "--opset", str(OPSET),
            "--output", str(output_path),
        ],
        check=True,
    )

    print("  Applying dilated-conv graph surgery (see module docstring)...")
    model = onnx.load(str(output_path))
    before = sum(1 for n in model.graph.node if n.op_type == "Transpose")
    model = simplify_dilated_convs(model)
    after = sum(1 for n in model.graph.node if n.op_type == "Transpose")
    print(f"  Transpose nodes: {before} -> {after}")
    onnx.save(model, str(output_path))


def check_graph_shape(onnx_path: Path) -> None:
    model = onnx.load(str(onnx_path))
    inp = model.graph.input[0]
    dims = inp.type.tensor_type.shape.dim
    seq_dim = dims[1]
    is_dynamic = seq_dim.dim_param != "" or (seq_dim.dim_value == 0 and seq_dim.dim_param == "")
    if not is_dynamic:
        raise SystemExit(
            f"{onnx_path.name}: sequence-length axis (dim 1) is NOT dynamic "
            f"(dim_value={seq_dim.dim_value!r}) -- every call site feeds a different "
            f"length, this would break past whatever length was frozen in at conversion."
        )

    n_transpose = sum(1 for node in model.graph.node if node.op_type == "Transpose")
    print(f"  {onnx_path.name}: input={[d.dim_param or d.dim_value for d in dims]}, "
          f"output={[d.dim_param or d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]}, "
          f"Transpose nodes={n_transpose}")
    if n_transpose > 60:
        print(f"  WARNING: {n_transpose} Transpose nodes is more than expected (~53, after "
              f"the dilated-conv graph surgery) -- worth investigating before relying on "
              f"GPU throughput.")


def random_one_hot(seq_len: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = rng.integers(0, NUM_CHANNELS, size=seq_len)
    arr = np.zeros((1, seq_len, NUM_CHANNELS), dtype=np.float32)
    arr[0, np.arange(seq_len), classes] = 1.0
    return arr


def run_tf(saved_model_dir: Path, x: np.ndarray) -> np.ndarray:
    model = tf.saved_model.load(str(saved_model_dir))
    infer = model.signatures["serving_default"]
    input_key = list(infer.structured_input_signature[1].keys())[0]
    result = infer(**{input_key: tf.constant(x)})
    output_key = list(result.keys())[0]
    return result[output_key].numpy()


def run_ort(onnx_path: Path, x: np.ndarray, providers=("CPUExecutionProvider",)) -> np.ndarray:
    session = ort.InferenceSession(str(onnx_path), providers=list(providers))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: x})[0]


def numeric_gate(saved_model_dir: Path, onnx_path: Path) -> None:
    for i, seq_len in enumerate(CHECK_SEQ_LENS):
        x = random_one_hot(seq_len, seed=1234 + i)
        tf_out = run_tf(saved_model_dir, x)
        ort_out = run_ort(onnx_path, x)
        max_diff = float(np.max(np.abs(tf_out - ort_out)))
        print(f"  seq_len={seq_len}: max|TF - ONNX(fp32)| = {max_diff:.3e}")
        if max_diff >= MAX_ABS_DIFF_FP32:
            raise SystemExit(
                f"{onnx_path.name}: TF vs ONNX(fp32) diverge by {max_diff:.3e} "
                f"(>= {MAX_ABS_DIFF_FP32:.0e}) at seq_len={seq_len} -- conversion is not "
                f"numerically equivalent, do not proceed."
            )


def convert_to_fp16(onnx_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(onnx_path))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=False)
    onnx.save(model_fp16, str(output_path))


def fp16_accuracy_probe(fp32_path: Path, fp16_path: Path) -> None:
    """Best-effort only: CPU EP fp16 kernel coverage is uneven, this is not a gate."""
    x = random_one_hot(CHECK_SEQ_LENS[0], seed=99)
    try:
        fp32_out = run_ort(fp32_path, x)
        fp16_out = run_ort(fp16_path, x)
        max_diff = float(np.max(np.abs(fp32_out - fp16_out)))
        print(f"  fp32 vs fp16 (CPU EP) max|diff| = {max_diff:.3e} (informational only -- "
              f"real verdict comes from the GPU/CUDA EP run)")
    except Exception as e:
        print(f"  fp16 CPU EP probe skipped ({type(e).__name__}: {e}) -- expected, CPU fp16 "
              f"kernel coverage is partial. Not a failure.")


def main() -> None:
    for i in range(1, NUM_MODELS + 1):
        name = f"spliceai{i}"
        print(f"=== {name} ===")
        saved_model_dir = TF_DIR / name
        onnx_path = ONNX_DIR / f"{name}.onnx"
        onnx_fp16_path = ONNX_FP16_DIR / f"{name}.onnx"

        print("Converting to ONNX (fp32) + applying dilated-conv graph surgery...")
        convert_to_onnx(saved_model_dir, onnx_path)
        check_graph_shape(onnx_path)

        print("Verifying TF vs ONNX(fp32) numeric parity...")
        numeric_gate(saved_model_dir, onnx_path)

        print("Converting to fp16 (keep_io_types=True)...")
        convert_to_fp16(onnx_path, onnx_fp16_path)
        fp16_accuracy_probe(onnx_path, onnx_fp16_path)

        print()

    print("All 5 models converted and verified.")
    print(f"fp32:  {ONNX_DIR}")
    print(f"fp16:  {ONNX_FP16_DIR}")


if __name__ == "__main__":
    main()
