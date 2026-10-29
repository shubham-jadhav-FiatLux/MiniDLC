"""Differential tests of the compiled code: every op, through lowering, the CPU schedules, LLVM and the ORC JIT (target
"cpu") and through the loop IR interpreter (target "interp"), against the reference interpreter in float64.

Shapes are chosen to hit the awkward paths: sizes that are not multiples of the vector width (8) or of the matmul
register tile (4x16), broadcasting, zero-size dimensions, every reduction layout. Contract checks cover NaN and Inf.
Needs the compiled _minidlc module (built by CMake); skipped otherwise.
"""
from __future__ import annotations

import numpy as np
import pytest

import dlcopt
import minidlc as dl
from minidlc import ir, nn

_minidlc = pytest.importorskip("minidlc._minidlc", reason="the _minidlc module is not built")
if not dlcopt.available():
    pytest.skip(f"dlc-opt not found at {dlcopt.DLC_OPT}", allow_module_level=True)

TARGETS = ["cpu", "interp"]


def compiled(text: str, feeds: dict, target: str, options: dict = None):
    exe, _ = _minidlc.compile(text, target, options or {})
    func = ir.parse_function(dlcopt.run(text))
    for a in func.args:
        if a.role != "input":
            exe.set(a.name, np.asarray(feeds[a.name], dtype=np.float32 if a.type.dtype == "f32" else np.int32, order="C"))
    inputs = {a.name: np.asarray(feeds[a.name], dtype=np.float32 if a.type.dtype == "f32" else np.int32, order="C")
              for a in func.args if a.role == "input"}
    return exe.run(inputs)


def reference(text: str, feeds: dict):
    func = ir.parse_function(dlcopt.run(text))
    return ir.execute(func, {k: np.asarray(v, dtype=np.float64) if np.asarray(v).dtype.kind == "f" else v
                             for k, v in feeds.items()})


def random_feeds(text: str, seed: int, positive: bool = False, classes: int = 5) -> dict:
    func = ir.parse_function(dlcopt.run(text))
    rng = np.random.default_rng(seed)
    feeds = {}
    for a in func.args:
        if a.type.dtype == "i32":
            feeds[a.name] = rng.integers(0, classes, a.type.shape).astype(np.int32)
        else:
            v = rng.standard_normal(a.type.shape).astype(np.float32)
            feeds[a.name] = np.abs(v) + 0.1 if positive else v
    return feeds


def check(text: str, rtol=1e-5, atol=1e-5, positive=False, classes=5, seed=0):
    feeds = random_feeds(text, seed, positive, classes)
    want = reference(text, feeds)
    for target in TARGETS:
        got = compiled(text, feeds, target)
        assert len(got) == len(want)
        for i, (g, w) in enumerate(zip(got, want)):
            assert g.shape == w.shape, (target, i, g.shape, w.shape)
            np.testing.assert_allclose(g, w, rtol=rtol, atol=atol, err_msg=f"{target}: output {i}")


def fn(args: str, body: str, results: str, ret: str) -> str:
    return f"func @f({args}) -> {results} {{\n{body}\n  return {ret}\n}}\n"


ELEMENTWISE = {
    "add_broadcast": ("%a: f32[7,13], %b: f32[13]", "  %0 = add(%a, %b)", "f32[7,13]"),
    "sub_column": ("%a: f32[7,13], %b: f32[7,1]", "  %0 = sub(%a, %b)", "f32[7,13]"),
    "mul_scalar": ("%a: f32[3,5,11], %b: f32[]", "  %0 = mul(%a, %b)", "f32[3,5,11]"),
    "div_3d": ("%a: f32[2,3,5], %b: f32[3,1]", "  %0 = div(%a, %b)", "f32[2,3,5]"),
    "relu": ("%a: f32[9,17]", "  %0 = relu(%a)", "f32[9,17]"),
    "gelu": ("%a: f32[37]", "  %0 = gelu(%a)", "f32[37]"),
    "exp": ("%a: f32[4,9]", "  %0 = exp(%a)", "f32[4,9]"),
    "tanh": ("%a: f32[4,9]", "  %0 = tanh(%a)", "f32[4,9]"),
    "cmp_select": ("%a: f32[6,7], %b: f32[7]",
                   "  %c = cmp_gt(%a, %b)\n  %0 = select(%c, %a, %b)", "f32[6,7]"),
    "broadcast_to": ("%a: f32[3,1]", "  %0 = broadcast_to(%a) {shape = [2, 3, 5]}", "f32[2,3,5]"),
    "chain": ("%a: f32[5,6], %b: f32[6]",
              "  %t = mul(%a, %b)\n  %u = exp(%t)\n  %v = add(%u, %a)\n  %0 = relu(%v)", "f32[5,6]"),
    "large_parallel": ("%a: f32[300,301], %b: f32[301]", "  %t = mul(%a, %b)\n  %0 = add(%t, %a)", "f32[300,301]"),
}


@pytest.mark.parametrize("name", list(ELEMENTWISE))
def test_elementwise(name):
    args, body, result = ELEMENTWISE[name]
    check(fn(args, body, result, "%0"))


def test_positive_domain_ops():
    check(fn("%a: f32[5,9]", "  %l = log(%a)\n  %s = sqrt(%a)\n  %r = rsqrt(%a)\n  %t = add(%l, %s)\n  %0 = add(%t, %r)",
             "f32[5,9]", "%0"), positive=True)


def test_multi_output_elementwise_kernel():
    check(fn("%a: f32[8,12], %b: f32[12]", "  %t = mul(%a, %b)\n  %u = exp(%t)\n  %s = sum(%u) {axes = [1]}",
             "(f32[8,12], f32[8])", "%t, %s"))


MATMULS = {
    "2d": ("%a: f32[5,19], %b: f32[19,23]", "", "f32[5,23]"),
    "2d_ta": ("%a: f32[19,5], %b: f32[19,23]", "{transpose_a = 1}", "f32[5,23]"),
    "2d_tb": ("%a: f32[5,19], %b: f32[23,19]", "{transpose_b = 1}", "f32[5,23]"),
    "2d_tab": ("%a: f32[19,5], %b: f32[23,19]", "{transpose_a = 1, transpose_b = 1}", "f32[5,23]"),
    "3x2": ("%a: f32[2,5,7], %b: f32[7,9]", "", "f32[2,5,9]"),
    "3x2_ta": ("%a: f32[2,7,5], %b: f32[7,9]", "{transpose_a = 1}", "f32[2,5,9]"),
    "3x3": ("%a: f32[3,5,7], %b: f32[3,7,6]", "", "f32[3,5,6]"),
    "3x3_tb": ("%a: f32[3,5,7], %b: f32[3,6,7]", "{transpose_b = 1}", "f32[3,5,6]"),
    "4x4": ("%a: f32[2,2,5,7], %b: f32[2,2,7,6]", "", "f32[2,2,5,6]"),
    "large": ("%a: f32[130,70], %b: f32[70,33]", "", "f32[130,33]"),
    "large_tb": ("%a: f32[130,70], %b: f32[33,70]", "{transpose_b = 1}", "f32[130,33]"),
}


@pytest.mark.parametrize("name", list(MATMULS))
def test_matmul(name):
    args, attrs, result = MATMULS[name]
    check(fn(args, f"  %0 = matmul(%a, %b) {attrs}".rstrip(), result, "%0"), rtol=1e-4, atol=1e-4)


def test_matmul_epilogue_with_broadcast_operands():
    check(fn("%a: f32[2,5,7], %w: f32[7,9], %b: f32[9], %r: f32[5,1]",
             "  %m = matmul(%a, %w)\n  %t = add(%m, %b)\n  %u = mul(%t, %r)\n  %0 = gelu(%u)", "f32[2,5,9]", "%0"),
          rtol=1e-4, atol=1e-4)


REDUCTIONS = {
    "rows": ("f32[6,10]", "sum", "{axes = [1]}", "f32[6]"),
    "columns": ("f32[6,10]", "sum", "{axes = [0]}", "f32[10]"),
    "middle_keepdims": ("f32[3,4,5]", "mean", "{axes = [1], keepdims = 1}", "f32[3,1,5]"),
    "outer_two": ("f32[3,4,5]", "sum", "{axes = [0, 1]}", "f32[5]"),
    "non_contiguous": ("f32[3,4,5]", "sum", "{axes = [0, 2]}", "f32[4]"),
    "all": ("f32[3,4,5]", "mean", "", "f32[]"),
    "max_rows": ("f32[6,10]", "reduce_max", "{axes = [-1]}", "f32[6]"),
    "max_columns": ("f32[6,10]", "reduce_max", "{axes = [0], keepdims = 1}", "f32[1,10]"),
    "large": ("f32[257,129]", "sum", "{axes = [1]}", "f32[257]"),
}


@pytest.mark.parametrize("name", list(REDUCTIONS))
def test_reductions(name):
    t, op, attrs, result = REDUCTIONS[name]
    check(fn(f"%a: {t}", f"  %0 = {op}(%a) {attrs}".rstrip(), result, "%0"), rtol=1e-4, atol=1e-5)


ROWS = {
    "softmax_last": ("%a: f32[3,4,5]", "  %0 = softmax(%a)", "f32[3,4,5]"),
    "softmax_axis1": ("%a: f32[3,4,5]", "  %0 = softmax(%a) {axis = 1}", "f32[3,4,5]"),
    "softmax_axis0": ("%a: f32[6,7]", "  %0 = softmax(%a) {axis = 0}", "f32[6,7]"),
    "log_softmax": ("%a: f32[5,11]", "  %0 = log_softmax(%a)", "f32[5,11]"),
    "layernorm": ("%a: f32[3,4,10], %g: f32[10], %b: f32[10]", "  %0 = layernorm(%a, %g, %b)", "f32[3,4,10]"),
    "cross_entropy_2d": ("%a: f32[9,5], %t: i32[9]", "  %0 = cross_entropy(%a, %t)", "f32[]"),
    "cross_entropy_3d": ("%a: f32[2,3,5], %t: i32[2,3]", "  %0 = cross_entropy(%a, %t)", "f32[]"),
}


@pytest.mark.parametrize("name", list(ROWS))
def test_row_kernels(name):
    args, body, result = ROWS[name]
    check(fn(args, body, result, "%0"), rtol=1e-5, atol=1e-5)


OTHERS = {
    "embedding": ("%t: f32[5,6], %i: i32[2,3]", "  %0 = embedding(%t, %i)", "f32[2,3,6]"),
    "onehot": ("%i: i32[7]", "  %0 = onehot(%i) {num_classes = 5}", "f32[7,5]"),
    "scatter_add": ("%i: i32[2,4], %u: f32[2,4,3]", "  %0 = scatter_add(%i, %u) {num_rows = 5}", "f32[5,3]"),
    "transpose": ("%a: f32[2,3,5]", "  %0 = transpose(%a) {perm = [2, 0, 1]}", "f32[5,2,3]"),
    "transpose_large": ("%a: f32[200,300]", "  %0 = transpose(%a)", "f32[300,200]"),
    "reshape_alias": ("%a: f32[4,6]", "  %e = exp(%a)\n  %r = reshape(%e) {shape = [6, 4]}\n  %0 = sum(%r) {axes = [1]}",
                      "f32[6]"),
    "cast": ("%i: i32[2,3]", "  %0 = cast(%i) {to = \"f32\"}", "f32[2,3]"),
    "constant_fill": ("%a: f32[3,4]", "  %c = constant() : f32[4,5] {value = 0.25}\n  %0 = matmul(%a, %c)", "f32[3,5]"),
    "returned_argument": ("%a: f32[3]", "  %0 = exp(%a)", "(f32[3], f32[3])"),
}


@pytest.mark.parametrize("name", list(OTHERS))
def test_other_kernels(name):
    args, body, result = OTHERS[name]
    ret = "%0, %a" if name == "returned_argument" else "%0"
    check(fn(args, body, result, ret), rtol=1e-5, atol=1e-5)


def test_zero_size_dimensions():
    check(fn("%a: f32[0,5], %b: f32[5]", "  %t = add(%a, %b)\n  %0 = sum(%t) {axes = [0]}", "f32[5]", "%0"))


def test_nan_and_inf_propagate():
    text = fn("%a: f32[8]", "  %r = relu(%a)\n  %m = reduce_max(%a) {axes = [0]}\n  %s = softmax(%a)", "(f32[8], f32[], f32[8])",
              "%r, %m, %s")
    a = np.array([np.nan, -1.0, 2.0, np.inf, -np.inf, 0.0, -0.0, 3.0], dtype=np.float32)
    for target in TARGETS:
        r, m, s = compiled(text, {"a": a}, target)
        assert np.isnan(r[0]) and r[1] == 0.0 and r[3] == np.inf  # relu(NaN) = NaN, as in PyTorch
        assert np.isnan(m)
        assert np.isnan(s).all()  # a NaN in the row makes the whole softmax row NaN


def test_softmax_is_stable_for_large_logits():
    text = fn("%a: f32[2,4]", "  %0 = softmax(%a)\n  %1 = log_softmax(%a)", "(f32[2,4], f32[2,4])", "%0, %1")
    a = np.array([[1e4, 1e4 - 1, -1e4, 0.0], [-3e4, -3e4, -3e4, -3e4]], dtype=np.float32)
    want = reference(text, {"a": a})
    for target in TARGETS:
        for g, w in zip(compiled(text, {"a": a}, target), want):
            assert np.isfinite(g).all()
            np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-5)


def test_parallel_kernels_are_deterministic():
    text = fn("%a: f32[256,128], %b: f32[128,96]", "  %m = matmul(%a, %b)\n  %0 = sum(%m) {axes = [0]}", "f32[96]", "%0")
    feeds = random_feeds(text, 3)
    first = compiled(text, feeds, "cpu")[0]
    for _ in range(3):
        np.testing.assert_array_equal(compiled(text, feeds, "cpu")[0], first)


@pytest.mark.parametrize("seed", range(42000, 42008))
def test_random_graphs(seed):
    from test_fuzzer import generate_random_dag
    loss, feed, _ = generate_random_dag(seed=seed, num_ops=8)
    func = ir.export(loss, name=f"fuzz_{seed}")
    text = func.to_text()
    feeds = {t.name: v for t, v in feed.items()}
    want = ir.execute(func, feeds)
    for target in TARGETS:
        got = compiled(text, feeds, target)
        np.testing.assert_allclose(got[0], want[0], rtol=1e-4, atol=1e-4, err_msg=target)
