"""Translation validation of the graph passes, and the IR text contract between the frontend and the C++ compiler.

Each graph is exported to IR text, run through `dlc-opt --passes=<pass>`, parsed back and executed in the NumPy
reference interpreter in float64; its outputs must match the unoptimized function's. The tests need a built dlc-opt:
set DLC_OPT or build into build/ (ctest runs this file with DLC_OPT set). Without one they are skipped.
"""
from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pytest

import dlcopt
import minidlc as dl
from minidlc import ir, nn
from minidlc.reference import ReferenceInterpreter
from minidlc.tensor import Tensor
from test_fuzzer import generate_random_dag

if not dlcopt.available():
    pytest.skip(f"dlc-opt not found at {dlcopt.DLC_OPT}: build it or set DLC_OPT", allow_module_level=True)

PIPELINE = "shape-infer,canonicalize,cse,dce,fusion,mem-plan"
PASSES = ["shape-infer", "canonicalize", "cse", "dce", "fusion", "mem-plan", PIPELINE]
FUZZ_SEEDS = range(42000, 42010)

# Miscompiles that translation validation catches; each entry names the phase that fixes it. Strict: a fix makes the
# test pass, which fails it until the entry is removed. (P2 fixed the two it started with: CSE merging nearby
# constants, and `x * 0` changing a result type; their graphs below stay as regression cases.)
KNOWN_FAILURES: Dict[tuple, str] = {}

dlc_opt = dlcopt.run


@dataclass
class Case:
    func: ir.IRFunction
    feeds: Dict[str, np.ndarray]  # input arguments; params and consts come from func.bindings
    outputs: List[Tensor] = field(default_factory=list)  # the traced tensors the function returns
    tensor_feeds: Dict[Tensor, np.ndarray] = field(default_factory=dict)

    def all_feeds(self) -> Dict[str, np.ndarray]:
        return {**self.func.bindings, **self.feeds}


def _traced(name: str, outputs: List[Tensor], placeholders: Dict[Tensor, np.ndarray], module=None) -> Case:
    func = ir.export(outputs, name=name, module=module)
    return Case(func, {t.name: v for t, v in placeholders.items()}, outputs, placeholders)


class _MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.layers = [nn.Linear(a, b) for a, b in zip(sizes, sizes[1:])]
        for i, layer in enumerate(self.layers):
            setattr(self, f"fc{i + 1}", layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = dl.relu(layer(x))
        return self.layers[-1](x)


def _classifier(name: str, sizes) -> Case:
    np.random.seed(7)
    rng = np.random.default_rng(7)
    model = _MLP(sizes)
    x = dl.placeholder((16, sizes[0]), name="x")
    y = dl.placeholder((16,), dtype="i32", name="y")
    logits = model(x)
    loss = dl.cross_entropy(logits, y)
    feeds = {x: rng.standard_normal((16, sizes[0])), y: rng.integers(0, sizes[-1], 16).astype(np.int32)}
    return _traced(name, [loss, logits], feeds, module=model)


def _chargpt() -> Case:
    np.random.seed(11)
    rng = np.random.default_rng(11)
    model = nn.CharGPT(vocab_size=11, block_size=8, d_model=16, num_heads=2, num_layers=2)
    idx = dl.placeholder((2, 8), dtype="i32", name="idx")
    targets = dl.placeholder((2, 8), dtype="i32", name="targets")
    logits = model(idx)
    loss = dl.cross_entropy(logits, targets)
    feeds = {idx: rng.integers(0, 11, (2, 8)).astype(np.int32), targets: rng.integers(0, 11, (2, 8)).astype(np.int32)}
    return _traced("chargpt", [loss, logits], feeds, module=model)


def _misc() -> Case:
    """The ops and patterns the models above miss: division, logs, reductions with keepdims, reshapes, 4-D matmuls,
    repeated subexpressions, identities and foldable constants."""
    rng = np.random.default_rng(3)
    x = dl.placeholder((2, 3, 4), name="x")
    q = dl.placeholder((2, 2, 3, 4), name="q")
    k = dl.placeholder((2, 2, 4, 3), name="k")
    a = dl.exp(x) / dl.sum(dl.exp(x), axis=-1, keepdims=True)
    b = dl.log(dl.add(dl.mean(a, axis=1, keepdims=True), 1.0))
    c = dl.log_softmax(dl.add(dl.mul(b, 1.0), 0.0), axis=-1)
    r = dl.reshape(dl.transpose(c, perm=[2, 0, 1]), (8,))
    s = dl.sub(dl.mul(2.0, 3.0), r)
    att = dl.sum(dl.softmax(dl.matmul(q, k), axis=-1), axis=0)
    total = dl.mean(dl.gelu(att)) + dl.sum(s) - dl.div(dl.sum(dl.relu(s)), 4.0)
    feeds = {x: rng.standard_normal((2, 3, 4)), q: rng.standard_normal((2, 2, 3, 4)),
             k: rng.standard_normal((2, 2, 4, 3))}
    return _traced("misc", [total, s, att], feeds)


# Regression cases for two v1 miscompiles: CSE merged these constants, and canonicalize rewrote x * 0 to a scalar.
def _nearby_constants() -> Case:
    x = dl.placeholder((4,), name="x")
    return _traced("nearby", [dl.sub(dl.mul(x, 0.1000001), dl.mul(x, 0.1000002))], {x: np.arange(1.0, 5.0)})


def _mul_by_zero() -> Case:
    x = dl.placeholder((4,), name="x")
    return _traced("mul0", [dl.mul(x, 0.0)], {x: np.arange(1.0, 5.0)})


def _fuzz(seed: int) -> Case:
    loss, feed, _ = generate_random_dag(seed=seed, num_ops=8)
    return _traced(f"fuzz_{seed}", [loss], {t: v for t, v in feed.items()})


BUILDERS = {
    "linear": lambda: _classifier("linear", [20, 5]),
    "mlp": lambda: _classifier("mlp", [20, 32, 5]),
    "chargpt": _chargpt,
    "misc": _misc,
    "nearby_constants": _nearby_constants,
    "mul_by_zero": _mul_by_zero,
    **{f"fuzz_{s}": functools.partial(_fuzz, s) for s in FUZZ_SEEDS},
}
CASES = list(BUILDERS)


@functools.lru_cache(maxsize=None)
def case(name: str) -> Case:
    return BUILDERS[name]()


@functools.lru_cache(maxsize=None)
def reference_outputs(name: str) -> List[np.ndarray]:
    c = case(name)
    return ir.execute(c.func, c.all_feeds())


def _known(reason):
    return [pytest.mark.xfail(strict=True, reason=reason)] if reason else []


VALIDATION_PARAMS = [
    pytest.param(name, passes, id=f"{name}-{passes.replace(',', '+')}", marks=_known(KNOWN_FAILURES.get((name, passes))))
    for name in CASES
    for passes in PASSES
]


@pytest.mark.parametrize("name, passes", VALIDATION_PARAMS)
def test_pass_preserves_semantics(name: str, passes: str):
    c = case(name)
    optimized = ir.parse_function(dlc_opt(c.func.to_text(), f"--passes={passes}"))
    assert [str(t) for t in optimized.result_types] == [str(t) for t in c.func.result_types]
    actual = ir.execute(optimized, c.all_feeds())
    for i, (got, want) in enumerate(zip(actual, reference_outputs(name))):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12, err_msg=f"output {i} after {passes}")


@pytest.mark.parametrize("name", CASES)
def test_exported_function_computes_the_traced_graph(name: str):
    c = case(name)
    traced = ReferenceInterpreter(dtype=np.float64).forward(c.outputs, c.tensor_feeds)
    for i, (got, t) in enumerate(zip(reference_outputs(name), c.outputs)):
        np.testing.assert_allclose(got, traced[t], rtol=1e-12, atol=1e-12, err_msg=f"output {i}")


@pytest.mark.parametrize("name", CASES)
def test_exported_text_is_canonical(name: str):
    """dlc-opt with no passes prints the exporter's text back unchanged, and the Python parser reads it back."""
    text = case(name).func.to_text()
    assert dlc_opt(text) == ir.module_text([case(name).func])
    assert ir.parse_function(text).to_text() == text


@pytest.mark.parametrize("name", ["mlp", "chargpt", "misc", "fuzz_42000"])
def test_optimized_text_is_a_fixpoint(name: str):
    """The annotated output of the full pipeline parses back to the same text, in C++ and in Python."""
    once = dlc_opt(case(name).func.to_text(), f"--passes={PIPELINE}")
    assert "fusion.group" in once and "mem.offset" in once
    assert dlc_opt(once) == once
    assert ir.module_text(ir.parse_module(once)) == once


def test_python_op_table_matches_the_cpp_registry():
    listed = {}
    for line in dlc_opt("", "--print-op-registry").splitlines():
        fields = dict(part.split("=", 1) for part in line.split()[1:])
        attrs = {}
        for spec in filter(None, fields["attrs"].split(",")):
            attr, kind = spec.split(":")
            attrs[attr] = (kind.rstrip("!"), kind.endswith("!"))
        operands = -1 if fields["operands"] == "*" else int(fields["operands"])
        listed[line.split()[0]] = (operands, attrs)
        assert fields["results"] == ("0" if line.split()[0] == "return" else "1")
    assert listed == ir.OPS
    assert list(listed) == list(ir.OPS)


def _inputs_as_params(func: ir.IRFunction, feeds: Dict[str, np.ndarray]) -> ir.IRFunction:
    """The function with every f32 input turned into a param, so autodiff differentiates it."""
    args = [dataclasses.replace(a, role="param") if a.role == "input" and a.type.dtype == "f32" else a
            for a in func.args]
    bindings = {**func.bindings, **{a.name: feeds[a.name] for a in func.args if a.role == "input"
                                    and a.type.dtype == "f32"}}
    return dataclasses.replace(func, args=args, bindings=bindings)


def _gradients(name: str):
    """Runs the autodiff pass. Returns the differentiated function, its outputs, and the reference interpreter's
    gradients (Python VJPs on the same graph) by argument name."""
    c = case(name)
    func = c.func if any(a.role == "param" for a in c.func.args) else _inputs_as_params(c.func, c.feeds)
    differentiated = ir.parse_function(dlc_opt(func.to_text(), "--passes=autodiff"))
    feeds = {**func.bindings, **c.feeds}
    outputs = ir.execute(differentiated, feeds)
    args, (loss,) = ir.to_graph(ir.IRFunction(func.name, func.args, func.result_types[:1],
                                              func.ops[:-1] + [ir.IROp("return", func.returns[:1])]))
    tensor_feeds = {t: np.asarray(feeds[n], dtype=np.float64) for n, t in args.items()}
    interp = ReferenceInterpreter(dtype=np.float64)
    grads = interp.backward(loss, interp.forward(loss, tensor_feeds))
    return differentiated, outputs, {n: grads.get(t) for n, t in args.items()}


LOSS_CASES = [n for n in CASES if str(case(n).func.result_types[0]) == "f32[]"]


@pytest.mark.parametrize("name", LOSS_CASES)
def test_autodiff_matches_the_reference_gradients(name: str):
    func, outputs, reference = _gradients(name)
    np.testing.assert_allclose(outputs[0], reference_outputs(name)[0], rtol=1e-12, atol=1e-12)
    grads = {arg: out for out, (kind, arg) in zip(outputs, func.result_kinds()) if kind == "grad"}
    assert set(grads) == {a.name for a in func.args if a.role == "param"}
    for arg, grad in grads.items():
        want = reference[arg] if reference[arg] is not None else np.zeros_like(grad)
        np.testing.assert_allclose(grad, want, rtol=1e-9, atol=1e-12, err_msg=f"d loss / d %{arg}")


def test_non_scalar_losses_are_rejected():
    with pytest.raises(AssertionError, match="the loss \\(the first result\\) must be f32"):
        dlc_opt(case("mul_by_zero").func.to_text(), "--passes=autodiff")
