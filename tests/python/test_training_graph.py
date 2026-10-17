"""The C++ training graph: every VJP rule of the autodiff pass against float64 central finite differences, and the
step function built by autodiff + optimizer lowering against eager training in the reference interpreter.

Needs a built dlc-opt (DLC_OPT or build/dlc-opt); skipped otherwise.
"""
from __future__ import annotations

import numpy as np
import pytest

import dlcopt
import minidlc as dl
from minidlc import ir, nn, optim
from minidlc.reference import ReferenceInterpreter

if not dlcopt.available():
    pytest.skip(f"dlc-opt not found at {dlcopt.DLC_OPT}: build it or set DLC_OPT", allow_module_level=True)

# One graph per VJP rule: each differentiates `%y = <op>(...)` through loss = sum(%y * %r), with %r random, so every
# output element gets a different upstream gradient. The params are differentiated; %r and integer inputs are data.
OP_GRAPHS = {
    "add_broadcast": ("%a: f32[3,4] {param}, %b: f32[4] {param}", "%y = add(%a, %b)", "f32[3,4]"),
    "sub_broadcast": ("%a: f32[3,1] {param}, %b: f32[2,1,4] {param}", "%y = sub(%a, %b)", "f32[2,3,4]"),
    "mul_broadcast": ("%a: f32[3,4] {param}, %b: f32[3,1] {param}", "%y = mul(%a, %b)", "f32[3,4]"),
    "div": ("%a: f32[3,4] {param}, %b: f32[4] {param}", "%y = div(%a, %b)", "f32[3,4]"),
    "relu": ("%a: f32[3,4] {param}", "%y = relu(%a)", "f32[3,4]"),
    "gelu": ("%a: f32[3,4] {param}", "%y = gelu(%a)", "f32[3,4]"),
    "exp": ("%a: f32[3,4] {param}", "%y = exp(%a)", "f32[3,4]"),
    "log": ("%a: f32[3,4] {param}", "%y = log(%a)", "f32[3,4]"),
    "tanh": ("%a: f32[3,4] {param}", "%y = tanh(%a)", "f32[3,4]"),
    "sqrt": ("%a: f32[3,4] {param}", "%y = sqrt(%a)", "f32[3,4]"),
    "rsqrt": ("%a: f32[3,4] {param}", "%y = rsqrt(%a)", "f32[3,4]"),
    "matmul": ("%a: f32[3,5] {param}, %b: f32[5,4] {param}", "%y = matmul(%a, %b)", "f32[3,4]"),
    "matmul_ta": ("%a: f32[5,3] {param}, %b: f32[5,4] {param}", "%y = matmul(%a, %b) {transpose_a = 1}", "f32[3,4]"),
    "matmul_tb": ("%a: f32[3,5] {param}, %b: f32[4,5] {param}", "%y = matmul(%a, %b) {transpose_b = 1}", "f32[3,4]"),
    "matmul_tab": ("%a: f32[5,3] {param}, %b: f32[4,5] {param}",
                   "%y = matmul(%a, %b) {transpose_a = 1, transpose_b = 1}", "f32[3,4]"),
    "matmul_3x2": ("%a: f32[2,3,5] {param}, %b: f32[5,4] {param}", "%y = matmul(%a, %b)", "f32[2,3,4]"),
    "matmul_3x2_tb": ("%a: f32[2,3,5] {param}, %b: f32[4,5] {param}", "%y = matmul(%a, %b) {transpose_b = 1}",
                      "f32[2,3,4]"),
    "matmul_3x3": ("%a: f32[2,3,5] {param}, %b: f32[2,5,4] {param}", "%y = matmul(%a, %b)", "f32[2,3,4]"),
    "matmul_4x4_tb": ("%a: f32[2,2,3,5] {param}, %b: f32[2,2,4,5] {param}",
                      "%y = matmul(%a, %b) {transpose_b = 1}", "f32[2,2,3,4]"),
    "softmax_last": ("%a: f32[3,4] {param}", "%y = softmax(%a)", "f32[3,4]"),
    "softmax_axis0": ("%a: f32[3,4] {param}", "%y = softmax(%a) {axis = 0}", "f32[3,4]"),
    "log_softmax": ("%a: f32[2,3,4] {param}", "%y = log_softmax(%a) {axis = 1}", "f32[2,3,4]"),
    "layernorm_2d": ("%a: f32[3,4] {param}, %g: f32[4] {param}, %b: f32[4] {param}",
                     "%y = layernorm(%a, %g, %b)", "f32[3,4]"),
    "layernorm_3d": ("%a: f32[2,3,4] {param}, %g: f32[4] {param}, %b: f32[4] {param}",
                     "%y = layernorm(%a, %g, %b) {eps = 0.01}", "f32[2,3,4]"),
    "sum_axes": ("%a: f32[2,3,4] {param}", "%y = sum(%a) {axes = [0, 2]}", "f32[3]"),
    "sum_keepdims": ("%a: f32[2,3,4] {param}", "%y = sum(%a) {axes = [1], keepdims = 1}", "f32[2,1,4]"),
    "mean_all": ("%a: f32[2,3,4] {param}", "%y = mean(%a)", "f32[]"),
    "mean_axis": ("%a: f32[2,3,4] {param}", "%y = mean(%a) {axes = [-1]}", "f32[2,3]"),
    "reduce_max": ("%a: f32[3,4] {param}", "%y = reduce_max(%a) {axes = [1]}", "f32[3]"),
    "transpose": ("%a: f32[2,3,4] {param}", "%y = transpose(%a) {perm = [1, 2, 0]}", "f32[3,4,2]"),
    "reshape": ("%a: f32[2,3,4] {param}", "%y = reshape(%a) {shape = [6, 4]}", "f32[6,4]"),
    "broadcast_to": ("%a: f32[3,1] {param}", "%y = broadcast_to(%a) {shape = [2, 3, 4]}", "f32[2,3,4]"),
    "embedding": ("%t: f32[5,4] {param}, %ids: i32[2,3]", "%y = embedding(%t, %ids)", "f32[2,3,4]"),
    "scatter_add": ("%ids: i32[2,3], %u: f32[2,3,4] {param}", "%y = scatter_add(%ids, %u) {num_rows = 5}",
                    "f32[5,4]"),
    "select": ("%c: f32[3,4], %a: f32[3,4] {param}, %b: f32[4] {param}", "%y = select(%c, %a, %b)", "f32[3,4]"),
}
SCALAR_LOSSES = {
    "cross_entropy_2d": ("%z: f32[4,5] {param}, %t: i32[4]", "%y = cross_entropy(%z, %t)"),
    "cross_entropy_3d": ("%z: f32[2,3,5] {param}, %t: i32[2,3]", "%y = cross_entropy(%z, %t)"),
}


def _graph(name: str) -> ir.IRFunction:
    """The test graph in canonical form: dlc-opt infers the omitted types and fills in default attributes."""
    if name in SCALAR_LOSSES:
        args, op = SCALAR_LOSSES[name]
        text = f"func @{name}({args}) -> f32[] {{\n  {op}\n  return %y\n}}\n"
    else:
        args, op, ytype = OP_GRAPHS[name]
        text = (f"func @{name}({args}, %r: {ytype}) -> f32[] {{\n  {op}\n  %l = mul(%y, %r)\n"
                f"  %s = sum(%l)\n  return %s\n}}\n")
    return ir.parse_function(dlcopt.run(text))


def _feeds(func: ir.IRFunction, rng: np.random.Generator) -> dict:
    feeds = {}
    for a in func.args:
        if a.type.dtype == "i32":
            feeds[a.name] = rng.integers(0, 5, a.type.shape)
        elif a.name == "c":
            feeds[a.name] = (rng.random(a.type.shape) > 0.5).astype(np.float64)
        else:
            # Away from the kinks of relu and the ties of reduce_max, and positive for log/sqrt/rsqrt/div.
            values = rng.uniform(0.5, 1.5, a.type.shape) * rng.choice([-1.0, 1.0], a.type.shape)
            positive = {"log", "sqrt", "rsqrt", "div"}
            feeds[a.name] = np.abs(values) if func.name in positive else values
    return feeds


def _numerical_gradient(func: ir.IRFunction, feeds: dict, arg: str, eps: float = 1e-6) -> np.ndarray:
    base = np.array(feeds[arg], dtype=np.float64)
    grad = np.zeros_like(base)
    for idx in np.ndindex(base.shape):
        values = []
        for delta in (eps, -eps):
            shifted = base.copy()
            shifted[idx] += delta
            values.append(float(ir.execute(func, {**feeds, arg: shifted})[0]))
        grad[idx] = (values[0] - values[1]) / (2 * eps)
    return grad


@pytest.mark.parametrize("name", list(OP_GRAPHS) + list(SCALAR_LOSSES))
def test_cpp_vjp_matches_finite_differences(name: str):
    func = _graph(name)
    feeds = _feeds(func, np.random.default_rng(len(name)))
    differentiated = ir.parse_function(dlcopt.run(func.to_text(), "--passes=autodiff,canonicalize,cse,dce"))
    outputs = ir.execute(differentiated, feeds)
    checked = 0
    for value, (kind, arg) in zip(outputs, differentiated.result_kinds()):
        if kind != "grad":
            continue
        numeric = _numerical_gradient(func, feeds, arg)
        np.testing.assert_allclose(value, numeric, rtol=1e-6, atol=1e-7, err_msg=f"{name}: d loss / d %{arg}")
        checked += 1
    assert checked == sum(a.role == "param" for a in func.args)


# --- One graph = one training step --------------------------------------------------------------------------------


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 16)
        self.fc2 = nn.Linear(16, 5)

    def forward(self, x):
        return self.fc2(dl.gelu(self.fc1(x)))


def _mlp():
    model = _MLP()
    x = dl.placeholder((8, 12), name="x")
    y = dl.placeholder((8,), dtype="i32", name="y")
    loss = dl.cross_entropy(model(x), y)

    def batch(rng):
        return {x: rng.standard_normal((8, 12)), y: rng.integers(0, 5, 8)}

    return model, loss, batch


def _chargpt():
    model = nn.CharGPT(vocab_size=11, block_size=8, d_model=16, num_heads=2, num_layers=2)
    x = dl.placeholder((2, 8), dtype="i32", name="x")
    y = dl.placeholder((2, 8), dtype="i32", name="y")
    loss = dl.cross_entropy(model(x), y)

    def batch(rng):
        return {x: rng.integers(0, 11, (2, 8)), y: rng.integers(0, 11, (2, 8))}

    return model, loss, batch


OPTIMIZERS = {
    "adam": ("adam{lr=0.01}", lambda params: optim.Adam(params, lr=0.01)),
    "sgd_momentum": ("sgd{lr=0.05,momentum=0.9}", lambda params: optim.SGD(params, lr=0.05, momentum=0.9)),
}


@pytest.mark.parametrize("model_name", ["mlp", "chargpt"])
@pytest.mark.parametrize("opt_name", list(OPTIMIZERS))
def test_one_graph_is_one_training_step(model_name: str, opt_name: str):
    """The C++ step graph (autodiff + optimizer lowering + cleanup), run for several steps with its donated results
    fed back, follows eager training (reference VJPs + the NumPy optimizer) step by step."""
    np.random.seed(0)
    model, loss, batch = {"mlp": _mlp, "chargpt": _chargpt}[model_name]()
    func = ir.export(loss, name=model_name, module=model)
    pass_spec, make_optimizer = OPTIMIZERS[opt_name]
    step = ir.parse_function(dlcopt.run(func.to_text(), f"--passes=autodiff,{pass_spec},canonicalize,cse,dce"))
    assert all(kind != "grad" for kind, _ in step.result_kinds())
    runner = ir.StepRunner(step, func.bindings)

    for p in model.parameters():
        p.data = p.data.astype(np.float64)
    optimizer = make_optimizer(model.parameters())
    interp = ReferenceInterpreter(dtype=np.float64)
    rng = np.random.default_rng(1)
    names = {id(p): n for n, p in model.named_parameters()}
    for _ in range(4):
        feeds = batch(rng)
        values = interp.forward(loss, feeds)
        grads = interp.backward(loss, values)
        optimizer.step({p: grads[p] for p in model.parameters()})
        (step_loss,) = runner({t.name: v for t, v in feeds.items()})
        np.testing.assert_allclose(step_loss, values[loss], rtol=1e-9)
    for p in model.parameters():
        np.testing.assert_allclose(runner.values[names[id(p)]], p.data, rtol=1e-7, atol=1e-8, err_msg=names[id(p)])
