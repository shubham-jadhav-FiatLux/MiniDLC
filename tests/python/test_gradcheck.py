import numpy as np
import minidlc as dl

def gradcheck_finite_diff(loss_node, param, feed_dict, eps=1e-5, rtol=1e-4, atol=1e-4):
    """Checks analytic gradient vs two-sided finite difference in float64."""
    interp = dl.ReferenceInterpreter(dtype=np.float64)

    # 1. Analytic gradient
    vals = interp.forward(loss_node, feed_dict)
    analytic_grads = interp.backward(loss_node, vals)
    analytic_grad = analytic_grads[param]

    # 2. Numerical gradient via central difference
    param_data = feed_dict[param]
    num_grad = np.zeros_like(param_data)
    it = np.nditer(param_data, flags=["multi_index"], op_flags=["readwrite"])

    orig = param_data.copy()
    while not it.finished:
        idx = it.multi_index
        orig_val = orig[idx]

        # f(x + eps)
        param_data[idx] = orig_val + eps
        v_plus = float(interp.forward(loss_node, feed_dict)[loss_node])

        # f(x - eps)
        param_data[idx] = orig_val - eps
        v_minus = float(interp.forward(loss_node, feed_dict)[loss_node])

        num_grad[idx] = (v_plus - v_minus) / (2.0 * eps)
        param_data[idx] = orig_val
        it.iternext()

    np.testing.assert_allclose(analytic_grad, num_grad, rtol=rtol, atol=atol)

def test_matmul_gradcheck():
    x = dl.placeholder((4, 8), name="x")
    w = dl.placeholder((8, 4), name="w")
    loss = dl.sum(dl.relu(x @ w))

    np.random.seed(42)
    x_val = np.random.randn(4, 8).astype(np.float64)
    w_val = np.random.randn(8, 4).astype(np.float64)

    gradcheck_finite_diff(loss, w, {x: x_val, w: w_val})
    gradcheck_finite_diff(loss, x, {x: x_val, w: w_val})

def test_binary_broadcasting_gradcheck():
    np.random.seed(42)
    x = dl.placeholder((3, 5), name="x")
    b = dl.placeholder((5,), name="b")
    target = np.random.randn(3, 5).astype(np.float64)

    # Add
    loss_add = dl.sum(dl.add(x, b))
    x_val = np.random.randn(3, 5).astype(np.float64)
    b_val = np.random.randn(5).astype(np.float64)
    gradcheck_finite_diff(loss_add, b, {x: x_val, b: b_val})

    # Sub
    loss_sub = dl.sum(dl.sub(x, b))
    gradcheck_finite_diff(loss_sub, b, {x: x_val, b: b_val})

    # Mul
    loss_mul = dl.sum(dl.mul(x, b))
    gradcheck_finite_diff(loss_mul, b, {x: x_val, b: b_val})

    # Div
    loss_div = dl.sum(dl.div(x, b))
    b_val_nz = np.abs(b_val) + 0.5
    gradcheck_finite_diff(loss_div, b, {x: x_val, b: b_val_nz})

def test_unary_ops_gradcheck():
    np.random.seed(42)
    x = dl.placeholder((4, 6), name="x")
    x_pos = np.abs(np.random.randn(4, 6).astype(np.float64)) + 0.1

    # ReLU
    loss_relu = dl.sum(dl.relu(x))
    gradcheck_finite_diff(loss_relu, x, {x: x_pos})

    # GeLU
    loss_gelu = dl.sum(dl.gelu(x))
    gradcheck_finite_diff(loss_gelu, x, {x: x_pos})

    # Exp
    loss_exp = dl.sum(dl.exp(x))
    gradcheck_finite_diff(loss_exp, x, {x: x_pos * 0.1})

    # Log
    loss_log = dl.sum(dl.log(x))
    gradcheck_finite_diff(loss_log, x, {x: x_pos})

def test_softmax_and_cross_entropy_gradcheck():
    np.random.seed(42)
    logits = dl.placeholder((4, 5), name="logits")
    targets = dl.placeholder((4,), dtype="i32", name="targets")

    logits_val = np.random.randn(4, 5).astype(np.float64)
    targets_val = np.array([0, 2, 4, 1], dtype=np.int32)

    # Softmax
    loss_smax = dl.sum(dl.softmax(logits, axis=-1))
    gradcheck_finite_diff(loss_smax, logits, {logits: logits_val})

    # Log-Softmax
    loss_logsmax = dl.sum(dl.log_softmax(logits, axis=-1))
    gradcheck_finite_diff(loss_logsmax, logits, {logits: logits_val})

    # Cross-Entropy
    loss_ce = dl.cross_entropy(logits, targets)
    gradcheck_finite_diff(loss_ce, logits, {logits: logits_val, targets: targets_val})

def test_layernorm_gradcheck():
    np.random.seed(42)
    x = dl.placeholder((3, 4), name="x")
    gamma = dl.placeholder((4,), name="gamma")
    beta = dl.placeholder((4,), name="beta")

    x_val = np.random.randn(3, 4).astype(np.float64)
    gamma_val = np.ones((4,), dtype=np.float64)
    beta_val = np.zeros((4,), dtype=np.float64)

    loss_ln = dl.sum(dl.layernorm(x, gamma, beta))
    gradcheck_finite_diff(loss_ln, x, {x: x_val, gamma: gamma_val, beta: beta_val})
    gradcheck_finite_diff(loss_ln, gamma, {x: x_val, gamma: gamma_val, beta: beta_val})
    gradcheck_finite_diff(loss_ln, beta, {x: x_val, gamma: gamma_val, beta: beta_val})

def test_fanout_accumulation_gradcheck():
    np.random.seed(42)
    x = dl.placeholder((4, 4), name="x")
    # Fan-out: x feeds into two distinct computation branches
    loss = dl.sum(dl.relu(x) + dl.mul(x, x))
    x_val = np.random.randn(4, 4).astype(np.float64)
    gradcheck_finite_diff(loss, x, {x: x_val})

def test_all_ops_gradcheck():
    test_matmul_gradcheck()
    test_binary_broadcasting_gradcheck()
    test_unary_ops_gradcheck()
    test_softmax_and_cross_entropy_gradcheck()
    test_layernorm_gradcheck()
    test_fanout_accumulation_gradcheck()
    print("[PASS] All Op VJP Gradient Checks Passed Successfully!")

if __name__ == "__main__":
    test_all_ops_gradcheck()
