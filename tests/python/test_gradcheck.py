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

def mixed_sign(rng, shape):
    """Random values of both signs, kept away from 0 so ReLU's kink is never crossed by the finite difference."""
    return np.sign(rng.randn(*shape)) * (np.abs(rng.randn(*shape)) + 0.1)

def weighted_sum(node, rng):
    """sum(node * r) for a fixed random r, so every output element gets a different upstream gradient."""
    r = dl.Tensor(shape=node.shape, dtype="f32", op="constant", data=rng.randn(*node.shape))
    return dl.sum(dl.mul(node, r))

def test_matmul_gradcheck():
    x = dl.placeholder((4, 8), name="x")
    w = dl.placeholder((8, 4), name="w")
    loss = dl.sum(dl.relu(x @ w))

    np.random.seed(42)
    x_val = np.random.randn(4, 8).astype(np.float64)
    w_val = np.random.randn(8, 4).astype(np.float64)

    gradcheck_finite_diff(loss, w, {x: x_val, w: w_val})
    gradcheck_finite_diff(loss, x, {x: x_val, w: w_val})

def test_batched_matmul_gradcheck():
    rng = np.random.RandomState(7)
    for a_shape, b_shape in [((2, 3, 4), (2, 4, 5)), ((2, 3, 4), (4, 5)), ((2, 2, 3, 4), (2, 2, 4, 5))]:
        a = dl.placeholder(a_shape, name="a")
        b = dl.placeholder(b_shape, name="b")
        loss = weighted_sum(a @ b, rng)
        feed = {a: rng.randn(*a_shape), b: rng.randn(*b_shape)}
        gradcheck_finite_diff(loss, a, feed)
        gradcheck_finite_diff(loss, b, feed)

def test_binary_broadcasting_gradcheck():
    np.random.seed(42)
    x = dl.placeholder((3, 5), name="x")
    b = dl.placeholder((5,), name="b")

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
    rng = np.random.RandomState(42)
    x = dl.placeholder((4, 6), name="x")
    x_mixed = mixed_sign(rng, (4, 6))
    x_pos = np.abs(x_mixed)

    # ReLU and GELU on both signs, so the ReLU mask is exercised
    gradcheck_finite_diff(weighted_sum(dl.relu(x), rng), x, {x: x_mixed.copy()})
    gradcheck_finite_diff(weighted_sum(dl.gelu(x), rng), x, {x: x_mixed.copy()})

    # Exp
    loss_exp = dl.sum(dl.exp(x))
    gradcheck_finite_diff(loss_exp, x, {x: x_mixed * 0.1})

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

def test_sequence_cross_entropy_gradcheck():
    rng = np.random.RandomState(3)
    logits = dl.placeholder((2, 3, 5), name="logits")
    targets = dl.placeholder((2, 3), dtype="i32", name="targets")
    feed = {logits: rng.randn(2, 3, 5), targets: rng.randint(0, 5, size=(2, 3)).astype(np.int32)}
    gradcheck_finite_diff(dl.cross_entropy(logits, targets), logits, feed)

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

def test_reductions_gradcheck():
    rng = np.random.RandomState(5)
    x = dl.placeholder((3, 4, 5), name="x")
    x_val = rng.randn(3, 4, 5)
    gradcheck_finite_diff(weighted_sum(dl.sum(x, axis=1, keepdims=True), rng), x, {x: x_val.copy()})
    gradcheck_finite_diff(weighted_sum(dl.sum(x, axis=-1), rng), x, {x: x_val.copy()})
    gradcheck_finite_diff(weighted_sum(dl.mean(x, axis=0), rng), x, {x: x_val.copy()})
    gradcheck_finite_diff(weighted_sum(dl.mean(x, axis=-2, keepdims=True), rng), x, {x: x_val.copy()})
    gradcheck_finite_diff(dl.mean(x), x, {x: x_val.copy()})

def test_transpose_and_reshape_gradcheck():
    rng = np.random.RandomState(11)
    x = dl.placeholder((2, 3, 4), name="x")
    x_val = rng.randn(2, 3, 4)
    gradcheck_finite_diff(weighted_sum(dl.transpose(x, perm=[2, 0, 1]), rng), x, {x: x_val.copy()})
    gradcheck_finite_diff(weighted_sum(dl.reshape(x, (6, 4)), rng), x, {x: x_val.copy()})

def test_embedding_gradcheck():
    rng = np.random.RandomState(13)
    table = dl.placeholder((6, 4), name="table")
    idx = dl.placeholder((2, 5), dtype="i32", name="idx")
    idx_val = np.array([[0, 2, 2, 5, 1], [3, 3, 3, 0, 4]], dtype=np.int32)  # repeated rows exercise scatter-add
    loss = weighted_sum(dl.embedding(table, idx), rng)
    gradcheck_finite_diff(loss, table, {table: rng.randn(6, 4), idx: idx_val})

def test_fanout_accumulation_gradcheck():
    np.random.seed(42)
    x = dl.placeholder((4, 4), name="x")
    # Fan-out: x feeds into two distinct computation branches
    loss = dl.sum(dl.relu(x) + dl.mul(x, x))
    x_val = np.random.randn(4, 4).astype(np.float64)
    gradcheck_finite_diff(loss, x, {x: x_val})

def run_all():
    test_matmul_gradcheck()
    test_batched_matmul_gradcheck()
    test_binary_broadcasting_gradcheck()
    test_unary_ops_gradcheck()
    test_softmax_and_cross_entropy_gradcheck()
    test_sequence_cross_entropy_gradcheck()
    test_layernorm_gradcheck()
    test_reductions_gradcheck()
    test_transpose_and_reshape_gradcheck()
    test_embedding_gradcheck()
    test_fanout_accumulation_gradcheck()
    print("[PASS] All Op VJP Gradient Checks Passed Successfully!")

if __name__ == "__main__":
    run_all()
