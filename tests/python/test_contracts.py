import numpy as np
import minidlc as dl
from minidlc import nn

def test_nan_inf_propagation():
    """Verify IEEE 754 NaN and Inf semantics across ops and autograd."""
    x = dl.placeholder((2, 2), name="x")
    r = dl.relu(x)
    interp = dl.ReferenceInterpreter()

    # Forward: relu(NaN) -> NaN, relu(-Inf) -> 0.0, relu(+Inf) -> Inf
    x_val = np.array([[np.nan, -np.inf], [np.inf, -2.0]], dtype=np.float32)
    vals = interp.forward(r, {x: x_val})
    res = vals[r]

    assert np.isnan(res[0, 0]), "relu(NaN) must be NaN"
    assert res[0, 1] == 0.0, "relu(-Inf) must be 0.0"
    assert np.isneginf(x_val[0, 1]), "Input was -inf"
    assert np.isposinf(res[1, 0]), "relu(+Inf) must be +Inf"
    assert res[1, 1] == 0.0, "relu(-2.0) must be 0.0"

    # Backward: NaN safety (PyTorch convention: grad * (x > 0.0) -> NaN > 0 is False -> 0.0)
    grads = interp.backward(r, vals)
    grad_x = grads[x]
    assert grad_x[0, 0] == 0.0, "relu_backward at NaN should yield 0.0"
    assert grad_x[0, 1] == 0.0, "relu_backward at -Inf should yield 0.0"
    assert grad_x[1, 0] == 1.0, "relu_backward at +Inf should yield 1.0"
    assert grad_x[1, 1] == 0.0, "relu_backward at negative should yield 0.0"
    print("[PASS] test_nan_inf_propagation")

def test_extreme_numerical_stability():
    """Verify log-sum-exp numerical stability with logits up to +/- 10,000."""
    x = dl.placeholder((2, 4), name="x")
    s = dl.softmax(x, axis=-1)
    ls = dl.log_softmax(x, axis=-1)

    interp = dl.ReferenceInterpreter()
    # Large positive logits: would overflow float32 exp(10000) = inf if unshifted
    feed_large = {x: np.array([[10000.0, 10001.0, 9999.0, 10000.0],
                               [-10000.0, -10000.0, -10000.0, -10000.0]], dtype=np.float32)}

    vals_s = interp.forward(s, feed_large)
    probs = vals_s[s]
    assert not np.isnan(probs).any(), "NaN in extreme softmax!"
    assert not np.isinf(probs).any(), "Inf in extreme softmax!"
    np.testing.assert_allclose(np.sum(probs, axis=-1), np.array([1.0, 1.0]), rtol=1e-5)

    vals_ls = interp.forward(ls, feed_large)
    log_probs = vals_ls[ls]
    assert not np.isnan(log_probs).any(), "NaN in extreme log_softmax!"
    assert not np.isinf(log_probs).any(), "Inf in extreme log_softmax!"

    # Cross entropy with extreme logits
    logits = dl.placeholder((2, 4), name="logits")
    targets = dl.placeholder((2,), dtype="i32", name="targets")
    ce_loss = dl.cross_entropy(logits, targets)

    feed_ce = {logits: feed_large[x], targets: np.array([1, 0], dtype=np.int32)}
    vals_ce = interp.forward(ce_loss, feed_ce)
    loss_val = float(vals_ce[ce_loss])
    assert not np.isnan(loss_val), "NaN in extreme cross entropy!"
    assert not np.isinf(loss_val), "Inf in extreme cross entropy!"
    assert loss_val >= 0.0, "Loss must be non-negative"

    # Backward gradients must also remain finite
    grads = interp.backward(ce_loss, vals_ce)
    assert not np.isnan(grads[logits]).any(), "NaN in cross entropy gradient!"
    assert not np.isinf(grads[logits]).any(), "Inf in cross entropy gradient!"
    print("[PASS] test_extreme_numerical_stability")

def test_softmax_non_default_axes():
    """Verify softmax and log_softmax along non-default axes (e.g. axis=0, axis=1)."""
    interp = dl.ReferenceInterpreter()
    np.random.seed(42)
    x_val = np.random.randn(3, 4, 5).astype(np.float32)

    for ax in [0, 1, 2, -1, -2]:
        x = dl.placeholder((3, 4, 5), name=f"x_ax_{ax}")
        s = dl.softmax(x, axis=ax)
        ls = dl.log_softmax(x, axis=ax)

        vals = interp.forward(s, {x: x_val})
        probs = vals[s]
        # Sum along axis should be 1.0
        axis_sums = np.sum(probs, axis=ax)
        np.testing.assert_allclose(axis_sums, np.ones_like(axis_sums), rtol=1e-5, atol=1e-5)

        vals_ls = interp.forward(ls, {x: x_val})
        log_probs = vals_ls[ls]
        np.testing.assert_allclose(np.exp(log_probs), probs, rtol=1e-5, atol=1e-5)
    print("[PASS] test_softmax_non_default_axes")

def test_high_rank_batched_matmuls():
    """Verify 4D batched matmuls [B, H, M, K] @ [B, H, K, N] forward and backward."""
    B, H, M, K, N = 2, 4, 8, 16, 6
    np.random.seed(42)
    a_val = np.random.randn(B, H, M, K).astype(np.float64)
    b_val = np.random.randn(B, H, K, N).astype(np.float64)

    a = dl.placeholder((B, H, M, K), name="a")
    b = dl.placeholder((B, H, K, N), name="b")
    c = a @ b
    loss = dl.sum(c)

    interp = dl.ReferenceInterpreter(dtype=np.float64)
    feed = {a: a_val, b: b_val}
    vals = interp.forward(loss, feed)

    expected_c = a_val @ b_val
    np.testing.assert_allclose(vals[c], expected_c, rtol=1e-6, atol=1e-6)

    # Gradients
    grads = interp.backward(loss, vals)
    grad_a = grads[a]
    grad_b = grads[b]

    # For sum(a @ b), dL/dc = ones(B, H, M, N)
    # dL/da = ones @ b.T
    # dL/db = a.T @ ones
    ones = np.ones((B, H, M, N), dtype=np.float64)
    expected_grad_a = ones @ np.swapaxes(b_val, -1, -2)
    expected_grad_b = np.swapaxes(a_val, -1, -2) @ ones

    np.testing.assert_allclose(grad_a, expected_grad_a, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(grad_b, expected_grad_b, rtol=1e-6, atol=1e-6)
    print("[PASS] test_high_rank_batched_matmuls")

def test_broadcasting_corner_cases():
    """Verify scalar x tensor, 1xN into MxN, and higher-rank broadcast adjoint accumulations."""
    interp = dl.ReferenceInterpreter(dtype=np.float64)
    np.random.seed(42)

    # 1. Scalar float + Tensor
    x = dl.placeholder((3, 4), name="x")
    s = dl.add(x, 2.5)
    loss1 = dl.sum(s)
    x_val = np.random.randn(3, 4).astype(np.float64)
    vals1 = interp.forward(loss1, {x: x_val})
    np.testing.assert_allclose(vals1[s], x_val + 2.5)
    grads1 = interp.backward(loss1, vals1)
    np.testing.assert_allclose(grads1[x], np.ones((3, 4)))

    # 2. Row broadcast: (1, 4) + (3, 4)
    r = dl.placeholder((1, 4), name="r")
    loss2 = dl.sum(dl.mul(x, r))
    r_val = np.random.randn(1, 4).astype(np.float64)
    vals2 = interp.forward(loss2, {x: x_val, r: r_val})
    grads2 = interp.backward(loss2, vals2)
    # grad wrt r must sum out the batch dimension (axis 0)
    expected_dr = np.sum(x_val, axis=0, keepdims=True)
    np.testing.assert_allclose(grads2[r], expected_dr, rtol=1e-6)
    assert grads2[r].shape == (1, 4)

    # 3. 1D broadcast into 3D: (5,) into (2, 3, 5)
    t = dl.placeholder((2, 3, 5), name="t")
    v = dl.placeholder((5,), name="v")
    loss3 = dl.sum(dl.add(t, v))
    t_val = np.random.randn(2, 3, 5).astype(np.float64)
    v_val = np.random.randn(5).astype(np.float64)
    vals3 = interp.forward(loss3, {t: t_val, v: v_val})
    grads3 = interp.backward(loss3, vals3)
    assert grads3[v].shape == (5,)
    np.testing.assert_allclose(grads3[v], np.full((5,), 6.0))
    print("[PASS] test_broadcasting_corner_cases")

def test_sum_mean_keepdims_and_negative_axis():
    """Verify sum and mean with keepdims=True and negative axes."""
    interp = dl.ReferenceInterpreter(dtype=np.float64)
    np.random.seed(42)
    x_val = np.random.randn(2, 3, 4).astype(np.float64)

    # Full reduction keepdims=True -> shape (1, 1, 1)
    x = dl.placeholder((2, 3, 4), name="x")
    s_kd = dl.sum(x, keepdims=True)
    assert s_kd.shape == (1, 1, 1), f"Expected (1, 1, 1), got {s_kd.shape}"

    vals = interp.forward(s_kd, {x: x_val})
    assert vals[s_kd].shape == (1, 1, 1)
    np.testing.assert_allclose(vals[s_kd], np.sum(x_val, keepdims=True))

    # Mean along axis=-2 with keepdims=True -> shape (2, 1, 4)
    m_kd = dl.mean(x, axis=-2, keepdims=True)
    assert m_kd.shape == (2, 1, 4), f"Expected (2, 1, 4), got {m_kd.shape}"
    vals_m = interp.forward(m_kd, {x: x_val})
    assert vals_m[m_kd].shape == (2, 1, 4)
    np.testing.assert_allclose(vals_m[m_kd], np.mean(x_val, axis=-2, keepdims=True))

    # Backward of mean
    loss_m = dl.sum(m_kd)
    grads_m = interp.backward(loss_m, vals_m)
    # Each element of axis=-2 has 3 elements, so grad is 1.0 / 3.0
    np.testing.assert_allclose(grads_m[x], np.full((2, 3, 4), 1.0 / 3.0), rtol=1e-6)
    print("[PASS] test_sum_mean_keepdims_and_negative_axis")

def test_non_tile_multiple_shapes():
    """Verify prime and non-power-of-two dimensions (e.g. 17x33x19)."""
    M, K, N = 17, 33, 19
    np.random.seed(42)
    a_val = np.random.randn(M, K).astype(np.float64)
    b_val = np.random.randn(K, N).astype(np.float64)

    a = dl.placeholder((M, K), name="a")
    b = dl.placeholder((K, N), name="b")
    c = a @ b
    relu_c = dl.relu(c)
    loss = dl.sum(relu_c)

    interp = dl.ReferenceInterpreter(dtype=np.float64)
    feed = {a: a_val, b: b_val}
    vals = interp.forward(loss, feed)

    expected_c = np.maximum(a_val @ b_val, 0.0)
    np.testing.assert_allclose(vals[relu_c], expected_c, rtol=1e-6)

    grads = interp.backward(loss, vals)
    dC = (a_val @ b_val > 0.0).astype(np.float64)
    expected_da = dC @ b_val.T
    expected_db = a_val.T @ dC

    np.testing.assert_allclose(grads[a], expected_da, rtol=1e-6)
    np.testing.assert_allclose(grads[b], expected_db, rtol=1e-6)
    print("[PASS] test_non_tile_multiple_shapes")

def test_fanout_multi_consumer_accumulation():
    """Verify that multiple consumers of a node correctly accumulate gradients (dL/dx = sum dL/dy_i * dy_i/dx)."""
    np.random.seed(42)
    x_val = np.random.randn(4, 4).astype(np.float64)

    x = dl.placeholder((4, 4), name="x")
    # Branch 1: x
    # Branch 2: relu(x)
    # Branch 3: x * x
    # Loss = sum(x + relu(x) + x * x)
    y1 = x
    y2 = dl.relu(x)
    y3 = dl.mul(x, x)
    total = dl.add(dl.add(y1, y2), y3)
    loss = dl.sum(total)

    interp = dl.ReferenceInterpreter(dtype=np.float64)
    feed = {x: x_val}
    vals = interp.forward(loss, feed)

    grads = interp.backward(loss, vals)
    grad_x = grads[x]

    # Analytic expectation: dL/dx = 1.0 + (x > 0) + 2 * x
    expected_grad = 1.0 + (x_val > 0.0).astype(np.float64) + 2.0 * x_val
    np.testing.assert_allclose(grad_x, expected_grad, rtol=1e-6)
    print("[PASS] test_fanout_multi_consumer_accumulation")

def test_zero_size_dimensions():
    """Verify that empty / 0-sized dimension tensors do not crash forward or backward."""
    interp = dl.ReferenceInterpreter(dtype=np.float32)
    x = dl.placeholder((0, 8), name="x")
    s = dl.sum(x)

    x_val = np.zeros((0, 8), dtype=np.float32)
    vals = interp.forward(s, {x: x_val})
    assert vals[s] == 0.0

    grads = interp.backward(s, vals)
    assert grads[x].shape == (0, 8)
    print("[PASS] test_zero_size_dimensions")

if __name__ == "__main__":
    print("Running MiniDLC Rigorous Contract & Corner Case Stress Tests...")
    test_nan_inf_propagation()
    test_extreme_numerical_stability()
    test_softmax_non_default_axes()
    test_high_rank_batched_matmuls()
    test_broadcasting_corner_cases()
    test_sum_mean_keepdims_and_negative_axis()
    test_non_tile_multiple_shapes()
    test_fanout_multi_consumer_accumulation()
    test_zero_size_dimensions()
    print("All Contract and Corner Case Stress Tests Passed Successfully!")
