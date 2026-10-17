import numpy as np
import minidlc as dl

def test_reference_matmul_relu():
    x_val = np.random.randn(8, 16).astype(np.float32)
    w_val = np.random.randn(16, 4).astype(np.float32)
    b_val = np.random.randn(4).astype(np.float32)

    x = dl.placeholder((8, 16), name="x")
    w = dl.placeholder((16, 4), name="w")
    b = dl.placeholder((4,), name="b")

    out = dl.relu((x @ w) + b)

    interp = dl.ReferenceInterpreter()
    vals = interp.forward(out, {x: x_val, w: w_val, b: b_val})

    expected = np.maximum((x_val @ w_val) + b_val, 0.0)
    np.testing.assert_allclose(vals[out], expected, rtol=1e-5, atol=1e-5)

def test_reference_softmax_cross_entropy():
    logits_val = np.random.randn(10, 5).astype(np.float32)
    targets_val = np.random.randint(0, 5, size=(10,)).astype(np.int32)

    logits = dl.placeholder((10, 5), name="logits")
    targets = dl.placeholder((10,), dtype="i32", name="targets")

    loss = dl.cross_entropy(logits, targets)

    interp = dl.ReferenceInterpreter()
    vals = interp.forward(loss, {logits: logits_val, targets: targets_val})

    shift = logits_val - np.max(logits_val, axis=-1, keepdims=True)
    lse = np.log(np.sum(np.exp(shift), axis=-1)) + np.max(logits_val, axis=-1)
    expected_loss = np.mean(lse - logits_val[np.arange(10), targets_val])

    np.testing.assert_allclose(vals[loss], expected_loss, rtol=1e-5, atol=1e-5)

def test_reference_layernorm():
    x_val = np.random.randn(4, 8).astype(np.float32)
    gamma_val = np.ones((8,), dtype=np.float32) * 1.2
    beta_val = np.ones((8,), dtype=np.float32) * 0.3

    x = dl.placeholder((4, 8), name="x")
    gamma = dl.placeholder((8,), name="gamma")
    beta = dl.placeholder((8,), name="beta")

    out = dl.layernorm(x, gamma, beta, eps=1e-5)
    interp = dl.ReferenceInterpreter()
    vals = interp.forward(out, {x: x_val, gamma: gamma_val, beta: beta_val})

    mu = np.mean(x_val, axis=-1, keepdims=True)
    var = np.var(x_val, axis=-1, keepdims=True)
    expected = gamma_val * ((x_val - mu) / np.sqrt(var + 1e-5)) + beta_val
    np.testing.assert_allclose(vals[out], expected, rtol=1e-5, atol=1e-5)

def test_reference_transpose_reshape():
    x_val = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    x = dl.placeholder((2, 3, 4), name="x")

    t = dl.transpose(x, perm=[2, 0, 1])
    r = dl.reshape(t, (4, 6))

    interp = dl.ReferenceInterpreter()
    vals = interp.forward(r, {x: x_val})

    expected = np.transpose(x_val, (2, 0, 1)).reshape(4, 6)
    np.testing.assert_allclose(vals[r], expected, rtol=1e-5, atol=1e-5)

def run_all():
    test_reference_matmul_relu()
    test_reference_softmax_cross_entropy()
    test_reference_layernorm()
    test_reference_transpose_reshape()
    print("[PASS] All Reference Interpreter Tests Passed Successfully!")

if __name__ == "__main__":
    run_all()
