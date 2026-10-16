import numpy as np
import minidlc as dl
from minidlc import ops, nn
from minidlc.reference import ReferenceInterpreter

def test_layernorm_forward():
    np.random.seed(42)
    B, T, D = 2, 4, 16
    x_np = np.random.randn(B, T, D).astype(np.float32)
    gamma_np = np.random.uniform(0.5, 1.5, (D,)).astype(np.float32)
    beta_np = np.random.randn(D).astype(np.float32)

    x = dl.Tensor(shape=(B, T, D), dtype="f32", op="constant", data=x_np.copy())
    gamma = dl.Tensor(shape=(D,), dtype="f32", op="constant", data=gamma_np.copy())
    beta = dl.Tensor(shape=(D,), dtype="f32", op="constant", data=beta_np.copy())

    out = ops.layernorm(x, gamma, beta, eps=1e-5)
    interp = ReferenceInterpreter()
    vals = interp.forward(out, {})
    y_actual = vals[out]

    # Reference computation
    mean = np.mean(x_np, axis=-1, keepdims=True)
    var = np.var(x_np, axis=-1, keepdims=True)
    x_hat = (x_np - mean) / np.sqrt(var + 1e-5)
    y_expected = gamma_np * x_hat + beta_np

    np.testing.assert_allclose(y_actual, y_expected, rtol=1e-5, atol=1e-5)
    print("[PASS] test_layernorm_forward")

def test_layernorm_backward():
    np.random.seed(42)
    B, T, D = 2, 3, 6
    x_np = np.random.randn(B, T, D).astype(np.float64)
    gamma_np = np.random.uniform(0.5, 1.5, (D,)).astype(np.float64)
    beta_np = np.random.randn(D).astype(np.float64)

    x = dl.placeholder((B, T, D), dtype="f64", name="x")
    gamma = dl.placeholder((D,), dtype="f64", name="gamma")
    beta = dl.placeholder((D,), dtype="f64", name="beta")

    out = ops.layernorm(x, gamma, beta, eps=1e-5)
    loss = ops.sum(out)

    interp = ReferenceInterpreter(dtype=np.float64)
    feed = {x: x_np.copy(), gamma: gamma_np.copy(), beta: beta_np.copy()}
    vals = interp.forward(loss, feed)
    grads = interp.backward(loss, vals)

    dx = grads[x]
    dgamma = grads[gamma]
    dbeta = grads[beta]

    eps = 1e-5
    # Check dgamma
    for i in range(D):
        g_plus = gamma_np.copy(); g_plus[i] += eps
        g_minus = gamma_np.copy(); g_minus[i] -= eps
        v_plus = interp.forward(loss, {x: x_np, gamma: g_plus, beta: beta_np})[loss]
        v_minus = interp.forward(loss, {x: x_np, gamma: g_minus, beta: beta_np})[loss]
        num_grad = (v_plus - v_minus) / (2.0 * eps)
        np.testing.assert_allclose(dgamma[i], num_grad, rtol=1e-3, atol=1e-3)

    # Check dbeta
    for i in range(D):
        b_plus = beta_np.copy(); b_plus[i] += eps
        b_minus = beta_np.copy(); b_minus[i] -= eps
        v_plus = interp.forward(loss, {x: x_np, gamma: gamma_np, beta: b_plus})[loss]
        v_minus = interp.forward(loss, {x: x_np, gamma: gamma_np, beta: b_minus})[loss]
        num_grad = (v_plus - v_minus) / (2.0 * eps)
        np.testing.assert_allclose(dbeta[i], num_grad, rtol=1e-3, atol=1e-3)

    # Check dx on sample points
    for b in range(B):
        for t in range(T):
            for d in range(D):
                x_plus = x_np.copy(); x_plus[b, t, d] += eps
                x_minus = x_np.copy(); x_minus[b, t, d] -= eps
                v_plus = interp.forward(loss, {x: x_plus, gamma: gamma_np, beta: beta_np})[loss]
                v_minus = interp.forward(loss, {x: x_minus, gamma: gamma_np, beta: beta_np})[loss]
                num_grad = (v_plus - v_minus) / (2.0 * eps)
                np.testing.assert_allclose(dx[b, t, d], num_grad, rtol=1e-3, atol=1e-3)

    print("[PASS] test_layernorm_backward")

def test_gelu_forward_and_backward():
    x_np = np.linspace(-3.0, 3.0, 25, dtype=np.float64)
    x = dl.placeholder((25,), dtype="f64", name="x")
    out = ops.gelu(x)
    loss = ops.sum(out)

    interp = ReferenceInterpreter(dtype=np.float64)
    feed = {x: x_np.copy()}
    vals = interp.forward(loss, feed)
    y_actual = vals[out]

    ref_gelu = 0.5 * x_np * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x_np + 0.044715 * (x_np ** 3))))
    np.testing.assert_allclose(y_actual, ref_gelu, rtol=1e-5, atol=1e-5)

    grads = interp.backward(loss, vals)
    dx = grads[x]

    eps = 1e-5
    for i in range(len(x_np)):
        x_plus = x_np.copy(); x_plus[i] += eps
        x_minus = x_np.copy(); x_minus[i] -= eps
        v_plus = interp.forward(loss, {x: x_plus})[loss]
        v_minus = interp.forward(loss, {x: x_minus})[loss]
        num_grad = (v_plus - v_minus) / (2.0 * eps)
        np.testing.assert_allclose(dx[i], num_grad, rtol=1e-3, atol=1e-3)

    print("[PASS] test_gelu_forward_and_backward")

def test_causal_multi_head_attention():
    np.random.seed(42)
    B, T, D, H = 2, 6, 16, 2
    mha = nn.MultiHeadAttention(d_model=D, num_heads=H, causal=True)
    x_base = np.random.randn(B, T, D).astype(np.float32)
    interp = ReferenceInterpreter()

    # Step-wise causality assertion
    for t in range(T - 1):
        x_mod = x_base.copy()
        # Perturb all future tokens strictly after step t
        x_mod[:, t+1:, :] += np.random.randn(B, T - 1 - t, D).astype(np.float32)

        t_base = dl.Tensor(shape=(B, T, D), dtype="f32", op="constant", data=x_base)
        t_mod = dl.Tensor(shape=(B, T, D), dtype="f32", op="constant", data=x_mod)

        out_base = interp.forward(mha(t_base), {})
        out_mod = interp.forward(mha(t_mod), {})

        y_base = list(out_base.values())[-1]
        y_mod = list(out_mod.values())[-1]

        # Prior tokens 0..t must be completely identical
        np.testing.assert_allclose(y_base[:, :t+1, :], y_mod[:, :t+1, :], atol=1e-5)
        # Future tokens t+1..T must differ
        assert not np.allclose(y_base[:, t+1:, :], y_mod[:, t+1:, :], atol=1e-3)

    print("[PASS] test_causal_multi_head_attention")

def test_transformer_block():
    np.random.seed(42)
    B, T, D = 2, 8, 16
    block = nn.TransformerBlock(d_model=D, num_heads=2, causal=True)
    x_np = np.random.randn(B, T, D).astype(np.float32)
    x = dl.Tensor(shape=(B, T, D), dtype="f32", op="constant", data=x_np)

    out = block(x)
    assert out.shape == (B, T, D)

    interp = ReferenceInterpreter()
    vals = interp.forward(out, {})
    y = vals[out]
    assert y.shape == (B, T, D)
    assert not np.isnan(y).any()
    print("[PASS] test_transformer_block")

def test_char_gpt_single_step_training():
    np.random.seed(42)
    B, T, V = 4, 16, 65
    x_data = np.random.randint(0, V, (B, T), dtype=np.int32)
    y_data = np.random.randint(0, V, (B, T), dtype=np.int32)

    model = nn.CharGPT(vocab_size=V, block_size=T, d_model=32, num_heads=2, num_layers=2)

    x_node = dl.placeholder((B, T), dtype="i32", name="x")
    y_node = dl.placeholder((B, T), dtype="i32", name="y")

    logits = model(x_node)
    loss = ops.cross_entropy(logits, y_node)

    interp = ReferenceInterpreter()
    feed = {x_node: x_data, y_node: y_data}

    # Initial forward loss
    vals = interp.forward(loss, feed)
    loss_val_before = float(vals[loss])
    # Expected initial cross entropy for random weights is ~ln(65) = 4.17
    assert 3.5 < loss_val_before < 5.0, f"Unexpected initial loss: {loss_val_before}"

    # Backward gradient calculation
    grads = interp.backward(loss, vals)

    # Check that all model parameters received finite gradients
    lr = 0.05
    for p in model.parameters():
        assert p in grads, f"Parameter {p.name} missing from gradients"
        g = grads[p]
        assert not np.isnan(g).any(), f"NaN in gradient for {p.name}"
        assert not np.isinf(g).any(), f"Inf in gradient for {p.name}"
        p.data = p.data - lr * g

    # Forward after update
    vals_after = interp.forward(loss, feed)
    loss_val_after = float(vals_after[loss])
    assert loss_val_after < loss_val_before, f"Loss did not decrease: {loss_val_before:.4f} -> {loss_val_after:.4f}"
    print(f"[PASS] test_char_gpt_single_step_training (loss: {loss_val_before:.4f} -> {loss_val_after:.4f})")

def test_char_gpt_generation():
    np.random.seed(42)
    model = nn.CharGPT(vocab_size=65, block_size=16, d_model=32, num_heads=2, num_layers=1)
    interp = ReferenceInterpreter()

    prompt = [10, 25, 30]
    generated = model.generate(interp, prompt, max_new_tokens=10, temperature=0.8)

    assert len(generated) == len(prompt) + 10
    for tok in generated:
        assert 0 <= tok < 65
    print("[PASS] test_char_gpt_generation")

if __name__ == "__main__":
    print("Running Transformer and Char-GPT Unit Tests...")
    test_layernorm_forward()
    test_layernorm_backward()
    test_gelu_forward_and_backward()
    test_causal_multi_head_attention()
    test_transformer_block()
    test_char_gpt_single_step_training()
    test_char_gpt_generation()
    print("All Transformer and Char-GPT Unit Tests Passed Successfully!")
