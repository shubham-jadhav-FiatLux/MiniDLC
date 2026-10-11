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
