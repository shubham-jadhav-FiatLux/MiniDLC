import random
from typing import List, Dict, Tuple
import numpy as np
import minidlc as dl

def finite_difference_grad(loss_node: dl.Tensor, param: dl.Tensor, feed_dict: Dict[dl.Tensor, np.ndarray], eps: float = 1e-6) -> np.ndarray:
    """Calculates numerical gradient via central finite difference in float64."""
    interp = dl.ReferenceInterpreter(dtype=np.float64)
    param_data = feed_dict[param]
    num_grad = np.zeros_like(param_data)
    orig = param_data.copy()

    it = np.nditer(param_data, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        val = orig[idx]

        param_data[idx] = val + eps
        v_plus = float(interp.forward(loss_node, feed_dict)[loss_node])

        param_data[idx] = val - eps
        v_minus = float(interp.forward(loss_node, feed_dict)[loss_node])

        num_grad[idx] = (v_plus - v_minus) / (2.0 * eps)
        param_data[idx] = val
        it.iternext()

    return num_grad

def generate_random_dag(seed: int, num_ops: int = 7) -> Tuple[dl.Tensor, Dict[dl.Tensor, np.ndarray], List[dl.Tensor]]:
    """Synthesizes a valid random computational DAG ending in a scalar loss."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    # Base shapes
    shapes = [(4, 4), (4, 8), (8, 4), (4, 1)]
    inputs: List[dl.Tensor] = []
    feed: Dict[dl.Tensor, np.ndarray] = {}

    for i, shp in enumerate(shapes):
        t = dl.placeholder(shp, name=f"input_{i}")
        # Initialize away from 0.0 to prevent ReLU gradient kink at non-differentiable point
        data = np_rng.randn(*shp).astype(np.float64) * 0.5
        data[np.abs(data) < 0.05] += 0.1
        inputs.append(t)
        feed[t] = data

    pool: List[dl.Tensor] = list(inputs)

    for op_idx in range(num_ops):
        op_type = rng.choice(["relu", "gelu", "add", "sub", "mul", "matmul", "layernorm", "softmax"])

        if op_type in ["relu", "gelu"]:
            inp = rng.choice(pool)
            new_node = dl.relu(inp) if op_type == "relu" else dl.gelu(inp)
            pool.append(new_node)

        elif op_type in ["add", "sub", "mul"]:
            # Pick two nodes with compatible shapes
            t1 = rng.choice(pool)
            t2 = rng.choice(pool)
            try:
                # Test broadcast compatibility
                dl.ops._broadcast_shape(t1.shape, t2.shape)
                if op_type == "add":
                    new_node = dl.add(t1, t2)
                elif op_type == "sub":
                    new_node = dl.sub(t1, t2)
                else:
                    # Scale down multiplication to prevent numerical explosion across deep graphs
                    new_node = dl.mul(t1, dl.mul(t2, 0.25))
                pool.append(new_node)
            except ValueError:
                continue

        elif op_type == "matmul":
            # Find two 2D nodes where t1.cols == t2.rows
            compat = []
            for t1 in pool:
                if len(t1.shape) == 2:
                    for t2 in pool:
                        if len(t2.shape) == 2 and t1.shape[1] == t2.shape[0]:
                            compat.append((t1, t2))
            if compat:
                t1, t2 = rng.choice(compat)
                # Scale down matmuls to stabilize values
                mm = dl.matmul(t1, t2)
                new_node = dl.mul(mm, 0.25)
                pool.append(new_node)

        elif op_type == "layernorm":
            candidates = [t for t in pool if len(t.shape) == 2]
            if candidates:
                inp = rng.choice(candidates)
                d_dim = inp.shape[-1]
                gamma = dl.placeholder((d_dim,), name=f"gamma_{op_idx}")
                beta = dl.placeholder((d_dim,), name=f"beta_{op_idx}")
                feed[gamma] = np.ones(d_dim, dtype=np.float64)
                feed[beta] = np.zeros(d_dim, dtype=np.float64)
                inputs.append(gamma)
                inputs.append(beta)
                new_node = dl.layernorm(inp, gamma, beta)
                pool.append(new_node)

        elif op_type == "softmax":
            candidates = [t for t in pool if len(t.shape) == 2]
            if candidates:
                inp = rng.choice(candidates)
                new_node = dl.softmax(inp, axis=-1)
                pool.append(new_node)

    # Combine multiple active nodes to maximize parameter graph coverage
    candidates = [p for p in pool if len(p.shape) >= 2]
    final_shape = candidates[-1].shape
    matching = [p for p in candidates if p.shape == final_shape]
    total = matching[-1]
    for node in matching[:-1]:
        try:
            total = dl.add(total, node)
        except Exception:
            pass

    # Reduce to a scalar
    loss = dl.sum(total)

    return loss, feed, inputs

def test_random_graph_fuzzing(num_trials: int = 25):
    """Stress tests random multi-op computational DAGs against finite differences."""
    interp = dl.ReferenceInterpreter(dtype=np.float64)

    print(f"Fuzzing {num_trials} randomized computational DAGs...")
    for trial in range(num_trials):
        seed = 42000 + trial
        loss, feed, inputs = generate_random_dag(seed=seed, num_ops=6)

        # 1. Forward pass
        vals = interp.forward(loss, feed)
        loss_val = float(vals[loss])
        assert not np.isnan(loss_val), f"Trial {trial}: NaN in loss"
        assert not np.isinf(loss_val), f"Trial {trial}: Inf in loss"

        # 2. Backward pass
        analytic_grads = interp.backward(loss, vals)

        # 3. Check gradients against numerical finite differences for active inputs
        checked_params = 0
        for p in inputs:
            if p in analytic_grads:
                a_grad = analytic_grads[p]
                if np.isnan(a_grad).any() or np.isinf(a_grad).any():
                    raise AssertionError(f"Trial {trial}: Non-finite gradient in {p.name}")

                n_grad = finite_difference_grad(loss, p, feed, eps=1e-5)
                # Verify relative and absolute tolerance
                np.testing.assert_allclose(
                    a_grad,
                    n_grad,
                    rtol=2e-3,
                    atol=2e-3,
                    err_msg=f"Trial {trial} (seed {seed}) gradient mismatch for {p.name}"
                )
                checked_params += 1

        print(f"  [PASS] Fuzzer Trial {trial + 1}/{num_trials} (seed {seed}, verified {checked_params} parameter gradients)")

    print(f"[PASS] All {num_trials} Random Graph Fuzzing Trials Passed Successfully!")

if __name__ == "__main__":
    test_random_graph_fuzzing(num_trials=25)
