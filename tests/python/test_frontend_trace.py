import minidlc as dl
from minidlc import nn

def test_mlp_tracing():
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            return self.fc2(dl.relu(self.fc1(x)))

    model = MLP()
    x = dl.placeholder((128, 784), dtype="f32", name="x")
    out = model(x)

    assert out.shape == (128, 10)
    assert out.op == "add"  # fc2 bias addition
    assert len(model.parameters()) == 4

def test_operator_overloading():
    a = dl.placeholder((10, 20), name="a")
    b = dl.placeholder((20, 30), name="b")
    c = dl.placeholder((30,), name="c")

    res = (a @ b) + c
    assert res.shape == (10, 30)
    assert res.op == "add"
    assert res.inputs[0].op == "matmul"

if __name__ == "__main__":
    test_mlp_tracing()
    test_operator_overloading()
    print("[PASS] All Frontend Tracing Tests Passed Successfully!")
