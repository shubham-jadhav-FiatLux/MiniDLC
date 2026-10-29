"""The GPU runtime: runs a compiled step (a plan plus PTX) on an NVIDIA GPU through the CUDA driver API.

The compiler (`_minidlc.compile_gpu`) turns a graph into a *GPU artifact*: the plan — memory sizes, where each
argument and result lives, the kernel launches in order with their grids and blocks, the final copies — and the PTX
of every kernel. This module needs neither LLVM nor the compiled module: it allocates two device regions (persistent
and transient), loads the PTX with `cuModuleLoadData` (the driver compiles it for the GPU at hand), and replays the
plan. Artifacts can be saved to a directory and run elsewhere (e.g. on Colab; see docs/gpu.md).

Devices:
- `CudaDevice`: libcuda through ctypes.
- `SimDevice`: host memory, kernels run by the compiler's lockstep GPU interpreter. It needs the compiled module
  (a saved artifact is recompiled from its source.dlc) and exists to test this runtime — slots, launches, copies,
  state — without a GPU.
"""
from __future__ import annotations

import ctypes
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

_DTYPES = {"f32": np.float32, "i32": np.int32}


# --- devices ---------------------------------------------------------------------------------------------------------

class CudaError(RuntimeError):
    pass


class CudaDevice:
    """The CUDA driver API through ctypes: one device, its primary context, one module per artifact."""

    def __init__(self, ordinal: int = 0, library: Optional[str] = None):
        self.lib = ctypes.CDLL(library or ("nvcuda.dll" if os.name == "nt" else "libcuda.so.1"))
        self._check(self.lib.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        self._check(self.lib.cuDeviceGet(ctypes.byref(dev), ordinal), "cuDeviceGet")
        self.device = dev.value
        ctx = ctypes.c_void_p()
        self._check(self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), self.device), "cuDevicePrimaryCtxRetain")
        self.context = ctx
        self._check(self.lib.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
        name = ctypes.create_string_buffer(256)
        self._check(self.lib.cuDeviceGetName(name, 256, self.device), "cuDeviceGetName")
        self.name = name.value.decode()
        major, minor = ctypes.c_int(), ctypes.c_int()
        self._check(self.lib.cuDeviceGetAttribute(ctypes.byref(major), 75, self.device), "compute capability")
        self._check(self.lib.cuDeviceGetAttribute(ctypes.byref(minor), 76, self.device), "compute capability")
        self.compute_capability = (major.value, minor.value)
        self.lib.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self.lib.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        self.lib.cuMemcpyDtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t]
        self.lib.cuMemsetD8_v2.argtypes = [ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t]
        self.lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]

    def _check(self, status: int, what: str) -> None:
        if status != 0:
            message = ctypes.c_char_p()
            self.lib.cuGetErrorString(status, ctypes.byref(message))
            raise CudaError(f"{what} failed: {message.value.decode() if message.value else status}")

    def load(self, artifact: "GpuArtifact") -> List[ctypes.c_void_p]:
        """Loads the artifact's PTX; returns its kernels (the module handle for launch)."""
        log = ctypes.create_string_buffer(1 << 14)
        options = (ctypes.c_int * 2)(5, 6)  # CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
        values = (ctypes.c_void_p * 2)(ctypes.cast(log, ctypes.c_void_p), ctypes.c_void_p(len(log)))
        module = ctypes.c_void_p()
        status = self.lib.cuModuleLoadDataEx(ctypes.byref(module), ctypes.c_char_p(artifact.ptx.encode() + b"\0"), 2,
                                             options, values)
        if status != 0:
            self._check(status, "cuModuleLoadDataEx: " + log.value.decode(errors="replace"))
        functions = []
        for name in artifact.kernel_names:
            fn = ctypes.c_void_p()
            self._check(self.lib.cuModuleGetFunction(ctypes.byref(fn), module, name.encode()), f"loading {name}")
            functions.append(fn)
        return functions

    def alloc(self, nbytes: int) -> int:
        ptr = ctypes.c_uint64()
        self._check(self.lib.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(max(nbytes, 256))), "cuMemAlloc")
        self._check(self.lib.cuMemsetD8_v2(ptr.value, 0, max(nbytes, 256)), "cuMemset")
        return ptr.value

    def free(self, ptr: int) -> None:
        self.lib.cuMemFree_v2(ptr)

    def htod(self, ptr: int, array: np.ndarray) -> None:
        array = np.ascontiguousarray(array)
        self._check(self.lib.cuMemcpyHtoD_v2(ptr, array.ctypes.data, array.nbytes), "cuMemcpyHtoD")

    def dtoh(self, ptr: int, out: np.ndarray) -> None:
        self._check(self.lib.cuMemcpyDtoH_v2(out.ctypes.data, ptr, out.nbytes), "cuMemcpyDtoH")

    def dtod(self, dst: int, src: int, nbytes: int) -> None:
        self._check(self.lib.cuMemcpyDtoD_v2(dst, src, nbytes), "cuMemcpyDtoD")

    def launch(self, module, kernel: int, grid, block, pointers: List[int]) -> None:
        values = [ctypes.c_uint64(p) for p in pointers]
        params = (ctypes.c_void_p * len(values))(*[ctypes.addressof(v) for v in values])
        self._check(self.lib.cuLaunchKernel(module[kernel], *grid, *block, 0, None, params, None),
                    f"launching kernel {kernel}")

    def synchronize(self) -> None:
        self._check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")


class SimDevice:
    """Host memory and the lockstep GPU interpreter, behind the CudaDevice interface (for testing the runtime)."""

    name = "simulated GPU (lockstep interpreter)"
    compute_capability = (0, 0)

    def __init__(self):
        self._memory: Dict[int, np.ndarray] = {}

    def load(self, artifact: "GpuArtifact"):
        if artifact.compiled is None:
            # A saved artifact: recompile its source (without PTX) and check that the plan is the same.
            from . import _minidlc
            artifact.compiled, _ = _minidlc.compile_gpu(artifact.source, artifact.options, False, artifact.arch)
            if json.loads(artifact.compiled.plan_json) != artifact.plan:
                raise RuntimeError("the artifact's plan differs from its source's: rebuild the artifact")
        return artifact.compiled

    def alloc(self, nbytes: int) -> int:
        buf = np.zeros(max(nbytes, 256) + 256, dtype=np.uint8)
        ptr = buf.ctypes.data + (-buf.ctypes.data) % 256
        self._memory[ptr] = buf
        return ptr

    def free(self, ptr: int) -> None:
        self._memory.pop(ptr, None)

    def htod(self, ptr: int, array: np.ndarray) -> None:
        array = np.ascontiguousarray(array)
        ctypes.memmove(ptr, array.ctypes.data, array.nbytes)

    def dtoh(self, ptr: int, out: np.ndarray) -> None:
        ctypes.memmove(out.ctypes.data, ptr, out.nbytes)

    def dtod(self, dst: int, src: int, nbytes: int) -> None:
        ctypes.memmove(dst, src, nbytes)

    def launch(self, module, kernel: int, grid, block, pointers: List[int]) -> None:
        module.simulate(kernel, pointers)

    def synchronize(self) -> None:
        pass


# --- artifacts and executables ---------------------------------------------------------------------------------------

@dataclass
class GpuArtifact:
    """A compiled step for the GPU: the plan, the PTX, and what a caller needs to feed it.

    `inputs` maps placeholder names to (argument name, dtype); `values` holds the initial values of the params and
    consts by argument name; `params` numbers the parameter arguments (their index in the model's parameter list);
    `training` says whether the first result is a loss.
    """
    plan: Dict[str, Any]
    ptx: str
    arch: str
    inputs: Dict[str, List[str]] = field(default_factory=dict)
    values: Dict[str, np.ndarray] = field(default_factory=dict)
    training: bool = False
    params: Dict[str, int] = field(default_factory=dict)
    source: str = ""  # the graph IR it was compiled from, and the compile options
    options: Dict[str, Any] = field(default_factory=dict)
    compiled: Any = None  # the _minidlc.GpuArtifact, when compiled in this process

    @property
    def kernel_names(self) -> List[str]:
        names: Dict[int, str] = {}
        for launch in self.plan["launches"]:
            names[launch["kernel"]] = launch["name"]
        return [names[i] for i in range(len(names))]

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "plan.json"), "w") as f:
            json.dump(self.plan, f, indent=1)
        with open(os.path.join(directory, "kernels.ptx"), "w") as f:
            f.write(self.ptx)
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump({"arch": self.arch, "inputs": self.inputs, "training": self.training, "params": self.params,
                       "options": self.options}, f, indent=1)
        with open(os.path.join(directory, "source.dlc"), "w") as f:
            f.write(self.source)
        np.savez_compressed(os.path.join(directory, "values.npz"), **self.values)

    @staticmethod
    def load(directory: str) -> "GpuArtifact":
        with open(os.path.join(directory, "plan.json")) as f:
            plan = json.load(f)
        with open(os.path.join(directory, "kernels.ptx")) as f:
            ptx = f.read()
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)
        with np.load(os.path.join(directory, "values.npz")) as npz:
            values = {k: npz[k] for k in npz.files}
        source_path = os.path.join(directory, "source.dlc")
        source = open(source_path).read() if os.path.exists(source_path) else ""
        return GpuArtifact(plan, ptx, meta["arch"], meta["inputs"], values, meta["training"], meta.get("params", {}),
                           source, meta.get("options", {}))


class GpuExecutable:
    """Replays a plan on a device: the counterpart of the C++ runtime's Executable."""

    def __init__(self, artifact: GpuArtifact, device=None, profile: bool = False):
        self.artifact = artifact
        self.plan = artifact.plan
        self.device = device or CudaDevice()
        self.module = self.device.load(artifact)
        self.regions = {"persistent": self.device.alloc(self.plan["persistent_bytes"]),
                        "transient": self.device.alloc(self.plan["transient_bytes"])}
        self.args = {a["name"]: a for a in self.plan["args"]}
        self.profiling = profile
        self.kernel_seconds: Dict[str, float] = {}
        self.calls = 0
        self.reset_state()
        for name, value in artifact.values.items():
            self.set(name, value)

    def __del__(self):
        try:
            for ptr in self.regions.values():
                self.device.free(ptr)
        except Exception:  # pragma: no cover - interpreter shutdown
            pass

    def address(self, slot: Dict[str, Any]) -> int:
        return self.regions[slot["region"]] + slot["offset"]

    def reset_state(self) -> None:
        for a in self.plan["args"]:
            if a["role"] == "state":
                init = a["init"] if a["init"] is not None else 0.0
                self.device.htod(self.address(a["slot"]), np.full(a["shape"], init, dtype=_DTYPES[a["dtype"]]))

    def set(self, name: str, value: np.ndarray) -> None:
        a = self.args[name]
        value = np.asarray(value, dtype=_DTYPES[a["dtype"]], order="C")
        if value.nbytes != a["slot"]["bytes"]:
            raise ValueError(f"'{name}' takes {a['slot']['bytes']} bytes, got {value.nbytes}")
        self.device.htod(self.address(a["slot"]), value)

    def get(self, name: str) -> np.ndarray:
        a = self.args[name]
        out = np.empty(a["shape"], dtype=_DTYPES[a["dtype"]])
        self.device.dtoh(self.address(a["slot"]), out)
        return out

    def run(self, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        for a in self.plan["args"]:
            if a["role"] != "input":
                continue
            if a["name"] not in inputs:
                raise ValueError(f"missing input '{a['name']}'")
            value = np.asarray(inputs[a["name"]], dtype=_DTYPES[a["dtype"]], order="C")
            if tuple(value.shape) != tuple(a["shape"]):
                raise ValueError(f"input '{a['name']}' has shape {value.shape}, expected {tuple(a['shape'])}")
            self.device.htod(self.address(a["slot"]), value)
        for launch in self.plan["launches"]:
            start = time.perf_counter() if self.profiling else 0.0
            self.device.launch(self.module, launch["kernel"], launch["grid"], launch["block"],
                               [self.address(s) for s in launch["args"]])
            if self.profiling:
                self.device.synchronize()
                self.kernel_seconds[launch["name"]] = self.kernel_seconds.get(launch["name"], 0.0) + (
                    time.perf_counter() - start)
        for c in self.plan["copies"]:
            self.device.dtod(self.address(c["to"]), self.address(c["from"]), c["from"]["bytes"])
        self.calls += 1
        results = []
        for r in self.plan["results"]:
            if r["kind"] == "donates":
                continue
            out = np.empty(r["shape"], dtype=_DTYPES[r["dtype"]])
            self.device.dtoh(self.address(r["slot"]), out)  # synchronizes with the launches
            results.append(out)
        return results

    def set_profiling(self, enabled: bool) -> None:
        """Times each launch (synchronizing after it, which serializes the step)."""
        self.profiling = enabled

    def profile(self) -> List[Dict[str, Any]]:
        flops = {l["name"]: l["flops"] for l in self.plan["launches"]}
        traffic = {l["name"]: l["bytes"] for l in self.plan["launches"]}
        return [{"name": n, "calls": self.calls, "seconds": s, "flops": flops[n], "bytes": traffic[n]}
                for n, s in self.kernel_seconds.items()]


class GpuFunction:
    """A GPU artifact behind the interface of `dl.compile`'s result: `step(x=..., y=...)` returns the loss (training)
    or the outputs."""

    def __init__(self, artifact: GpuArtifact, device=None, profile: bool = False):
        self.artifact = artifact
        self.executable = GpuExecutable(artifact, device, profile)

    def __call__(self, **feeds: np.ndarray):
        missing = set(self.artifact.inputs) - set(feeds)
        if missing:
            raise ValueError(f"missing inputs: {sorted(missing)}")
        arrays = {arg: np.asarray(feeds[name], dtype=_DTYPES[dtype], order="C")
                  for name, (arg, dtype) in self.artifact.inputs.items()}
        outputs = self.executable.run(arrays)
        if self.artifact.training:
            return float(outputs[0])
        return outputs[0] if len(outputs) == 1 else outputs


    def get_params(self) -> Dict[int, np.ndarray]:
        """The current parameters, by their index in the model's parameter list."""
        return {i: self.executable.get(arg) for arg, i in self.artifact.params.items()}

    def set_params(self, params: Dict[int, np.ndarray]) -> None:
        for arg, i in self.artifact.params.items():
            if i in params:
                self.executable.set(arg, params[i])


def load(directory: str, device=None, profile: bool = False) -> GpuFunction:
    """Loads a saved artifact and prepares it on `device` (default: the first CUDA GPU)."""
    return GpuFunction(GpuArtifact.load(directory), device, profile)
