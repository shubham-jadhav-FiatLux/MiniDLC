"""
CUDA Engine & Dynamic Driver API Runtime for MiniDLC.
Provides dynamic loading of libcuda / libnvrtc, device arena allocation,
host-device memory transfers, and kernel dispatching.
"""

import os
import ctypes
from typing import Optional, Dict, Any, Tuple
import numpy as np


class CUDAEngine:
    """Manages CUDA driver API loading, compilation via NVRTC, and device execution."""

    _instance: Optional["CUDAEngine"] = None
    _cuda_lib: Optional[ctypes.CDLL] = None
    _nvrtc_lib: Optional[ctypes.CDLL] = None

    def __init__(self):
        self._init_driver_api()

    @classmethod
    def get(cls) -> "CUDAEngine":
        if cls._instance is None:
            cls._instance = CUDAEngine()
        return cls._instance

    @classmethod
    def is_available(cls) -> bool:
        try:
            eng = cls.get()
            return eng._cuda_lib is not None
        except Exception:
            return False

    def _init_driver_api(self):
        cuda_paths = [
            "libcuda.so",
            "libcuda.so.1",
            "/usr/lib/x86_64-linux-gnu/libcuda.so",
            "/usr/local/cuda/lib64/libcuda.so",
        ]
        for p in cuda_paths:
            try:
                self._cuda_lib = ctypes.CDLL(p)
                break
            except OSError:
                continue

        nvrtc_paths = [
            "libnvrtc.so",
            "/usr/local/cuda/lib64/libnvrtc.so",
            "/usr/lib/x86_64-linux-gnu/libnvrtc.so",
        ]
        for p in nvrtc_paths:
            try:
                self._nvrtc_lib = ctypes.CDLL(p)
                break
            except OSError:
                continue

    def allocate_device_arena(self, size_bytes: int) -> int:
        """Allocates contiguous memory on the CUDA device arena."""
        if not self.is_available():
            raise RuntimeError(
                "CUDA GPU device or driver library (libcuda.so) is not available on this machine. "
                "To execute CUDA kernels, run on a CUDA-equipped system or build inside docker/Dockerfile.cuda."
            )
        d_ptr = ctypes.c_void_p()
        # cuMemAlloc(&d_ptr, size_bytes)
        res = self._cuda_lib.cuMemAlloc(ctypes.byref(d_ptr), ctypes.c_size_t(size_bytes))
        if res != 0:
            raise RuntimeError(f"cuMemAlloc failed with error code: {res}")
        return d_ptr.value

    def free_device_arena(self, d_ptr: int) -> None:
        """Frees device arena memory."""
        if self._cuda_lib and d_ptr:
            self._cuda_lib.cuMemFree(ctypes.c_void_p(d_ptr))

    def copy_htod(self, d_dst: int, h_src: np.ndarray) -> None:
        """Host to Device memory copy (cudaMemcpyHostToDevice)."""
        if not self.is_available():
            raise RuntimeError("CUDA is not available.")
        size_bytes = h_src.nbytes
        res = self._cuda_lib.cuMemcpyHtoD(
            ctypes.c_void_p(d_dst),
            h_src.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(size_bytes)
        )
        if res != 0:
            raise RuntimeError(f"cuMemcpyHtoD failed with code {res}")

    def copy_dtoh(self, h_dst: np.ndarray, d_src: int) -> None:
        """Device to Host memory copy (cudaMemcpyDeviceToHost)."""
        if not self.is_available():
            raise RuntimeError("CUDA is not available.")
        size_bytes = h_dst.nbytes
        res = self._cuda_lib.cuMemcpyDtoH(
            h_dst.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(d_src),
            ctypes.c_size_t(size_bytes)
        )
        if res != 0:
            raise RuntimeError(f"cuMemcpyDtoH failed with code {res}")

    def compile_nvrtc(self, cuda_source: str, kernel_name: str) -> bytes:
        """Compiles CUDA C++ source to PTX bytecode at runtime using NVRTC."""
        if self._nvrtc_lib is None:
            raise RuntimeError("libnvrtc.so (CUDA Runtime Compiler) is not available.")
        # NVRTC program compilation
        prog = ctypes.c_void_p()
        src_bytes = cuda_source.encode("utf-8")
        name_bytes = kernel_name.encode("utf-8")

        res = self._nvrtc_lib.nvrtcCreateProgram(
            ctypes.byref(prog), src_bytes, name_bytes, 0, None, None
        )
        if res != 0:
            raise RuntimeError(f"nvrtcCreateProgram failed: {res}")

        # Compile options: -default-device, -use_fast_math
        options = [b"--use_fast_math", b"-arch=compute_75"]
        opt_arr = (ctypes.c_char_p * len(options))(*options)
        compile_res = self._nvrtc_lib.nvrtcCompileProgram(prog, len(options), opt_arr)
        if compile_res != 0:
            # Retrieve log
            log_size = ctypes.c_size_t()
            self._nvrtc_lib.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
            log_buf = ctypes.create_string_buffer(log_size.value)
            self._nvrtc_lib.nvrtcGetProgramLog(prog, log_buf)
            raise RuntimeError(f"NVRTC compilation failed:\n{log_buf.value.decode('utf-8')}")

        ptx_size = ctypes.c_size_t()
        self._nvrtc_lib.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
        ptx_buf = ctypes.create_string_buffer(ptx_size.value)
        self._nvrtc_lib.nvrtcGetPTX(prog, ptx_buf)
        return ptx_buf.raw
