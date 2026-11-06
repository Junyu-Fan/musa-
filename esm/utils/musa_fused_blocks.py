import ctypes
import os
from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache
def _launcher(path, symbol, argtypes):
    launch = getattr(ctypes.CDLL(path), symbol)
    launch.argtypes = argtypes
    launch.restype = ctypes.c_int
    return launch


def residual(x, branch, scale):
    path = os.getenv("ESM_MUSA_RESIDUAL_LIBRARY")
    if (
        not path or torch.is_grad_enabled() or x.device.type != "musa"
        or x.device != branch.device or x.shape != branch.shape
        or x.dtype not in (torch.float32, torch.bfloat16)
        or branch.dtype not in (torch.float32, torch.bfloat16)
        or not x.is_contiguous() or not branch.is_contiguous()
        or not 0 < x.numel() < 2**31
    ):
        return None
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    launch = _launcher(path, "esm_residual",
                       (ptr, ptr, ptr, integer, integer, integer, ctypes.c_double, integer, ptr))
    out = torch.empty_like(x, dtype=torch.float32)
    status = launch(x.data_ptr(), branch.data_ptr(), out.data_ptr(),
                    x.dtype == torch.bfloat16, branch.dtype == torch.bfloat16,
                    x.numel(), scale, 0, torch.musa.current_stream(x.device).musa_stream)
    if status:
        raise RuntimeError(f"MUSA residual launch failed: {status}")
    return out


_tables = {}


def prepare_swiglu(device):
    """Prepare the BF16 lookup table before inference or graph capture."""
    device = torch.device(device)
    if device.type != "musa" or device.index is None:
        raise ValueError("Pass an explicit MUSA device, e.g. musa:0")
    if device not in _tables:
        with torch.inference_mode():
            bits = torch.arange(65536, dtype=torch.int64).to(torch.int16)
            table = F.silu(bits.view(torch.bfloat16).to(device)).contiguous()
            torch.musa.current_stream(device).synchronize()
        _tables[device] = table


def swiglu(x):
    path = os.getenv("ESM_MUSA_SWIGLU_LIBRARY")
    table = _tables.get(x.device)
    if (
        not path or table is None or torch.is_grad_enabled()
        or x.dtype != torch.bfloat16 or not x.is_contiguous()
        or x.ndim == 0 or x.shape[-1] % 2 or not 0 < x.numel() < 2**31
    ):
        return None
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    launch = _launcher(path, "esm_swiglu_lut_fast", (ptr, ptr, ptr, integer, integer, ptr))
    out = x.new_empty((*x.shape[:-1], x.shape[-1] // 2))
    status = launch(x.data_ptr(), table.data_ptr(), out.data_ptr(), out.numel(),
                    out.shape[-1], torch.musa.current_stream(x.device).musa_stream)
    if status:
        raise RuntimeError(f"MUSA SwiGLU launch failed: {status}")
    return out
