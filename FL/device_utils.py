"""
Cross-platform device selection for Federated Learning.

Supports four backends:
    - cuda : NVIDIA GPU
    - dml  : DirectML (AMD / Intel Arc on Windows)
    - mps  : Apple Silicon (M1/M2/M3/M4)
    - cpu  : Fallback

Usage in your FL code:
    from device_utils import get_device, device_name

    device = get_device()               # auto-detect best available
    print(f"Training on: {device_name(device)}")

    model = model.to(device)
    x, y = x.to(device), y.to(device)

Forcing a specific backend (for testing or per-teammate config):
    # Option 1: Python argument
    device = get_device(prefer="cpu")

    # Option 2: Environment variable (recommended for teammates)
    # On Windows PowerShell:  $env:FL_DEVICE = "cuda"
    # On Mac/Linux bash:      export FL_DEVICE=mps
    device = get_device()  # picks up FL_DEVICE automatically

Priority when auto-detecting: cuda > dml > mps > cpu
"""

import os
from typing import Optional, List


# ------------------------------------------------------------
# Backend availability checks
# ------------------------------------------------------------

def _has_cuda() -> bool:
    """NVIDIA CUDA available?"""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _has_directml() -> bool:
    """DirectML (AMD / Intel Arc on Windows) available?"""
    try:
        import torch_directml
        return torch_directml.device_count() > 0
    except Exception:
        return False


def _has_mps() -> bool:
    """Apple Silicon MPS available?"""
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


def list_available_backends() -> List[str]:
    """Return list of backends actually available on this machine, in priority order."""
    backends = []
    if _has_cuda():
        backends.append("cuda")
    if _has_directml():
        backends.append("dml")
    if _has_mps():
        backends.append("mps")
    backends.append("cpu")  # always available
    return backends


# ------------------------------------------------------------
# Main API
# ------------------------------------------------------------

def get_device(prefer: Optional[str] = None):
    """
    Resolve to a torch device object.

    Args:
        prefer: One of "auto", "cuda", "dml", "mps", "cpu", or None.
                If None, reads the FL_DEVICE environment variable.
                If that is also unset, falls back to "auto".

    Returns:
        torch.device for cuda/mps/cpu.
        torch.device(type="privateuseone") for DirectML.
        (Both work identically with .to(device) calls.)

    Raises:
        RuntimeError if a specific backend is requested but unavailable.
        ValueError on unknown preference string.
    """
    import torch

    # Resolve preference order: arg > env var > auto
    if prefer is None:
        prefer = os.environ.get("FL_DEVICE", "auto")
    prefer = prefer.lower().strip()

    # Explicit backends
    if prefer == "cpu":
        return torch.device("cpu")

    if prefer == "cuda":
        if not _has_cuda():
            raise RuntimeError(
                "CUDA was requested but is not available. "
                "Check NVIDIA driver and that you ran install_nvidia.bat."
            )
        return torch.device("cuda")

    if prefer in ("dml", "directml"):
        if not _has_directml():
            raise RuntimeError(
                "DirectML was requested but is not available. "
                "Run install_amd.bat to install torch-directml."
            )
        import torch_directml
        return torch_directml.device()

    if prefer == "mps":
        if not _has_mps():
            raise RuntimeError(
                "MPS was requested but is not available. "
                "MPS requires Apple Silicon + macOS 12.3+."
            )
        return torch.device("mps")

    # Auto-detect: cuda > dml > mps > cpu
    if prefer in ("auto", ""):
        if _has_cuda():
            return torch.device("cuda")
        if _has_directml():
            import torch_directml
            return torch_directml.device()
        if _has_mps():
            return torch.device("mps")
        return torch.device("cpu")

    raise ValueError(
        f"Unknown device preference: '{prefer}'. "
        f"Use one of: auto, cuda, dml, mps, cpu."
    )


def device_name(device) -> str:
    """Return a human-readable name for a device object (for logging)."""
    import torch

    if not isinstance(device, torch.device):
        return str(device)

    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        return f"CUDA: {torch.cuda.get_device_name(idx)}"

    if device.type == "mps":
        return "MPS (Apple Silicon)"

    if device.type == "cpu":
        return "CPU"

    # DirectML devices report type "privateuseone"
    if device.type == "privateuseone":
        try:
            import torch_directml
            return f"DirectML: {torch_directml.device_name(0)}"
        except Exception:
            return "DirectML"

    return str(device)


def sync(device) -> None:
    """
    Force all pending operations on the device to complete.
    Used for accurate timing. No-op for CPU.
    """
    import torch
    if not isinstance(device, torch.device):
        return
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    # CPU and DirectML: implicit sync on tensor read (.item(), .cpu(), etc.)
