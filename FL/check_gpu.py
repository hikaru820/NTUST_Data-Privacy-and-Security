"""
GPU detection and validation script.

Run after installing dependencies to verify the environment:
    python check_gpu.py

What it does:
    1. Prints Python / torch / torch-directml versions
    2. Lists every available backend (CUDA / DirectML / MPS / CPU)
    3. Picks the auto-selected device via device_utils.get_device()
    4. Runs a 2048x2048 matmul on CPU vs the selected device
       to confirm the GPU actually works AND is faster than CPU.

Exit codes:
    0 = success
    1 = torch not installed
    2 = selected device failed the matmul test
"""

import sys
import time


# ============================================================
# Section 1: Version info
# ============================================================
print("=" * 60)
print("  FL Environment GPU Check")
print("=" * 60)
print(f"\nPython:   {sys.version.split()[0]}")
print(f"Platform: {sys.platform}")

try:
    import torch
    print(f"torch:    {torch.__version__}")
except ImportError:
    print("\n[ERROR] torch is not installed.")
    print("Run install_amd.bat / install_nvidia.bat / install_cpu.bat / install_mac.command first.")
    sys.exit(1)

# torch-directml is optional (AMD-only)
try:
    import torch_directml
    dml_ver = getattr(torch_directml, "__version__", "installed")
    print(f"torch-directml: {dml_ver}")
    HAS_DML = True
except ImportError:
    HAS_DML = False

from device_utils import (
    get_device,
    device_name,
    list_available_backends,
    sync,
)


# ============================================================
# Section 2: Enumerate backends
# ============================================================
print("\n--- Available backends (priority order) ---")
for b in list_available_backends():
    print(f"  - {b}")

print("\n--- CUDA ---")
if torch.cuda.is_available():
    print(f"  Available:    YES")
    print(f"  Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}:     {torch.cuda.get_device_name(i)}")
else:
    print("  Available:    no")

print("\n--- DirectML ---")
if HAS_DML:
    count = torch_directml.device_count()
    print(f"  Available:    YES")
    print(f"  Device count: {count}")
    for i in range(count):
        print(f"  Device {i}:     {torch_directml.device_name(i)}")
else:
    print("  Available:    no (torch-directml not installed)")

print("\n--- MPS (Apple Silicon) ---")
if torch.backends.mps.is_available():
    print("  Available:    YES")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_built():
    print("  Available:    no (built but unavailable — check macOS version)")
else:
    print("  Available:    no")


# ============================================================
# Section 3: Auto-selected device
# ============================================================
print("\n" + "=" * 60)
print("  Auto-selected device")
print("=" * 60)

device = get_device()
print(f"\nDevice:       {device_name(device)}")
print(f"torch object: {device}")


# ============================================================
# Section 4: Speed test
# ============================================================
print("\n--- Matmul speed test (2048x2048, 10 iterations) ---")

ITERATIONS = 10
MATRIX_SIZE = 2048


def benchmark(dev, label: str):
    """Run matmul on device, return ms/iter or None on failure."""
    try:
        a = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=dev)
        b = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=dev)
    except Exception as e:
        print(f"  {label:35s} FAILED to allocate ({type(e).__name__}: {e})")
        return None

    try:
        # Warmup (first kernel launch is always slow)
        for _ in range(3):
            c = a @ b
        sync(dev)

        start = time.time()
        for _ in range(ITERATIONS):
            c = a @ b
        sync(dev)
        # Force sync via host read (works for all backends including DML)
        _ = c.sum().item()
        elapsed = (time.time() - start) * 1000 / ITERATIONS
        print(f"  {label:35s} {elapsed:8.2f} ms/iter")
        return elapsed
    except Exception as e:
        print(f"  {label:35s} FAILED during compute ({type(e).__name__}: {e})")
        return None


cpu_time = benchmark(torch.device("cpu"), "CPU (baseline)")
sel_time = benchmark(device, f"Selected: {device_name(device)}")


# ============================================================
# Section 5: Verdict
# ============================================================
print("\n" + "=" * 60)
if sel_time is None:
    print("  [FAIL] Selected device could not run matmul.")
    print("=" * 60)
    sys.exit(2)

if device.type == "cpu":
    print("  [OK] CPU works. No GPU acceleration available.")
elif cpu_time is None:
    print(f"  [OK] {device_name(device)} works.")
else:
    speedup = cpu_time / sel_time
    print(f"  [OK] {device_name(device)} works.")
    print(f"       Speedup vs CPU: {speedup:.1f}x")
    if speedup < 1.0:
        print("       WARNING: GPU slower than CPU — driver or backend may be misconfigured.")

print("=" * 60)
print("\nNext step: build the Flower FL server + clients.")
