#!/bin/bash
#SBATCH --job-name=mxfp8-smoke
#SBATCH --output=/home/kyle/mxfp8-smoke-%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -uo pipefail
export HF_HOME=/home/kyle/fastvideo-lab/models
cd /home/kyle/fastvideo-lab/FastVideo
source /home/kyle/fastvideo-lab/.venv/bin/activate

# Pull just the one file from PR 1796. It imports only torch, triton and
# quack, so nothing else from that branch is needed and the checkout is
# left untouched.
git fetch https://github.com/hao-ai-lab/FastVideo.git 'pull/1796/head' 2>&1 | tail -1
git show FETCH_HEAD:fastvideo/layers/mxfp8linear.py > /tmp/mxfp8linear_1796.py

nice -n 19 python - <<'PY'
import sys, traceback, importlib.util
import torch

print("torch", torch.__version__)
print("device", torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
print("capability sm_%d%d" % cap)

try:
    import quack
    print("quack ok:", quack.__name__)
except Exception as e:
    print("VERDICT: DEPENDENCY MISSING - quack-kernels not installed:", e)
    sys.exit(0)

spec = importlib.util.spec_from_file_location("mxfp8linear_1796", "/tmp/mxfp8linear_1796.py")
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception:
    print("VERDICT: IMPORT FAILED")
    traceback.print_exc()
    sys.exit(0)

torch.manual_seed(0)
dev = "cuda"
# H3 FFN-shaped, small M. Multiples of 32 and 128 to satisfy the block layout.
M, K, N = 512, 5376, 10752
x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
w = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.02
ref = (x.float() @ w.float().T)

try:
    wq, ws = mod.quantize_mxfp8_weight_blockwise(w)
    xq, xs = mod.quantize_mxfp8_blockwise(x)
    print("quantize ok:", tuple(wq.shape), wq.dtype, "|", tuple(xq.shape), xq.dtype)
except Exception:
    print("VERDICT: QUANTIZE FAILED (triton side)")
    traceback.print_exc()
    sys.exit(0)

try:
    out = mod.mxfp8_scaled_mm(xq, xs, wq, ws, None)
    torch.cuda.synchronize()
except Exception:
    print("VERDICT: NO MX GEMM KERNEL ON THIS ARCH - F.scaled_mm BlockWise1x32 failed")
    traceback.print_exc()
    sys.exit(0)

err = (out.float() - ref).abs().mean() / ref.abs().mean()
print("output", tuple(out.shape), out.dtype, "mean rel err %.4f" % err.item())
if err.item() < 0.05:
    print("VERDICT: PASS - MXFP8 scaled_mm runs on sm_%d%d and is numerically sane" % cap)
else:
    print("VERDICT: RUNS BUT WRONG - kernel executed, rel err %.4f is too high" % err.item())

# M sweep: H3's real FFN sees tens to hundreds of thousands of rows per
# forward, so a single small M says nothing about the workload. Timing is
# GEMM only; quantization cost excluded on both sides.
import time
def bench(fn, n=30, warm=8):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n
print("\nM sweep, K=%d N=%d, GEMM only:" % (K, N))
print("%10s %12s %12s %8s" % ("M", "bf16 ms", "mxfp8 ms", "ratio"))
for M2 in (512, 4096, 32768, 131072, 262144):
    x2 = torch.randn(M2, K, dtype=torch.bfloat16, device=dev)
    x2q, x2s = mod.quantize_mxfp8_blockwise(x2)
    t_bf = bench(lambda: x2 @ w.T)
    t_mx = bench(lambda: mod.mxfp8_scaled_mm(x2q, x2s, wq, ws, None))
    tf = 2*M2*K*N/t_bf/1e12
    print("%10d %9.3f (%3.0fTF) %9.3f %7.2fx" % (M2, t_bf*1e3, tf, t_mx*1e3, t_bf/t_mx))
    del x2, x2q, x2s
    torch.cuda.empty_cache()
PY
