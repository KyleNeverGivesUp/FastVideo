#!/bin/bash
#SBATCH --job-name=mxfp8-smoke
#SBATCH --output=/home/kyle/mxfp8-smoke-%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -uo pipefail
export HF_HOME=/home/kyle/fastvideo-lab/models
cd /home/kyle/fastvideo-lab/FastVideo
source .venv/bin/activate

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

# Quick timing vs bf16 at the same shape, n=50 after warmup.
import time
for _ in range(10): mod.mxfp8_scaled_mm(xq, xs, wq, ws, None)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(50): mod.mxfp8_scaled_mm(xq, xs, wq, ws, None)
torch.cuda.synchronize(); t_mx = (time.perf_counter() - t0) / 50
xb, wb = x, w
for _ in range(10): xb @ wb.T
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(50): xb @ wb.T
torch.cuda.synchronize(); t_bf = (time.perf_counter() - t0) / 50
print("gemm only: bf16 %.3f ms, mxfp8 %.3f ms, ratio %.2fx" % (t_bf*1e3, t_mx*1e3, t_bf/t_mx))
PY
