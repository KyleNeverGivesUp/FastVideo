#!/bin/bash
# under-64 first measured row: r16 checkpoint, FP8 on vs off, 832x480x124, one GB10.
# Run from inside ~/fastvideo-lab/FastVideo with the fp8-verify branch checked out.
set -uo pipefail
export HF_HOME=/home/kyle/fastvideo-lab/models
cd /home/kyle/fastvideo-lab/FastVideo
source /home/kyle/fastvideo-lab/.venv/bin/activate

MODEL="KyleNeverGivesUp/FastH3-4-step-Preview-v1-r16"
echo "== disk =="; df -h / | tail -1
echo "== cache check =="
python - <<PY
from huggingface_hub import snapshot_download
import os
try:
    p = snapshot_download("$MODEL", local_files_only=True)
    print("cached at:", p)
except Exception as e:
    print("NOT CACHED - would download ~114GB. Aborting is safer; rerun with ALLOW_DOWNLOAD=1 to fetch.")
    raise SystemExit(3)
PY
[ $? -eq 3 ] && [ "${ALLOW_DOWNLOAD:-0}" != "1" ] && exit 3

run_arm () {
  local tag="$1" quant="$2"
  echo "=================== ARM: $tag ==================="
  nice -n 19 python - "$tag" "$quant" <<'PY'
import importlib.util, sys, time
tag, quant = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("fasth3", "examples/inference/basic/basic_fasth3.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

args = mod.parse_args([
    "--prompt", "an alpine lake at sunrise, gentle wind over the water",
    "--model-path", "KyleNeverGivesUp/FastH3-4-step-Preview-v1-r16",
    "--height", "480", "--width", "832", "--num-frames", "124",
    "--num-gpus", "1", "--vsa-kernel", "triton", "--no-fa4",
    "--no-warmup", "--repeats", "1", "--seed", "2026",
    "--output", f"outputs/u64_{tag}",
])
config = mod.build_generator_config(args)
if quant == "fp8":
    from fastvideo.api.schema import QuantizationConfig
    config.engine.quantization = QuantizationConfig(transformer_quant="FP8")
print(f"[{tag}] quantization =", config.engine.quantization)

from fastvideo import VideoGenerator
gen = VideoGenerator.from_config(config)
t0 = time.perf_counter()
result = gen.generate(mod.build_request(args, __import__("pathlib").Path(f"outputs/u64_{tag}/clip.mp4"), args.seed))
wall = time.perf_counter() - t0
peak = getattr(result, "peak_memory_mb", None)
print(f"[{tag}] E2E {wall:.1f}s   peak_memory_mb = {peak}")
try:
    gen.shutdown()
except Exception:
    pass
PY
}

run_arm bf16 none
run_arm fp8  fp8
echo "done"
