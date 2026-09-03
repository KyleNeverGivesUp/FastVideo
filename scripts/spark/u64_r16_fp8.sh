#!/bin/bash
# under-64 first measured row: r16 checkpoint, FP8 on vs off, 832x480x124, one GB10.
# Requires the fp8-verify branch (PR 1780) checked out in ~/fastvideo-lab/FastVideo.
set -uo pipefail
export HF_HOME=/home/kyle/fastvideo-lab/models
cd /home/kyle/fastvideo-lab/FastVideo
source /home/kyle/fastvideo-lab/.venv/bin/activate

echo "== branch =="; git log --oneline -1
echo "== disk ==";  df -h / | tail -1

# The executor spawns workers, and spawn re-imports the main script by path,
# so the runner must be a real file rather than stdin.
RUNNER=/tmp/u64_runner.py
cat > "$RUNNER" <<'PYEOF'
import importlib.util, sys, time
from pathlib import Path

def main() -> None:
    tag, quant = sys.argv[1], sys.argv[2]
    spec = importlib.util.spec_from_file_location(
        "fasth3", "examples/inference/basic/basic_fasth3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

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

    gen = mod.VideoGenerator.from_config(config)
    t0 = time.perf_counter()
    result = gen.generate(mod.build_request(
        args, Path(f"outputs/u64_{tag}/clip.mp4"), args.seed))
    wall = time.perf_counter() - t0
    peak = getattr(result, "peak_memory_mb", None)
    print(f"[{tag}] E2E {wall:.1f}s   peak_memory_mb = {peak}")
    try:
        gen.shutdown()
    except Exception:
        pass

if __name__ == "__main__":
    main()
PYEOF

nice -n 19 python "$RUNNER" bf16 none
nice -n 19 python "$RUNNER" fp8  fp8
echo "done"
