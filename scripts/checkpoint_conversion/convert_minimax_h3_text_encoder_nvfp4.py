# SPDX-License-Identifier: Apache-2.0
"""Serialize the MiniMax-H3 Qwen3-VL text encoder as NVFP4.

H3 conditions on hidden state 50 of a 64-layer Qwen3-VL and never predicts a
token, so the conditioner builds 50 decoder layers, no ``lm_head``, and reads
nothing above (#1711). Every linear in those layers is quantized with
``flashinfer.nvfp4_quantize`` exactly as ``convert_model_to_nvfp4`` would at
runtime and stored as ``weight_packed`` / ``weight_scale`` /
``weight_global_scale``; the byte layout is documented in
``fastvideo/models/encoders/minimax_h3_checkpoint_nvfp4.py``. Everything else
(token embedding, norms, the vision tower) is copied unchanged. ``config.json``
gains the ``quantization_config`` block that makes the loader select the
serialized NVFP4 path, so no inference flag is needed.

For the FastH3 conditioner: 50 layers x 487.6M values = 24.4B values, 12.2 GB
packed plus 1.5 GB of scales, next to about 3 GB of unquantized tensors,
against 48.8 GB of bf16 language layers (63 GB on disk with the 14 dropped
layers and ``lm_head``).

Needs a Blackwell GPU with FlashInfer: the quantizer is a CUDA kernel, and the
error report runs the same ``mm_fp4`` path the loader executes.

Usage::

    python scripts/checkpoint_conversion/convert_minimax_h3_text_encoder_nvfp4.py \
        --src /path/to/FastH3/text_encoder \
        --dst /path/to/FastH3-nvfp4/text_encoder

    # then assemble a model dir whose other components point at the original
    ln -s /path/to/FastH3/{transformer,tokenizer,processor,vae,audio_vae,\
scheduler,audio_scheduler,modular_model_index.json} /path/to/FastH3-nvfp4/

Use ``--report-only`` to quantize every language linear, print the relative
error of the FP4 GEMM against the bf16 product, and write nothing.

``--keep-bf16 mlp.down_proj`` leaves one projection kind in bf16 in every
layer and records it in ``modules_to_not_convert`` so the loader builds those
linears unquantized. The SwiGLU product feeding ``down_proj`` carries the
widest activation outliers in the stack, so it is the first candidate when
4-bit everywhere costs too much quality.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fastvideo.layers.quantization.nvfp4_config import _nvfp4_quantize, _require_flashinfer
from fastvideo.models.encoders.minimax_h3_checkpoint_nvfp4 import (
    _nvfp4_linear,
    _quantize_activation_nvfp4,
    nvfp4_packed_weight_shape,
    nvfp4_scale_shape,
    nvfp4_weight_global_scale,
    serialized_nvfp4_quantization_config,
    validate_nvfp4_geometry,
)

INDEX_NAME = "model.safetensors.index.json"
SINGLE_FILE_NAME = "model.safetensors"
# MINIMAX_H3_TEXT_ENCODER_LAYER in fastvideo/pipelines/basic/minimax_h3/packing.py:
# H3 reads hidden state 50 and nothing above it.
DEFAULT_NUM_LAYERS = 50
LANGUAGE_LINEAR = re.compile(r"^model\.language_model\.layers\.(?P<layer>\d+)\."
                             r"(?P<proj>self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)\.weight$")
LANGUAGE_LAYER = re.compile(r"^model\.language_model\.layers\.(?P<layer>\d+)\.")
PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
               "mlp.up_proj", "mlp.down_proj")

# Keys the converter drops on purpose, with the reason each.
SKIPPED_KEYS = {
    "lm_head.weight": "the conditioner never predicts tokens; H3 reads a hidden state",
    "model.language_model.layers.{N >= --num-layers}.*": "layers above the one H3 reads are never built (#1711)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="text_encoder directory of the bf16 checkpoint")
    parser.add_argument("--dst", type=Path, help="text_encoder directory to write (required unless --report-only)")
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS,
                        help="keep language layers below this index (default: %(default)s, the layer H3 reads)")
    parser.add_argument("--device", default="cuda", help="CUDA device that runs the quantizer")
    parser.add_argument("--shard-size-gb", type=float, default=4.0, help="safetensors shard size")
    parser.add_argument("--probe-rows", type=int, default=512,
                        help="rows of random activations per linear for the error report, 0 disables it")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-only", action="store_true", help="quantize and report errors, write nothing")
    parser.add_argument("--keep-bf16", action="append", default=[], metavar="PROJ",
                        help="projection kind to leave in bf16 in every layer, e.g. mlp.down_proj; "
                        "repeat or comma-separate for several")
    args = parser.parse_args()
    if not args.report_only and args.dst is None:
        parser.error("--dst is required unless --report-only is given")
    keep = [name.strip() for entry in args.keep_bf16 for name in entry.split(",") if name.strip()]
    unknown = sorted(set(keep) - set(PROJECTIONS))
    if unknown:
        parser.error(f"--keep-bf16 got unknown projection(s) {unknown}; choose from {list(PROJECTIONS)}")
    args.keep_bf16 = tuple(dict.fromkeys(keep))
    return args


def source_shards(src: Path) -> list[Path]:
    index = src / INDEX_NAME
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return [src / name for name in sorted(set(weight_map.values()))]
    single = src / SINGLE_FILE_NAME
    if single.is_file():
        return [single]
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors weights under {src}")
    return shards


class ShardWriter:
    """Accumulate tensors and flush them as fixed-size safetensors shards plus an index."""

    def __init__(self, dst: Path | None, shard_bytes: int) -> None:
        self.dst = dst
        self.shard_bytes = shard_bytes
        self.pending: dict[str, torch.Tensor] = {}
        self.pending_bytes = 0
        self.weight_map: dict[str, str] = {}
        self.total_bytes = 0
        self.shard_index = 0

    def add(self, name: str, tensor: torch.Tensor) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        if self.pending and self.pending_bytes + nbytes > self.shard_bytes:
            self.flush()
        self.pending[name] = tensor
        self.pending_bytes += nbytes
        self.total_bytes += nbytes

    def flush(self) -> None:
        if not self.pending:
            return
        self.shard_index += 1
        filename = f"model-nvfp4-{self.shard_index:05d}.safetensors"
        if self.dst is not None:
            save_file(self.pending, str(self.dst / filename), metadata={"format": "pt"})
        for name in self.pending:
            self.weight_map[name] = filename
        self.pending = {}
        self.pending_bytes = 0

    def finish(self) -> None:
        self.flush()
        if self.dst is not None:
            index = {"metadata": {"total_size": self.total_bytes}, "weight_map": self.weight_map}
            (self.dst / INDEX_NAME).write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


def probe_relative_error(
    weight: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    rows: int,
    generator: torch.Generator,
) -> float:
    """||fp4(x) @ fp4(W).T - x @ W.T|| / ||x @ W.T|| on random bf16 rows, through the loader's own GEMM path."""
    x = torch.randn(rows, weight.shape[1], generator=generator, device=weight.device,
                    dtype=torch.float32).to(torch.bfloat16)
    reference = x.float() @ weight.float().t()
    x_fp4, x_scale = _quantize_activation_nvfp4(x)
    alpha = (1.0 / global_scale).reshape(())
    output = _nvfp4_linear(x_fp4, x_scale, packed, scale, alpha)
    return ((output.float() - reference).norm() / reference.norm().clamp_min(1e-12)).item()


def quantize_language_linear(
    weight: torch.Tensor,
    device: torch.device,
    sf_layout: object,
    probe_rows: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float | None]:
    output_size, input_size = weight.shape
    validate_nvfp4_geometry(output_size, input_size)
    weight_device = weight.to(device=device, dtype=torch.bfloat16)
    global_scale = nvfp4_weight_global_scale(weight_device)
    packed, scale = _nvfp4_quantize(weight_device, global_scale, sfLayout=sf_layout.layout_128x4, do_shuffle=False)
    if packed.dtype != torch.uint8:
        packed = packed.view(torch.uint8)
    if scale.dtype != torch.uint8:
        scale = scale.view(torch.uint8)
    expected_packed = nvfp4_packed_weight_shape(output_size, input_size)
    expected_scale = nvfp4_scale_shape(output_size, input_size)
    if tuple(packed.shape) != expected_packed or tuple(scale.shape) != expected_scale:
        raise RuntimeError("flashinfer.nvfp4_quantize returned an unexpected layout for a "
                           f"{output_size}x{input_size} weight: packed {tuple(packed.shape)} "
                           f"(expected {expected_packed}), scale {tuple(scale.shape)} (expected {expected_scale}). "
                           "The loader allocates exactly these shapes; check the FlashInfer version.")
    error = None
    if probe_rows and generator is not None:
        error = probe_relative_error(weight_device, packed, scale, global_scale, probe_rows, generator)
    return packed.cpu().contiguous(), scale.cpu().contiguous(), global_scale.reshape(1).cpu(), error


def main() -> None:
    args = parse_args()
    src: Path = args.src
    if not (src / "config.json").is_file():
        raise FileNotFoundError(f"{src} has no config.json; point --src at the text_encoder directory")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("The NVFP4 quantizer is a CUDA kernel; pass --device cuda")
    sf_layout, _, _ = _require_flashinfer()
    try:
        import flashinfer
        flashinfer_version = str(getattr(flashinfer, "__version__", "unknown"))
    except ImportError:
        flashinfer_version = "unknown"

    dst: Path | None = None
    if not args.report_only:
        dst = args.dst
        dst.mkdir(parents=True, exist_ok=True)
        if any(dst.glob("*.safetensors")):
            raise SystemExit(f"{dst} already holds safetensors shards; refusing to mix outputs")
    writer = ShardWriter(dst, int(args.shard_size_gb * (1 << 30)))
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.probe_rows else None

    quantized = 0
    kept_bf16 = 0
    kept_bf16_bytes = 0
    copied = 0
    skipped: dict[str, list[str]] = defaultdict(list)
    source_language_bytes = 0
    written_language_bytes = 0
    errors: dict[str, list[float]] = defaultdict(list)
    started = time.perf_counter()

    for shard in source_shards(src):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in sorted(handle.keys()):
                layer_match = LANGUAGE_LAYER.match(key)
                if key == "lm_head.weight":
                    skipped["lm_head.weight"].append(key)
                    continue
                if layer_match is not None and int(layer_match["layer"]) >= args.num_layers:
                    skipped["model.language_model.layers.{N >= --num-layers}.*"].append(key)
                    continue
                tensor = handle.get_tensor(key)
                linear_match = LANGUAGE_LINEAR.match(key)
                if linear_match is None:
                    writer.add(key, tensor.contiguous())
                    copied += 1
                    continue
                if linear_match["proj"] in args.keep_bf16:
                    writer.add(key, tensor.contiguous())
                    kept_bf16 += 1
                    kept_bf16_bytes += tensor.numel() * tensor.element_size()
                    continue
                packed, scale, global_scale, error = quantize_language_linear(
                    tensor, device, sf_layout, args.probe_rows, generator)
                prefix = key[:-len(".weight")]
                writer.add(prefix + ".weight_packed", packed)
                writer.add(prefix + ".weight_scale", scale)
                writer.add(prefix + ".weight_global_scale", global_scale)
                source_language_bytes += tensor.numel() * tensor.element_size()
                written_language_bytes += packed.numel() + scale.numel() + 4
                quantized += 1
                if error is not None:
                    errors[linear_match["proj"]].append(error)
                if quantized % 35 == 0:
                    print(f"  quantized {quantized} language linears "
                          f"({time.perf_counter() - started:.0f}s)", flush=True)
    writer.finish()

    if dst is not None:
        config = json.loads((src / "config.json").read_text(encoding="utf-8"))
        config["quantization_config"] = serialized_nvfp4_quantization_config(
            keep_bf16=args.keep_bf16,
            producer={
                "converter": Path(__file__).name,
                "flashinfer": flashinfer_version,
                "kept_language_layers": args.num_layers,
            })
        (dst / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        for extra in src.iterdir():
            if extra.is_file() and extra.name not in {INDEX_NAME, "config.json"} \
                    and not extra.name.endswith(".safetensors"):
                shutil.copy2(extra, dst / extra.name)

    print(f"language linears quantized: {quantized}")
    if args.keep_bf16:
        print(f"language linears kept bf16: {kept_bf16} ({', '.join(args.keep_bf16)}; "
              f"{kept_bf16_bytes / 1e9:.2f} GB)")
    print(f"tensors copied unchanged:   {copied}")
    for reason, keys in skipped.items():
        print(f"skipped {len(keys):>4} keys: {reason}: {SKIPPED_KEYS[reason]}")
    print(f"language linear bytes: {source_language_bytes / 1e9:.2f} GB bf16 -> "
          f"{written_language_bytes / 1e9:.2f} GB NVFP4 (packed values + E4M3 scales)")
    print(f"artifact total: {writer.total_bytes / 1e9:.2f} GB in {writer.shard_index} shard(s)"
          f"{'' if dst is not None else ' (not written, --report-only)'}")
    if errors:
        print(f"FP4 GEMM relative error vs bf16, {args.probe_rows} random rows per linear:")
        print(f"  {'projection':<20}{'linears':>8}{'max':>12}{'mean':>12}")
        for proj in sorted(errors):
            values = errors[proj]
            print(f"  {proj:<20}{len(values):>8}{max(values):>12.3e}{sum(values) / len(values):>12.3e}")
    print(f"done in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
