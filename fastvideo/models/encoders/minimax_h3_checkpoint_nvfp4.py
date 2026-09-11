# SPDX-License-Identifier: Apache-2.0
"""Serialized NVFP4 execution for the MiniMax-H3 Qwen3-VL encoder.

The bf16 conditioner is 48 GB resident and sets the single-GB10 peak once the
DiT runs in FP8. A checkpoint written by
``scripts/checkpoint_conversion/convert_minimax_h3_text_encoder_nvfp4.py``
stores every ``language_model.layers.*`` linear as three tensors and no
``weight``::

    <prefix>.weight_packed        uint8   [out, in // 2]   two E2M1 values per byte
    <prefix>.weight_scale         uint8   [out, in // 16]  one E4M3 scale per 16 values,
                                                           FlashInfer 128x4 swizzled layout
    <prefix>.weight_global_scale  float32 [1]              (448 * 6) / amax(|W|)

The bytes are exactly what ``flashinfer.nvfp4_quantize(W, global_scale,
sfLayout=SfLayout.layout_128x4)`` returns, so loading them reproduces the state
``nvfp4_config.convert_model_to_nvfp4`` builds at runtime without ever
materializing the bf16 weight. Activations are quantized per call with a unit
global scale and multiplied with ``flashinfer.mm_fp4``.

The checkpoint selects this path through ``config.json``::

    "quantization_config": {"quant_method": "nvfp4", "group_size": 16,
                            "scale_layout": "128x4", "activation_scheme": "dynamic",
                            "modules_to_not_convert": ["model.visual", "lm_head"]}

Whole projection kinds may stay bf16 by listing their suffix in
``modules_to_not_convert`` (for example ``"mlp.down_proj"``); those linears
keep a plain ``weight`` and the unquantized method.

Single GPU only: packed columns and swizzled scale rows cannot be narrowed per
tensor-parallel rank without repacking.
"""

from typing import Any

import torch
from torch import nn
from torch.nn.parameter import Parameter

from fastvideo.distributed import get_tp_world_size
from fastvideo.layers.linear import LinearBase, LinearMethodBase
from fastvideo.layers.quantization.base_config import QuantizationConfig
from fastvideo.layers.quantization.nvfp4_config import (
    _coerce_fp4_input_dtype,
    _mm_fp4,
    _nvfp4_quantize,
    _require_flashinfer,
)
from fastvideo.models.utils import set_weight_attrs

NVFP4_GROUP_SIZE = 16
NVFP4_SCALE_LAYOUT = "128x4"
# FlashInfer's 128x4 layout tiles scales in 128 rows by 4 scale columns. Keeping
# every weight a whole number of tiles means the serialized scale tensor is
# exactly [out, in // 16] with no padding to describe. Every Qwen3-VL language
# linear satisfies this (out in {1024, 5120, 8192, 25600}, in in {5120, 8192, 25600}).
NVFP4_ROW_TILE = 128
NVFP4_COLUMN_MULTIPLE = 4 * NVFP4_GROUP_SIZE
_E2M1_MAX = 6.0
_E4M3_MAX = 448.0
# An E4M3 byte of 0xFF is NaN, so a scale row that is still all 0xFF after
# loading is one the checkpoint never filled.
_UNLOADED_SCALE_BYTE = 0xFF


def validate_nvfp4_geometry(output_size: int, input_size: int) -> None:
    if output_size % NVFP4_ROW_TILE:
        raise ValueError(f"MiniMax-H3 serialized NVFP4 requires output_size divisible by {NVFP4_ROW_TILE}, "
                         f"got {output_size}")
    if input_size % NVFP4_COLUMN_MULTIPLE:
        raise ValueError(f"MiniMax-H3 serialized NVFP4 requires input_size divisible by {NVFP4_COLUMN_MULTIPLE}, "
                         f"got {input_size}")


def nvfp4_packed_weight_shape(output_size: int, input_size: int) -> tuple[int, int]:
    return output_size, input_size // 2


def nvfp4_scale_shape(output_size: int, input_size: int) -> tuple[int, int]:
    return output_size, input_size // NVFP4_GROUP_SIZE


def nvfp4_weight_global_scale(weight: torch.Tensor) -> torch.Tensor:
    """The per-tensor scale ``convert_model_to_nvfp4`` uses: E4M3 max times E2M1 max over amax."""
    amax = weight.float().abs().nan_to_num().max()
    if not torch.isfinite(amax) or amax <= 0:
        raise ValueError("MiniMax-H3 NVFP4 global scale needs a finite, non-zero weight amax")
    return ((_E4M3_MAX * _E2M1_MAX) / amax).to(torch.float32)


def serialized_nvfp4_quantization_config(
    *,
    keep_bf16: tuple[str, ...] | list[str] = (),
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``config.json`` ``quantization_config`` the converter writes and ``from_config`` accepts.

    ``keep_bf16`` lists projection suffixes such as ``mlp.down_proj`` that stay
    bf16 in every language layer; they are appended to ``modules_to_not_convert``.
    """
    config: dict[str, Any] = {
        "quant_method": "nvfp4",
        "activation_scheme": "dynamic",
        "fmt": "e2m1",
        "group_size": NVFP4_GROUP_SIZE,
        "scale_fmt": "e4m3",
        "scale_layout": NVFP4_SCALE_LAYOUT,
        "modules_to_not_convert": ["model.visual", "lm_head", *keep_bf16],
    }
    if producer:
        config["producer"] = dict(producer)
    return config


def _require_flashinfer_fp4() -> Any:
    """Resolve FlashInfer's FP4 entry points; raises with an install hint when absent."""
    return _require_flashinfer()


def _quantize_activation_nvfp4(x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one bf16 activation the way ``NVFP4QuantizeMethod`` does: unit global scale, 128x4 layout."""
    sf_layout, _, _ = _require_flashinfer_fp4()
    global_scale = torch.ones((), dtype=torch.float32, device=x_2d.device)
    return _nvfp4_quantize(x_2d, global_scale, sfLayout=sf_layout.layout_128x4, do_shuffle=False)


def _nvfp4_linear(
    x_fp4: torch.Tensor,
    x_scale: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """``x @ W.T`` on packed operands. Mirrors ``NVFP4QuantizeMethod.apply``: the weight is
    handed to ``mm_fp4`` transposed, the output is bf16, the backend is FlashInfer's choice."""
    return _mm_fp4(
        x_fp4,
        weight_packed.t(),
        x_scale,
        weight_scale.t(),
        alpha,
        torch.bfloat16,
        None,
        backend="auto",
    )


class MiniMaxH3SerializedNVFP4Config(QuantizationConfig):
    """Serialized 16-group NVFP4 contract for the H3 text encoder."""

    def __init__(self, group_size: int, scale_layout: str, bf16_suffixes: tuple[str, ...] = ()) -> None:
        super().__init__()
        if group_size != NVFP4_GROUP_SIZE:
            raise ValueError(f"MiniMax-H3 serialized NVFP4 requires group_size={NVFP4_GROUP_SIZE}, got {group_size}")
        if scale_layout != NVFP4_SCALE_LAYOUT:
            raise ValueError(f"MiniMax-H3 serialized NVFP4 requires scale_layout={NVFP4_SCALE_LAYOUT!r}, "
                             f"got {scale_layout!r}")
        self.group_size = group_size
        self.scale_layout = scale_layout
        # Projection suffixes the checkpoint kept in bf16 in every language
        # layer, e.g. ("mlp.down_proj",). Those linears load a plain weight.
        self.bf16_suffixes = tuple(bf16_suffixes)
        self.is_checkpoint_nvfp4_serialized = True
        self.activation_scheme = "dynamic"

    @classmethod
    def get_name(cls) -> str:
        return "nvfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MiniMaxH3SerializedNVFP4Config":
        quant_method = str(config.get("quant_method", "")).lower()
        if quant_method != "nvfp4":
            raise ValueError(f"MiniMax-H3 serialized NVFP4 config got quant_method {quant_method!r}")
        if str(config.get("activation_scheme", "dynamic")).lower() != "dynamic":
            raise ValueError("MiniMax-H3 serialized NVFP4 requires dynamic activation quantization")
        if str(config.get("fmt", "e2m1")).lower() not in ("e2m1", "float4_e2m1fn", "nvfp4"):
            raise ValueError(f"MiniMax-H3 serialized NVFP4 requires E2M1 weights, got {config.get('fmt')!r}")
        if str(config.get("scale_fmt", "e4m3")).lower() not in ("e4m3", "float8_e4m3fn"):
            raise ValueError(f"MiniMax-H3 serialized NVFP4 requires E4M3 block scales, got {config.get('scale_fmt')!r}")
        group_size = config.get("group_size", NVFP4_GROUP_SIZE)
        if not isinstance(group_size, int) or isinstance(group_size, bool):
            raise ValueError("MiniMax-H3 serialized NVFP4 group_size must be an integer")
        scale_layout = str(config.get("scale_layout", NVFP4_SCALE_LAYOUT))
        ignored_layers = config.get("modules_to_not_convert", config.get("ignored_layers", []))
        if not isinstance(ignored_layers, list | tuple):
            raise ValueError("MiniMax-H3 serialized NVFP4 modules_to_not_convert must be a sequence")
        if not any(isinstance(name, str) and "visual" in name for name in ignored_layers):
            raise ValueError("MiniMax-H3 serialized NVFP4 requires the vision stack to be listed in "
                             "modules_to_not_convert")
        bf16_suffixes: list[str] = []
        for name in ignored_layers:
            if not isinstance(name, str) or "visual" in name or name == "lm_head" or name.endswith(".lm_head"):
                continue
            # A whole projection kind may stay bf16 across every language layer
            # (``mlp.down_proj``); excluding individual layers is not a contract
            # this loader supports.
            if "layers." in name or name.startswith("language_model") or ".language_model." in name:
                raise ValueError("MiniMax-H3 serialized NVFP4 keeps bf16 per projection kind, not per layer; "
                                 f"got modules_to_not_convert entry {name!r}. Use a suffix such as "
                                 "'mlp.down_proj'.")
            bf16_suffixes.append(name)
        return cls(group_size, scale_layout, tuple(bf16_suffixes))

    def validate_runtime(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise RuntimeError(f"MiniMax-H3 serialized NVFP4 requires a CUDA device; got {device.type!r}")
        capability = torch.cuda.get_device_capability(device)
        capability_number = capability[0] * 10 + capability[1]
        if capability_number < self.get_min_capability():
            raise RuntimeError("MiniMax-H3 serialized NVFP4 requires GPU capability "
                               f"sm{self.get_min_capability()} or newer, got sm{capability_number}")
        if capability[0] not in (10, 12):
            raise RuntimeError("MiniMax-H3 serialized NVFP4 runs FlashInfer's Blackwell FP4 GEMM; "
                               f"got unsupported sm{capability_number}")
        _require_flashinfer_fp4()

    def is_kept_bf16(self, prefix: str) -> bool:
        return any(prefix == suffix or prefix.endswith("." + suffix) for suffix in self.bf16_suffixes)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if not isinstance(layer, LinearBase) or ".language_model.layers." not in prefix:
            return None
        if self.is_kept_bf16(prefix):
            return None
        return MiniMaxH3SerializedNVFP4LinearMethod(self.group_size)


class MiniMaxH3SerializedNVFP4LinearMethod(LinearMethodBase):
    """Execute serialized NVFP4 weights without re-quantizing them."""

    def __init__(self, group_size: int) -> None:
        super().__init__()
        self.group_size = group_size

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if get_tp_world_size() > 1:
            raise NotImplementedError("MiniMax-H3 serialized NVFP4 supports a single GPU: packed FP4 columns and "
                                      "128x4 swizzled scale rows cannot be narrowed per tensor-parallel rank")
        output_size_per_partition = sum(output_partition_sizes)
        validate_nvfp4_geometry(output_size_per_partition, input_size_per_partition)

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        weight_loader = extra_weight_attrs.get("weight_loader")
        weight_packed = Parameter(
            torch.zeros(nvfp4_packed_weight_shape(output_size_per_partition, input_size_per_partition),
                        dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(weight_packed, {"input_dim": 1, "output_dim": 0, "weight_loader": weight_loader})
        layer.register_parameter("weight_packed", weight_packed)

        weight_scale = Parameter(
            torch.full(nvfp4_scale_shape(output_size_per_partition, input_size_per_partition),
                       _UNLOADED_SCALE_BYTE,
                       dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(weight_scale, {"input_dim": 1, "output_dim": 0, "weight_loader": weight_loader})
        layer.register_parameter("weight_scale", weight_scale)

        weight_global_scale = Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)
        set_weight_attrs(weight_global_scale, {"weight_loader": weight_loader})
        layer.register_parameter("weight_global_scale", weight_global_scale)
        # No bf16 weight ever exists on this layer; ``None`` keeps ``layer.weight``
        # readable for code that inspects it, matching the purged NVFP4 path.
        layer.register_parameter("weight", None)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        weight_packed = getattr(layer, "weight_packed", None)
        weight_scale = getattr(layer, "weight_scale", None)
        weight_global_scale = getattr(layer, "weight_global_scale", None)
        if weight_packed is None or weight_scale is None or weight_global_scale is None:
            raise ValueError("Serialized MiniMax-H3 NVFP4 linear is missing weight_packed, weight_scale "
                             "or weight_global_scale")
        if weight_packed.dtype != torch.uint8 or weight_scale.dtype != torch.uint8:
            raise ValueError("Serialized MiniMax-H3 NVFP4 weight_packed and weight_scale must be uint8, got "
                             f"{weight_packed.dtype} and {weight_scale.dtype}")
        if weight_global_scale.dtype != torch.float32:
            raise ValueError("Serialized MiniMax-H3 NVFP4 weight_global_scale must be float32, "
                             f"got {weight_global_scale.dtype}")
        output_size = layer.output_size_per_partition
        input_size = layer.input_size_per_partition
        expected_packed = nvfp4_packed_weight_shape(output_size, input_size)
        expected_scale = nvfp4_scale_shape(output_size, input_size)
        if tuple(weight_packed.shape) != expected_packed:
            raise ValueError("Serialized MiniMax-H3 NVFP4 weight_packed shape mismatch: "
                             f"expected {expected_packed}, got {tuple(weight_packed.shape)}")
        if tuple(weight_scale.shape) != expected_scale:
            raise ValueError("Serialized MiniMax-H3 NVFP4 weight_scale shape mismatch: "
                             f"expected {expected_scale}, got {tuple(weight_scale.shape)}")
        if not bool(torch.isfinite(weight_global_scale).all()) or bool((weight_global_scale <= 0).any()):
            raise ValueError("Serialized MiniMax-H3 NVFP4 weight_global_scale was not loaded: "
                             "it must be a finite positive value")
        if bool((weight_scale == _UNLOADED_SCALE_BYTE).all()):
            raise ValueError("Serialized MiniMax-H3 NVFP4 weight_scale was not loaded: every byte is still 0xFF")
        layer.weight_packed.data = weight_packed.data
        layer.weight_scale.data = weight_scale.data
        layer.weight_global_scale.data = weight_global_scale.data
        # ``mm_fp4`` folds both global scales into one multiplier. Activations use a
        # unit global scale, so the multiplier is the inverse weight global scale.
        alpha = (1.0 / weight_global_scale.data.float()).reshape(())
        layer.register_buffer("_nvfp4_alpha", alpha, persistent=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.device.type != "cuda":
            raise RuntimeError("MiniMax-H3 serialized NVFP4 execution requires CUDA")
        capability = torch.cuda.get_device_capability(x.device)
        capability_number = capability[0] * 10 + capability[1]
        if capability_number < MiniMaxH3SerializedNVFP4Config.get_min_capability():
            raise RuntimeError("MiniMax-H3 serialized NVFP4 requires GPU capability "
                               f"sm{MiniMaxH3SerializedNVFP4Config.get_min_capability()} or newer, "
                               f"got sm{capability_number}")
        alpha = getattr(layer, "_nvfp4_alpha", None)
        if alpha is None:
            raise RuntimeError("MiniMax-H3 serialized NVFP4 linear was not finalized: "
                               "process_weights_after_loading has not run")

        x = _coerce_fp4_input_dtype(x)
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        x_fp4, x_scale = _quantize_activation_nvfp4(x_2d)
        output = _nvfp4_linear(x_fp4, x_scale, layer.weight_packed, layer.weight_scale, alpha)
        if bias is not None:
            output = output + bias
        return output.view(*original_shape[:-1], output.shape[-1])


__all__ = [
    "MiniMaxH3SerializedNVFP4Config",
    "MiniMaxH3SerializedNVFP4LinearMethod",
    "NVFP4_GROUP_SIZE",
    "NVFP4_SCALE_LAYOUT",
    "nvfp4_packed_weight_shape",
    "nvfp4_scale_shape",
    "nvfp4_weight_global_scale",
    "serialized_nvfp4_quantization_config",
    "validate_nvfp4_geometry",
]
