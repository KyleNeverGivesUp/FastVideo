# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os

import pytest
import torch

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29515")

import fastvideo.models.encoders.minimax_h3_checkpoint_nvfp4 as h3_nvfp4
from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConfig
from fastvideo.layers.linear import ColumnParallelLinear, UnquantizedLinearMethod
from fastvideo.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod, VocabParallelEmbedding
from fastvideo.models.encoders.base import TextEncoder
from fastvideo.models.encoders.minimax_h3_checkpoint_fp8 import MiniMaxH3SerializedFP8Config
from fastvideo.models.encoders.minimax_h3_checkpoint_nvfp4 import (
    MiniMaxH3SerializedNVFP4Config,
    MiniMaxH3SerializedNVFP4LinearMethod,
    serialized_nvfp4_quantization_config,
)
from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConditioner
from fastvideo.models.loader.text_encoder_quantization import (
    _configure_text_encoder_quantization,
    _process_quantized_text_encoder_weights,
    _read_text_encoder_checkpoint_quantization_config,
)

LANGUAGE_PREFIX = "minimax_h3_qwen3_vl.language_model.layers.0.self_attn.q_proj"


def _checkpoint_quantization_config(**overrides) -> dict:
    config = serialized_nvfp4_quantization_config()
    config.update(overrides)
    return config


def _language_linear(config: MiniMaxH3SerializedNVFP4Config, input_size: int = 128,
                     output_size: int = 256) -> ColumnParallelLinear:
    return ColumnParallelLinear(
        input_size=input_size,
        output_size=output_size,
        bias=False,
        quant_config=config,
        prefix=LANGUAGE_PREFIX,
    )


def _fill_loaded(layer: ColumnParallelLinear, global_scale: float = 2.0) -> None:
    layer.weight_packed.data.fill_(0x11)
    layer.weight_scale.data.fill_(0x38)  # E4M3 1.0
    layer.weight_global_scale.data.fill_(global_scale)


def test_h3_accepts_only_the_serialized_nvfp4_contract() -> None:
    config = MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config())
    assert config.group_size == 16
    assert config.scale_layout == "128x4"
    assert config.get_name() == "nvfp4"
    assert config.get_supported_act_dtypes() == [torch.bfloat16]

    with pytest.raises(ValueError, match="group_size=16"):
        MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config(group_size=32))
    with pytest.raises(ValueError, match="scale_layout='128x4'"):
        MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config(scale_layout="linear"))
    with pytest.raises(ValueError, match="dynamic activation"):
        MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config(activation_scheme="static"))
    with pytest.raises(ValueError, match="E2M1"):
        MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config(fmt="e4m3"))
    with pytest.raises(ValueError, match="vision stack"):
        MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config(modules_to_not_convert=["lm_head"]))
    with pytest.raises(ValueError, match="per projection kind, not per layer"):
        MiniMaxH3SerializedNVFP4Config.from_config(
            _checkpoint_quantization_config(modules_to_not_convert=["model.visual", "language_model.layers.3"]))
    kept = MiniMaxH3SerializedNVFP4Config.from_config(
        _checkpoint_quantization_config(modules_to_not_convert=["model.visual", "lm_head", "mlp.down_proj"]))
    assert kept.bf16_suffixes == ("mlp.down_proj", )
    assert config.bf16_suffixes == ()
    with pytest.raises(ValueError, match="quant_method 'fp8'"):
        MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config(quant_method="fp8"))


def test_converter_metadata_round_trips_and_tolerates_producer_notes() -> None:
    metadata = serialized_nvfp4_quantization_config(producer={"converter": "x.py", "flashinfer": "0.6.13rc2"})
    config = MiniMaxH3SerializedNVFP4Config.from_config(metadata)
    assert config.group_size == 16
    assert metadata["producer"]["flashinfer"] == "0.6.13rc2"

    mixed = serialized_nvfp4_quantization_config(keep_bf16=("mlp.down_proj", ))
    assert mixed["modules_to_not_convert"] == ["model.visual", "lm_head", "mlp.down_proj"]
    assert MiniMaxH3SerializedNVFP4Config.from_config(mixed).bf16_suffixes == ("mlp.down_proj", )


def test_kept_bf16_projection_builds_a_plain_linear(distributed_setup) -> None:
    config = MiniMaxH3SerializedNVFP4Config.from_config(
        _checkpoint_quantization_config(modules_to_not_convert=["model.visual", "lm_head", "mlp.down_proj"]))
    down = ColumnParallelLinear(
        input_size=128,
        output_size=128,
        bias=False,
        quant_config=config,
        prefix="minimax_h3_qwen3_vl.language_model.layers.0.mlp.down_proj",
    )
    up = ColumnParallelLinear(
        input_size=128,
        output_size=128,
        bias=False,
        quant_config=config,
        prefix="minimax_h3_qwen3_vl.language_model.layers.0.mlp.up_proj",
    )

    assert isinstance(down.quant_method, UnquantizedLinearMethod)
    assert down.weight is not None and down.weight.shape == (128, 128)
    assert not hasattr(down, "weight_packed")
    assert isinstance(up.quant_method, MiniMaxH3SerializedNVFP4LinearMethod)
    assert up.weight is None


def test_serialized_nvfp4_allocates_packed_weight_and_scales_without_a_bf16_weight(distributed_setup) -> None:
    config = MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config())
    layer = _language_linear(config)

    assert isinstance(layer.quant_method, MiniMaxH3SerializedNVFP4LinearMethod)
    assert layer.weight is None
    assert layer.weight_packed.dtype == torch.uint8
    assert layer.weight_packed.shape == (256, 64)
    assert layer.weight_scale.dtype == torch.uint8
    assert layer.weight_scale.shape == (256, 8)
    assert layer.weight_global_scale.dtype == torch.float32
    assert layer.weight_global_scale.shape == (1, )
    # Bound methods are fresh objects on every attribute access, so compare by equality.
    assert getattr(layer.weight_packed, "weight_loader", None) == layer.weight_loader
    assert getattr(layer.weight_scale, "weight_loader", None) == layer.weight_loader
    assert getattr(layer.weight_global_scale, "weight_loader", None) == layer.weight_loader

    _fill_loaded(layer, global_scale=4.0)
    packed_pointer = layer.weight_packed.data_ptr()
    scale_pointer = layer.weight_scale.data_ptr()
    layer.quant_method.process_weights_after_loading(layer)

    assert layer.weight_packed.data_ptr() == packed_pointer
    assert layer.weight_scale.data_ptr() == scale_pointer
    assert layer.weight is None
    assert layer._nvfp4_alpha.dtype == torch.float32
    assert layer._nvfp4_alpha.item() == pytest.approx(0.25)


def test_serialized_nvfp4_finalization_rejects_tensors_the_checkpoint_never_filled(distributed_setup) -> None:
    config = MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config())
    layer = _language_linear(config)

    with pytest.raises(ValueError, match="weight_global_scale was not loaded"):
        layer.quant_method.process_weights_after_loading(layer)

    layer.weight_global_scale.data.fill_(2.0)
    with pytest.raises(ValueError, match="weight_scale was not loaded"):
        layer.quant_method.process_weights_after_loading(layer)

    layer.weight_scale.data.fill_(0x38)
    layer.quant_method.process_weights_after_loading(layer)
    assert layer._nvfp4_alpha.item() == pytest.approx(0.5)


def test_serialized_nvfp4_quantizes_only_language_linears(distributed_setup) -> None:
    config = MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config())
    visual_linear = ColumnParallelLinear(
        input_size=128,
        output_size=128,
        bias=False,
        quant_config=config,
        prefix="minimax_h3_qwen3_vl.visual.blocks.0.attn.proj",
    )
    embedding = VocabParallelEmbedding(
        num_embeddings=128,
        embedding_dim=128,
        org_num_embeddings=128,
        quant_config=config,
        prefix="minimax_h3_qwen3_vl.language_model.embed_tokens",
    )

    assert isinstance(visual_linear.quant_method, UnquantizedLinearMethod)
    assert visual_linear.weight.dtype == torch.get_default_dtype()
    assert isinstance(embedding.quant_method, UnquantizedEmbeddingMethod)
    assert embedding.weight.dtype == torch.get_default_dtype()


def test_serialized_nvfp4_rejects_geometry_outside_flashinfer_tiles(distributed_setup) -> None:
    config = MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config())
    with pytest.raises(ValueError, match="input_size divisible by 64"):
        _language_linear(config, input_size=96, output_size=256)
    with pytest.raises(ValueError, match="output_size divisible by 128"):
        _language_linear(config, input_size=128, output_size=200)


def test_serialized_nvfp4_refuses_tensor_parallel(distributed_setup, monkeypatch) -> None:
    config = MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config())
    monkeypatch.setattr(h3_nvfp4, "get_tp_world_size", lambda: 2)
    with pytest.raises(NotImplementedError, match="single GPU"):
        _language_linear(config)


def test_serialized_nvfp4_cpu_execution_fails_closed(distributed_setup) -> None:
    config = MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config())
    layer = _language_linear(config, input_size=128, output_size=128)
    _fill_loaded(layer)
    layer.quant_method.process_weights_after_loading(layer)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        layer(torch.zeros(2, 128, dtype=torch.bfloat16))


def test_runtime_preflight_reports_capability_and_missing_flashinfer(monkeypatch) -> None:
    config = MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config())
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        config.validate_runtime(torch.device("cpu"))

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    with pytest.raises(RuntimeError, match="sm100 or newer"):
        config.validate_runtime(torch.device("cuda"))

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 1))

    def missing_flashinfer():
        raise ImportError("NVFP4 quantization requires flashinfer")

    monkeypatch.setattr(h3_nvfp4, "_require_flashinfer_fp4", missing_flashinfer)
    with pytest.raises(ImportError, match="requires flashinfer"):
        config.validate_runtime(torch.device("cuda"))

    monkeypatch.setattr(h3_nvfp4, "_require_flashinfer_fp4", lambda: (None, None, None))
    config.validate_runtime(torch.device("cuda"))


def test_loader_selects_nvfp4_from_checkpoint_metadata(tmp_path) -> None:
    checkpoint_config = _checkpoint_quantization_config()
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": checkpoint_config}), encoding="utf-8")

    assert _read_text_encoder_checkpoint_quantization_config(str(tmp_path)) == checkpoint_config
    model_config = MiniMaxH3Qwen3VLConfig()
    quant_config = _configure_text_encoder_quantization(model_config, MiniMaxH3Qwen3VLConditioner, str(tmp_path))
    assert isinstance(quant_config, MiniMaxH3SerializedNVFP4Config)
    assert model_config.quant_config is quant_config

    with pytest.raises(ValueError, match="does not support serialized 'nvfp4'"):
        _configure_text_encoder_quantization(MiniMaxH3Qwen3VLConfig(), TextEncoder, str(tmp_path))


def test_conditioner_still_routes_fp8_metadata_to_the_fp8_config() -> None:
    fp8_metadata = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": ["model.visual", "lm_head"],
    }
    assert isinstance(MiniMaxH3Qwen3VLConditioner.checkpoint_quantization_config_from_metadata(fp8_metadata),
                      MiniMaxH3SerializedFP8Config)
    assert isinstance(
        MiniMaxH3Qwen3VLConditioner.checkpoint_quantization_config_from_metadata(_checkpoint_quantization_config()),
        MiniMaxH3SerializedNVFP4Config)


def test_post_load_processing_visits_only_serialized_nvfp4_linears(distributed_setup) -> None:
    config = MiniMaxH3SerializedNVFP4Config.from_config(_checkpoint_quantization_config())
    quantized = _language_linear(config, input_size=128, output_size=128)
    plain = ColumnParallelLinear(input_size=128, output_size=128, bias=False, prefix="plain")
    _fill_loaded(quantized)
    model = torch.nn.ModuleList([quantized, plain])

    assert _process_quantized_text_encoder_weights(model, torch.device("cpu")) == 1
    assert quantized.weight_packed.device.type == "cpu"
    assert quantized._nvfp4_alpha.device.type == "cpu"
    assert plain.weight.device.type == "cpu"


def test_nvfp4_linear_hands_mm_fp4_transposed_operands_and_a_bf16_output(monkeypatch) -> None:
    x_fp4 = torch.zeros(4, 64, dtype=torch.uint8)
    x_scale = torch.zeros(128, 8, dtype=torch.uint8)
    weight_packed = torch.zeros(256, 64, dtype=torch.uint8)
    weight_scale = torch.zeros(256, 8, dtype=torch.uint8)
    alpha = torch.tensor(0.5, dtype=torch.float32)
    receipt: dict[str, object] = {}

    def fake_mm_fp4(a, b, a_scale, b_scale, alpha_arg, out_dtype, out, **kwargs):
        receipt.update(a=a, b=b, a_scale=a_scale, b_scale=b_scale, alpha=alpha_arg, out_dtype=out_dtype, out=out,
                       kwargs=kwargs)
        return torch.zeros(a.shape[0], b.shape[1], dtype=out_dtype)

    monkeypatch.setattr(h3_nvfp4, "_mm_fp4", fake_mm_fp4)
    output = h3_nvfp4._nvfp4_linear(x_fp4, x_scale, weight_packed, weight_scale, alpha)

    assert output.shape == (4, 256)
    assert output.dtype == torch.bfloat16
    assert receipt["a"] is x_fp4
    assert receipt["b"].shape == (64, 256)
    assert receipt["b"].data_ptr() == weight_packed.data_ptr()
    assert receipt["b_scale"].shape == (8, 256)
    assert receipt["b_scale"].data_ptr() == weight_scale.data_ptr()
    assert receipt["alpha"] is alpha
    assert receipt["out_dtype"] == torch.bfloat16
    assert receipt["out"] is None
    assert receipt["kwargs"] == {"backend": "auto"}
