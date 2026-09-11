# SPDX-License-Identifier: Apache-2.0
"""Smoke test for a converted NVFP4 MiniMax-H3 text encoder.

Loads the bf16 conditioner and a checkpoint written by
``scripts/checkpoint_conversion/convert_minimax_h3_text_encoder_nvfp4.py``
through the production ``TextEncoderLoader`` and compares the layer-50 hidden
states on a few prompts. A layout, transposition or scale regression in the
serialized NVFP4 path shows up here as a cosine near zero; genuine 4-bit
noise stays above the thresholds.

Run on a Blackwell GPU::

    MINIMAX_H3_RUN_NVFP4_PARITY=1 \
    MINIMAX_H3_MODEL_ROOT=/path/to/FastH3 \
    MINIMAX_H3_NVFP4_TEXT_ENCODER=/path/to/FastH3-nvfp4/text_encoder \
    pytest tests/local_tests/minimax_h3/test_minimax_h3_text_encoder_nvfp4_parity.py -s
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConfig
from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
from fastvideo.models.loader.component_loader import TextEncoderLoader

PROMPTS = (
    "an alpine lake at sunrise, gentle wind over the water",
    "a red fox crosses fresh snow at sunrise",
    "a busy night market in Taipei, neon signs reflecting on wet pavement, handheld camera",
)
MIN_MEAN_COSINE = float(os.environ.get("MINIMAX_H3_NVFP4_MIN_COSINE", "0.98"))
MAX_RELATIVE_ERROR = float(os.environ.get("MINIMAX_H3_NVFP4_MAX_REL_ERR", "0.30"))


def _require_assets() -> tuple[torch.device, Path, Path]:
    if os.environ.get("MINIMAX_H3_RUN_NVFP4_PARITY") != "1":
        pytest.skip("set MINIMAX_H3_RUN_NVFP4_PARITY=1 on a Blackwell GPU node")
    if not torch.cuda.is_available():
        pytest.fail("MiniMax-H3 NVFP4 parity requires a CUDA GPU", pytrace=False)
    root = os.environ.get("MINIMAX_H3_MODEL_ROOT")
    converted = os.environ.get("MINIMAX_H3_NVFP4_TEXT_ENCODER")
    if not root or not converted:
        pytest.fail("set MINIMAX_H3_MODEL_ROOT and MINIMAX_H3_NVFP4_TEXT_ENCODER", pytrace=False)
    root_path, converted_path = Path(root), Path(converted)
    missing = [str(p) for p in (root_path / "text_encoder", root_path / "tokenizer", converted_path) if not p.is_dir()]
    if missing:
        pytest.fail(f"directories are missing: {missing}", pytrace=False)
    if not (converted_path / "config.json").is_file():
        pytest.fail(f"{converted_path} has no config.json; point at the converted text_encoder directory",
                    pytrace=False)
    return torch.device("cuda"), root_path, converted_path


@pytest.fixture(scope="module", autouse=True)
def _distributed_runtime():
    if os.environ.get("MINIMAX_H3_RUN_NVFP4_PARITY") != "1":
        yield
        return
    for name, value in {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29624",
            "RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "0"
    }.items():
        os.environ.setdefault(name, value)
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()


def _loader_args() -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_config=SimpleNamespace(
            text_encoder_configs=(MiniMaxH3Qwen3VLConfig(), ),
            text_encoder_precisions=("bf16", ),
        ),
        text_encoder_cpu_offload=False,
        override_text_encoder_quant=None,
        override_text_encoder_safetensors=None,
        pin_cpu_memory=False,
        disable_offload_on_unified_memory=lambda device_id=0, *, offload_flag=None: True,
    )


def _encode(text_encoder_dir: Path, tokenizer, device: torch.device) -> list[torch.Tensor]:
    model = TextEncoderLoader().load(str(text_encoder_dir), _loader_args())
    outputs = []
    for prompt in PROMPTS:
        ids = torch.tensor(tokenizer(prompt, add_special_tokens=False)["input_ids"], dtype=torch.long, device=device)
        with torch.inference_mode():
            outputs.append(model(input_ids=ids).float().cpu())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def test_minimax_h3_text_encoder_nvfp4_parity() -> None:
    from transformers import AutoTokenizer

    device, root, converted = _require_assets()
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    reference = _encode(root / "text_encoder", tokenizer, device)
    quantized = _encode(converted, tokenizer, device)

    print(f"\n{'prompt':<60}{'tokens':>7}{'cos mean':>10}{'cos min':>9}{'rel err':>10}", flush=True)
    for prompt, expected, actual in zip(PROMPTS, reference, quantized, strict=True):
        assert actual.shape == expected.shape
        cosine = torch.nn.functional.cosine_similarity(expected, actual, dim=-1)
        relative_error = ((expected - actual).norm() / expected.norm()).item()
        print(f"{prompt[:58]:<60}{expected.shape[0]:>7}{cosine.mean().item():>10.4f}{cosine.min().item():>9.4f}"
              f"{relative_error:>10.3e}", flush=True)
        assert cosine.mean().item() >= MIN_MEAN_COSINE, f"{prompt!r}: mean cosine {cosine.mean().item():.4f}"
        assert relative_error <= MAX_RELATIVE_ERROR, f"{prompt!r}: relative error {relative_error:.3e}"
