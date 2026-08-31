# SPDX-License-Identifier: Apache-2.0
"""FastVideo composed pipelines for MiniMax H3."""

from __future__ import annotations

from pathlib import Path

from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.hf_transformer_utils import get_diffusers_config
from fastvideo.pipelines.basic.minimax_h3.stages import (
    MiniMaxH3AudioDecodingStage,
    MiniMaxH3ConditioningStage,
    MiniMaxH3DenoisingStage,
    MiniMaxH3InputPreparationStage,
    MiniMaxH3LatentPreparationStage,
    MiniMaxH3VideoDecodingStage,
)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.lora_pipeline import LoRAPipeline

logger = init_logger(__name__)


def _apply_h3_checkpoint_arch_configs(model_path: str, fastvideo_args: FastVideoArgs,
                                      extra_config_module_map: dict[str, str]) -> None:
    """Overlay checkpoint config.json onto pipeline configs without loading weights."""
    root = Path(model_path)
    vae_dir = root / "vae"
    if (vae_dir / "config.json").is_file():
        fastvideo_args.pipeline_config.vae_config.update_model_arch(get_diffusers_config(str(vae_dir)))
    transformer_dir = root / extra_config_module_map.get("transformer", "transformer")
    if (transformer_dir / "config.json").is_file():
        fastvideo_args.pipeline_config.dit_config.update_model_arch(get_diffusers_config(str(transformer_dir)))
    logger.info(
        "MiniMax-H3 geometry from config: patch_size=%s spatial_compression_ratio=%s latent_channels=%s",
        tuple(fastvideo_args.pipeline_config.dit_config.patch_size),
        int(fastvideo_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio),
        int(fastvideo_args.pipeline_config.vae_config.arch_config.latent_channels),
    )


class MiniMaxH3BasePipeline(LoRAPipeline, ComposedPipelineBase):
    """Shared loading and target-generation path for MiniMax H3.

    Inherits ``LoRAPipeline`` so acceleration and distillation adapters can be merged
    in; without it every adapter is rejected with "pipeline is not a LoRAPipeline".
    """

    # The linears every published H3 adapter targets. Left unset, ``LoRAPipeline``
    # wraps *every* linear in the DiT -- including ``proj_in``, whose ``.weight`` the
    # forward pass reads directly. ``BaseLayerWithLoRA`` exposes no ``.weight``, so
    # that wrapping turns generation into an AttributeError before the first step.
    lora_target_modules = [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out",
        "ff.fc_in",
        "ff.fc_out",
        "adaln_proj.linear",
        # The final AdaLN projection. Published community adapters (larryvrh's Turbo)
        # target it as `final_layer.adaln_proj.linear`.
        "norm_out.linear",
    ]

    pipeline_config_cls: type[MiniMaxH3PipelineConfig] = MiniMaxH3PipelineConfig
    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "processor",
        "vae",
        "audio_vae",
        "transformer",
        "scheduler",
        "audio_scheduler",
    ]
    # Deferral is safe here: geometry scalars come from checkpoint config.json
    # (applied in initialize_pipeline without loading weights), no stage
    # constructor reads a deferred component, and initialize_pipeline only
    # inspects the schedulers, which are never deferred.
    _lazy_module_names = ("text_encoder", "transformer", "vae", "audio_vae")

    @classmethod
    def get_hf_download_component_dirs(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._extra_config_module_map.get(name, name) for name in cls._required_config_modules))

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        _apply_h3_checkpoint_arch_configs(self.model_path, fastvideo_args, self._extra_config_module_map)
        for module_name, modality, expected_shift in (
            ("scheduler", "video", 12.0),
            ("audio_scheduler", "audio", 3.0),
        ):
            shift = getattr(self.get_module(module_name), "shift", None)
            if shift is None or float(shift) != expected_shift:
                raise ValueError(f"MiniMax-H3 {modality} scheduler must expose shift={expected_shift:g}, got {shift}.")

    def _add_stages(self, fastvideo_args: FastVideoArgs, *, ref2va: bool) -> None:
        transformer = self.get_module("transformer")
        vae = self.get_module("vae")
        audio_vae = self.get_module("audio_vae")
        scheduler = self.get_module("scheduler")
        audio_scheduler = self.get_module("audio_scheduler")
        # Geometry scalars live on the checkpoint-updated arch config. Holding
        # the live VAE/DiT here would materialize them on the first attribute
        # read. Encode still needs the live VAE for FL2VA/Ref2VA.
        video_geometry = fastvideo_args.pipeline_config.vae_config.arch_config

        self.add_stage(
            "input_preparation_stage",
            MiniMaxH3InputPreparationStage(
                vae=video_geometry,
                audio_vae=audio_vae if ref2va else None,
                ref2va=ref2va,
            ),
        )
        self.add_stage(
            "conditioning_stage",
            MiniMaxH3ConditioningStage(
                conditioner=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                processor=self.get_module("processor"),
                ref2va=ref2va,
            ),
        )
        self.add_stage(
            "latent_preparation_stage",
            MiniMaxH3LatentPreparationStage(
                vae=vae,
                audio_vae=audio_vae,
                scheduler=scheduler,
                ref2va=ref2va,
            ),
        )
        self.add_stage(
            "denoising_stage",
            MiniMaxH3DenoisingStage(
                transformer=transformer,
                scheduler=scheduler,
                audio_scheduler=audio_scheduler,
            ),
        )
        self.add_stage("video_decoding_stage", MiniMaxH3VideoDecodingStage(vae=vae))
        self.add_stage("audio_decoding_stage", MiniMaxH3AudioDecodingStage(audio_vae=audio_vae))


class MiniMaxH3Pipeline(MiniMaxH3BasePipeline):
    """One-request joint video/stereo-audio pipeline for T2VA and FL2VA."""

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self._add_stages(fastvideo_args, ref2va=False)


class MiniMaxH3RefPipeline(MiniMaxH3BasePipeline):
    """Ordered-reference joint video/stereo-audio pipeline for Ref2VA."""

    _extra_config_module_map = {"transformer": "transformer_ref"}

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self._add_stages(fastvideo_args, ref2va=True)


class MiniMaxH3ModularPipeline(MiniMaxH3Pipeline):
    """Public T2VA/FL2VA entry matching the official manifest class name."""


class MiniMaxH3Ref2VAModularPipeline(MiniMaxH3RefPipeline):
    """Public Ref2VA entry using the checkpoint's ``transformer_ref`` partition."""


EntryClass = [MiniMaxH3ModularPipeline, MiniMaxH3Ref2VAModularPipeline]

__all__ = [
    "EntryClass",
    "MiniMaxH3BasePipeline",
    "MiniMaxH3ModularPipeline",
    "MiniMaxH3Pipeline",
    "MiniMaxH3Ref2VAModularPipeline",
    "MiniMaxH3RefPipeline",
]
