# Copyright 2026 The EASYDEL / EASYMLX Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native EasyMLX configuration objects for the FLUX.2 klein stack."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config
from easymlx.modules.qwen3 import Qwen3Config
from easymlx.utils.hf_composite import resolve_local_hf_path


def _load_hf_json(
    source: str | Path,
    filename: str,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    path = resolve_local_hf_path(source)
    if path.is_dir():
        resolved_path = path / filename
        if resolved_path.exists():
            return json.loads(resolved_path.read_text(encoding="utf-8"))

    from huggingface_hub import hf_hub_download

    resolved = hf_hub_download(
        repo_id=str(source),
        filename=filename,
        revision=revision,
        local_files_only=local_files_only,
        repo_type="model",
    )
    return json.loads(Path(resolved).read_text(encoding="utf-8"))


@register_config("flux2_transformer")
class Flux2TransformerConfig(EasyMLXBaseConfig):
    """Configuration for the FLUX.2 image transformer."""

    model_type = "flux2_transformer"

    def __init__(
        self,
        *,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: int | None = None,
        num_layers: int = 8,
        num_single_layers: int = 48,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        joint_attention_dim: int = 15360,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: tuple[int, ...] | list[int] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
        guidance_embeds: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.out_channels = None if out_channels is None else int(out_channels)
        self.num_layers = int(num_layers)
        self.num_single_layers = int(num_single_layers)
        self.attention_head_dim = int(attention_head_dim)
        self.num_attention_heads = int(num_attention_heads)
        self.joint_attention_dim = int(joint_attention_dim)
        self.timestep_guidance_channels = int(timestep_guidance_channels)
        self.mlp_ratio = float(mlp_ratio)
        self.axes_dims_rope = tuple(int(v) for v in axes_dims_rope)
        self.rope_theta = int(rope_theta)
        self.eps = float(eps)
        self.guidance_embeds = bool(guidance_embeds)
        if sum(self.axes_dims_rope) != self.attention_head_dim:
            raise ValueError(
                "The sum of `axes_dims_rope` must match `attention_head_dim` "
                f"({sum(self.axes_dims_rope)} != {self.attention_head_dim})."
            )


@register_config("autoencoder_kl_flux2")
class AutoencoderKLFlux2Config(EasyMLXBaseConfig):
    """Configuration for the FLUX.2 VAE."""

    model_type = "autoencoder_kl_flux2"

    def __init__(
        self,
        *,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: tuple[str, ...] | list[str] = (
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        up_block_types: tuple[str, ...] | list[str] = (
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ),
        block_out_channels: tuple[int, ...] | list[int] = (128, 256, 512, 512),
        decoder_block_out_channels: tuple[int, ...] | list[int] | None = None,
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 32,
        norm_num_groups: int = 32,
        sample_size: int = 1024,
        force_upcast: bool = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        mid_block_add_attention: bool = True,
        batch_norm_eps: float = 1e-4,
        batch_norm_momentum: float = 0.1,
        patch_size: tuple[int, int] | list[int] = (2, 2),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.block_out_channels = tuple(int(v) for v in block_out_channels)
        normalized_down_block_types = tuple(str(v) for v in down_block_types)
        normalized_up_block_types = tuple(str(v) for v in up_block_types)
        if len(normalized_down_block_types) != len(self.block_out_channels):
            normalized_down_block_types = ("DownEncoderBlock2D",) * len(self.block_out_channels)
        if len(normalized_up_block_types) != len(self.block_out_channels):
            normalized_up_block_types = ("UpDecoderBlock2D",) * len(self.block_out_channels)
        self.down_block_types = normalized_down_block_types
        self.up_block_types = normalized_up_block_types
        self.decoder_block_out_channels = (
            None if decoder_block_out_channels is None else tuple(int(v) for v in decoder_block_out_channels)
        )
        self.layers_per_block = int(layers_per_block)
        self.act_fn = str(act_fn)
        self.latent_channels = int(latent_channels)
        self.norm_num_groups = int(norm_num_groups)
        self.sample_size = int(sample_size)
        self.force_upcast = bool(force_upcast)
        self.use_quant_conv = bool(use_quant_conv)
        self.use_post_quant_conv = bool(use_post_quant_conv)
        self.mid_block_add_attention = bool(mid_block_add_attention)
        self.batch_norm_eps = float(batch_norm_eps)
        self.batch_norm_momentum = float(batch_norm_momentum)
        self.patch_size = tuple(int(v) for v in patch_size)


@register_config("flow_match_euler_discrete_scheduler")
class FlowMatchEulerDiscreteSchedulerConfig(EasyMLXBaseConfig):
    """Configuration for the FLUX.2 scheduler."""

    model_type = "flow_match_euler_discrete_scheduler"

    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: float | None = 0.5,
        max_shift: float | None = 1.15,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
        invert_sigmas: bool = False,
        shift_terminal: float | None = None,
        use_karras_sigmas: bool = False,
        use_exponential_sigmas: bool = False,
        use_beta_sigmas: bool = False,
        time_shift_type: str = "exponential",
        stochastic_sampling: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.use_dynamic_shifting = bool(use_dynamic_shifting)
        self.base_shift = None if base_shift is None else float(base_shift)
        self.max_shift = None if max_shift is None else float(max_shift)
        self.base_image_seq_len = int(base_image_seq_len)
        self.max_image_seq_len = int(max_image_seq_len)
        self.invert_sigmas = bool(invert_sigmas)
        self.shift_terminal = None if shift_terminal is None else float(shift_terminal)
        self.use_karras_sigmas = bool(use_karras_sigmas)
        self.use_exponential_sigmas = bool(use_exponential_sigmas)
        self.use_beta_sigmas = bool(use_beta_sigmas)
        self.time_shift_type = str(time_shift_type)
        self.stochastic_sampling = bool(stochastic_sampling)


@register_config("flux2_klein")
class Flux2KleinConfig(EasyMLXBaseConfig):
    """Composite configuration for a FLUX.2 klein repository."""

    model_type = "flux2_klein"

    def __init__(
        self,
        *,
        text_config: Qwen3Config | dict[str, Any] | None = None,
        transformer_config: Flux2TransformerConfig | dict[str, Any] | None = None,
        vae_config: AutoencoderKLFlux2Config | dict[str, Any] | None = None,
        scheduler_config: FlowMatchEulerDiscreteSchedulerConfig | dict[str, Any] | None = None,
        is_distilled: bool = True,
        **kwargs,
    ):
        self.text_config = text_config if isinstance(text_config, Qwen3Config) else Qwen3Config(**(text_config or {}))
        self.transformer_config = (
            transformer_config
            if isinstance(transformer_config, Flux2TransformerConfig)
            else Flux2TransformerConfig(**(transformer_config or {}))
        )
        self.vae_config = (
            vae_config
            if isinstance(vae_config, AutoencoderKLFlux2Config)
            else AutoencoderKLFlux2Config(**(vae_config or {}))
        )
        self.scheduler_config = (
            scheduler_config
            if isinstance(scheduler_config, FlowMatchEulerDiscreteSchedulerConfig)
            else FlowMatchEulerDiscreteSchedulerConfig(**(scheduler_config or {}))
        )
        self.is_distilled = bool(is_distilled)
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        **kwargs,
    ) -> "Flux2KleinConfig":
        revision = kwargs.pop("revision", None)
        local_files_only = bool(kwargs.pop("local_files_only", False))

        model_index = _load_hf_json(
            pretrained_model_name_or_path,
            "model_index.json",
            revision=revision,
            local_files_only=local_files_only,
        )
        text_config = _load_hf_json(
            pretrained_model_name_or_path,
            "text_encoder/config.json",
            revision=revision,
            local_files_only=local_files_only,
        )
        transformer_config = _load_hf_json(
            pretrained_model_name_or_path,
            "transformer/config.json",
            revision=revision,
            local_files_only=local_files_only,
        )
        vae_config = _load_hf_json(
            pretrained_model_name_or_path,
            "vae/config.json",
            revision=revision,
            local_files_only=local_files_only,
        )
        scheduler_config = _load_hf_json(
            pretrained_model_name_or_path,
            "scheduler/scheduler_config.json",
            revision=revision,
            local_files_only=local_files_only,
        )

        return cls(
            text_config=text_config,
            transformer_config=transformer_config,
            vae_config=vae_config,
            scheduler_config=scheduler_config,
            is_distilled=bool(model_index.get("is_distilled", True)),
            **kwargs,
        )


__all__ = (
    "AutoencoderKLFlux2Config",
    "FlowMatchEulerDiscreteSchedulerConfig",
    "Flux2KleinConfig",
    "Flux2TransformerConfig",
)
