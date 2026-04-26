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

"""Tests for the native FLUX.2 klein EasyMLX stack."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from easymlx.infra.factory import TaskType, registry
from easymlx.modules.auto import AutoEasyMLXConfig, AutoEasyMLXModel
from easymlx.modules.flux2_klein import (
    AutoencoderKLFlux2,
    AutoencoderKLFlux2Config,
    FlowMatchEulerDiscreteScheduler,
    FlowMatchEulerDiscreteSchedulerConfig,
    Flux2KleinConfig,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
    Flux2TransformerConfig,
)
from easymlx.modules.qwen3 import Qwen3Config, Qwen3ForCausalLM
from mlx.utils import tree_flatten
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


def _build_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "flux": 4,
        "klein": 5,
        "test": 6,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    fast.chat_template = "{{ messages[0]['content'] }}"
    return fast


def _build_flux2_klein_repo(tmp_path: Path, *, quantization_friendly: bool = False) -> Path:
    repo_name = "flux2-klein-native-quant" if quantization_friendly else "flux2-klein-native"
    repo = tmp_path / repo_name
    repo.mkdir()

    text_hidden_size = 32 if quantization_friendly else 8
    text_intermediate_size = 64 if quantization_friendly else 16
    text_head_dim = 8 if quantization_friendly else 4
    text_num_heads = 4 if quantization_friendly else 2
    text_num_kv_heads = 2 if quantization_friendly else 1
    transformer_in_channels = 32 if quantization_friendly else 16
    transformer_head_dim = 8
    transformer_num_heads = 4 if quantization_friendly else 2
    transformer_joint_attention_dim = 96 if quantization_friendly else 24
    transformer_timestep_guidance_channels = 32 if quantization_friendly else 8
    vae_latent_channels = transformer_in_channels // 4
    vae_block_out_channels = (32, 64) if quantization_friendly else (8, 16)
    vae_norm_num_groups = 8 if quantization_friendly else 4

    text_config = Qwen3Config(
        vocab_size=64 if quantization_friendly else 16,
        hidden_size=text_hidden_size,
        intermediate_size=text_intermediate_size,
        num_hidden_layers=4,
        num_attention_heads=text_num_heads,
        num_key_value_heads=text_num_kv_heads,
        head_dim=text_head_dim,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    text_encoder = Qwen3ForCausalLM(text_config)
    text_encoder.save_pretrained(repo / "text_encoder")

    transformer_config = Flux2TransformerConfig(
        in_channels=transformer_in_channels,
        out_channels=transformer_in_channels,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=transformer_head_dim,
        num_attention_heads=transformer_num_heads,
        joint_attention_dim=transformer_joint_attention_dim,
        timestep_guidance_channels=transformer_timestep_guidance_channels,
        axes_dims_rope=(2, 2, 2, 2),
        guidance_embeds=False,
    )
    transformer = Flux2Transformer2DModel(transformer_config)
    transformer.save_pretrained(repo / "transformer")

    vae_config = AutoencoderKLFlux2Config(
        block_out_channels=vae_block_out_channels,
        latent_channels=vae_latent_channels,
        layers_per_block=1,
        norm_num_groups=vae_norm_num_groups,
        sample_size=32,
    )
    vae = AutoencoderKLFlux2(vae_config)
    vae.save_pretrained(repo / "vae")

    scheduler = FlowMatchEulerDiscreteScheduler(
        FlowMatchEulerDiscreteSchedulerConfig(
            num_train_timesteps=1000,
            shift=3.0,
            use_dynamic_shifting=True,
        )
    )
    scheduler.save_pretrained(repo / "scheduler")

    tokenizer = _build_tokenizer()
    tokenizer.save_pretrained(str(repo / "tokenizer"))

    (repo / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "Flux2KleinPipeline",
                "is_distilled": True,
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "Qwen3ForCausalLM"],
                "tokenizer": ["transformers", "Qwen2TokenizerFast"],
                "transformer": ["diffusers", "Flux2Transformer2DModel"],
                "vae": ["diffusers", "AutoencoderKLFlux2"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return repo


def test_flux2_klein_registry_and_core_modules() -> None:
    base_pipeline = registry.get_module_registration(TaskType.BASE_MODULE, "flux2_klein")
    base_transformer = registry.get_module_registration(TaskType.BASE_MODULE, "flux2_transformer")
    base_vae = registry.get_module_registration(TaskType.BASE_MODULE, "autoencoder_kl_flux2")

    assert base_pipeline.module is Flux2KleinPipeline
    assert base_pipeline.config is Flux2KleinConfig
    assert base_transformer.module is Flux2Transformer2DModel
    assert base_transformer.config is Flux2TransformerConfig
    assert base_vae.module is AutoencoderKLFlux2
    assert base_vae.config is AutoencoderKLFlux2Config


def test_flux2_klein_config_from_composite_repo(tmp_path: Path) -> None:
    repo = _build_flux2_klein_repo(tmp_path)

    config = Flux2KleinConfig.from_pretrained(repo)

    assert isinstance(config.text_config, Qwen3Config)
    assert isinstance(config.transformer_config, Flux2TransformerConfig)
    assert isinstance(config.vae_config, AutoencoderKLFlux2Config)
    assert isinstance(config.scheduler_config, FlowMatchEulerDiscreteSchedulerConfig)
    assert config.is_distilled is True


def test_flux2_klein_pipeline_loads_and_generates_latents(tmp_path: Path) -> None:
    repo = _build_flux2_klein_repo(tmp_path)

    pipeline = Flux2KleinPipeline.from_pretrained(repo)
    output = pipeline(
        prompt="flux klein test",
        height=32,
        width=32,
        num_inference_steps=2,
        seed=0,
        output_type="latent",
        text_encoder_out_layers=(1, 2, 3),
    )

    assert output.images.shape == (1, 4, 16, 16)


def test_flux2_klein_pipeline_save_pretrained_roundtrip(tmp_path: Path) -> None:
    repo = _build_flux2_klein_repo(tmp_path)
    pipeline = Flux2KleinPipeline.from_pretrained(repo)

    saved = tmp_path / "flux2-klein-roundtrip"
    pipeline.save_pretrained(saved)

    assert (saved / "model_index.json").exists()
    assert (saved / "text_encoder" / "config.json").exists()
    assert (saved / "transformer" / "config.json").exists()
    assert (saved / "vae" / "config.json").exists()
    assert (saved / "scheduler" / "scheduler_config.json").exists()
    assert (saved / "tokenizer" / "tokenizer_config.json").exists()

    reloaded = Flux2KleinPipeline.from_pretrained(saved, auto_convert_hf=False)
    output = reloaded(
        prompt="flux klein test",
        height=32,
        width=32,
        num_inference_steps=2,
        seed=0,
        output_type="latent",
        text_encoder_out_layers=(1, 2, 3),
    )

    assert output.images.shape == (1, 4, 16, 16)


def test_flux2_klein_pipeline_loads_from_tilde_hf_cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home_dir = tmp_path / "home"
    cache_root = home_dir / ".cache" / "huggingface" / "hub" / "models--black-forest-labs--FLUX.2-klein-4B"
    snapshot_dir = cache_root / "snapshots" / "testsnapshot"
    refs_dir = cache_root / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text("testsnapshot\n", encoding="utf-8")

    repo = _build_flux2_klein_repo(tmp_path)
    shutil.copytree(repo, snapshot_dir)
    monkeypatch.setenv("HOME", str(home_dir))

    pipeline = Flux2KleinPipeline.from_pretrained(
        "~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B",
        auto_convert_hf=False,
    )
    output = pipeline(
        prompt="flux klein test",
        height=32,
        width=32,
        num_inference_steps=2,
        seed=0,
        output_type="latent",
        text_encoder_out_layers=(1, 2, 3),
    )

    assert output.images.shape == (1, 4, 16, 16)


def test_flux2_klein_pipeline_loads_from_repo_id_using_local_hf_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home_dir = tmp_path / "home"
    hub_root = home_dir / ".cache" / "huggingface" / "hub"
    cache_root = hub_root / "models--black-forest-labs--FLUX.2-klein-4B"
    snapshot_dir = cache_root / "snapshots" / "testsnapshot"
    refs_dir = cache_root / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text("testsnapshot\n", encoding="utf-8")

    repo = _build_flux2_klein_repo(tmp_path)
    shutil.copytree(repo, snapshot_dir)

    monkeypatch.setenv("HOME", str(home_dir))
    try:
        import huggingface_hub.constants as hf_constants

        monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(hub_root))
    except Exception:
        pass

    pipeline = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        auto_convert_hf=False,
        local_files_only=True,
    )
    output = pipeline(
        prompt="flux klein test",
        height=32,
        width=32,
        num_inference_steps=2,
        seed=0,
        output_type="latent",
        text_encoder_out_layers=(1, 2, 3),
    )

    assert output.images.shape == (1, 4, 16, 16)


def test_flux2_klein_auto_base_loader_resolves_composite_repo(tmp_path: Path) -> None:
    repo = _build_flux2_klein_repo(tmp_path)

    config = AutoEasyMLXConfig.from_pretrained(str(repo))
    model = AutoEasyMLXModel.from_pretrained(str(repo), auto_convert_hf=False)
    output = model(
        prompt="flux klein test",
        height=32,
        width=32,
        num_inference_steps=2,
        seed=0,
        output_type="latent",
        text_encoder_out_layers=(1, 2, 3),
    )

    assert isinstance(config, Flux2KleinConfig)
    assert isinstance(model, Flux2KleinPipeline)
    assert output.images.shape == (1, 4, 16, 16)


def test_flux2_klein_pipeline_supports_mxfp4_loading(tmp_path: Path) -> None:
    repo = _build_flux2_klein_repo(tmp_path, quantization_friendly=True)

    pipeline = Flux2KleinPipeline.from_pretrained(
        repo,
        auto_convert_hf=False,
        quantization="mxfp4",
    )

    text_params = tree_flatten(pipeline.text_encoder.parameters(), destination={})
    transformer_params = tree_flatten(pipeline.transformer.parameters(), destination={})
    vae_params = tree_flatten(pipeline.vae.parameters(), destination={})

    assert any(key.endswith(".scales") for key in text_params)
    assert any(key.endswith(".scales") for key in transformer_params)
    assert not any(key.endswith(".scales") for key in vae_params)

    latents = pipeline(
        prompt="flux klein test",
        height=32,
        width=32,
        num_inference_steps=2,
        seed=0,
        output_type="latent",
        text_encoder_out_layers=(1, 2, 3),
    ).images
    decoded = pipeline.vae.decode(latents, return_dict=False)[0]
    assert decoded.shape == (1, 3, 32, 32)


def test_flux2_klein_pipeline_save_pretrained_roundtrip_for_quantized_pipeline(tmp_path: Path) -> None:
    repo = _build_flux2_klein_repo(tmp_path, quantization_friendly=True)

    pipeline = Flux2KleinPipeline.from_pretrained(
        repo,
        auto_convert_hf=False,
        quantization="mxfp4",
    )

    saved = tmp_path / "flux2-klein-roundtrip-mxfp4"
    pipeline.save_pretrained(saved)
    saved_text_config = json.loads((saved / "text_encoder" / "config.json").read_text(encoding="utf-8"))
    saved_transformer_config = json.loads((saved / "transformer" / "config.json").read_text(encoding="utf-8"))
    saved_vae_config = json.loads((saved / "vae" / "config.json").read_text(encoding="utf-8"))

    assert saved_text_config["easymlx_quantization"]["mode"] == "mxfp4"
    assert saved_transformer_config["easymlx_quantization"]["mode"] == "mxfp4"
    assert "easymlx_quantization" not in saved_vae_config

    reloaded = Flux2KleinPipeline.from_pretrained(
        saved,
        auto_convert_hf=False,
    )

    text_params = tree_flatten(reloaded.text_encoder.parameters(), destination={})
    transformer_params = tree_flatten(reloaded.transformer.parameters(), destination={})
    vae_params = tree_flatten(reloaded.vae.parameters(), destination={})

    assert any(key.endswith(".scales") for key in text_params)
    assert any(key.endswith(".scales") for key in transformer_params)
    assert not any(key.endswith(".scales") for key in vae_params)

    latents = reloaded(
        prompt="flux klein test",
        height=32,
        width=32,
        num_inference_steps=2,
        seed=0,
        output_type="latent",
        text_encoder_out_layers=(1, 2, 3),
    ).images
    decoded = reloaded.vae.decode(latents, return_dict=False)[0]
    assert decoded.shape == (1, 3, 32, 32)


def test_flux2_transformer_vae_and_scheduler_shape_smoke() -> None:
    transformer = Flux2Transformer2DModel(
        Flux2TransformerConfig(
            in_channels=8,
            out_channels=8,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=24,
            timestep_guidance_channels=8,
            axes_dims_rope=(2, 2, 2, 2),
            guidance_embeds=False,
        )
    )
    transformer_out = transformer(
        mx.random.normal(shape=(1, 16, 8)),
        encoder_hidden_states=mx.random.normal(shape=(1, 6, 24)),
        timestep=mx.array([0.5]),
        img_ids=mx.zeros((1, 16, 4), dtype=mx.int32),
        txt_ids=mx.zeros((1, 6, 4), dtype=mx.int32),
        return_dict=False,
    )[0]
    assert transformer_out.shape == (1, 16, 8)

    vae = AutoencoderKLFlux2(
        AutoencoderKLFlux2Config(
            block_out_channels=(16, 32),
            latent_channels=4,
            layers_per_block=1,
            norm_num_groups=8,
            sample_size=32,
        )
    )
    image = mx.random.normal(shape=(1, 3, 32, 32))
    latents = vae.encode(image).latent_dist.mode()
    reconstruction = vae.decode(latents, return_dict=False)[0]
    assert latents.shape == (1, 4, 16, 16)
    assert reconstruction.shape == (1, 3, 32, 32)

    scheduler = FlowMatchEulerDiscreteScheduler(
        FlowMatchEulerDiscreteSchedulerConfig(use_dynamic_shifting=True, shift=3.0)
    )
    scheduler.set_timesteps(4, mu=1.0)
    stepped = scheduler.step(
        mx.random.normal(shape=(1, 16, 8)),
        scheduler.timesteps[0],
        mx.random.normal(shape=(1, 16, 8)),
    )
    assert stepped.prev_sample.shape == (1, 16, 8)


def test_flux2_vae_diffusers_parity_on_tiny_checkpoint(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")

    torch.manual_seed(0)
    torch_vae = diffusers.AutoencoderKLFlux2(
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 16),
        decoder_block_out_channels=(8, 16),
        latent_channels=4,
        layers_per_block=1,
        norm_num_groups=4,
        sample_size=32,
        force_upcast=False,
        patch_size=(2, 2),
    )
    torch_vae.eval()
    torch_vae.save_pretrained(tmp_path / "torch-flux2-vae")

    mlx_vae = AutoencoderKLFlux2.from_pretrained(
        tmp_path / "torch-flux2-vae",
        auto_convert_hf=False,
        dtype=mx.float32,
    )

    latents = np.random.default_rng(0).standard_normal((1, 4, 16, 16), dtype=np.float32)
    with torch.no_grad():
        torch_out = torch_vae.decode(torch.from_numpy(latents), return_dict=False)[0].float().cpu().numpy()
    mlx_out = np.array(mlx_vae.decode(mx.array(latents), return_dict=False)[0], dtype=np.float32)

    diff = np.abs(torch_out - mlx_out)
    assert float(diff.max()) < 1e-3
    assert float(diff.mean()) < 1e-4


def test_flux2_klein_pipeline_passes_flux_sigma_schedule(tmp_path: Path) -> None:
    repo = _build_flux2_klein_repo(tmp_path)
    pipeline = Flux2KleinPipeline.from_pretrained(repo)

    class RecordingScheduler(FlowMatchEulerDiscreteScheduler):
        def __init__(self, config: FlowMatchEulerDiscreteSchedulerConfig):
            super().__init__(config)
            self.recorded_sigmas: list[float] | None = None

        def set_timesteps(
            self,
            num_inference_steps: int | None = None,
            *,
            sigmas: list[float] | None = None,
            mu: float | None = None,
            timesteps: list[float] | None = None,
        ) -> None:
            self.recorded_sigmas = sigmas
            super().set_timesteps(
                num_inference_steps,
                sigmas=sigmas,
                mu=mu,
                timesteps=timesteps,
            )

    pipeline.scheduler = RecordingScheduler(pipeline.scheduler.config)
    num_inference_steps = 4
    pipeline(
        prompt="flux klein test",
        height=32,
        width=32,
        num_inference_steps=num_inference_steps,
        seed=0,
        output_type="latent",
        text_encoder_out_layers=(1, 2, 3),
    )

    assert pipeline.scheduler.recorded_sigmas is not None
    assert np.allclose(
        np.array(pipeline.scheduler.recorded_sigmas, dtype=np.float32),
        np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=np.float32),
    )
