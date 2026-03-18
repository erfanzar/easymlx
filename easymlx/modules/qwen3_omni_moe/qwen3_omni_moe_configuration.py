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

"""Qwen3 Omni MoE configuration (serving/inference only).

This module defines the hierarchical configuration classes for the
Qwen3 Omni MoE multimodal model. The top-level ``Qwen3OmniMoeConfig``
aggregates separate configs for the thinker (vision + audio + text),
talker, and code-to-waveform sub-models.
"""

from __future__ import annotations

from typing import Any

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


class Qwen3OmniMoeAudioEncoderConfig(EasyMLXBaseConfig):
    """Configuration for the Qwen3 Omni MoE audio encoder.

    Attributes:
        model_type: Identifier string for the audio encoder model type.
        num_mel_bins: Number of mel-frequency bins for audio input.
        encoder_layers: Number of transformer layers in the audio encoder.
        encoder_attention_heads: Number of attention heads per encoder layer.
        encoder_ffn_dim: Dimensionality of the feed-forward network in encoder layers.
        d_model: Hidden dimensionality of the audio encoder.
        dropout: Dropout probability for encoder layers.
        attention_dropout: Dropout probability for attention weights.
        activation_function: Activation function name used in feed-forward layers.
        activation_dropout: Dropout probability applied after activation.
        scale_embedding: Whether to scale embeddings by sqrt(d_model).
        initializer_range: Standard deviation for weight initialization.
        max_source_positions: Maximum number of source positions for positional encoding.
        n_window: Window size for audio processing.
        output_dim: Output projection dimensionality.
        n_window_infer: Window size used during inference.
        conv_chunksize: Chunk size for convolutional processing.
        downsample_hidden_size: Hidden size for the downsampling layer.
    """

    def __init__(
        self,
        *,
        num_mel_bins: int = 128,
        encoder_layers: int = 32,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        d_model: int = 1280,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_function: str = "gelu",
        activation_dropout: float = 0.0,
        scale_embedding: bool = False,
        initializer_range: float = 0.02,
        max_source_positions: int = 1500,
        n_window: int = 100,
        output_dim: int = 3584,
        n_window_infer: int = 400,
        conv_chunksize: int = 500,
        downsample_hidden_size: int = 480,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.d_model = d_model
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.scale_embedding = scale_embedding
        self.initializer_range = initializer_range
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.downsample_hidden_size = downsample_hidden_size


Qwen3OmniMoeAudioConfig = Qwen3OmniMoeAudioEncoderConfig


class Qwen3OmniMoeVisionEncoderConfig(EasyMLXBaseConfig):
    """Configuration for the Qwen3 Omni MoE vision encoder.

    Attributes:
        model_type: Identifier string for the vision encoder model type.
        depth: Number of transformer blocks in the vision encoder.
        hidden_size: Hidden dimensionality of vision transformer layers.
        hidden_act: Activation function name for the vision MLP layers.
        intermediate_size: Dimensionality of the feed-forward hidden layer.
        num_heads: Number of attention heads in each vision block.
        in_channels: Number of input image channels (e.g., 3 for RGB).
        patch_size: Size of each image patch for the patch embedding.
        image_size: Expected input image resolution (height and width).
        spatial_merge_size: Factor for spatial merging of vision tokens.
        temporal_patch_size: Patch size along the temporal axis for video.
        out_hidden_size: Output projection dimensionality after the vision encoder.
        num_position_embeddings: Maximum number of position embeddings.
        deepstack_visual_indexes: Layer indices used for deep-stack visual features.
        initializer_range: Standard deviation for weight initialization.
    """

    def __init__(
        self,
        *,
        depth: int = 27,
        hidden_size: int = 1152,
        hidden_act: str = "gelu_pytorch_tanh",
        intermediate_size: int = 4304,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 16,
        image_size: int = 224,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 3584,
        num_position_embeddings: int = 2304,
        deepstack_visual_indexes: list[int] | None = None,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.deepstack_visual_indexes = deepstack_visual_indexes
        self.initializer_range = initializer_range

    def finalize(self) -> None:
        """Finalize configuration by setting default values for optional fields.

        Sets ``deepstack_visual_indexes`` to ``[8, 16, 24]`` if not provided.
        """
        if self.deepstack_visual_indexes is None:
            self.deepstack_visual_indexes = [8, 16, 24]


Qwen3OmniMoeVisionConfig = Qwen3OmniMoeVisionEncoderConfig


class Qwen3OmniMoeTextConfig(EasyMLXBaseConfig):
    """Configuration for the Qwen3 Omni MoE text decoder.

    Attributes:
        model_type: Identifier string for the text model type.
        vocab_size: Size of the token vocabulary.
        hidden_size: Hidden dimensionality of transformer layers.
        intermediate_size: Dimensionality of the dense MLP hidden layer.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads for grouped-query attention.
        head_dim: Dimensionality of each attention head. Computed from
            ``hidden_size // num_attention_heads`` if not provided.
        hidden_act: Activation function name for MLP layers.
        max_position_embeddings: Maximum sequence length for position embeddings.
        initializer_range: Standard deviation for weight initialization.
        rms_norm_eps: Epsilon for RMSNorm layers.
        use_cache: Whether to use KV caching during generation.
        tie_word_embeddings: Whether input and output embeddings share weights.
        rope_theta: Base frequency for rotary position embeddings.
        rope_scaling: Optional dictionary configuring RoPE scaling.
        attention_bias: Whether attention projections include bias terms.
        attention_dropout: Dropout probability for attention weights.
        use_sliding_window: Whether to enable sliding window attention.
        sliding_window: Size of the sliding window (in tokens).
        max_window_layers: Layer index threshold above which sliding window is used.
        decoder_sparse_step: Interval at which MoE layers replace dense MLPs.
        moe_intermediate_size: Intermediate size for each MoE expert.
        num_experts_per_tok: Number of experts activated per token (top-k).
        num_experts: Total number of experts in the MoE layers.
        norm_topk_prob: Whether to normalize the top-k routing probabilities.
        output_router_logits: Whether to output router logits for auxiliary losses.
        router_aux_loss_coef: Coefficient for the router auxiliary loss.
        mlp_only_layers: Layer indices that always use a dense MLP (no MoE).
        layer_types: Per-layer attention type list (``"full_attention"`` or
            ``"sliding_attention"``).
    """

    def __init__(
        self,
        *,
        vocab_size: int = 151936,
        hidden_size: int = 3584,
        intermediate_size: int = 18944,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 28,
        num_key_value_heads: int = 4,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1000000.0,
        rope_scaling: dict[str, Any] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        use_sliding_window: bool = False,
        sliding_window: int | None = None,
        max_window_layers: int = 28,
        decoder_sparse_step: int = 1,
        moe_intermediate_size: int = 768,
        num_experts_per_tok: int = 8,
        num_experts: int = 128,
        norm_topk_prob: bool = True,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        mlp_only_layers: list[int] | None = None,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = mlp_only_layers
        self.layer_types = layer_types

    def finalize(self) -> None:
        """Finalize configuration by computing derived values.

        Computes ``head_dim`` from ``hidden_size`` and ``num_attention_heads``
        if not explicitly set. Initializes ``mlp_only_layers`` to an empty list
        and builds ``layer_types`` based on sliding window settings. Also
        normalizes ``rope_scaling`` dictionary keys for compatibility.
        """
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.mlp_only_layers is None:
            self.mlp_only_layers = []

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and self.use_sliding_window and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        if self.rope_scaling is not None and "type" in self.rope_scaling and "rope_type" not in self.rope_scaling:
            self.rope_scaling = dict(self.rope_scaling)
            rtype = self.rope_scaling["type"]
            if rtype == "mrope" or ("mrope_section" in self.rope_scaling and rtype == "default"):
                self.rope_scaling["type"] = "mrope"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]


class Qwen3OmniMoeThinkerConfig(EasyMLXBaseConfig):
    """Configuration for the Qwen3 Omni MoE thinker sub-model.

    The thinker encapsulates the vision encoder, audio encoder, and text
    decoder configurations along with special token IDs for multimodal inputs.

    Attributes:
        audio_config: Audio encoder configuration.
        vision_config: Vision encoder configuration.
        text_config: Text decoder configuration.
        audio_token_id: Token ID marking audio placeholders in the input.
        image_token_id: Token ID marking image placeholders in the input.
        video_token_id: Token ID marking video placeholders in the input.
        vision_start_token_id: Token ID indicating the start of vision tokens.
        vision_end_token_id: Token ID indicating the end of vision tokens.
        position_id_per_seconds: Number of position IDs per second of audio.
        audio_start_token_id: Token ID indicating the start of audio tokens.
        user_token_id: Token ID for the user role marker.
        initializer_range: Standard deviation for weight initialization.
    """

    def __init__(
        self,
        *,
        audio_config: Qwen3OmniMoeAudioEncoderConfig | dict[str, Any] | None = None,
        vision_config: Qwen3OmniMoeVisionEncoderConfig | dict[str, Any] | None = None,
        text_config: Qwen3OmniMoeTextConfig | dict[str, Any] | None = None,
        audio_token_id: int = 151646,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        position_id_per_seconds: int = 25,
        audio_start_token_id: int = 151647,
        user_token_id: int = 872,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if audio_config is not None and isinstance(audio_config, dict):
            audio_config = Qwen3OmniMoeAudioEncoderConfig(**audio_config)
        if vision_config is not None and isinstance(vision_config, dict):
            vision_config = Qwen3OmniMoeVisionEncoderConfig(**vision_config)
        if text_config is not None and isinstance(text_config, dict):
            text_config = Qwen3OmniMoeTextConfig(**text_config)
        self.audio_config = audio_config
        self.vision_config = vision_config
        self.text_config = text_config
        self.audio_token_id = audio_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.position_id_per_seconds = position_id_per_seconds
        self.audio_start_token_id = audio_start_token_id
        self.user_token_id = user_token_id
        self.initializer_range = initializer_range

    def finalize(self) -> None:
        """Finalize the thinker configuration.

        Converts any dictionary-typed sub-configs to their proper class
        instances and calls ``finalize()`` on vision and text sub-configs.
        """
        if isinstance(self.audio_config, dict):
            self.audio_config = Qwen3OmniMoeAudioEncoderConfig(**self.audio_config)
        if isinstance(self.vision_config, dict):
            self.vision_config = Qwen3OmniMoeVisionEncoderConfig(**self.vision_config)
        if isinstance(self.text_config, dict):
            self.text_config = Qwen3OmniMoeTextConfig(**self.text_config)

        if self.vision_config is not None:
            self.vision_config.finalize()
        if self.text_config is not None:
            self.text_config.finalize()

    def get_text_config(self) -> Qwen3OmniMoeTextConfig:
        """Return the text decoder configuration.

        Returns:
            The ``Qwen3OmniMoeTextConfig`` instance for the text decoder.
        """
        return self.text_config


class Qwen3OmniMoeTalkerCodePredictorConfig(EasyMLXBaseConfig):
    """Configuration for the talker code predictor sub-model.

    This transformer predicts discrete audio codec codes from the thinker's
    hidden representations.

    Attributes:
        model_type: Identifier string for the code predictor model type.
        vocab_size: Size of the codec code vocabulary.
        hidden_size: Hidden dimensionality of the predictor transformer.
        intermediate_size: Feed-forward network hidden dimensionality.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads for GQA.
        head_dim: Dimensionality per attention head.
        hidden_act: Activation function name for MLP layers.
        max_position_embeddings: Maximum sequence length.
        initializer_range: Standard deviation for weight initialization.
        rms_norm_eps: Epsilon for RMSNorm layers.
        use_cache: Whether to use KV caching.
        tie_word_embeddings: Whether to tie input/output embeddings.
        rope_theta: Base frequency for rotary position embeddings.
        rope_scaling: Optional RoPE scaling configuration.
        attention_bias: Whether attention projections have bias.
        attention_dropout: Dropout probability for attention.
        sliding_window: Sliding window size, or None to disable.
        layer_types: Per-layer attention type list.
        num_code_groups: Number of parallel codec code groups.
    """

    def __init__(
        self,
        *,
        vocab_size: int = 2048,
        hidden_size: int = 1024,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 5,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, Any] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window: int | None = None,
        layer_types: list[str] | None = None,
        num_code_groups: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        self.num_code_groups = num_code_groups


class Qwen3OmniMoeTalkerTextConfig(EasyMLXBaseConfig):
    """Configuration for the talker text sub-model.

    The talker text model generates text tokens alongside audio codes.

    Attributes:
        model_type: Identifier string for the talker text model type.
        vocab_size: Size of the token vocabulary.
        hidden_size: Hidden dimensionality of transformer layers.
        intermediate_size: Feed-forward network hidden dimensionality.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads for GQA.
        head_dim: Dimensionality per attention head (computed if None).
        hidden_act: Activation function name.
        max_position_embeddings: Maximum sequence length.
        initializer_range: Standard deviation for weight initialization.
        rms_norm_eps: Epsilon for RMSNorm layers.
        use_cache: Whether to use KV caching.
        tie_word_embeddings: Whether to tie input/output embeddings.
        rope_theta: Base frequency for rotary position embeddings.
        rope_scaling: Optional RoPE scaling configuration.
        attention_bias: Whether attention projections have bias.
        attention_dropout: Dropout probability for attention.
        sliding_window: Sliding window size, or None to disable.
        decoder_sparse_step: Interval at which MoE layers replace dense MLPs.
        moe_intermediate_size: Intermediate size for each MoE expert.
        num_experts_per_tok: Number of experts activated per token (top-k).
        num_experts: Total number of experts in MoE layers.
        norm_topk_prob: Whether to normalize top-k routing probabilities.
        output_router_logits: Whether to output router logits.
        router_aux_loss_coef: Coefficient for router auxiliary loss.
        mlp_only_layers: Layer indices that always use a dense MLP.
        shared_expert_intermediate_size: Intermediate size for the shared expert.
    """

    def __init__(
        self,
        *,
        vocab_size: int = 3072,
        hidden_size: int = 1024,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 20,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, Any] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window: int | None = None,
        decoder_sparse_step: int = 1,
        moe_intermediate_size: int = 384,
        num_experts_per_tok: int = 8,
        num_experts: int = 128,
        norm_topk_prob: bool = False,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        mlp_only_layers: list[int] | None = None,
        shared_expert_intermediate_size: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = mlp_only_layers
        self.shared_expert_intermediate_size = shared_expert_intermediate_size

    def finalize(self) -> None:
        """Finalize configuration by computing derived values.

        Computes ``head_dim`` from ``hidden_size`` and ``num_attention_heads``
        if not set, and initializes ``mlp_only_layers`` to an empty list.
        """
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.mlp_only_layers is None:
            self.mlp_only_layers = []


class Qwen3OmniMoeTalkerConfig(EasyMLXBaseConfig):
    """Configuration for the Qwen3 Omni MoE talker sub-model.

    The talker generates audio output from the thinker's representations,
    coordinating a code predictor and a text model.

    Attributes:
        code_predictor_config: Configuration for the codec code predictor.
        text_config: Configuration for the talker text model.
        num_code_groups: Number of parallel codec code groups.
        thinker_hidden_size: Hidden size of the thinker model (for projection).
        codec_eos_token_id: End-of-sequence token ID for codec codes.
        accept_hidden_layer: Layer index from which to extract hidden states.
        codec_nothink_id: Codec token ID for the no-think state.
        codec_think_bos_id: Codec token ID for beginning of thinking.
        codec_think_eos_id: Codec token ID for end of thinking.
        codec_pad_id: Codec padding token ID.
        codec_bos_id: Codec beginning-of-sequence token ID.
        audio_token_id: Token ID marking audio placeholders.
        image_token_id: Token ID marking image placeholders.
        video_token_id: Token ID marking video placeholders.
        vision_start_token_id: Token ID for the start of vision tokens.
        position_id_per_seconds: Number of position IDs per second of audio.
        audio_start_token_id: Token ID for the start of audio tokens.
        speaker_id: Optional mapping of speaker names to integer IDs.
        spatial_merge_size: Factor for spatial merging of vision tokens.
    """

    def __init__(
        self,
        *,
        code_predictor_config: Qwen3OmniMoeTalkerCodePredictorConfig | dict[str, Any] | None = None,
        text_config: Qwen3OmniMoeTalkerTextConfig | dict[str, Any] | None = None,
        num_code_groups: int = 32,
        thinker_hidden_size: int = 2048,
        codec_eos_token_id: int = 4198,
        accept_hidden_layer: int = 18,
        codec_nothink_id: int = 4203,
        codec_think_bos_id: int = 4204,
        codec_think_eos_id: int = 4205,
        codec_pad_id: int = 4196,
        codec_bos_id: int = 4197,
        audio_token_id: int = 151646,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        position_id_per_seconds: int = 25,
        audio_start_token_id: int = 151669,
        speaker_id: dict[str, int] | None = None,
        spatial_merge_size: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if code_predictor_config is not None and isinstance(code_predictor_config, dict):
            code_predictor_config = Qwen3OmniMoeTalkerCodePredictorConfig(**code_predictor_config)
        if text_config is not None and isinstance(text_config, dict):
            text_config = Qwen3OmniMoeTalkerTextConfig(**text_config)
        self.code_predictor_config = code_predictor_config
        self.text_config = text_config
        self.num_code_groups = num_code_groups
        self.thinker_hidden_size = thinker_hidden_size
        self.codec_eos_token_id = codec_eos_token_id
        self.accept_hidden_layer = accept_hidden_layer
        self.codec_nothink_id = codec_nothink_id
        self.codec_think_bos_id = codec_think_bos_id
        self.codec_think_eos_id = codec_think_eos_id
        self.codec_pad_id = codec_pad_id
        self.codec_bos_id = codec_bos_id
        self.audio_token_id = audio_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.position_id_per_seconds = position_id_per_seconds
        self.audio_start_token_id = audio_start_token_id
        self.speaker_id = speaker_id
        self.spatial_merge_size = spatial_merge_size


class Qwen3OmniMoeCode2WavConfig(EasyMLXBaseConfig):
    """Configuration for the code-to-waveform synthesis sub-model.

    Converts discrete codec codes into continuous audio waveforms.

    Attributes:
        model_type: Identifier string for the code2wav model type.
        codebook_size: Size of the audio codec codebook.
        hidden_size: Hidden dimensionality of transformer layers.
        intermediate_size: Feed-forward network hidden dimensionality.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads for GQA.
        head_dim: Dimensionality per attention head (computed if None).
        hidden_act: Activation function name.
        max_position_embeddings: Maximum sequence length.
        rms_norm_eps: Epsilon for RMSNorm layers.
        rope_theta: Base frequency for rotary position embeddings.
        attention_bias: Whether attention projections have bias.
        attention_dropout: Dropout probability for attention.
        sliding_window: Sliding window size for attention.
        layer_scale_initial_scale: Initial scale factor for layer scaling.
        num_quantizers: Number of residual vector quantizers.
        upsample_rates: Upsampling rates for the waveform decoder stages.
        upsampling_ratios: Upsampling ratios for additional decoder stages.
        decoder_dim: Hidden dimensionality of the waveform decoder.
    """

    def __init__(
        self,
        *,
        codebook_size: int = 2048,
        hidden_size: int = 1024,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 8000,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window: int = 72,
        layer_scale_initial_scale: float = 0.01,
        num_quantizers: int = 16,
        upsample_rates: tuple[int, ...] = (8, 5, 4, 3),
        upsampling_ratios: tuple[int, ...] = (2, 2),
        decoder_dim: int = 1536,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.codebook_size = codebook_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.layer_scale_initial_scale = layer_scale_initial_scale
        self.num_quantizers = num_quantizers
        self.upsample_rates = upsample_rates
        self.upsampling_ratios = upsampling_ratios
        self.decoder_dim = decoder_dim


@register_config("qwen3_omni_moe")
class Qwen3OmniMoeConfig(EasyMLXBaseConfig):
    """Top-level configuration for the Qwen3 Omni MoE multimodal model.

    Aggregates the thinker, talker, and code-to-waveform sub-model
    configurations. Sub-configs can be passed as class instances or
    plain dictionaries and will be normalized during initialization.

    Args:
        thinker_config: Configuration for the thinker (vision + audio + text).
        talker_config: Configuration for the talker (audio generation).
        code2wav_config: Configuration for code-to-waveform synthesis.
        enable_audio_output: Whether audio output generation is enabled.
        im_start_token_id: Token ID for the start of an IM (instant message) turn.
        im_end_token_id: Token ID for the end of an IM turn.
        tts_pad_token_id: Padding token ID for TTS sequences.
        tts_bos_token_id: Beginning-of-sequence token ID for TTS.
        tts_eos_token_id: End-of-sequence token ID for TTS.
        system_token_id: Token ID for the system role marker.
        user_token_id: Token ID for the user role marker.
        assistant_token_id: Token ID for the assistant role marker.
        tie_word_embeddings: Whether to tie input/output embeddings.
        **kwargs: Additional keyword arguments passed to the base config.
    """

    model_type = "qwen3_omni_moe"

    def __init__(
        self,
        *,
        thinker_config: Qwen3OmniMoeThinkerConfig | dict[str, Any] | None = None,
        talker_config: Qwen3OmniMoeTalkerConfig | dict[str, Any] | None = None,
        code2wav_config: Qwen3OmniMoeCode2WavConfig | dict[str, Any] | None = None,
        enable_audio_output: bool = True,
        im_start_token_id: int = 151644,
        im_end_token_id: int = 151645,
        tts_pad_token_id: int = 151671,
        tts_bos_token_id: int = 151672,
        tts_eos_token_id: int = 151673,
        system_token_id: int = 8948,
        user_token_id: int = 872,
        assistant_token_id: int = 77091,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if thinker_config is None:
            thinker_cfg = Qwen3OmniMoeThinkerConfig(
                audio_config=Qwen3OmniMoeAudioEncoderConfig(),
                vision_config=Qwen3OmniMoeVisionEncoderConfig(),
                text_config=Qwen3OmniMoeTextConfig(),
            )
        elif isinstance(thinker_config, Qwen3OmniMoeThinkerConfig):
            thinker_cfg = thinker_config
        else:
            thinker_cfg = Qwen3OmniMoeThinkerConfig(**thinker_config)

        thinker_cfg.finalize()

        if talker_config is None:
            talker_cfg = Qwen3OmniMoeTalkerConfig(
                code_predictor_config=Qwen3OmniMoeTalkerCodePredictorConfig(),
                text_config=Qwen3OmniMoeTalkerTextConfig(),
            )
        elif isinstance(talker_config, Qwen3OmniMoeTalkerConfig):
            talker_cfg = talker_config
        else:
            talker_cfg = Qwen3OmniMoeTalkerConfig(**talker_config)

        if code2wav_config is None:
            code2wav_cfg = Qwen3OmniMoeCode2WavConfig()
        elif isinstance(code2wav_config, Qwen3OmniMoeCode2WavConfig):
            code2wav_cfg = code2wav_config
        else:
            code2wav_cfg = Qwen3OmniMoeCode2WavConfig(**code2wav_config)

        self.thinker_config = thinker_cfg.to_dict()
        self.talker_config = talker_cfg.to_dict()
        self.code2wav_config = code2wav_cfg.to_dict()
        self.enable_audio_output = bool(enable_audio_output)
        self.im_start_token_id = int(im_start_token_id)
        self.im_end_token_id = int(im_end_token_id)
        self.tts_pad_token_id = int(tts_pad_token_id)
        self.tts_bos_token_id = int(tts_bos_token_id)
        self.tts_eos_token_id = int(tts_eos_token_id)
        self.system_token_id = int(system_token_id)
        self.user_token_id = int(user_token_id)
        self.assistant_token_id = int(assistant_token_id)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def to_thinker_config(self) -> Qwen3OmniMoeThinkerConfig:
        """Reconstruct a finalized ``Qwen3OmniMoeThinkerConfig`` from stored dicts.

        Returns:
            A fully initialized and finalized thinker configuration.
        """
        thinker_cfg = Qwen3OmniMoeThinkerConfig(
            audio_config=Qwen3OmniMoeAudioEncoderConfig(**self.thinker_config["audio_config"]),
            vision_config=Qwen3OmniMoeVisionEncoderConfig(**self.thinker_config["vision_config"]),
            text_config=Qwen3OmniMoeTextConfig(**self.thinker_config["text_config"]),
            audio_token_id=self.thinker_config.get("audio_token_id", 151646),
            image_token_id=self.thinker_config.get("image_token_id", 151655),
            video_token_id=self.thinker_config.get("video_token_id", 151656),
            vision_start_token_id=self.thinker_config.get("vision_start_token_id", 151652),
            vision_end_token_id=self.thinker_config.get("vision_end_token_id", 151653),
            position_id_per_seconds=self.thinker_config.get("position_id_per_seconds", 25),
            audio_start_token_id=self.thinker_config.get("audio_start_token_id", 151647),
            user_token_id=self.thinker_config.get("user_token_id", 872),
            initializer_range=self.thinker_config.get("initializer_range", 0.02),
        )
        thinker_cfg.finalize()
        return thinker_cfg

    def get_text_config(self) -> Qwen3OmniMoeTextConfig:
        """Return the text decoder configuration via the thinker config.

        Returns:
            The ``Qwen3OmniMoeTextConfig`` from the thinker sub-config.
        """
        return self.to_thinker_config().get_text_config()


__all__ = (
    "Qwen3OmniMoeAudioConfig",
    "Qwen3OmniMoeAudioEncoderConfig",
    "Qwen3OmniMoeCode2WavConfig",
    "Qwen3OmniMoeConfig",
    "Qwen3OmniMoeTalkerCodePredictorConfig",
    "Qwen3OmniMoeTalkerConfig",
    "Qwen3OmniMoeTalkerTextConfig",
    "Qwen3OmniMoeTextConfig",
    "Qwen3OmniMoeThinkerConfig",
    "Qwen3OmniMoeVisionConfig",
    "Qwen3OmniMoeVisionEncoderConfig",
)
