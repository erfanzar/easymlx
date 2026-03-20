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

"""Gemma3N configuration (serving/inference only).

Gemma3N is a VLM wrapper that delegates text generation to a complex
text backbone with AltUp, Laurel, per-layer inputs, and KV sharing.
"""

from __future__ import annotations

from typing import Any

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("gemma3n")
class Gemma3NConfig(EasyMLXBaseConfig):
    """Configuration for the Gemma3N VLM wrapper model.

    Gemma3N is a complex architecture with AltUp (Alternating Updates)
    multi-stream processing, Laurel (Learned Augmented Residual Layer)
    blocks, per-layer token and model-projected inputs, KV sharing
    across later layers, sliding + full attention patterns, and
    activation sparsity. Vision and audio towers are stripped for
    text-only inference. Registered as model type ``"gemma3n"``.

    Attributes:
        model_type: Identifier string (``"gemma3n"``).
        text_config: Serialized dictionary of the full text backbone
            configuration, parsed into ``Gemma3NTextConfig`` at runtime.
        vocab_size: Vocabulary size from text config.
        hidden_size: Hidden dimensionality from text config.
        num_hidden_layers: Number of decoder layers from text config.

    Args:
        text_config: Dictionary of text backbone parameters. See
            ``Gemma3NTextConfig`` fields for details. Defaults to empty.
        tie_word_embeddings: Whether to tie embeddings. Defaults to True.
        pad_token_id: Padding token id. Defaults to None.
        bos_token_id: Beginning-of-sequence token id. Defaults to None.
        eos_token_id: End-of-sequence token id(s). Defaults to None.

    Example::

        >>> config = Gemma3NConfig(text_config={"hidden_size": 1536})
        >>> config.model_type
        'gemma3n'
    """

    model_type = "gemma3n"

    def __init__(
        self,
        *,
        text_config: dict[str, Any] | None = None,
        tie_word_embeddings: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        if text_config is None:
            text_config = {}
        self.text_config = dict(text_config)

        self.vocab_size = int(text_config.get("vocab_size", 262144))
        self.hidden_size = int(text_config.get("hidden_size", 1536))
        self.num_hidden_layers = int(text_config.get("num_hidden_layers", 26))
        self.intermediate_size = text_config.get("intermediate_size", 6144)
        self.num_attention_heads = int(text_config.get("num_attention_heads", 8))
        self.num_key_value_heads = int(text_config.get("num_key_value_heads", 4))
        self.head_dim = int(text_config.get("head_dim", 256))
        self.rms_norm_eps = float(text_config.get("rms_norm_eps", 1e-6))
        self.rope_theta = float(text_config.get("rope_theta", 1_000_000.0))
        self.max_position_embeddings = int(text_config.get("max_position_embeddings", 32768))
        self.tie_word_embeddings = bool(tie_word_embeddings)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            **kwargs,
        )


__all__ = ("Gemma3NConfig",)
