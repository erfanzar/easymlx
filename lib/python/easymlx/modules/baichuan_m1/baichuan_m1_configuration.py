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

"""Baichuan M1 configuration for EasyMLX.

Baichuan M1 uses a packed QKV projection (W_pack), a custom 1-D
convolution on K/V, and sliding window attention on designated layers.
"""

from __future__ import annotations

from easymlx.infra import EasyMLXBaseConfig
from easymlx.infra.factory import register_config


@register_config("baichuan_m1")
class BaichuanM1Config(EasyMLXBaseConfig):
    """Configuration for the Baichuan M1 transformer model.

    Baichuan M1 uses packed QKV projection (``W_pack``), a custom 1-D
    convolution on K/V with a window of 2, and sliding window attention
    on designated layers. Registered under model type ``"baichuan_m1"``.

    Args:
        vocab_size: Number of tokens in the vocabulary. Defaults to 102400.
        hidden_size: Dimensionality of hidden representations. Defaults to 4096.
        intermediate_size: SwiGLU MLP intermediate dimensionality. Defaults to 11008.
        num_hidden_layers: Number of transformer decoder layers. Defaults to 32.
        num_attention_heads: Number of attention heads. Defaults to 32.
        num_key_value_heads: Number of KV heads for GQA. Defaults to 32.
        rope_theta: RoPE base frequency. Defaults to 10000.0.
        sliding_window: Sliding window size for local attention. Defaults to 512.
        sliding_window_layers: List of layer indices that use sliding window
            attention. Defaults to empty list.
        conv_window: Convolution window size for K/V (must be 2). Defaults to 2.
        rms_norm_eps: RMSNorm epsilon. Defaults to 1e-5.
        num_swa_attention_heads: Number of Q heads for sliding window layers.
            If ``None``, uses ``num_attention_heads``. Defaults to ``None``.
        num_swa_key_value_heads: Number of KV heads for sliding window layers.
            If ``None``, uses ``num_key_value_heads``. Defaults to ``None``.
        tie_word_embeddings: Whether to tie input/output embeddings. Defaults to ``False``.

    Attributes:
        model_type: The model type identifier (``"baichuan_m1"``).

    Example::

        >>> config = BaichuanM1Config(hidden_size=2048, num_hidden_layers=24)
        >>> config.model_type
        'baichuan_m1'
    """

    model_type = "baichuan_m1"

    def __init__(
        self,
        *,
        vocab_size: int = 102400,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        rope_theta: float = 10000.0,
        sliding_window: int = 512,
        sliding_window_layers: list[int] | None = None,
        conv_window: int = 2,
        rms_norm_eps: float = 1e-5,
        num_swa_attention_heads: int | None = None,
        num_swa_key_value_heads: int | None = None,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        bos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize the Baichuan M1 configuration.

        See class docstring for full parameter documentation.
        """
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.rope_theta = float(rope_theta)
        self.sliding_window = int(sliding_window)
        self.sliding_window_layers = sliding_window_layers or []
        self.conv_window = int(conv_window)
        self.rms_norm_eps = float(rms_norm_eps)
        self.num_swa_attention_heads = num_swa_attention_heads
        self.num_swa_key_value_heads = num_swa_key_value_heads

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ("BaichuanM1Config",)
