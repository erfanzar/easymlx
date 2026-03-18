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

"""Rotary positional embedding implementations for MLX.

This module provides various RoPE (Rotary Positional Embedding) variants used
by modern transformer architectures, including Su-scaled RoPE, Llama-3 RoPE,
and YaRN RoPE. It also provides a factory function ``get_rope`` that selects
the appropriate RoPE variant based on a scaling configuration dictionary.

Supported RoPE types:
    - ``default`` / ``linear``: Standard or linearly scaled RoPE via ``nn.RoPE``.
    - ``llama3``: Llama-3 piecewise frequency scaling.
    - ``yarn``: YaRN (Yet another RoPE extensioN) with frequency interpolation.
    - ``longrope``: Su-scaled RoPE with separate short/long frequency factors.
    - ``mrope``: Multi-resolution RoPE (delegates to standard ``nn.RoPE``).
"""

import math

import mlx.core as mx
import mlx.nn as nn


class SuScaledRoPE(nn.Module):
    """Su-scaled Rotary Positional Embedding layer.

    Applies separate frequency scaling factors for short and long sequences,
    with optional magnitude scaling. This variant is used by models that
    employ the LongRoPE strategy for extended context lengths.

    Attributes:
        original_max_position_embeddings: Threshold for switching between
            short and long frequency factors.
        dim: Number of feature dimensions to rotate.
    """

    def __init__(
        self,
        dims: int,
        base: float = 10000.0,
        max_position_embeddings: int = 131072,
        original_max_position_embeddings: int = 4096,
        short_factor: list["float"] | float = 1.0,
        long_factor: list["float"] | float = 1.0,
        short_mscale: float | None = None,
        long_mscale: float | None = None,
    ):
        """
        Su Scaled Rotary Embedding layer.

        Args:
            dims (int): The feature dimensions to be rotated.
            base (int, optional): Base for the exponential scaling.
            max_position_embeddings (int, optional): The maximum sequence
              length that this model was trained with. This is used to determine
              the size of the original RoPE embeddings when using long scaling.
              Default: ``131072``.
            original_max_position_embeddings (int, optional): The maximum
              sequence length that this model was trained with. This is used to
              determine the size of the original RoPE embeddings when using long
              scaling. Default: ``4096``.
            short_factor (float or list["float"], optional): List of scaling
              factors for sequences of length lesser than
              ``original_max_position_embeddings``. Default: ``1.0``.
            long_factor (float or list["float"], optional): List of scaling
              factors for sequences of length greater than
              ``original_max_position_embeddings``.  Default: ``1.0``.
            short_mscale (float, optional): Scale the input prior to embedding.
            long_mscale (float, optional): Scale the input prior to embedding.
        """
        super().__init__()
        self.original_max_position_embeddings = original_max_position_embeddings
        self.dim = dims

        freqs = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        self._short_freqs = mx.array(short_factor, dtype=mx.float32) * freqs
        self._long_freqs = mx.array(long_factor, dtype=mx.float32) * freqs

        def default_scale(factor):
            return math.sqrt(1 + math.log(factor) / math.log(original_max_position_embeddings))

        factor = max_position_embeddings / original_max_position_embeddings
        self._short_scale = short_mscale or (1.0 if factor <= 1.0 else default_scale(factor))
        self._long_scale = long_mscale or (1.0 if factor <= 1.0 else default_scale(factor))

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        """Apply Su-scaled rotary embeddings to the input tensor.

        Selects short or long frequency factors based on the total sequence
        length (offset + current length) relative to
        ``original_max_position_embeddings``.

        Args:
            x: Input tensor of shape ``(..., seq_len, dim)`` to apply
                rotary embeddings to.
            offset: Position offset for cached/incremental decoding.

        Returns:
            Tensor of the same shape as ``x`` with rotary embeddings applied
            to the first ``self.dim`` dimensions.
        """
        seq_len = offset + x.shape[-2]
        if seq_len > self.original_max_position_embeddings:
            freqs = self._long_freqs
            scale = self._long_scale
        else:
            freqs = self._short_freqs
            scale = self._short_scale

        x[..., : self.dim] = scale * x[..., : self.dim]
        return mx.fast.rope(
            x,
            self.dim,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=freqs,
        )


class Llama3RoPE(nn.Module):
    """Llama-3 piecewise frequency-scaled Rotary Positional Embedding.

    Implements the Llama-3 RoPE scaling strategy that applies different
    scaling factors to different frequency bands: high-frequency components
    are left unscaled, low-frequency components are fully scaled, and
    medium-frequency components are smoothly interpolated between the two.

    Attributes:
        dims: Number of feature dimensions to rotate.
        max_position_embeddings: Maximum sequence length for this model.
        traditional: Whether to use the traditional (interleaved) RoPE layout.
    """

    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 2048,
        traditional: bool = False,
        base: float = 10000,
        scaling_config: dict | None = None,
    ):
        """Initialize Llama-3 RoPE.

        Args:
            dims: Number of feature dimensions to rotate.
            max_position_embeddings: Maximum sequence length this model supports.
                Defaults to 2048.
            traditional: Whether to use the traditional (interleaved) RoPE
                layout. Defaults to False.
            base: Base frequency for the exponential scaling. Defaults to 10000.
            scaling_config: Dictionary containing scaling parameters. Must
                include ``"factor"`` key. Optional keys: ``"low_freq_factor"``
                (default 1.0), ``"high_freq_factor"`` (default 4.0), and
                ``"original_max_position_embeddings"`` (default 8192).

        Raises:
            KeyError: If ``scaling_config`` does not contain ``"factor"``.
        """
        super().__init__()
        self.dims = dims
        self.max_position_embeddings = max_position_embeddings
        self.traditional = traditional

        factor = scaling_config["factor"]
        low_freq_factor = scaling_config.get("low_freq_factor", 1.0)
        high_freq_factor = scaling_config.get("high_freq_factor", 4.0)
        old_context_len = scaling_config.get("original_max_position_embeddings", 8192)

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        freqs = base ** (mx.arange(0, dims, 2) / dims)
        wavelens = 2 * mx.pi * freqs

        freqs = mx.where(wavelens > low_freq_wavelen, freqs * factor, freqs)
        is_medium_freq = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)
        smooth_factors = (old_context_len / wavelens - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth_freqs = freqs / ((1 - smooth_factors) / factor + smooth_factors)
        self._freqs = mx.where(is_medium_freq, smooth_freqs, freqs)

    def extra_repr(self) -> str:
        """Return a string representation of the module's parameters.

        Returns:
            Formatted string with dims, traditional, and max_position_embeddings.
        """
        return f"{self.dims}, traditional={self.traditional}, max_position_embeddings={self.max_position_embeddings}"

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        """Apply Llama-3 rotary embeddings to the input tensor.

        Args:
            x: Input tensor of shape ``(..., seq_len, dim)``.
            offset: Position offset for cached/incremental decoding.

        Returns:
            Tensor with rotary embeddings applied.
        """
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class YarnRoPE(nn.Module):
    """YaRN (Yet another RoPE extensioN) Rotary Positional Embedding.

    Implements the YaRN strategy for extending context length by combining
    linear interpolation with NTK-aware scaling. Uses a frequency-dependent
    ramp mask to smoothly transition between interpolated and extrapolated
    frequencies.

    Attributes:
        mscale: Magnitude scale factor applied to the rotated dimensions.
        dims: Number of feature dimensions to rotate.
        traditional: Whether to use the traditional (interleaved) RoPE layout.
    """

    def __init__(
        self,
        dims: int,
        traditional: bool = False,
        max_position_embeddings: int = 2048,
        base: float = 10000,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: int = 1,
        mscale_all_dim: int = 0,
    ):
        """Initialize YaRN RoPE.

        Args:
            dims: Number of feature dimensions to rotate.
            traditional: Whether to use the traditional (interleaved) RoPE
                layout. Defaults to False.
            max_position_embeddings: Maximum sequence length. Defaults to 2048.
            base: Base frequency for exponential scaling. Defaults to 10000.
            scaling_factor: Context length scaling factor. Defaults to 1.0.
            original_max_position_embeddings: Original training context length.
                Defaults to 4096.
            beta_fast: Fast beta parameter controlling the upper correction
                boundary. Defaults to 32.
            beta_slow: Slow beta parameter controlling the lower correction
                boundary. Defaults to 1.
            mscale: Magnitude scaling parameter. Defaults to 1.
            mscale_all_dim: Magnitude scaling parameter for all dimensions.
                Defaults to 0.
        """
        super().__init__()

        def yarn_find_correction_dim(num_rotations):
            return (dims * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))) / (
                2 * math.log(base)
            )

        def yarn_find_correction_range():
            low = math.floor(yarn_find_correction_dim(beta_fast))
            high = math.ceil(yarn_find_correction_dim(beta_slow))
            return max(low, 0), min(high, dims - 1)

        def yarn_get_mscale(scale=1, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        def yarn_linear_ramp_mask(min_val, max_val, dim):
            if min_val == max_val:
                max_val += 0.001

            linear_func = (mx.arange(dim, dtype=mx.float32) - min_val) / (max_val - min_val)
            return mx.clip(linear_func, 0, 1)

        self.mscale = yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(scaling_factor, mscale_all_dim)
        freq_extra = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        freq_inter = scaling_factor * freq_extra
        low, high = yarn_find_correction_range()
        freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dims // 2)
        self._freqs = (freq_inter * freq_extra) / (freq_inter * freq_mask + freq_extra * (1 - freq_mask))
        self.dims = dims
        self.traditional = traditional

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        """Apply YaRN rotary embeddings to the input tensor.

        Args:
            x: Input tensor of shape ``(..., seq_len, dim)``.
            offset: Position offset for cached/incremental decoding.
                Defaults to 0.

        Returns:
            Tensor with YaRN rotary embeddings applied.
        """
        if self.mscale != 1.0:
            x[..., : self.dims] = self.mscale * x[..., : self.dims]
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


def get_rope(
    dims: float | int,
    base: float | int,
    traditional: bool,
    scaling_config: dict | None = None,
    max_position_embeddings: int | None = None,
) -> nn.Module:
    """Factory function to create the appropriate RoPE module.

    Selects and instantiates the correct RoPE variant based on the
    ``rope_type`` field in ``scaling_config``.

    Args:
        dims: Number of feature dimensions to rotate.
        base: Base frequency for the exponential scaling.
        traditional: Whether to use the traditional (interleaved) RoPE layout.
        scaling_config: Optional dictionary containing RoPE scaling parameters.
            Must include a ``"type"`` or ``"rope_type"`` key. If None, defaults
            to standard RoPE.
        max_position_embeddings: Maximum sequence length. Required for some
            RoPE variants (llama3, yarn, longrope).

    Returns:
        An ``nn.Module`` implementing the selected RoPE variant.

    Raises:
        ValueError: If the specified ``rope_type`` is not supported.

    Example:
        >>> rope = get_rope(dims=64, base=10000, traditional=False)
        >>> rope = get_rope(
        ...     dims=64, base=10000, traditional=False,
        ...     scaling_config={"type": "llama3", "factor": 8.0},
        ...     max_position_embeddings=8192,
        ... )
    """
    if scaling_config is not None:
        rope_type = scaling_config.get("type") or scaling_config.get("rope_type", "default")
    else:
        rope_type = "default"

    if rope_type in ["default", "linear"]:
        scale = 1 / scaling_config["factor"] if rope_type == "linear" else 1.0
        return nn.RoPE(dims, traditional=traditional, base=base, scale=scale)

    elif rope_type == "llama3":
        return Llama3RoPE(
            dims=dims,
            max_position_embeddings=max_position_embeddings,
            traditional=traditional,
            base=base,
            scaling_config=scaling_config,
        )
    elif rope_type == "yarn":
        scaling_factor = scaling_config["factor"]
        rope_kwargs = {
            key: scaling_config[key]
            for key in [
                "original_max_position_embeddings",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            ]
            if key in scaling_config
        }
        return YarnRoPE(
            dims=dims,
            max_position_embeddings=max_position_embeddings,
            traditional=traditional,
            scaling_factor=scaling_factor,
            base=base,
            **rope_kwargs,
        )
    elif rope_type == "longrope":
        return SuScaledRoPE(
            dims=dims,
            base=base,
            max_position_embeddings=max_position_embeddings,
            original_max_position_embeddings=scaling_config["original_max_position_embeddings"],
            short_factor=scaling_config["short_factor"],
            long_factor=scaling_config["long_factor"],
        )
    elif rope_type == "mrope":
        mrope_section = scaling_config.get("mrope_section", [])
        assert len(mrope_section) == 3, f"MRoPE currently only supports 3 sections, got {len(mrope_section)}."
        return nn.RoPE(dims, traditional=traditional, base=base)
    else:
        raise ValueError(f"Unsupported RoPE type {rope_type}")
