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

"""Utility kernels for VLM models (serving-only).

Provides a Metal-accelerated bilinear grid sampling implementation used
by vision-language model image preprocessors.
"""

from __future__ import annotations

import mlx.core as mx


def grid_sample(x: mx.array, grid: mx.array) -> mx.array:
    """Bilinear grid sampling using an MLX Metal kernel.

    Performs spatial transformer-style sampling from the input feature
    map *x* at the locations specified by *grid*, using bilinear
    interpolation with zero-padding for out-of-bounds coordinates.

    This is functionally equivalent to ``torch.nn.functional.grid_sample``
    with ``mode='bilinear'``, ``padding_mode='zeros'``, and
    ``align_corners=False``.

    Args:
        x: Input feature map of shape ``[B, H, W, C]``.
        grid: Sampling grid of shape ``[B, gN, gM, 2]``, where the
            last dimension contains ``(x, y)`` coordinates normalized
            to ``[-1, 1]``.

    Returns:
        Interpolated output of shape ``[B, gN, gM, C]``.

    Raises:
        AssertionError: If *x* is not 4-D or *grid* is not 4-D.
        ValueError: If the last dimension of *grid* is not 2.

    Example::

        import mlx.core as mx
        x = mx.random.normal((1, 8, 8, 3))
        grid = mx.random.uniform(shape=(1, 4, 4, 2), low=-1.0, high=1.0)
        out = grid_sample(x, grid)  # shape: (1, 4, 4, 3)
    """

    assert x.ndim == 4, "`x` must be 4D."
    assert grid.ndim == 4, "`grid` must be 4D."

    bsz, _, _, channels = x.shape
    _, g_h, g_w, dims = grid.shape
    out_shape = (bsz, g_h, g_w, channels)
    if dims != 2:
        raise ValueError("Last dim of `grid` must be size 2.")

    source = """
        uint elem = thread_position_in_grid.x;
        int H = x_shape[1];
        int W = x_shape[2];
        int C = x_shape[3];
        int gH = grid_shape[1];
        int gW = grid_shape[2];

        int w_stride = C;
        int h_stride = W * w_stride;
        int b_stride = H * h_stride;

        uint grid_idx = elem / C * 2;
        float ix = ((grid["grid_idx"] + 1) * W - 1) / 2;
        float iy = ((grid[grid_idx + 1] + 1) * H - 1) / 2;

        int ix_nw = floor(ix);
        int iy_nw = floor(iy);

        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;

        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;

        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;

        T nw = (ix_se - ix)    * (iy_se - iy);
        T ne = (ix    - ix_sw) * (iy_sw - iy);
        T sw = (ix_ne - ix)    * (iy    - iy_ne);
        T se = (ix    - ix_nw) * (iy    - iy_nw);

        int batch_idx = elem / C / gH / gW * b_stride;
        int channel_idx = elem % C;
        int base_idx = batch_idx + channel_idx;

        T I_nw = x[base_idx + iy_nw * h_stride + ix_nw * w_stride];
        T I_ne = x[base_idx + iy_ne * h_stride + ix_ne * w_stride];
        T I_sw = x[base_idx + iy_sw * h_stride + ix_sw * w_stride];
        T I_se = x[base_idx + iy_se * h_stride + ix_se * w_stride];

        I_nw = iy_nw >= 0 && iy_nw <= H - 1 && ix_nw >= 0 && ix_nw <= W - 1 ? I_nw : 0;
        I_ne = iy_ne >= 0 && iy_ne <= H - 1 && ix_ne >= 0 && ix_ne <= W - 1 ? I_ne : 0;
        I_sw = iy_sw >= 0 && iy_sw <= H - 1 && ix_sw >= 0 && ix_sw <= W - 1 ? I_sw : 0;
        I_se = iy_se >= 0 && iy_se <= H - 1 && ix_se >= 0 && ix_se <= W - 1 ? I_se : 0;

        out["elem"] = nw * I_nw + ne * I_ne + sw * I_sw + se * I_se;
    """

    kernel = mx.fast.metal_kernel(
        name="grid_sample",
        input_names=["x", "grid"],
        output_names=["out"],
        source=source,
    )

    outputs = kernel(
        inputs=[x, grid],
        template=[("T", x.dtype)],
        output_shapes=[out_shape],
        output_dtypes=[x.dtype],
        grid=(mx.prod(mx.array(out_shape)), 1, 1),
        threadgroup=(256, 1, 1),
    )
    return outputs[0]


__all__ = "grid_sample"
