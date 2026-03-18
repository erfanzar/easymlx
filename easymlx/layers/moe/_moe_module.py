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

"""MoE routing helpers for easymlx (serving-only).

Implements top-k expert selection with optional group-level filtering
and score correction, supporting both sigmoid and softmax scoring.

Attributes:
    ScoreFn: Type alias for supported scoring function names.
"""

from __future__ import annotations

import typing as tp

import mlx.core as mx
import mlx.nn as nn

ScoreFn = tp.Literal["sigmoid", "softmax"]


def _score_logits(logits: mx.array, scoring_func: ScoreFn) -> mx.array:
    """Apply a scoring function to raw router logits.

    Args:
        logits: Raw router logits of shape ``[..., num_experts]``.
        scoring_func: One of ``"sigmoid"`` or ``"softmax"``.

    Returns:
        Scores of the same shape as *logits*, in float32.

    Raises:
        ValueError: If *scoring_func* is not a supported value.
    """
    if scoring_func == "sigmoid":
        return mx.sigmoid(logits.astype(mx.float32))
    if scoring_func == "softmax":
        return mx.softmax(logits.astype(mx.float32), axis=-1)
    raise ValueError(f"Unsupported scoring_func={scoring_func!r}")


def topk_expert_select(
    logits: mx.array,
    *,
    top_k: int,
    scoring_func: ScoreFn = "sigmoid",
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
    n_group: int = 1,
    topk_group: int = 1,
    score_correction_bias: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Select the top-k experts for each token.

    Supports optional group-level filtering (when ``n_group > 1``) and
    score correction bias (for DeepSeekV2-style routing).

    Args:
        logits: Raw router logits of shape
            ``[batch, seq_len, num_experts]`` or
            ``[num_tokens, num_experts]``.
        top_k: Number of experts to select per token. Must be positive.
        scoring_func: Scoring function to apply before selection.
        norm_topk_prob: If ``True`` and ``top_k > 1``, normalize the
            selected scores to sum to 1.
        routed_scaling_factor: Multiplicative scaling applied to the
            final expert weights.
        n_group: Number of expert groups. When ``> 1``, experts are
            partitioned into groups and only the top ``topk_group``
            groups are kept.
        topk_group: Number of groups to keep when ``n_group > 1``.
        score_correction_bias: Optional per-expert bias added to scores
            before selection.

    Returns:
        A tuple ``(indices, scores)`` where *indices* has shape
        ``[..., top_k]`` containing the selected expert indices and
        *scores* has the same shape containing the corresponding
        routing weights.

    Raises:
        ValueError: If ``top_k`` is not positive.
    """
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    scores = _score_logits(logits, scoring_func)
    if score_correction_bias is not None:
        scores = scores + score_correction_bias

    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2)
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    top_scores = mx.take_along_axis(scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        top_scores = top_scores / top_scores.sum(axis=-1, keepdims=True)
    top_scores = top_scores * float(routed_scaling_factor)
    return inds, top_scores


class TopKRouter(nn.Module):
    """Top-k expert router for Mixture-of-Experts blocks.

    Projects hidden states to router logits and applies
    :func:`topk_expert_select` to choose the top-k experts per token.

    Args:
        hidden_size: Dimension of the input hidden states.
        num_experts: Total number of experts.
        top_k: Number of experts selected per token.
        scoring_func: Scoring function for the router.
        norm_topk_prob: Whether to normalize selected scores.
        routed_scaling_factor: Scaling factor for routing weights.
        n_group: Number of expert groups for group-level filtering.
        topk_group: Number of groups to keep per token.
        use_score_bias: If ``True``, include a learnable per-expert
            score correction bias.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        *,
        top_k: int,
        scoring_func: ScoreFn = "sigmoid",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        n_group: int = 1,
        topk_group: int = 1,
        use_score_bias: bool = False,
    ):
        """Initialize the top-k router.

        Args:
            hidden_size: Dimension of the input hidden states.
            num_experts: Total number of available experts.
            top_k: Number of experts to select per token.
            scoring_func: One of ``"sigmoid"`` or ``"softmax"``.
            norm_topk_prob: If ``True`` and ``top_k > 1``, normalize
                the selected routing weights.
            routed_scaling_factor: Multiplicative scaling for the
                routing weights.
            n_group: Number of expert groups.
            topk_group: Number of groups to retain.
            use_score_bias: Whether to add a learnable score correction
                bias.
        """
        super().__init__()
        self.top_k = int(top_k)
        self.scoring_func = scoring_func
        self.norm_topk_prob = bool(norm_topk_prob)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.n_group = int(n_group)
        self.topk_group = int(topk_group)
        self.weight = mx.zeros((num_experts, hidden_size))
        self.score_correction_bias = mx.zeros((num_experts,)) if use_score_bias else None

    def __call__(self, hidden_states: mx.array) -> tuple[mx.array, mx.array]:
        """Route tokens to experts.

        Args:
            hidden_states: Input hidden states of shape
                ``[..., hidden_size]``.

        Returns:
            A tuple ``(indices, scores)`` where *indices* has shape
            ``[..., top_k]`` and *scores* has the same shape.
        """
        logits = hidden_states @ self.weight.T
        return topk_expert_select(
            logits,
            top_k=self.top_k,
            scoring_func=self.scoring_func,
            norm_topk_prob=self.norm_topk_prob,
            routed_scaling_factor=self.routed_scaling_factor,
            n_group=self.n_group,
            topk_group=self.topk_group,
            score_correction_bias=self.score_correction_bias,
        )


__all__ = ("ScoreFn", "TopKRouter", "topk_expert_select")
