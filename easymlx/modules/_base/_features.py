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

"""Modular features for task-specific modules.

This module provides reusable feature implementations that can be mixed and matched
in task-specific model classes. Features encapsulate common functionality to avoid
code duplication and enable consistent behavior across different model architectures.

Available Features:
    LogitCapFeature: Clips logits to prevent extreme values
    TieEmbeddingsFeature: Shares weights between input embeddings and output head
    RouterAuxLossFeature: Computes auxiliary loss for MoE router load balancing
    GradientCheckpointingFeature: Configures activation checkpointing for memory efficiency
    SequenceLengthPoolingFeature: Pools sequences for classification tasks

Design Philosophy:
    Each feature class follows the single responsibility principle, encapsulating
    one specific functionality. Features are instantiated in BaseTaskModule and
    used by task-specific subclasses as needed.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn


class LogitCapFeature:
    """Apply logit capping to prevent extreme values.

    Logit capping is used to stabilize training by preventing logits from becoming
    too large or too small. This is particularly useful for:
        - Models that use temperature scaling
        - Models prone to numerical instability
        - Preventing overconfident predictions
        - Stabilizing gradients during training

    The capping operation clips logits to the range [-cap_value, cap_value],
    which bounds the maximum probability ratio between any two tokens.

    Attributes:
        cap_value (float): Maximum absolute value for logits. Must be positive.
    """

    def __init__(self, cap_value: float):
        """Initialize logit capping feature.

        Args:
            cap_value: Maximum absolute value for logits. Logits will be clipped
                to the range [-cap_value, cap_value]. Must be a positive number.

        Raises:
            ValueError: If cap_value is not positive.
        """
        if cap_value <= 0:
            raise ValueError(f"cap_value must be positive, got {cap_value}")
        self.cap_value = cap_value

    def apply(self, logits: mx.array) -> mx.array:
        """Apply logit capping to the given logits.

        Clips all values in the logits tensor to be within [-cap_value, cap_value].

        Args:
            logits: Input logits tensor of any shape. Values outside the cap range
                will be clipped.

        Returns:
            Clipped logits tensor of the same shape as input, with all values
            in the range [-cap_value, cap_value].
        """
        return mx.clip(logits, -self.cap_value, self.cap_value)

    def __repr__(self) -> str:
        return f"LogitCapFeature(cap_value={self.cap_value})"


class TieEmbeddingsFeature:
    """Tie input embeddings with output head weights.

    Weight tying is a technique where the input embedding matrix is shared with
    the output projection matrix (LM head). This reduces the number of parameters
    and can improve performance, especially for smaller models.

    Attributes:
        tie (bool): Whether weight tying is enabled.
    """

    def __init__(self, tie: bool = True):
        """Initialize embedding tying feature.

        Args:
            tie: Whether to tie input embeddings with output head weights.
                When True, the LM head uses the transpose of the embedding
                matrix instead of separate parameters. Defaults to True.
        """
        self.tie = tie

    def setup(self, embedding_module: nn.Module, lm_head_module: nn.Module) -> None:
        """Set up weight tying between embedding and LM head.

        In MLX, weight tying is handled by sharing the weight array between
        the embedding and the LM head output projection.

        Args:
            embedding_module: The input embedding layer.
            lm_head_module: The LM head linear layer.
        """
        if not self.tie:
            return
        # Placeholder: actual tying depends on the embedding/lm_head structure.
        # Typically: lm_head.weight = embedding.weight
        pass

    def __repr__(self) -> str:
        return f"TieEmbeddingsFeature(tie={self.tie})"


class RouterAuxLossFeature:
    """Compute auxiliary loss for MoE router load balancing.

    Mixture-of-Experts (MoE) models use routers to distribute inputs across
    different expert networks. Without regularization, the router may learn
    to always select the same few experts. The auxiliary loss encourages
    balanced load distribution.

    Attributes:
        coef (float): Coefficient multiplied with the auxiliary loss.
    """

    def __init__(self, coef: float):
        """Initialize router auxiliary loss feature.

        Args:
            coef: Coefficient to multiply the auxiliary loss by. Controls
                the trade-off between the main task loss and load balancing.

        Raises:
            ValueError: If coef is negative.
        """
        if coef < 0:
            raise ValueError(f"coef must be non-negative, got {coef}")
        self.coef = coef

    def compute_loss(
        self,
        router_losses: list | tuple | None,
    ) -> Any | None:
        """Compute the weighted auxiliary loss from router losses.

        Args:
            router_losses: List or tuple of router loss values from each MoE layer.
                Can be None if the model doesn't have routers or router losses
                weren't computed.

        Returns:
            Weighted sum of router losses (sum(losses) * coef), or None if
            router_losses is None or empty.
        """
        if router_losses is None or len(router_losses) == 0:
            return None

        total_loss = sum(router_losses)
        return total_loss * self.coef  # pyright: ignore[reportReturnType]

    def __repr__(self) -> str:
        return f"RouterAuxLossFeature(coef={self.coef})"


class GradientCheckpointingFeature:
    """Configure gradient checkpointing for model components.

    Gradient checkpointing (activation checkpointing) trades compute for memory
    by recomputing intermediate activations during the backward pass instead of
    storing them.

    Attributes:
        policy (str | None): Checkpointing policy controlling what to save/recompute.
        save_names (list[str]): Operation names to always save (not recompute).
        exclude_names (list[str]): Operation names to exclude from checkpointing.
    """

    def __init__(
        self,
        policy: str | None = None,
        save_names: list[str] | None = None,
        exclude_names: list[str] | None = None,
    ):
        """Initialize gradient checkpointing feature.

        Args:
            policy: Checkpointing policy string. Defaults to None (no checkpointing).
            save_names: List of operation names to always save during forward pass.
            exclude_names: List of operation names to exclude from checkpointing.
        """
        self.policy = policy
        self.save_names = save_names or []
        self.exclude_names = exclude_names or []

    def should_checkpoint(self) -> bool:
        """Check if gradient checkpointing should be applied.

        Returns:
            True if checkpointing should be applied (policy is set and not
            "nothing_saveable"), False otherwise.
        """
        return self.policy is not None and self.policy != "nothing_saveable"

    def get_config(self) -> dict[str, Any]:
        """Get checkpointing configuration as a dictionary.

        Returns:
            Dictionary containing policy, save_names, and exclude_names.
        """
        return {
            "policy": self.policy,
            "save_names": self.save_names,
            "exclude_names": self.exclude_names,
        }

    def __repr__(self) -> str:
        return f"GradientCheckpointingFeature(policy={self.policy})"


class SequenceLengthPoolingFeature:
    """Pool sequence representations for classification tasks.

    For sequence classification tasks, we need to reduce the sequence of hidden
    states to a single vector representation. This feature provides different
    pooling strategies:
        - "last": Use the last non-padding token (decoder-style)
        - "first": Use the first token, typically [CLS] (encoder-style)
        - "mean": Average all token representations
        - "max": Max pooling over the sequence dimension
        - "weighted_mean": Position-weighted mean pooling

    Attributes:
        strategy (str): Pooling strategy to use.
        pad_token_id (int | None): Token ID for padding (needed for "last" strategy).
    """

    def __init__(self, strategy: str = "last", pad_token_id: int | None = None):
        """Initialize sequence pooling feature.

        Args:
            strategy: Pooling strategy to use. Must be one of:
                "last", "first", "mean", "max", "weighted_mean".
                Defaults to "last".
            pad_token_id: Padding token ID. Required when using "last" strategy
                without attention_mask.

        Raises:
            ValueError: If strategy is not one of the valid options.
            ValueError: If strategy is "last" and pad_token_id is None.
        """
        valid_strategies = {"last", "first", "mean", "max", "weighted_mean"}
        if strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}, got {strategy}")

        if strategy == "last" and pad_token_id is None:
            raise ValueError("pad_token_id is required for 'last' pooling strategy")

        self.strategy = strategy
        self.pad_token_id = pad_token_id

    def pool(
        self,
        hidden_states: mx.array,
        input_ids: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """Pool hidden states to get sequence-level representation.

        Args:
            hidden_states: Sequence of hidden states of shape
                (batch_size, sequence_length, hidden_dim).
            input_ids: Input token IDs of shape (batch_size, sequence_length).
                Required for "last" strategy when attention_mask is not provided.
            attention_mask: Optional mask of shape (batch_size, sequence_length)
                where 1 indicates valid tokens and 0 indicates padding.

        Returns:
            Pooled representation of shape (batch_size, hidden_dim).

        Raises:
            ValueError: If using "last" strategy without input_ids or attention_mask.
            ValueError: If an unknown pooling strategy is configured.
        """
        batch_size = hidden_states.shape[0]

        if self.strategy == "first":
            return hidden_states[:, 0]

        elif self.strategy == "last":
            if attention_mask is not None:
                lengths = mx.sum(attention_mask.astype(mx.int32), axis=-1) - 1
                lengths = mx.maximum(lengths, 0)
                return hidden_states[mx.arange(batch_size), lengths]

            if input_ids is None:
                raise ValueError("input_ids required for 'last' pooling strategy")

            sequence_lengths = mx.argmax(mx.equal(input_ids, self.pad_token_id).astype(mx.int32), -1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            return hidden_states[mx.arange(batch_size), sequence_lengths]

        elif self.strategy == "mean":
            if attention_mask is not None:
                weights = attention_mask[:, :, None].astype(hidden_states.dtype)
                return mx.sum(hidden_states * weights, axis=1) / mx.clip(mx.sum(weights, axis=1), a_min=1e-9)
            return mx.mean(hidden_states, axis=1)

        elif self.strategy == "weighted_mean":
            if attention_mask is not None:
                seq_len = hidden_states.shape[1]
                pos_weights = mx.arange(1, seq_len + 1, dtype=hidden_states.dtype)[None, :, None]
                mask = attention_mask[:, :, None].astype(hidden_states.dtype)
                weights = pos_weights * mask
                return mx.sum(hidden_states * weights, axis=1) / mx.clip(mx.sum(weights, axis=1), a_min=1e-9)
            return mx.mean(hidden_states, axis=1)

        elif self.strategy == "max":
            return mx.max(hidden_states, axis=1)

        else:
            raise ValueError(f"Unknown pooling strategy: {self.strategy}")

    def __repr__(self) -> str:
        return f"SequenceLengthPoolingFeature(strategy={self.strategy}, pad_token_id={self.pad_token_id})"
