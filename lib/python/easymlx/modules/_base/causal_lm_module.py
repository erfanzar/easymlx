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

"""Generic base class for Causal Language Modeling tasks on MLX.

This module provides ``BaseCausalLMModule``, the foundation for all autoregressive
language models. It wraps a base transformer model with a language modeling head
that projects hidden states to vocabulary logits, supporting weight tying, logit
capping, and both standard and paged KV cache strategies.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from easymlx.caching import (
    PageCacheView,
    PageMetadata,
    TransformerCacheView,
)
from easymlx.infra.base_config import EasyMLXBaseConfig
from easymlx.infra.base_module import EasyMLXBaseModule
from easymlx.infra.modeling_outputs import CausalLMOutput

CacheView = TransformerCacheView | PageCacheView


class BaseCausalLMModule[ModelT: nn.Module, ConfigT: EasyMLXBaseConfig](EasyMLXBaseModule):
    """Base class for causal language models on MLX.

    Wraps a base transformer model with a language modeling head that projects
    hidden states to vocabulary logits. Supports weight tying, logit capping,
    paged cache initialization, and standard cache initialization.

    Subclasses only need to provide ``__init__`` — ``__call__``, ``init_paged_cache``,
    ``init_cache``, and ``sanitize`` are all inherited.
    """

    def __init__(
        self,
        config: ConfigT,
        base_model: ModelT | None = None,
        base_model_class: type[ModelT] | None = None,
        base_model_name: str = "model",
        *,
        tie_word_embeddings: bool = False,
        logit_cap: float | None = None,
        lm_head_name: str = "lm_head",
        lm_head_bias: bool = False,
    ):
        """Initialize the causal language model module.

        Args:
            config: Model configuration containing hyperparameters such as
                ``hidden_size``, ``vocab_size``, etc.
            base_model: Pre-instantiated base model instance. If provided,
                ``base_model_class`` is ignored.
            base_model_class: Base model class to instantiate. Required if
                ``base_model`` is not provided.
            base_model_name: Attribute name under which to store the base
                model. Defaults to ``"model"``.
            tie_word_embeddings: Whether to share weights between the input
                embedding layer and the LM head. Defaults to False.
            logit_cap: Maximum absolute value for output logits via tanh
                scaling. If None, no capping is applied. Defaults to None.
            lm_head_name: Attribute name for the LM head linear layer.
                Defaults to ``"lm_head"``.
            lm_head_bias: Whether to include bias in the LM head.
                Defaults to False.

        Raises:
            ValueError: If neither ``base_model`` nor ``base_model_class``
                is provided.
        """
        super().__init__(config=config)
        self._base_model_name = base_model_name
        self._lm_head_name = lm_head_name
        self._tie_word_embeddings = tie_word_embeddings
        self._logit_cap = logit_cap

        if base_model is not None:
            setattr(self, base_model_name, base_model)
        elif base_model_class is not None:
            setattr(self, base_model_name, base_model_class(config))
        else:
            raise ValueError("Either base_model or base_model_class must be provided.")

        if not tie_word_embeddings:
            lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=lm_head_bias)
            setattr(self, lm_head_name, lm_head)

    @property
    def base_model(self) -> ModelT:
        """Access the base model via the configured attribute name.

        Returns:
            The underlying transformer base model instance.
        """
        return getattr(self, self._base_model_name)

    def get_embedding(self) -> nn.Embedding:
        """Return the input embedding layer from the base model.

        Searches for the embedding layer using common attribute names
        and method conventions across different model architectures.

        Returns:
            The input embedding module.

        Raises:
            AttributeError: If no embedding layer can be found on the
                base model.
        """
        model = self.base_model
        for attr in ("get_embedding", "get_input_embeddings"):
            fn = getattr(model, attr, None)
            if callable(fn):
                return fn()
        for attr in ("embed_tokens", "wte", "word_embeddings", "embeddings"):
            if hasattr(model, attr):
                return getattr(model, attr)
        raise AttributeError("Cannot find embedding layer on base model.")

    def compute_lm_logits(self, hidden_states: mx.array) -> mx.array:
        """Project hidden states to vocabulary logits.

        Uses the LM head linear layer, or the embedding layer's ``as_linear``
        method if weight tying is enabled. Applies logit capping if configured.

        Args:
            hidden_states: Hidden state tensor of shape
                ``(batch_size, seq_len, hidden_size)`` or
                ``(num_tokens, hidden_size)``.

        Returns:
            Logits tensor of shape ``(..., vocab_size)``.
        """
        if self._tie_word_embeddings:
            embedding = self.get_embedding()
            logits = embedding.as_linear(hidden_states)
        else:
            lm_head = getattr(self, self._lm_head_name)
            logits = lm_head(hidden_states)
        return self.apply_logit_cap(logits)

    def apply_logit_cap(self, logits: mx.array) -> mx.array:
        """Apply logit capping via tanh scaling if configured.

        When a logit cap value is set, applies ``cap * tanh(logits / cap)``
        to smoothly constrain logits to ``(-cap, cap)``.

        Args:
            logits: Input logits tensor of any shape.

        Returns:
            Capped logits if logit capping is configured, otherwise
            the input unchanged.
        """
        if self._logit_cap is not None:
            cap = self._logit_cap
            logits = cap * mx.tanh(logits / cap)
        return logits

    def _extract_last_tokens(
        self,
        hidden_states: mx.array,
        cache_metadata: PageMetadata | None,
    ) -> mx.array:
        """Extract the last token hidden state per sequence for paged serving.

        When using paged attention with flattened (2D) hidden states, this
        method uses ``cache_metadata.query_start_loc`` to identify the last
        token position for each sequence in the batch.

        Args:
            hidden_states: Hidden states tensor. If 2D and cache_metadata is
                provided, assumed to be flattened across sequences.
            cache_metadata: Paged cache metadata containing query start
                locations. If None, hidden_states are returned as-is.

        Returns:
            Hidden states with only the last token per sequence, shaped
            ``(batch_size, hidden_size)``.
        """
        if (
            hidden_states.ndim == 2
            and cache_metadata is not None
            and not bool(getattr(cache_metadata, "is_single_token_decode", False))
        ):
            qsl = cache_metadata.query_start_loc
            if not isinstance(qsl, mx.array):
                qsl = mx.array(list(qsl), dtype=mx.int32)
            last_indices = (qsl[1:].astype(mx.int32) - 1).astype(mx.int32)
            hidden_states = mx.take(hidden_states, last_indices, axis=0)
        return hidden_states

    @staticmethod
    def _default_eagle3_feature_layers(num_layers: int) -> tuple[int, ...]:
        """Choose lower, middle, and final layer indices for EAGLE3 features."""
        if num_layers <= 0:
            return ()
        if num_layers == 1:
            return (1,)
        lower = max(1, num_layers // 3)
        middle = max(1, (2 * num_layers) // 3)
        return tuple(dict.fromkeys((lower, middle, num_layers)))

    def _normalize_eagle3_feature_layers(
        self,
        feature_layer_indices: tuple[int, ...] | list[int] | None,
        num_layers: int,
    ) -> tuple[int, ...]:
        """Normalize requested hidden-state layer indices.

        Layer ``0`` refers to the embedding output. Layer ``num_layers``
        refers to the final normalized transformer output.
        """
        if feature_layer_indices is None:
            feature_layer_indices = self._default_eagle3_feature_layers(num_layers)
        normalized: list[int] = []
        for layer_index in feature_layer_indices:
            idx = int(layer_index)
            if idx < 0:
                idx = num_layers + 1 + idx
            idx = min(max(idx, 0), num_layers)
            if idx not in normalized:
                normalized.append(idx)
        return tuple(normalized)

    def _forward_with_eagle3_hidden_states(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        feature_layer_indices: tuple[int, ...] | list[int] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, ...]]:
        """Run a full-context forward pass and capture EAGLE3 features.

        This generic path supports the common EasyMLX decoder layout
        (``embed_tokens`` + ``layers`` + optional ``norm``). Architectures
        with a different backbone can override ``eagle3_hidden_states``.
        """
        from easymlx.layers.attention import build_attention_mask

        model = self.base_model
        layers = getattr(model, "layers", None)
        embed_tokens = getattr(model, "embed_tokens", None)
        if layers is None or embed_tokens is None:
            raise AttributeError(
                f"{type(model).__name__} does not expose the generic decoder layout needed for EAGLE3 features."
            )

        if input_embeddings is not None:
            hidden_states = mx.array(input_embeddings)
        else:
            input_ids = mx.array(input_ids, dtype=mx.int32)
            if input_ids.ndim == 1:
                input_ids = input_ids[None, :]
            hidden_states = embed_tokens(input_ids)

        layers = list(layers)
        num_layers = len(layers)
        selected_layers = set(self._normalize_eagle3_feature_layers(feature_layer_indices, num_layers))
        captured: dict[int, mx.array] = {}
        if 0 in selected_layers:
            captured[0] = hidden_states

        mask: mx.array | str | None = None
        sliding_mask: mx.array | str | None = None
        if hidden_states.ndim == 3:
            batch_size, seq_len = hidden_states.shape[:2]
            attention_mask_arr = mx.array(attention_mask) if attention_mask is not None else None
            mask = build_attention_mask(attention_mask_arr, batch_size=batch_size, seq_len=seq_len)
            if any(bool(getattr(layer, "use_sliding", False)) for layer in layers):
                sliding_window = getattr(model, "sliding_window", None)
                sliding_mask = build_attention_mask(
                    attention_mask_arr,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    window_size=sliding_window,
                )

        for layer_idx, layer in enumerate(layers, start=1):
            layer_mask = sliding_mask if bool(getattr(layer, "use_sliding", False)) else mask
            hidden_states = layer(hidden_states, mask=layer_mask, cache_view=None, cache_metadata=None)
            if layer_idx in selected_layers and layer_idx != num_layers:
                captured[layer_idx] = hidden_states

        norm = getattr(model, "norm", None)
        if callable(norm):
            hidden_states = norm(hidden_states)
        if num_layers in selected_layers:
            captured[num_layers] = hidden_states

        hidden_features = tuple(captured[idx] for idx in sorted(selected_layers) if idx in captured)
        return hidden_states, hidden_features

    def eagle3_hidden_states(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        feature_layer_indices: tuple[int, ...] | list[int] | None = None,
    ) -> tuple[mx.array, ...]:
        """Return target hidden-state features for EAGLE3 proposers."""
        _, hidden_states = self._forward_with_eagle3_hidden_states(
            input_ids,
            attention_mask=attention_mask,
            input_embeddings=input_embeddings,
            feature_layer_indices=feature_layer_indices,
        )
        return hidden_states

    def __call__(
        self,
        input_ids: mx.ArrayLike,
        *,
        attention_mask: mx.ArrayLike | None = None,
        input_embeddings: mx.array | None = None,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
        return_dict: bool = True,
        output_hidden_states: bool = False,
        eagle3_feature_layer_indices: tuple[int, ...] | list[int] | None = None,
    ) -> mx.array | CausalLMOutput:
        """Forward pass through the causal language model.

        Runs the base model, extracts the last tokens (for paged serving),
        and projects to vocabulary logits.

        Args:
            input_ids: Input token IDs of shape ``(batch_size, seq_len)``
                or ``(num_tokens,)`` for paged serving.
            attention_mask: Optional attention mask of shape
                ``(batch_size, seq_len)``.
            input_embeddings: Optional pre-computed input embeddings.
                Overrides ``input_ids`` if provided.
            cache_views: Per-layer KV cache views for incremental decoding.
            cache_metadata: Paged cache metadata for paged attention serving.
            return_dict: If True, return a ``CausalLMOutput`` dataclass.
                If False, return raw logits tensor. Defaults to True.
            output_hidden_states: If True for a full-context call, also
                return EAGLE3 hidden-state features.
            eagle3_feature_layer_indices: Optional feature layer indices
                used when ``output_hidden_states=True``.

        Returns:
            A ``CausalLMOutput`` containing logits if ``return_dict`` is True,
            otherwise a raw logits ``mx.array``.
        """
        captured_hidden_states: tuple[mx.array, ...] | None = None
        if output_hidden_states and cache_views is None and cache_metadata is None:
            hidden_states, captured_hidden_states = self._forward_with_eagle3_hidden_states(
                input_ids,
                attention_mask=attention_mask,
                input_embeddings=input_embeddings,
                feature_layer_indices=eagle3_feature_layer_indices,
            )
        else:
            hidden_states = self.base_model(
                input_ids,
                attention_mask=attention_mask,
                input_embeddings=input_embeddings,
                cache_views=cache_views,
                cache_metadata=cache_metadata,
            )
        hidden_states = self._extract_last_tokens(hidden_states, cache_metadata)
        logits = self.compute_lm_logits(hidden_states)

        if return_dict:
            return CausalLMOutput(logits=logits, hidden_states=captured_hidden_states)
        return logits

    def decode_step(
        self,
        input_ids: mx.ArrayLike,
        *,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
    ) -> mx.array:
        """Run the raw next-token logits path with a fixed decode signature."""
        hidden_states = self.base_model(
            input_ids,
            cache_views=cache_views,
            cache_metadata=cache_metadata,
        )
        hidden_states = self._extract_last_tokens(hidden_states, cache_metadata)
        return self.compute_lm_logits(hidden_states)

    def get_decode_state(self):
        get_state = getattr(self.base_model, "get_decode_state", None)
        if not callable(get_state):
            raise AttributeError(f"{type(self.base_model).__name__} does not expose get_decode_state().")
        return get_state()

    def set_decode_state(self, state) -> None:
        set_state = getattr(self.base_model, "set_decode_state", None)
        if not callable(set_state):
            raise AttributeError(f"{type(self.base_model).__name__} does not expose set_decode_state().")
        set_state(state)

    def decode_step_with_state(
        self,
        input_ids: mx.ArrayLike,
        *,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
        decode_state,
    ):
        decode_fn = getattr(self.base_model, "decode_step_with_state", None)
        if not callable(decode_fn):
            raise AttributeError(f"{type(self.base_model).__name__} does not expose decode_step_with_state().")
        hidden_states, new_decode_state = decode_fn(
            input_ids,
            cache_views=cache_views,
            cache_metadata=cache_metadata,
            decode_state=decode_state,
        )
        hidden_states = self._extract_last_tokens(hidden_states, cache_metadata)
        return self.compute_lm_logits(hidden_states), new_decode_state

    def decode_step_with_state_and_hidden(
        self,
        input_ids: mx.ArrayLike,
        *,
        cache_views: list[CacheView] | None = None,
        cache_metadata: PageMetadata | None = None,
        decode_state=None,
        feature_layer_indices: tuple[int, ...] | list[int] | None = None,
    ) -> tuple[mx.array, object, tuple[mx.array, ...]]:
        """Decode with caches while capturing selected intermediate features.

        This is the cached counterpart to ``_forward_with_eagle3_hidden_states``.
        It is intentionally generic for decoder-only backbones that expose the
        usual ``embed_tokens`` + ``layers`` + optional ``norm`` layout.
        """
        model = self.base_model
        layers = getattr(model, "layers", None)
        embed_tokens = getattr(model, "embed_tokens", None)
        if layers is None or embed_tokens is None:
            raise AttributeError(
                f"{type(model).__name__} does not expose the generic decoder layout needed for hidden features."
            )
        layers = list(layers)
        if cache_views is not None and len(cache_views) != len(layers):
            raise ValueError("cache_views length must match number of layers.")

        input_ids = mx.array(input_ids, dtype=mx.int32)
        hidden_states = embed_tokens(input_ids)
        activation_dtype_fn = getattr(model, "_activation_dtype", None)
        if callable(activation_dtype_fn):
            activation_dtype = activation_dtype_fn(hidden_states.dtype)
            if hidden_states.dtype != activation_dtype:
                hidden_states = hidden_states.astype(activation_dtype)

        num_layers = len(layers)
        selected_layers = set(self._normalize_eagle3_feature_layers(feature_layer_indices, num_layers))
        captured: dict[int, mx.array] = {}
        if 0 in selected_layers:
            captured[0] = hidden_states

        linear_states = None
        if isinstance(decode_state, dict) and isinstance(decode_state.get("linear_layers"), list):
            linear_states = decode_state["linear_layers"]
        new_linear_states: list[dict[str, mx.array]] = []
        state_idx = 0

        for layer_idx, layer in enumerate(layers, start=1):
            layer_cache = None if cache_views is None else cache_views[layer_idx - 1]
            layer_decode = getattr(layer, "decode_step_with_state", None)
            if callable(layer_decode) and linear_states is not None:
                layer_state = None
                if not bool(getattr(layer, "use_full_attention", False)):
                    layer_state = linear_states[state_idx]
                    state_idx += 1
                hidden_states, new_layer_state = layer_decode(
                    hidden_states,
                    cache_view=layer_cache,
                    cache_metadata=cache_metadata,
                    decode_state=layer_state,
                )
                if new_layer_state is not None:
                    new_linear_states.append(new_layer_state)
            else:
                hidden_states = layer(
                    hidden_states,
                    mask=None,
                    cache_view=layer_cache,
                    cache_metadata=cache_metadata,
                )

            if layer_idx in selected_layers and layer_idx != num_layers:
                captured[layer_idx] = hidden_states

        norm = getattr(model, "norm", None)
        if callable(norm):
            hidden_states = norm(hidden_states)
        if num_layers in selected_layers:
            captured[num_layers] = hidden_states

        logits_hidden_states = self._extract_last_tokens(hidden_states, cache_metadata)
        logits = self.compute_lm_logits(logits_hidden_states)
        if linear_states is not None:
            new_decode_state = {"linear_layers": new_linear_states}
        else:
            try:
                new_decode_state = self.get_decode_state()
            except AttributeError:
                new_decode_state = None
        hidden_features = tuple(captured[idx] for idx in sorted(selected_layers) if idx in captured)
        return logits, new_decode_state, hidden_features

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Filter out rotary embedding frequencies and tied LM head weights.

        Removes weight keys that should not be loaded, such as
        ``rotary_emb.inv_freq`` (computed at init) and the LM head weight
        when weight tying is enabled.

        Args:
            weights: Dictionary mapping parameter names to weight arrays.

        Returns:
            Sanitized weights dictionary with irrelevant keys removed.
        """
        weights = {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}
        if self._tie_word_embeddings:
            weights.pop(f"{self._lm_head_name}.weight", None)
        return weights

    def get_task_head(self) -> nn.Module | None:
        """Return the LM head module, or None if weight tying is enabled.

        Returns:
            The LM head ``nn.Linear`` module, or None when weights are tied
            with the input embedding layer.
        """
        if self._tie_word_embeddings:
            return None
        return getattr(self, self._lm_head_name, None)

    def get_lm_head(self) -> nn.Module | None:
        """Return the LM head module. Alias for ``get_task_head``.

        Returns:
            The LM head module, or None if weight tying is enabled.
        """
        return self.get_task_head()
