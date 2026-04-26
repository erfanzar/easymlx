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

"""Operation cache requirements mixin for easymlx modules.

Provides mixins and data classes for discovering and managing operation cache
requirements in easymlx neural network models.  Enables dynamic discovery of
cache types needed by different layer operations, supporting standard
transformer attention, paged attention, recurrent/linear attention, and
hybrid architectures.

Primary components:

- ``LayerOperationInfo``: Information about a single layer's operation and cache needs.
- ``OperationsCacheInfo``: Aggregated cache requirements for all model layers.
- ``OperationCacheMixin``: Mixin class providing cache discovery methods for models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from easymlx.infra.etils import DEFAULT_ATTENTION_MECHANISM, canonical_attention_mechanism
from easymlx.operations import OperationRegistry
from easymlx.operations.requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
)

__all__ = [
    "LayerOperationInfo",
    "OperationCacheMixin",
    "OperationsCacheInfo",
]


@dataclass
class LayerOperationInfo:
    """Information about a single layer's operation and cache requirements.

    Attributes:
        layer_index: Index of the layer in the model (0-based).
        slot: Unique slot identifier for this operation within the layer.
        layer_type: Type of layer (e.g., ``"full_attention"``,
            ``"linear_attention"``).
        operation_name: Name of the prefill operation implementation.
        requirements: Operation requirements (metadata and cache).
        supported_cache_types: Cache types supported by this operation.
        decode_operation_name: Name of the decode operation (if
            different from prefill).
        requires_kv_cache: Whether this layer requires KV cache for
            attention.
        requires_state_cache: Whether this layer requires recurrent
            state cache.
    """

    layer_index: int
    slot: str | None
    layer_type: str
    operation_name: str
    requirements: OperationRequirements
    supported_cache_types: CacheType
    decode_operation_name: str | None = None
    requires_kv_cache: bool = True
    requires_state_cache: bool = False

    @property
    def is_attention_layer(self) -> bool:
        """Check if this layer uses standard attention caching.

        Returns:
            ``True`` if the layer supports ``TRANSFORMER`` or ``PAGED``
            cache types.
        """
        return CacheType.TRANSFORMER in self.supported_cache_types or CacheType.PAGED in self.supported_cache_types

    @property
    def is_recurrent_layer(self) -> bool:
        """Check if this layer uses recurrent-style caching exclusively.

        Returns:
            ``True`` if the layer supports ``RECURRENT`` cache but not
            standard attention caches (``TRANSFORMER`` or ``PAGED``).
        """
        supports_recurrent = CacheType.RECURRENT in self.supported_cache_types
        supports_attention = (
            CacheType.TRANSFORMER in self.supported_cache_types or CacheType.PAGED in self.supported_cache_types
        )
        return supports_recurrent and not supports_attention

    @property
    def has_separate_decode(self) -> bool:
        """Check whether prefill and decode use different operations.

        Returns:
            ``True`` if ``decode_operation_name`` differs from
            ``operation_name``.
        """
        return self.decode_operation_name is not None and self.decode_operation_name != self.operation_name


@dataclass
class OperationsCacheInfo:
    """Aggregated cache requirements for all layers in a model.

    Attributes:
        layers: List of :class:`LayerOperationInfo` for each layer.
        prefill_operation: Name of the operation used for prefill.
        decode_operation: Name of the operation used for decode.
        combined_cache_types: Intersection of cache types that work
            for all operations.
        combined_metadata: Union of metadata requirements across layers.
        is_hybrid_model: Whether the model mixes attention and
            recurrent layer types.
        supports_paged: Whether all operations support ``PageCacheView``.
        supports_transformer_cache: Whether all operations support
            ``TransformerCache``.
        requires_hybrid_cache: Whether the model needs ``HybridCache``
            (has both attention and recurrent layers).
        requires_state_management: Whether any layer needs recurrent
            state.
        has_separate_decode_ops: Whether any layer uses a different
            decode operation.
    """

    layers: list[LayerOperationInfo] = field(default_factory=list)
    prefill_operation: str = ""
    decode_operation: str = ""
    combined_cache_types: CacheType = field(default_factory=CacheType.any)
    combined_metadata: MetadataField = field(default_factory=MetadataField.basic)
    is_hybrid_model: bool = False
    supports_paged: bool = False
    supports_transformer_cache: bool = False
    requires_hybrid_cache: bool = False
    requires_state_management: bool = False
    has_separate_decode_ops: bool = False

    @property
    def num_attention_layers(self) -> int:
        """Count of attention-based layers.

        Returns:
            The number of layers where :attr:`LayerOperationInfo.is_attention_layer`
            is ``True``.
        """
        return sum(1 for layer in self.layers if layer.is_attention_layer)

    @property
    def num_recurrent_layers(self) -> int:
        """Count of recurrent/linear attention layers.

        Returns:
            The number of layers where :attr:`LayerOperationInfo.is_recurrent_layer`
            is ``True``.
        """
        return sum(1 for layer in self.layers if layer.is_recurrent_layer)

    @property
    def attention_ratio(self) -> float:
        """Ratio of attention layers to total layers.

        Returns:
            A float in ``[0.0, 1.0]``. Returns ``1.0`` if there are
            no layers.
        """
        if not self.layers:
            return 1.0
        return self.num_attention_layers / len(self.layers)

    def get_recommended_cache_type(self) -> str:
        """Get recommended cache type based on model requirements.

        Priority order:

        1. ``"hybrid"`` -- if model has recurrent layers requiring
           state management.
        2. ``"transformer"`` -- if supported (simplest, most
           compatible).
        3. ``"paged"`` -- if transformer not supported but paged is.

        Returns:
            One of ``"hybrid"``, ``"transformer"``, or ``"paged"``.
        """
        if self.requires_hybrid_cache or self.requires_state_management:
            return "hybrid"
        if self.supports_transformer_cache:
            return "transformer"
        if self.supports_paged:
            return "paged"
        return "transformer"

    def get_layer_by_index(self, index: int) -> LayerOperationInfo | None:
        """Get the first layer info matching a given index.

        Args:
            index: The layer index to look up.

        Returns:
            The matching :class:`LayerOperationInfo`, or ``None`` if
            not found.
        """
        for layer in self.layers:
            if layer.layer_index == index:
                return layer
        return None

    def get_layers_by_index(self, index: int) -> list[LayerOperationInfo]:
        """Get all layer infos for a given index (slot-aware).

        Args:
            index: The layer index to look up.

        Returns:
            A list of :class:`LayerOperationInfo` entries matching the
            index.
        """
        return [layer for layer in self.layers if layer.layer_index == index]


class OperationCacheMixin:
    """Mixin that provides operation cache requirements discovery.

    This mixin adds methods to :class:`EasyMLXBaseModule` for discovering
    what operations are used and what cache types they require. It supports
    both dynamic discovery (walking the module graph) and static discovery
    (reading from configuration).

    The config-based approach is the *primary* method for easymlx since
    models are simpler. Dynamic discovery is available as a fallback for
    complex architectures.
    """

    def _get_operation_requirements(
        self,
        name: str | None,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements | None:
        """Get requirements for an operation by name from the registry.

        Args:
            name: The operation name to look up (case-insensitive).
                If ``None``, returns ``None`` immediately.
            mode: The execution mode for requirements lookup.

        Returns:
            The :class:`OperationRequirements` for the named operation,
            or ``None`` if the operation is not registered or the lookup
            fails.
        """
        if name is None:
            return None
        try:
            op_class: type | None = OperationRegistry.get(name.lower())
            if op_class is not None:
                return op_class.get_requirements(mode)
        except (KeyError, ValueError):
            pass
        return None

    def get_operations_cache_info(
        self,
        mode: ExecutionMode = ExecutionMode.MIXED,
        dynamic: bool = False,
    ) -> OperationsCacheInfo:
        """Get complete information about operations and their cache requirements.

        Args:
            mode: Execution mode for requirements lookup.
            dynamic: If ``True``, discover from actual module graph
                first. Defaults to ``False`` (config-based is primary
                for easymlx).

        Returns:
            An :class:`OperationsCacheInfo` with per-layer details and
            recommendations.
        """
        if dynamic:
            info = self.get_operations_cache_info_dynamic(mode)
            if info.layers:
                return info
        return self._get_operations_cache_info_from_config(mode)

    def _get_operations_cache_info_from_config(
        self,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationsCacheInfo:
        """Get cache info by reading from model configuration.

        Reads ``config.attn_mechanism``, ``config.decode_attn_mechanism``,
        ``config.layer_types``, and ``config.num_hidden_layers`` to
        construct per-layer cache requirements without walking the
        module graph.

        Args:
            mode: The execution mode (unused directly, but passed to
                operation requirements lookup).

        Returns:
            An :class:`OperationsCacheInfo` derived from config values.
        """
        config = getattr(self, "config", None)
        if config is None:
            return OperationsCacheInfo()

        attn_mechanism = getattr(config, "attn_mechanism", None)
        decode_attn_mechanism = getattr(config, "decode_attn_mechanism", None)

        if attn_mechanism is not None and hasattr(attn_mechanism, "value"):
            attn_mechanism = str(attn_mechanism.value)
        if decode_attn_mechanism is not None and hasattr(decode_attn_mechanism, "value"):
            decode_attn_mechanism = str(decode_attn_mechanism.value)

        prefill_op = canonical_attention_mechanism(attn_mechanism) or canonical_attention_mechanism(
            DEFAULT_ATTENTION_MECHANISM
        )
        decode_op = canonical_attention_mechanism(decode_attn_mechanism) or prefill_op

        layer_types = getattr(config, "layer_types", None)
        num_hidden_layers = getattr(config, "num_hidden_layers", 1)

        layers: list[LayerOperationInfo] = []
        combined_cache = CacheType.any()
        combined_metadata = MetadataField.NONE
        has_attention = False
        has_recurrent = False

        if layer_types is not None:
            for idx, layer_type in enumerate(layer_types):
                prefill_reqs = self._get_operation_requirements(layer_type, ExecutionMode.PREFILL)
                op_name = layer_type if prefill_reqs is not None else prefill_op

                if prefill_reqs is None:
                    prefill_reqs = self._get_operation_requirements(prefill_op, ExecutionMode.PREFILL)
                if prefill_reqs is None:
                    prefill_reqs = OperationRequirements.default(op_name)

                layer_decode_op = decode_op if decode_op != prefill_op else op_name
                decode_reqs = self._get_operation_requirements(layer_decode_op, ExecutionMode.DECODE)
                if decode_reqs is None:
                    decode_reqs = OperationRequirements.default(layer_decode_op)

                reqs = prefill_reqs | decode_reqs

                prefill_supports_attention = CacheType.TRANSFORMER in _supported_caches(
                    prefill_reqs
                ) or CacheType.PAGED in _supported_caches(prefill_reqs)
                decode_supports_attention = CacheType.TRANSFORMER in _supported_caches(
                    decode_reqs
                ) or CacheType.PAGED in _supported_caches(decode_reqs)
                prefill_supports_recurrent = CacheType.RECURRENT in _supported_caches(prefill_reqs)
                decode_supports_recurrent = CacheType.RECURRENT in _supported_caches(decode_reqs)
                prefill_is_recurrent = prefill_supports_recurrent and not prefill_supports_attention
                decode_is_recurrent = decode_supports_recurrent and not decode_supports_attention

                layer_info = LayerOperationInfo(
                    layer_index=idx,
                    slot=None,
                    layer_type=layer_type,
                    operation_name=op_name,
                    decode_operation_name=layer_decode_op if layer_decode_op != op_name else None,
                    requirements=reqs,
                    supported_cache_types=_supported_caches(reqs),
                    requires_kv_cache=(
                        (prefill_reqs.requires_cache and prefill_supports_attention)
                        or (decode_reqs.requires_cache and decode_supports_attention)
                    ),
                    requires_state_cache=prefill_is_recurrent or decode_is_recurrent,
                )
                layers.append(layer_info)

                combined_cache &= _supported_caches(reqs)
                combined_metadata |= reqs.required_metadata

                if prefill_supports_attention or decode_supports_attention:
                    has_attention = True
                if prefill_is_recurrent or decode_is_recurrent:
                    has_recurrent = True
        else:
            prefill_reqs = self._get_operation_requirements(prefill_op, ExecutionMode.PREFILL)
            decode_reqs = self._get_operation_requirements(decode_op, ExecutionMode.DECODE)

            if prefill_reqs is None:
                prefill_reqs = OperationRequirements.default(prefill_op)
            if decode_reqs is None:
                decode_reqs = OperationRequirements.default(decode_op)

            reqs = prefill_reqs | decode_reqs

            prefill_supports_attention = CacheType.TRANSFORMER in _supported_caches(
                prefill_reqs
            ) or CacheType.PAGED in _supported_caches(prefill_reqs)
            decode_supports_attention = CacheType.TRANSFORMER in _supported_caches(
                decode_reqs
            ) or CacheType.PAGED in _supported_caches(decode_reqs)
            prefill_supports_recurrent = CacheType.RECURRENT in _supported_caches(prefill_reqs)
            decode_supports_recurrent = CacheType.RECURRENT in _supported_caches(decode_reqs)
            prefill_is_recurrent = prefill_supports_recurrent and not prefill_supports_attention
            decode_is_recurrent = decode_supports_recurrent and not decode_supports_attention

            for idx in range(num_hidden_layers):
                layer_info = LayerOperationInfo(
                    layer_index=idx,
                    slot=None,
                    layer_type=(
                        "attention" if (prefill_supports_attention or decode_supports_attention) else "recurrent"
                    ),
                    operation_name=prefill_op,
                    decode_operation_name=decode_op if decode_op != prefill_op else None,
                    requirements=reqs,
                    supported_cache_types=_supported_caches(reqs),
                    requires_kv_cache=(
                        (prefill_reqs.requires_cache and prefill_supports_attention)
                        or (decode_reqs.requires_cache and decode_supports_attention)
                    ),
                    requires_state_cache=prefill_is_recurrent or decode_is_recurrent,
                )
                layers.append(layer_info)

            combined_cache = _supported_caches(reqs)
            combined_metadata = reqs.required_metadata
            has_attention = prefill_supports_attention or decode_supports_attention
            has_recurrent = prefill_is_recurrent or decode_is_recurrent

        supports_paged = CacheType.PAGED in combined_cache
        supports_transformer = CacheType.TRANSFORMER in combined_cache
        needs_hybrid = has_attention and has_recurrent

        return OperationsCacheInfo(
            layers=layers,
            prefill_operation=prefill_op,
            decode_operation=decode_op,
            combined_cache_types=combined_cache,
            combined_metadata=combined_metadata,
            is_hybrid_model=has_attention and has_recurrent,
            supports_paged=supports_paged,
            supports_transformer_cache=supports_transformer,
            requires_hybrid_cache=needs_hybrid,
            requires_state_management=has_recurrent,
            has_separate_decode_ops=decode_op != prefill_op,
        )

    def get_operations_cache_info_dynamic(
        self,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationsCacheInfo:
        """Discover operations dynamically by walking the module graph.

        Walks the model's layer list (``layers``, ``h``, or ``blocks``)
        and inspects each layer for ``AttentionPerformer``,
        ``OperationExecutor``, and ``BaseOperation`` instances.

        Args:
            mode: The execution mode for requirements lookup.

        Returns:
            An :class:`OperationsCacheInfo` built from dynamically
            discovered layers. May have an empty ``layers`` list if
            the model has no recognizable layer container.
        """
        import mlx.nn as nn

        layers: list[LayerOperationInfo] = []
        seen_op_ids: set[int] = set()
        has_separate_decode = False

        model_layers: list | None = None
        for attr_name in ("layers", "h", "blocks"):
            candidate = getattr(self, attr_name, None)
            if isinstance(candidate, list) and candidate:
                model_layers = candidate
                break

        if model_layers is None:
            return OperationsCacheInfo()

        for layer_idx, layer_module in enumerate(model_layers):
            if not isinstance(layer_module, nn.Module):
                continue
            self._walk_layer_for_ops(
                layer_module,
                layer_idx,
                mode,
                layers,
                seen_op_ids,
            )

        layers.sort(key=lambda x: (x.layer_index, x.slot or "", x.operation_name))
        has_separate_decode = any(layer.has_separate_decode for layer in layers)
        return self._build_cache_info_from_layers(layers, has_separate_decode)

    def _walk_layer_for_ops(
        self,
        module,
        layer_idx: int,
        mode: ExecutionMode,
        layers: list[LayerOperationInfo],
        seen_op_ids: set[int],
    ) -> None:
        """Recursively walk a module looking for operation instances.

        Inspects *module* for ``AttentionPerformer``,
        ``OperationExecutor``, and ``BaseOperation`` instances. Child
        ``nn.Module`` instances are recursed into.

        Args:
            module: The ``nn.Module`` to inspect.
            layer_idx: The layer index to assign to discovered
                operations.
            mode: Execution mode for requirements lookup.
            layers: Accumulator list; discovered
                :class:`LayerOperationInfo` entries are appended here.
            seen_op_ids: Set of ``id()`` values for already-visited
                operation objects, used to avoid duplicates.
        """
        import mlx.nn as nn

        from easymlx.layers.attention import AttentionPerformer
        from easymlx.operations._base_operation import BaseOperation
        from easymlx.operations.executor import OperationExecutor

        for attr_name in vars(module):
            attr_value = getattr(module, attr_name, None)
            if attr_value is None:
                continue

            if isinstance(attr_value, AttentionPerformer):
                if id(attr_value) in seen_op_ids:
                    continue
                seen_op_ids.add(id(attr_value))
                reqs = OperationRequirements.default("attention_performer")
                layer_info = LayerOperationInfo(
                    layer_index=layer_idx,
                    slot=None,
                    layer_type="full_attention",
                    operation_name="attention_performer",
                    requirements=reqs,
                    supported_cache_types=CacheType.TRANSFORMER | CacheType.PAGED,
                    requires_kv_cache=True,
                    requires_state_cache=False,
                )
                layers.append(layer_info)
                continue

            if isinstance(attr_value, OperationExecutor):
                if id(attr_value) in seen_op_ids:
                    continue
                seen_op_ids.add(id(attr_value))
                reqs = attr_value.get_combined_requirements()
                op = attr_value.get_operation(mode)
                if op is not None:
                    op_name = op.get_impl_name()
                    if isinstance(op_name, tuple):
                        op_name = op_name[0]
                    decode_op = attr_value.decode_operation
                    decode_op_name = None
                    if decode_op is not None and id(decode_op) != id(op):
                        dname = decode_op.get_impl_name()
                        decode_op_name = dname[0] if isinstance(dname, tuple) else dname

                    supports_attention = CacheType.TRANSFORMER in _supported_caches(
                        reqs
                    ) or CacheType.PAGED in _supported_caches(reqs)
                    supports_recurrent = CacheType.RECURRENT in _supported_caches(reqs)
                    is_recurrent = supports_recurrent and not supports_attention

                    layer_info = LayerOperationInfo(
                        layer_index=layer_idx,
                        slot=None,
                        layer_type="linear_attention" if is_recurrent else "full_attention",
                        operation_name=op_name,
                        decode_operation_name=decode_op_name,
                        requirements=reqs,
                        supported_cache_types=_supported_caches(reqs),
                        requires_kv_cache=reqs.requires_cache and supports_attention,
                        requires_state_cache=is_recurrent,
                    )
                    layers.append(layer_info)
                continue

            if isinstance(attr_value, BaseOperation):
                if id(attr_value) in seen_op_ids:
                    continue
                seen_op_ids.add(id(attr_value))
                reqs = attr_value.get_requirements(mode)
                op_name = attr_value.get_impl_name()
                if isinstance(op_name, tuple):
                    op_name = op_name[0]

                supports_attention = CacheType.TRANSFORMER in _supported_caches(
                    reqs
                ) or CacheType.PAGED in _supported_caches(reqs)
                supports_recurrent = CacheType.RECURRENT in _supported_caches(reqs)
                is_recurrent = supports_recurrent and not supports_attention

                layer_info = LayerOperationInfo(
                    layer_index=layer_idx,
                    slot=None,
                    layer_type="linear_attention" if is_recurrent else "full_attention",
                    operation_name=op_name,
                    requirements=reqs,
                    supported_cache_types=_supported_caches(reqs),
                    requires_kv_cache=reqs.requires_cache and supports_attention,
                    requires_state_cache=is_recurrent,
                )
                layers.append(layer_info)
                continue

            if isinstance(attr_value, nn.Module):
                self._walk_layer_for_ops(
                    attr_value,
                    layer_idx,
                    mode,
                    layers,
                    seen_op_ids,
                )

    def _build_cache_info_from_layers(
        self,
        layers: list[LayerOperationInfo],
        has_separate_decode: bool = False,
    ) -> OperationsCacheInfo:
        """Build an :class:`OperationsCacheInfo` from discovered layers.

        Args:
            layers: The list of per-layer operation info entries.
            has_separate_decode: Whether any layer uses different
                decode operations.

        Returns:
            A fully populated :class:`OperationsCacheInfo`.
        """
        if not layers:
            return OperationsCacheInfo()

        combined_cache = CacheType.any()
        combined_metadata = MetadataField.NONE
        has_attention = False
        has_recurrent = False

        for layer in layers:
            combined_cache &= layer.supported_cache_types
            combined_metadata |= layer.requirements.required_metadata

            if layer.is_attention_layer:
                has_attention = True
            if layer.is_recurrent_layer:
                has_recurrent = True

        prefill_op = layers[0].operation_name if layers else ""
        decode_op = layers[0].decode_operation_name or prefill_op if layers else ""

        supports_paged = CacheType.PAGED in combined_cache
        supports_transformer = CacheType.TRANSFORMER in combined_cache

        return OperationsCacheInfo(
            layers=layers,
            prefill_operation=prefill_op,
            decode_operation=decode_op,
            combined_cache_types=combined_cache,
            combined_metadata=combined_metadata,
            is_hybrid_model=has_attention and has_recurrent,
            supports_paged=supports_paged,
            supports_transformer_cache=supports_transformer,
            requires_hybrid_cache=has_attention and has_recurrent,
            requires_state_management=has_recurrent,
            has_separate_decode_ops=has_separate_decode,
        )

    def get_layer_cache_requirements(self, layer_index: int) -> LayerOperationInfo | None:
        """Get cache requirements for a specific layer by index.

        Args:
            layer_index: The 0-based layer index.

        Returns:
            The :class:`LayerOperationInfo` for the layer, or ``None``
            if not found.
        """
        cache_info = self.get_operations_cache_info()
        return cache_info.get_layer_by_index(layer_index)

    def get_required_cache_class(self) -> type:
        """Get the required cache class based on operation requirements.

        Inspects the ``cache_view_cls`` of each layer's requirements to
        determine whether the model needs a ``TransformerCache``,
        ``RecurrentCache``, or ``HybridCache``.

        Returns:
            One of ``TransformerCache``, ``RecurrentCache``, or
            ``HybridCache``.
        """
        cache_info = self.get_operations_cache_info()
        cache_view_classes: set[type | None] = set()

        for layer in cache_info.layers:
            cache_view_cls = layer.requirements.cache_view_cls
            if cache_view_cls is not None:
                cache_view_classes.add(cache_view_cls)

        from easymlx.caching import (
            HybridCache,
            RecurrentCache,
            RecurrentCacheView,
            TransformerCache,
            TransformerCacheView,
        )

        has_transformer = any(issubclass(cls, TransformerCacheView) for cls in cache_view_classes if cls is not None)
        has_recurrent = any(issubclass(cls, RecurrentCacheView) for cls in cache_view_classes if cls is not None)

        if not has_transformer and has_recurrent:
            return RecurrentCache
        elif has_transformer and not has_recurrent:
            return TransformerCache
        else:
            return HybridCache

    def get_operations_cache_view(self) -> dict[int, type]:
        """Get the cache view class required for each layer.

        Returns:
            A dictionary mapping layer indices to their required cache
            view class. Only layers with an explicit ``cache_view_cls``
            are included.
        """
        cache_info = self.get_operations_cache_info()
        result: dict[int, type] = {}

        for op_info in cache_info.layers:
            cache_view_cls = op_info.requirements.cache_view_cls
            if cache_view_cls is not None:
                result.setdefault(op_info.layer_index, cache_view_cls)

        return result

    def get_unique_cache_view_classes(self) -> set[type]:
        """Get all unique cache view classes used by this model.

        Returns:
            A set of cache view class types used across all layers.
        """
        cache_info = self.get_operations_cache_info()
        cache_view_classes: set[type] = set()

        for layer in cache_info.layers:
            cache_view_cls = layer.requirements.cache_view_cls
            if cache_view_cls is not None:
                cache_view_classes.add(cache_view_cls)

        return cache_view_classes


def _supported_caches(reqs: OperationRequirements) -> CacheType:
    """Infer supported cache types from an :class:`OperationRequirements`.

    easymlx ``OperationRequirements`` is flat (no nested
    ``.cache.supported``). This helper derives supported ``CacheType``
    flags from the ``cache_view_cls`` and ``requires_cache`` fields.
    When neither is set, it conservatively assumes
    ``TRANSFORMER | PAGED`` support (the common case for vanilla
    attention).

    Args:
        reqs: The operation requirements to inspect.

    Returns:
        A ``CacheType`` flag set representing the supported cache
        backends.
    """
    from easymlx.caching import (
        HybridCacheView,
        PageCacheView,
        RecurrentCacheView,
        TransformerCacheView,
    )

    cls = reqs.cache_view_cls
    if cls is None:
        return CacheType.TRANSFORMER | CacheType.PAGED

    result = CacheType.NONE
    if issubclass(cls, TransformerCacheView):
        result |= CacheType.TRANSFORMER | CacheType.PAGED
    if issubclass(cls, PageCacheView):
        result |= CacheType.PAGED
    if issubclass(cls, RecurrentCacheView):
        result |= CacheType.RECURRENT
    if issubclass(cls, HybridCacheView):
        result |= CacheType.HYBRID | CacheType.TRANSFORMER | CacheType.RECURRENT

    if result == CacheType.NONE:
        return CacheType.TRANSFORMER | CacheType.PAGED

    return result
