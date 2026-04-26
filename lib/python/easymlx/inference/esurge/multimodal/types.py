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

"""Multimodal data types used by the eSurge preprocessing layer.

Defines the immutable and mutable data classes that flow through the
multimodal preprocessing pipeline:

* :class:`PlaceholderRange` -- token-level span marking a single
  multimodal placeholder in the prompt.
* :class:`MultiModalFeature` -- per-image or per-video feature bundle
  including pixel values, grid metadata, and optional cached embeddings.
* :class:`BatchedMultiModalInputs` -- aggregated tensors ready for a
  batched model forward pass.
* :class:`MultimodalInput` -- top-level normalized input payload
  consumed by :class:`~easymlx.inference.esurge.multimodal.manager.MultimodalManager`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np


@dataclass(frozen=True, slots=True)
class PlaceholderRange:
    """Token range containing one multimodal placeholder span.

    Attributes:
        offset: Start token index of the placeholder in the prompt.
        length: Number of tokens the placeholder occupies.
        modality: The media modality (``"image"`` or ``"video"``).
    """

    offset: int
    length: int
    modality: str = "image"

    @property
    def end(self) -> int:
        """Return the exclusive end index of the placeholder span.

        Returns:
            ``offset + length``.
        """
        return int(self.offset) + int(self.length)


@dataclass(slots=True)
class MultiModalFeature:
    """Single image/video feature with optional cached embeddings.

    Each feature corresponds to exactly one image or one video clip and
    carries the raw pixel values, spatial/temporal grid metadata, an
    optional token-level placeholder range, and a content hash used for
    cache lookups.

    Attributes:
        mm_hash: MD5 hex digest of the pixel-value content.
        modality: ``"image"`` or ``"video"``.
        pixel_values: Preprocessed pixel array, or ``None`` after
            clearing.
        grid_thw: Optional ``(T, H, W)`` grid shape array.
        placeholder_range: Optional token placeholder span.
        cached_embeddings: Encoder output previously stored in the
            vision cache.
        request_idx: Index of the originating request.
    """

    mm_hash: str
    modality: str
    pixel_values: mx.array | np.ndarray | None
    grid_thw: mx.array | np.ndarray | None
    placeholder_range: PlaceholderRange | None = None
    cached_embeddings: Any | None = None
    request_idx: int = 0

    @staticmethod
    def _compute_hash(pixel_values: Any) -> str:
        """Compute an MD5 hash over pixel-value shape and content.

        For arrays with more than 1 million elements, a strided
        subsample is hashed to reduce latency.

        Args:
            pixel_values: Pixel array to hash.

        Returns:
            Hex-encoded MD5 digest string.
        """
        array = np.asarray(pixel_values)
        shape_bytes = np.asarray(array.shape, dtype=np.int32).tobytes()
        if array.size > 1_000_000:
            sampled = array.flat[::100]
            content_bytes = sampled.tobytes()
        else:
            content_bytes = array.tobytes()
        return hashlib.md5(shape_bytes + content_bytes).hexdigest()

    @classmethod
    def from_image(
        cls,
        pixel_values: mx.array | np.ndarray,
        grid_thw: mx.array | np.ndarray | None = None,
        placeholder_range: PlaceholderRange | None = None,
        request_idx: int = 0,
        mm_hash: str | None = None,
    ) -> MultiModalFeature:
        """Construct a feature for an image.

        Args:
            pixel_values: Preprocessed image pixel array.
            grid_thw: Optional spatial grid shape.
            placeholder_range: Optional token placeholder span.
            request_idx: Index of the originating request.

        Returns:
            A :class:`MultiModalFeature` with ``modality="image"``.
        """
        return cls(
            mm_hash=cls._compute_hash(pixel_values) if mm_hash is None else str(mm_hash),
            modality="image",
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            placeholder_range=placeholder_range,
            request_idx=int(request_idx),
        )

    @classmethod
    def from_video(
        cls,
        pixel_values: mx.array | np.ndarray,
        grid_thw: mx.array | np.ndarray | None = None,
        placeholder_range: PlaceholderRange | None = None,
        request_idx: int = 0,
        mm_hash: str | None = None,
    ) -> MultiModalFeature:
        """Construct a feature for a video clip.

        Args:
            pixel_values: Preprocessed video pixel array.
            grid_thw: Optional spatiotemporal grid shape.
            placeholder_range: Optional token placeholder span.
            request_idx: Index of the originating request.

        Returns:
            A :class:`MultiModalFeature` with ``modality="video"``.
        """
        return cls(
            mm_hash=cls._compute_hash(pixel_values) if mm_hash is None else str(mm_hash),
            modality="video",
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            placeholder_range=placeholder_range,
            request_idx=int(request_idx),
        )

    @property
    def has_cached_embeddings(self) -> bool:
        """Return whether cached encoder embeddings are available.

        Returns:
            ``True`` if ``cached_embeddings`` is not ``None``.
        """
        return self.cached_embeddings is not None

    def set_cached_embeddings(self, embeddings: Any) -> None:
        """Attach cached encoder embeddings to this feature.

        Args:
            embeddings: The encoder output to cache on this feature.
        """
        self.cached_embeddings = embeddings

    def clear_pixel_values(self) -> None:
        """Release pixel values and grid metadata to free memory.

        After calling this method, ``pixel_values`` and ``grid_thw``
        will both be ``None``.
        """
        self.pixel_values = None
        self.grid_thw = None


@dataclass(slots=True)
class BatchedMultiModalInputs:
    """Batched image/video tensors derived from multimodal features.

    Aggregates preprocessed pixel values and grid metadata across all
    images and videos in a request batch, along with mappings from
    request indices back to individual feature indices.

    Attributes:
        pixel_values: Concatenated image pixel values, or ``None``.
        image_grid_thw: Concatenated image grid shapes, or ``None``.
        pixel_values_videos: Concatenated video pixel values, or
            ``None``.
        video_grid_thw: Concatenated video grid shapes, or ``None``.
        image_features: Per-image :class:`MultiModalFeature` objects.
        video_features: Per-video :class:`MultiModalFeature` objects.
        request_to_image_indices: Mapping from request index to the
            indices within ``image_features``.
        request_to_video_indices: Mapping from request index to the
            indices within ``video_features``.
    """

    pixel_values: mx.array | np.ndarray | None = None
    image_grid_thw: mx.array | np.ndarray | None = None
    pixel_values_videos: mx.array | np.ndarray | None = None
    video_grid_thw: mx.array | np.ndarray | None = None
    image_features: list[MultiModalFeature] = field(default_factory=list)
    video_features: list[MultiModalFeature] = field(default_factory=list)
    request_to_image_indices: dict[int, list["int"]] = field(default_factory=dict)
    request_to_video_indices: dict[int, list["int"]] = field(default_factory=dict)

    @classmethod
    def from_features(cls, features: list[MultiModalFeature]) -> BatchedMultiModalInputs:
        """Construct a batched input from a list of individual features.

        Features are partitioned by modality, and their pixel values and
        grid arrays are concatenated along the leading axis.

        Args:
            features: List of :class:`MultiModalFeature` instances.

        Returns:
            A populated :class:`BatchedMultiModalInputs`.
        """
        image_features: list[MultiModalFeature] = []
        video_features: list[MultiModalFeature] = []
        request_to_image_indices: dict[int, list["int"]] = {}
        request_to_video_indices: dict[int, list["int"]] = {}

        for feature in features:
            if feature.pixel_values is None:
                continue
            if feature.modality == "image":
                idx = len(image_features)
                image_features.append(feature)
                request_to_image_indices.setdefault(int(feature.request_idx), []).append(idx)
            elif feature.modality == "video":
                idx = len(video_features)
                video_features.append(feature)
                request_to_video_indices.setdefault(int(feature.request_idx), []).append(idx)

        pixel_values = None
        image_grid_thw = None
        if image_features:
            image_pixels = [f.pixel_values for f in image_features if f.pixel_values is not None]
            if image_pixels:
                pixel_values = mx.concatenate(
                    [mx.array(pv) if not isinstance(pv, mx.array) else pv for pv in image_pixels], axis=0
                )
            image_grids = [f.grid_thw for f in image_features if f.grid_thw is not None]
            if image_grids:
                image_grid_thw = mx.concatenate(
                    [mx.array(grid) if not isinstance(grid, mx.array) else grid for grid in image_grids],
                    axis=0,
                )

        pixel_values_videos = None
        video_grid_thw = None
        if video_features:
            video_pixels = [f.pixel_values for f in video_features if f.pixel_values is not None]
            if video_pixels:
                pixel_values_videos = mx.concatenate(
                    [mx.array(pv) if not isinstance(pv, mx.array) else pv for pv in video_pixels],
                    axis=0,
                )
            video_grids = [f.grid_thw for f in video_features if f.grid_thw is not None]
            if video_grids:
                video_grid_thw = mx.concatenate(
                    [mx.array(grid) if not isinstance(grid, mx.array) else grid for grid in video_grids],
                    axis=0,
                )

        return cls(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            image_features=image_features,
            video_features=video_features,
            request_to_image_indices=request_to_image_indices,
            request_to_video_indices=request_to_video_indices,
        )

    @classmethod
    def empty(cls) -> BatchedMultiModalInputs:
        """Return an empty :class:`BatchedMultiModalInputs` with no features.

        Returns:
            A default-constructed instance with all arrays set to
            ``None`` and empty feature lists.
        """
        return cls()

    @property
    def has_images(self) -> bool:
        """Return whether the batch contains any image data.

        Returns:
            ``True`` if ``pixel_values`` is not ``None``.
        """
        return self.pixel_values is not None

    @property
    def has_videos(self) -> bool:
        """Return whether the batch contains any video data.

        Returns:
            ``True`` if ``pixel_values_videos`` is not ``None``.
        """
        return self.pixel_values_videos is not None

    @property
    def has_vision(self) -> bool:
        """Return whether the batch contains any vision data.

        Returns:
            ``True`` if images or videos are present.
        """
        return self.has_images or self.has_videos

    @property
    def num_images(self) -> int:
        """Return the total number of images in the batch.

        Returns:
            Length of ``image_features``.
        """
        return len(self.image_features)

    @property
    def num_videos(self) -> int:
        """Return the total number of videos in the batch.

        Returns:
            Length of ``video_features``.
        """
        return len(self.video_features)

    def get_request_image_count(self, request_idx: int) -> int:
        """Return the number of images belonging to a specific request.

        Args:
            request_idx: The request index to query.

        Returns:
            Count of images for the given request.
        """
        return len(self.request_to_image_indices.get(int(request_idx), []))

    def get_request_video_count(self, request_idx: int) -> int:
        """Return the number of videos belonging to a specific request.

        Args:
            request_idx: The request index to query.

        Returns:
            Count of videos for the given request.
        """
        return len(self.request_to_video_indices.get(int(request_idx), []))


@dataclass(slots=True)
class MultimodalInput:
    """Normalized multimodal input payload used by the manager.

    Serves as both the initial raw input to
    :meth:`~easymlx.inference.esurge.multimodal.manager.MultimodalManager.prepare`
    and the enriched output with populated ``features`` and ``batched``
    fields.

    Attributes:
        text: Optional plain-text prompt.
        media: Optional raw media items in heterogeneous formats.
        messages: Optional chat-style message list (OpenAI format).
        images: Optional list of normalized PIL images.
        videos: Optional list of normalized ``(T, H, W, 3)`` arrays.
        features: Preprocessed :class:`MultiModalFeature` instances.
        batched: Aggregated :class:`BatchedMultiModalInputs` for the
            model runner.
    """

    text: str | None = None
    media: list[Any] | None = None
    messages: list[dict[str, Any]] | None = None
    images: list[Any] | None = None
    videos: list[np.ndarray] | None = None
    features: list[MultiModalFeature] | None = None
    batched: BatchedMultiModalInputs | None = None


__all__ = ("BatchedMultiModalInputs", "MultiModalFeature", "MultimodalInput", "PlaceholderRange")
