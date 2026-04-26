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

"""Multimodal preprocessing manager for eSurge vision-language inputs.

This module implements the full multimodal preprocessing pipeline used by
the eSurge serving runtime.  It handles:

* Normalizing heterogeneous image/video payloads (PIL images, NumPy
  arrays, base64 data URIs, local file paths) into canonical formats.
* Bucket-based resolution alignment for efficient batching.
* CLIP-style RGB normalization and spatiotemporal patchification for
  models that accept flat patch inputs (e.g. Qwen2-VL, GLM-4V).
* Optional processor-delegated preprocessing with automatic fallback.
* An LRU embedding cache (:class:`VisionEncoderCache`) to avoid
  redundant encoder forward passes.

The public entry point is :meth:`MultimodalManager.prepare`, which
accepts a :class:`MultimodalInput` and returns an enriched copy with
preprocessed features and batched tensors ready for the model runner.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image

from .cache import VisionEncoderCache
from .types import BatchedMultiModalInputs, MultiModalFeature, MultimodalInput

CLIP_IMAGE_MEAN = mx.array([0.48145466, 0.4578275, 0.40821073], dtype=mx.float32)
CLIP_IMAGE_STD = mx.array([0.26862954, 0.26130258, 0.27577711], dtype=mx.float32)
DEFAULT_RESOLUTION_BUCKETS = [(32, 32), (64, 64), (128, 128), (384, 384), (512, 512), (768, 768), (1024, 1024)]


class MultimodalManager:
    """Real multimodal preprocessing layer inspired by EasyDeL's manager.

    Orchestrates image and video normalization, optional processor-based
    preprocessing, spatiotemporal patchification fallback, and feature
    caching for the eSurge serving runtime.

    Attributes:
        processor: Optional HuggingFace-style processor used for
            image/video preprocessing.
        model: Optional model instance whose vision config guides
            resolution buckets and patch parameters.
        resolution_buckets: Sorted list of ``(height, width)`` tuples
            used for resolution alignment.
        cache: Optional :class:`VisionEncoderCache` for embedding reuse.
    """

    def __init__(
        self,
        processor: Any | None = None,
        model: Any | None = None,
        resolution_buckets: list[tuple[int, int]] | None = None,
        cache_capacity_mb: int = 1024,
        enable_cache: bool = True,
    ):
        """Initialize the multimodal manager.

        Args:
            processor: HuggingFace-compatible image/video processor.
                When provided, preprocessing is delegated to the
                processor before falling back to manual patchification.
            model: Model instance that exposes a ``config`` attribute
                with an optional ``vision_config`` sub-config.
            resolution_buckets: Custom resolution buckets.  When
                ``None``, :data:`DEFAULT_RESOLUTION_BUCKETS` is used.
            cache_capacity_mb: Vision encoder cache capacity in
                megabytes.
            enable_cache: Whether to enable the embedding cache.
        """
        self.processor = processor
        self.model = model
        self.resolution_buckets = resolution_buckets or list(DEFAULT_RESOLUTION_BUCKETS)
        self.cache = VisionEncoderCache(cache_capacity_mb) if enable_cache else None

    def supported(self) -> bool:
        """Return whether multimodal inputs are supported.

        Returns:
            Always ``True`` for this manager implementation.
        """
        return True

    def _get_vision_config(self) -> Any | None:
        """Retrieve the vision sub-config from the model.

        Returns:
            The vision config object, or ``None`` if unavailable.
        """
        if self.model is None:
            return None
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return None
        if hasattr(cfg, "get_vision_config"):
            try:
                return cfg.get_vision_config()
            except Exception:
                pass
        return getattr(cfg, "vision_config", None)

    def _supports_flat_patch_inputs(self) -> bool:
        """Check whether the model accepts flat spatiotemporal patch tensors.

        Returns:
            ``True`` if the model type is one of the known flat-patch
            architectures (e.g. ``qwen2_vl``, ``glm4v``).
        """
        if self.model is None:
            return False
        model_type = str(getattr(getattr(self.model, "config", None), "model_type", "")).lower()
        return model_type in {
            "glm4v",
            "glm4v_moe",
            "glm46v",
            "qwen2_vl",
            "qwen3_vl",
            "qwen3_vl_moe",
        }

    @staticmethod
    def _align_dim_to_multiple(dim: int, multiple: int) -> int:
        """Round *dim* to the nearest multiple of *multiple*.

        Args:
            dim: The dimension value to align.
            multiple: The alignment factor.  Values ``<= 1`` cause *dim*
                to be returned unchanged (clamped to at least ``1``).

        Returns:
            The aligned dimension value.
        """
        dim = int(dim)
        multiple = int(multiple)
        if multiple <= 1:
            return max(1, dim)
        lower = (dim // multiple) * multiple
        upper = lower + multiple
        if lower <= 0:
            lower = multiple
        return lower if (dim - lower) < (upper - dim) else upper

    def _get_resize_buckets_for_model(self) -> list[tuple[int, int]]:
        """Build the effective resolution bucket list for the current model.

        Resolution buckets are augmented with the model's native
        ``image_size`` and, for flat-patch models, aligned to the
        ``patch_size * spatial_merge_size`` multiple.

        Returns:
            Sorted, deduplicated list of ``(height, width)`` tuples.
        """
        buckets = [(int(h), int(w)) for h, w in self.resolution_buckets]
        vision_cfg = self._get_vision_config()
        image_size = getattr(vision_cfg, "image_size", None) if vision_cfg is not None else None
        if isinstance(image_size, (int, float)) and int(image_size) > 0:
            sz = int(image_size)
            buckets.append((sz, sz))

        if self._supports_flat_patch_inputs() and vision_cfg is not None:
            patch_size = int(getattr(vision_cfg, "patch_size", 1) or 1)
            spatial_merge_size = int(getattr(vision_cfg, "spatial_merge_size", 1) or 1)
            multiple = patch_size * max(1, spatial_merge_size)
            if multiple > 1:
                buckets = [
                    (
                        self._align_dim_to_multiple(h, multiple),
                        self._align_dim_to_multiple(w, multiple),
                    )
                    for h, w in buckets
                ]
        return sorted(set(buckets))

    @staticmethod
    def _to_mx_array(value: Any, *, dtype: Any | None = None) -> mx.array:
        """Convert *value* to an MLX array, preserving existing MLX arrays."""
        if isinstance(value, mx.array):
            return value.astype(dtype) if dtype is not None and value.dtype != dtype else value
        array = mx.array(value)
        return array.astype(dtype) if dtype is not None and array.dtype != dtype else array

    def resize_to_bucket(self, image: Image.Image) -> Image.Image:
        """Resize an image to the nearest model-aware resolution bucket.

        Args:
            image: Input PIL image.

        Returns:
            The resized PIL image.
        """
        return self._resize_to_buckets(image, self._get_resize_buckets_for_model())

    @staticmethod
    def _resize_to_buckets(image: Image.Image, buckets: list[tuple[int, int]]) -> Image.Image:
        """Resize *image* to the bucket closest in total pixel count.

        Args:
            image: Input PIL image.
            buckets: Available ``(height, width)`` resolution buckets.

        Returns:
            The (possibly unchanged) resized PIL image.
        """
        width, height = image.size
        original_pixels = width * height
        target_h, target_w = min(buckets, key=lambda b: abs(b[0] * b[1] - original_pixels))
        if (height, width) == (target_h, target_w):
            return image
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    @staticmethod
    def _to_uint8_image_array(arr: np.ndarray) -> np.ndarray:
        """Convert a generic numeric array into an ``(H, W, 3)`` uint8 RGB array.

        Handles channel-first layouts, single-channel inputs, RGBA
        stripping, and float-to-uint8 rescaling.

        Args:
            arr: Input array with 2 or 3 dimensions.

        Returns:
            A ``uint8`` NumPy array with shape ``(H, W, 3)``.

        Raises:
            ValueError: If the array cannot be coerced to a 3-channel
                RGB layout.
        """
        out = np.asarray(arr)
        if out.ndim == 2:
            out = np.repeat(out[..., None], 3, axis=-1)
        if out.ndim != 3:
            raise ValueError(f"Expected 3D image array, got shape {out.shape}")
        if out.shape[0] in (1, 3, 4) and out.shape[-1] not in (1, 3, 4):
            out = np.transpose(out, (1, 2, 0))
        if out.shape[-1] == 1:
            out = np.repeat(out, 3, axis=-1)
        if out.shape[-1] == 4:
            out = out[..., :3]
        if out.shape[-1] != 3:
            raise ValueError(f"Expected RGB-like image with 3 channels, got shape {out.shape}")
        if np.issubdtype(out.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(out)) <= 1.0 else 1.0
            out = np.clip(out * scale, 0, 255).astype(np.uint8)
        else:
            out = np.clip(out, 0, 255).astype(np.uint8)
        return out

    def normalize_image_input(self, payload: Any) -> Image.Image:
        """Normalize a heterogeneous image payload into a PIL RGB image.

        Accepts PIL images, NumPy arrays, raw bytes, base64 data URIs,
        ``file://`` URIs, and local filesystem paths.

        Args:
            payload: The image payload in any supported format.

        Returns:
            A PIL ``Image.Image`` in RGB mode.

        Raises:
            ValueError: If the payload format is not recognized or the
                data URI is malformed.
        """
        if isinstance(payload, Image.Image):
            return payload.convert("RGB")
        if isinstance(payload, np.ndarray):
            return Image.fromarray(self._to_uint8_image_array(payload)).convert("RGB")
        if isinstance(payload, (bytes, bytearray)):
            return Image.open(io.BytesIO(payload)).convert("RGB")
        if isinstance(payload, str):
            if payload.startswith("data:"):
                try:
                    _header, raw = payload.split(",", 1)
                    image_data = base64.b64decode(raw)
                except Exception as exc:
                    raise ValueError("Invalid image data URL payload") from exc
                return Image.open(io.BytesIO(image_data)).convert("RGB")
            path = payload[len("file://") :] if payload.startswith("file://") else payload
            if os.path.exists(path):
                return Image.open(path).convert("RGB")
        raise ValueError("Unsupported image payload; expected PIL image, ndarray, bytes, data URL, or local path.")

    def normalize_video_input(self, payload: Any) -> np.ndarray:
        """Normalize a heterogeneous video payload into a ``(T, H, W, 3)`` uint8 array.

        Accepts a 4-D NumPy array or a list/tuple of image-like frames.

        Args:
            payload: Video data as a NumPy array with shape
                ``[T, H, W, C]`` (or ``[T, C, H, W]``) or a sequence
                of image-like objects accepted by
                :meth:`normalize_image_input`.

        Returns:
            A ``uint8`` NumPy array with shape ``(T, H, W, 3)``.

        Raises:
            ValueError: If the payload shape or format is unsupported,
                or the frame list is empty.
        """
        if isinstance(payload, np.ndarray):
            video = np.asarray(payload)
            if video.ndim != 4:
                raise ValueError(f"Expected video array with shape [T,H,W,C], got {video.shape}")
            if video.shape[-1] not in (1, 3, 4) and video.shape[1] in (1, 3, 4):
                video = np.transpose(video, (0, 2, 3, 1))
            if video.shape[-1] == 1:
                video = np.repeat(video, 3, axis=-1)
            if video.shape[-1] == 4:
                video = video[..., :3]
            if video.shape[-1] != 3:
                raise ValueError(f"Expected RGB video, got shape {video.shape}")
            if np.issubdtype(video.dtype, np.floating):
                scale = 255.0 if float(np.nanmax(video)) <= 1.0 else 1.0
                video = np.clip(video * scale, 0, 255).astype(np.uint8)
            else:
                video = np.clip(video, 0, 255).astype(np.uint8)
            return video

        if isinstance(payload, (list, tuple)):
            frames: list[np.ndarray] = []
            for frame in payload:
                image = self.normalize_image_input(frame)
                frames.append(np.asarray(image, dtype=np.uint8))
            if not frames:
                raise ValueError("Video frame list is empty")
            return np.stack(frames, axis=0)

        raise ValueError("Unsupported video payload; expected ndarray or list of image-like frames.")

    def extract_media_from_messages(self, messages: list[dict[str, Any]]) -> tuple[list[Image.Image], list[np.ndarray]]:
        """Extract image and video payloads from a chat-message list.

        Iterates over OpenAI-style multi-content messages, detecting
        ``image``, ``image_url``, ``video``, and ``video_url`` content
        blocks and normalizing each payload.

        Args:
            messages: List of message dicts, each with a ``"content"``
                key that may be a string or a list of typed content
                blocks.

        Returns:
            A ``(images, videos)`` tuple where *images* is a list of
            PIL images and *videos* is a list of ``(T, H, W, 3)``
            uint8 arrays.

        Raises:
            ValueError: If a ``video_url`` block is encountered (not
                supported for local preprocessing) or if any payload
                normalization fails.
        """
        images: list[Image.Image] = []
        videos: list[np.ndarray] = []

        for message in messages:
            content = message.get("content", [])
            if isinstance(content, str):
                continue
            if not isinstance(content, list):
                continue

            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type", "")).lower()

                if item_type in {"image", "input_image"} or "image" in item:
                    image_payload = item.get("image")
                    if image_payload is not None:
                        images.append(self.normalize_image_input(image_payload))
                        continue

                if item_type in {"image_url", "input_image"} or "image_url" in item:
                    image_url_payload = item.get("image_url")
                    url = image_url_payload
                    if isinstance(image_url_payload, dict):
                        url = image_url_payload.get("url") or image_url_payload.get("image_url")
                    images.append(self.normalize_image_input(url))
                    continue

                if item_type in {"video", "input_video"} or "video" in item:
                    video_payload = item.get("video")
                    if video_payload is not None:
                        videos.append(self.normalize_video_input(video_payload))
                        continue

                if item_type in {"video_url", "input_video"} or "video_url" in item:
                    raise ValueError("video_url is not supported in local preprocessing. Provide inline video arrays.")

        return images, videos

    @staticmethod
    def _normalize_rgb(rgb: Any) -> mx.array:
        """Apply CLIP-style mean/std normalization to an RGB float array.

        Args:
            rgb: Float32 array with values in ``[0, 255]`` (or already
                ``float32``).

        Returns:
            Normalized float32 array with CLIP mean subtracted and
            divided by CLIP std.
        """
        rgb = MultimodalManager._to_mx_array(rgb, dtype=mx.float32)
        rgb = rgb / 255.0
        rgb = (rgb - CLIP_IMAGE_MEAN) / CLIP_IMAGE_STD
        return rgb

    def _patchify_spatiotemporal(
        self,
        frames: Any,
        *,
        patch_size: int,
        temporal_patch_size: int,
        spatial_merge_size: int = 1,
    ) -> tuple[mx.array, mx.array]:
        """Decompose a video tensor into flat spatiotemporal patches.

        The input is reshaped into non-overlapping 3-D patches along the
        temporal, height, and width axes, then flattened into a 2-D
        ``(num_patches, patch_dim)`` matrix.

        Args:
            frames: Video tensor of shape ``(T, H, W, C)`` where
                ``C == 3``.
            patch_size: Spatial patch edge length in pixels.
            temporal_patch_size: Temporal patch size in frames.
            spatial_merge_size: Optional spatial merge factor used to
                compute the effective alignment multiple.

        Returns:
            A tuple ``(flat_patches, grid_thw)`` where *flat_patches*
            has shape ``(num_patches, patch_dim)`` and *grid_thw* is a
            1-D int64 array ``[t_groups, grid_h, grid_w]``.

        Raises:
            ValueError: If *frames* does not have 4 dimensions, is not
                3-channel, or has non-positive patch sizes.
        """
        frames = self._to_mx_array(frames, dtype=mx.float32)
        if frames.ndim != 4:
            raise ValueError(f"Expected frames with shape [T,H,W,C], got {frames.shape}.")
        t_total, height, width, channels = frames.shape
        if channels != 3:
            raise ValueError(f"Expected 3-channel RGB input, got channels={channels}.")

        patch_size = int(patch_size)
        temporal_patch_size = int(temporal_patch_size)
        spatial_merge_size = int(spatial_merge_size or 1)
        if patch_size <= 0 or temporal_patch_size <= 0 or spatial_merge_size <= 0:
            raise ValueError(
                f"Invalid patch sizes: patch_size={patch_size}, temporal_patch_size={temporal_patch_size}, "
                f"spatial_merge_size={spatial_merge_size}"
            )

        spatial_multiple = patch_size * spatial_merge_size
        t_pad = (-t_total) % temporal_patch_size
        h_pad = (-height) % spatial_multiple
        w_pad = (-width) % spatial_multiple
        if t_pad or h_pad or w_pad:
            frames = mx.pad(frames, ((0, t_pad), (0, h_pad), (0, w_pad), (0, 0)), mode="edge")

        t_padded, h_padded, w_padded, _ = frames.shape
        t_groups = t_padded // temporal_patch_size
        grid_h = h_padded // patch_size
        grid_w = w_padded // patch_size

        patches = frames.reshape(t_groups, temporal_patch_size, grid_h, patch_size, grid_w, patch_size, channels)
        patches = patches.transpose(0, 2, 4, 1, 3, 5, 6)
        patches = patches.reshape(t_groups * grid_h * grid_w, temporal_patch_size, patch_size, patch_size, channels)
        patches = patches.transpose(0, 4, 1, 2, 3)
        flat = patches.reshape(patches.shape[0], -1).astype(mx.float32)
        grid_thw = mx.array([t_groups, grid_h, grid_w], dtype=mx.int64)
        return flat, grid_thw

    def _pad_flat_patches_for_merge(
        self,
        pixel_values: Any,
        grid_thw: Any,
        *,
        spatial_merge_size: int,
    ) -> tuple[mx.array, mx.array]:
        """Pad flat patch grids so spatial dims are divisible by *spatial_merge_size*.

        This ensures that the downstream spatial merge layer receives
        grids whose height and width are exact multiples of
        ``spatial_merge_size``.

        Args:
            pixel_values: Flat patch tensor of shape
                ``(total_patches, patch_dim)``.
            grid_thw: Grid shape array of shape ``(num_items, 3)`` with
                columns ``[T, H, W]``.
            spatial_merge_size: Required spatial divisibility factor.

        Returns:
            A ``(pixel_values, grid_thw)`` tuple, potentially padded.
        """
        spatial_merge_size = int(spatial_merge_size or 1)
        pixel_values = self._to_mx_array(pixel_values)
        grid_thw = self._to_mx_array(grid_thw, dtype=mx.int64)
        if spatial_merge_size <= 1 or pixel_values.ndim != 2:
            return pixel_values, grid_thw
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            return pixel_values, grid_thw

        grid_np = np.asarray(grid_thw, dtype=np.int64)
        sizes = grid_np.prod(axis=1).astype(int)
        if int(sizes.sum()) != int(pixel_values.shape[0]):
            return pixel_values, grid_thw

        patch_dim = int(pixel_values.shape[1])
        out_chunks: list[mx.array] = []
        out_grid: list[mx.array] = []
        offset = 0
        for (t, h, w), n in zip(grid_np, sizes, strict=False):
            t_i, h_i, w_i = int(t), int(h), int(w)
            n_i = int(n)
            chunk = pixel_values[offset : offset + n_i]
            offset += n_i

            new_h = ((h_i + spatial_merge_size - 1) // spatial_merge_size) * spatial_merge_size
            new_w = ((w_i + spatial_merge_size - 1) // spatial_merge_size) * spatial_merge_size
            if new_h == h_i and new_w == w_i:
                out_chunks.append(chunk)
                out_grid.append(mx.array([t_i, h_i, w_i], dtype=mx.int64))
                continue

            try:
                reshaped = chunk.reshape(t_i, h_i, w_i, patch_dim)
            except Exception:
                out_chunks.append(chunk)
                out_grid.append(mx.array([t_i, h_i, w_i], dtype=mx.int64))
                continue

            padded = mx.pad(reshaped, ((0, 0), (0, new_h - h_i), (0, new_w - w_i), (0, 0)), mode="edge")
            out_chunks.append(padded.reshape(t_i * new_h * new_w, patch_dim))
            out_grid.append(mx.array([t_i, new_h, new_w], dtype=mx.int64))

        return mx.concatenate(out_chunks, axis=0), mx.stack(out_grid, axis=0)

    def process_images(self, images: list[Image.Image] | None) -> tuple[mx.array | None, mx.array | None]:
        """Preprocess a list of PIL images into pixel-value tensors.

        First attempts to use the configured ``processor``.  If the
        processor fails or is unavailable and the model supports flat
        patch inputs, falls back to manual CLIP normalization and
        spatiotemporal patchification.

        Args:
            images: List of PIL images, or ``None``.

        Returns:
            A ``(pixel_values, image_grid_thw)`` tuple.  Both are
            ``None`` when *images* is empty or ``None``.

        Raises:
            ValueError: If neither the processor nor the fallback path
                can handle the images.
        """
        if not images:
            return None, None

        resize_buckets = self._get_resize_buckets_for_model()
        bucketed = [self._resize_to_buckets(self.normalize_image_input(img), resize_buckets) for img in images]
        vision_cfg = self._get_vision_config()
        spatial_merge_size = int(getattr(vision_cfg, "spatial_merge_size", 1) or 1) if vision_cfg is not None else 1

        if self.processor is not None:
            try:
                processed = self.processor(images=bucketed, return_tensors="np")
            except Exception as exc:
                if isinstance(exc, ValueError) and "either `text` or `text_target`" in str(exc):
                    try:
                        processed = self.processor(text=[""] * len(bucketed), images=bucketed, return_tensors="np")
                    except Exception:
                        processed = None
                else:
                    processed = None
            if isinstance(processed, dict):
                pixel_values = processed.get("pixel_values")
                image_grid_thw = processed.get("image_grid_thw")
                if pixel_values is not None:
                    pixel_values = self._to_mx_array(pixel_values)
                    if image_grid_thw is not None:
                        image_grid_thw = self._to_mx_array(image_grid_thw, dtype=mx.int64)
                    if image_grid_thw is not None and self._supports_flat_patch_inputs() and spatial_merge_size > 1:
                        pixel_values, image_grid_thw = self._pad_flat_patches_for_merge(
                            pixel_values,
                            image_grid_thw,
                            spatial_merge_size=spatial_merge_size,
                        )
                    return pixel_values, image_grid_thw

        if not self._supports_flat_patch_inputs():
            raise ValueError(
                "Processor failed to preprocess images and model does not expose a flat-patch vision fallback."
            )
        if vision_cfg is None:
            raise ValueError("Vision config not available for fallback image preprocessing.")

        patch_size = int(getattr(vision_cfg, "patch_size", 14) or 14)
        temporal_patch_size = int(getattr(vision_cfg, "temporal_patch_size", 2) or 1)
        pixel_values_list: list[mx.array] = []
        grids: list[mx.array] = []

        for img in bucketed:
            rgb = mx.array(np.asarray(img.convert("RGB"), dtype=np.float32))
            rgb = self._normalize_rgb(rgb)
            frames = mx.repeat(rgb[None, ...], temporal_patch_size, axis=0)
            flat, grid = self._patchify_spatiotemporal(
                frames,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                spatial_merge_size=spatial_merge_size,
            )
            pixel_values_list.append(flat)
            grids.append(grid[None, :])

        pixel_values = mx.concatenate(pixel_values_list, axis=0) if pixel_values_list else None
        image_grid_thw = mx.concatenate(grids, axis=0).astype(mx.int64) if grids else None
        if pixel_values is not None and image_grid_thw is not None and spatial_merge_size > 1:
            pixel_values, image_grid_thw = self._pad_flat_patches_for_merge(
                pixel_values, image_grid_thw, spatial_merge_size=spatial_merge_size
            )
        return pixel_values, image_grid_thw

    def process_videos(self, videos: list[np.ndarray] | None) -> tuple[mx.array | None, mx.array | None]:
        """Preprocess a list of video arrays into pixel-value tensors.

        Mirrors :meth:`process_images` but operates on ``(T, H, W, C)``
        video arrays.  Falls back to manual patchification when the
        processor is unavailable or fails.

        Args:
            videos: List of ``(T, H, W, C)`` uint8 video arrays, or
                ``None``.

        Returns:
            A ``(pixel_values_videos, video_grid_thw)`` tuple.  Both
            are ``None`` when *videos* is empty or ``None``.

        Raises:
            ValueError: If neither the processor nor the fallback path
                can handle the videos.
        """
        if not videos:
            return None, None

        normalized_videos = [self.normalize_video_input(video) for video in videos]
        vision_cfg = self._get_vision_config()
        spatial_merge_size = int(getattr(vision_cfg, "spatial_merge_size", 1) or 1) if vision_cfg is not None else 1
        resize_buckets = self._get_resize_buckets_for_model()

        if self.processor is not None:
            try:
                processed = self.processor(videos=normalized_videos, return_tensors="np")
            except Exception as exc:
                if isinstance(exc, ValueError) and "either `text` or `text_target`" in str(exc):
                    try:
                        processed = self.processor(
                            text=[""] * len(normalized_videos),
                            videos=normalized_videos,
                            return_tensors="np",
                        )
                    except Exception:
                        processed = None
                else:
                    processed = None
            if isinstance(processed, dict):
                pixel_values_videos = processed.get("pixel_values_videos")
                video_grid_thw = processed.get("video_grid_thw")
                if pixel_values_videos is not None:
                    pixel_values_videos = self._to_mx_array(pixel_values_videos)
                    if video_grid_thw is not None:
                        video_grid_thw = self._to_mx_array(video_grid_thw, dtype=mx.int64)
                    if video_grid_thw is not None and self._supports_flat_patch_inputs() and spatial_merge_size > 1:
                        pixel_values_videos, video_grid_thw = self._pad_flat_patches_for_merge(
                            pixel_values_videos,
                            video_grid_thw,
                            spatial_merge_size=spatial_merge_size,
                        )
                    return pixel_values_videos, video_grid_thw

        if not self._supports_flat_patch_inputs():
            raise ValueError(
                "Processor failed to preprocess videos and model does not expose a flat-patch vision fallback."
            )
        if vision_cfg is None:
            raise ValueError("Vision config not available for fallback video preprocessing.")

        patch_size = int(getattr(vision_cfg, "patch_size", 14) or 14)
        temporal_patch_size = int(getattr(vision_cfg, "temporal_patch_size", 2) or 1)
        pixel_values_list: list[mx.array] = []
        grids: list[mx.array] = []

        for video in normalized_videos:
            _, height, width, _ = video.shape
            target_h, target_w = min(resize_buckets, key=lambda b: abs(b[0] * b[1] - height * width))
            if (height, width) != (target_h, target_w):
                resized = np.stack(
                    [
                        np.asarray(
                            Image.fromarray(frame).convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
                        )
                        for frame in video
                    ],
                    axis=0,
                )
            else:
                resized = video

            normalized = self._normalize_rgb(mx.array(resized.astype(np.float32, copy=False)))
            flat, grid = self._patchify_spatiotemporal(
                normalized,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                spatial_merge_size=spatial_merge_size,
            )
            pixel_values_list.append(flat)
            grids.append(grid[None, :])

        pixel_values_videos = mx.concatenate(pixel_values_list, axis=0) if pixel_values_list else None
        video_grid_thw = mx.concatenate(grids, axis=0).astype(mx.int64) if grids else None
        if pixel_values_videos is not None and video_grid_thw is not None and spatial_merge_size > 1:
            pixel_values_videos, video_grid_thw = self._pad_flat_patches_for_merge(
                pixel_values_videos, video_grid_thw, spatial_merge_size=spatial_merge_size
            )
        return pixel_values_videos, video_grid_thw

    def _attach_cached_embeddings(self, features: list[MultiModalFeature]) -> None:
        """Attach previously cached encoder embeddings to features.

        For each feature whose hash exists in the cache, the cached
        embedding tensor is set on the feature object, allowing the
        model runner to skip the encoder forward pass.

        Args:
            features: List of :class:`MultiModalFeature` instances to
                check against the cache.
        """
        if self.cache is None:
            return
        for feature in features:
            cached = self.cache.get(feature.mm_hash)
            if cached is not None:
                feature.set_cached_embeddings(cached)

    def process_images_to_features(
        self, images: list[Image.Image] | None, request_idx: int = 0
    ) -> list[MultiModalFeature]:
        """Preprocess images and wrap results as :class:`MultiModalFeature` objects.

        Each image produces one feature containing its pixel values and
        optional grid metadata.  Cached embeddings are attached when
        available.

        Args:
            images: List of PIL images, or ``None``.
            request_idx: Index of the originating request, used to map
                features back to requests in batched contexts.

        Returns:
            List of :class:`MultiModalFeature` instances (empty if no
            images).
        """
        if not images:
            return []
        pixel_values, image_grid_thw = self.process_images(images)
        if pixel_values is None:
            return []

        features: list[MultiModalFeature] = []
        patch_offsets: list["int"] | None = None
        if (
            image_grid_thw is not None
            and image_grid_thw.ndim == 2
            and image_grid_thw.shape[1] == 3
            and pixel_values.ndim == 2
        ):
            sizes = (np.asarray(image_grid_thw, dtype=np.int64).prod(axis=1)).astype(int).tolist()
            patch_offsets = [0]
            for size in sizes[:-1]:
                patch_offsets.append(patch_offsets[-1] + int(size))

        for idx in range(len(images)):
            if patch_offsets is not None:
                assert image_grid_thw is not None
                start = int(patch_offsets[idx])
                size = int(np.asarray(image_grid_thw[idx]).prod())
                single_pv = pixel_values[start : start + size]
            elif pixel_values.ndim > 3:
                single_pv = pixel_values[idx : idx + 1]
            else:
                single_pv = pixel_values

            single_grid = None
            if image_grid_thw is not None and idx < len(image_grid_thw):
                single_grid = image_grid_thw[idx : idx + 1]
            features.append(
                MultiModalFeature.from_image(
                    pixel_values=single_pv,
                    grid_thw=single_grid,
                    request_idx=request_idx,
                    mm_hash="" if self.cache is None else None,
                )
            )

        self._attach_cached_embeddings(features)
        return features

    def process_videos_to_features(
        self, videos: list[np.ndarray] | None, request_idx: int = 0
    ) -> list[MultiModalFeature]:
        """Preprocess videos and wrap results as :class:`MultiModalFeature` objects.

        Each video produces one feature containing its pixel values and
        optional grid metadata.  Cached embeddings are attached when
        available.

        Args:
            videos: List of ``(T, H, W, 3)`` uint8 video arrays, or
                ``None``.
            request_idx: Index of the originating request for batched
                mapping.

        Returns:
            List of :class:`MultiModalFeature` instances (empty if no
            videos).
        """
        if not videos:
            return []
        pixel_values_videos, video_grid_thw = self.process_videos(videos)
        if pixel_values_videos is None:
            return []

        features: list[MultiModalFeature] = []
        patch_offsets: list["int"] | None = None
        if (
            video_grid_thw is not None
            and video_grid_thw.ndim == 2
            and video_grid_thw.shape[1] == 3
            and pixel_values_videos.ndim == 2
        ):
            sizes = (np.asarray(video_grid_thw, dtype=np.int64).prod(axis=1)).astype(int).tolist()
            patch_offsets = [0]
            for size in sizes[:-1]:
                patch_offsets.append(patch_offsets[-1] + int(size))

        for idx in range(len(videos)):
            if patch_offsets is not None:
                assert video_grid_thw is not None
                start = int(patch_offsets[idx])
                size = int(np.asarray(video_grid_thw[idx]).prod())
                single_pv = pixel_values_videos[start : start + size]
            elif pixel_values_videos.ndim > 4:
                single_pv = pixel_values_videos[idx : idx + 1]
            else:
                single_pv = pixel_values_videos

            single_grid = None
            if video_grid_thw is not None and idx < len(video_grid_thw):
                single_grid = video_grid_thw[idx : idx + 1]
            features.append(
                MultiModalFeature.from_video(
                    pixel_values=single_pv,
                    grid_thw=single_grid,
                    request_idx=request_idx,
                    mm_hash="" if self.cache is None else None,
                )
            )

        self._attach_cached_embeddings(features)
        return features

    def clear_cache(self) -> None:
        """Clear the vision encoder embedding cache."""
        if self.cache is not None:
            self.cache.clear()

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Return cache statistics, or ``None`` if caching is disabled.

        Returns:
            A dictionary of cache statistics (see
            :meth:`VisionEncoderCache.get_stats`), or ``None``.
        """
        if self.cache is None:
            return None
        return self.cache.get_stats()

    def prepare(self, payload: MultimodalInput) -> MultimodalInput:
        """Run the full multimodal preprocessing pipeline on a payload.

        Extracts images and videos from all supported payload fields
        (``messages``, ``images``, ``videos``, ``media``), preprocesses
        them into :class:`MultiModalFeature` objects, and batches them
        into a :class:`BatchedMultiModalInputs` ready for model
        consumption.

        Args:
            payload: The raw :class:`MultimodalInput` to preprocess.

        Returns:
            A new :class:`MultimodalInput` with populated ``features``
            and ``batched`` fields.
        """
        images: list[Image.Image] = []
        videos: list[np.ndarray] = []

        if payload.messages:
            msg_images, msg_videos = self.extract_media_from_messages(payload.messages)
            images.extend(msg_images)
            videos.extend(msg_videos)

        if payload.images:
            images.extend([self.normalize_image_input(item) for item in payload.images])
        if payload.videos:
            videos.extend([self.normalize_video_input(item) for item in payload.videos])

        if payload.media:
            for item in payload.media:
                if isinstance(item, dict):
                    kind = str(item.get("type", "")).lower()
                    if kind in {"image", "input_image"} or "image" in item:
                        image_payload = item.get("image", item.get("image_url"))
                        if isinstance(image_payload, dict):
                            image_payload = image_payload.get("url") or image_payload.get("image_url")
                        images.append(self.normalize_image_input(image_payload))
                        continue
                    if kind in {"video", "input_video"} or "video" in item:
                        videos.append(self.normalize_video_input(item.get("video")))
                        continue
                elif isinstance(item, np.ndarray):
                    if item.ndim == 4:
                        videos.append(self.normalize_video_input(item))
                    else:
                        images.append(self.normalize_image_input(item))
                else:
                    images.append(self.normalize_image_input(item))

        features: list[MultiModalFeature] = []
        features.extend(self.process_images_to_features(images, request_idx=0))
        features.extend(self.process_videos_to_features(videos, request_idx=0))
        batched = BatchedMultiModalInputs.from_features(features) if features else BatchedMultiModalInputs.empty()

        return MultimodalInput(
            text=payload.text,
            media=payload.media,
            messages=payload.messages,
            images=images or None,
            videos=videos or None,
            features=features or None,
            batched=batched,
        )


MultiModalManager = MultimodalManager


__all__ = ("MultiModalManager", "MultimodalManager")
