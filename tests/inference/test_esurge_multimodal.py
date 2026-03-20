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

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np
from PIL import Image

from easymlx.inference.esurge.multimodal.cache import VisionEncoderCache
from easymlx.inference.esurge.multimodal.manager import MultimodalManager
from easymlx.inference.esurge.multimodal.types import BatchedMultiModalInputs, MultimodalInput


@dataclass
class _DummyVisionConfig:
    patch_size: int = 14
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    image_size: int = 224


@dataclass
class _DummyConfig:
    model_type: str = "qwen2_vl"
    vision_config: _DummyVisionConfig = field(default_factory=_DummyVisionConfig)

    def get_vision_config(self) -> _DummyVisionConfig:
        return self.vision_config


@dataclass
class _DummyModel:
    config: _DummyConfig = field(default_factory=_DummyConfig)


def _make_image(width: int = 60, height: int = 40, value: int = 127) -> Image.Image:
    arr = np.full((height, width, 3), fill_value=value, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _as_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    raw = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{raw}"


def test_vision_encoder_cache_lru_and_stats() -> None:
    cache = VisionEncoderCache(capacity_mb=1)
    emb1 = np.zeros((64, 64), dtype=np.float32)
    emb2 = np.ones((64, 64), dtype=np.float32)
    emb3 = np.full((64, 64), 2.0, dtype=np.float32)

    cache.put("a", emb1)
    cache.put("b", emb2)
    _ = cache.get("a")
    cache.put("c", emb3)

    assert cache.get("a") is not None
    assert cache.get("missing") is None
    stats = cache.get_stats()
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1
    assert stats["capacity_mb"] == 1


def test_extract_media_from_messages_supports_data_url_and_video() -> None:
    manager = MultimodalManager(processor=None, model=_DummyModel())
    image = _make_image()
    url = _as_data_url(image)
    video = np.zeros((2, 24, 24, 3), dtype=np.uint8)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "video", "video": video},
            ],
        }
    ]

    images, videos = manager.extract_media_from_messages(messages)
    assert len(images) == 1
    assert len(videos) == 1
    assert videos[0].shape == (2, 24, 24, 3)


def test_process_images_flat_patch_fallback_returns_grid() -> None:
    manager = MultimodalManager(processor=None, model=_DummyModel())
    pixel_values, image_grid_thw = manager.process_images([_make_image(50, 50)])
    assert isinstance(pixel_values, mx.array)
    assert isinstance(image_grid_thw, mx.array)
    assert pixel_values.ndim == 2
    assert image_grid_thw.shape[1] == 3


def test_process_images_to_features_attaches_cached_embeddings() -> None:
    manager = MultimodalManager(processor=None, model=_DummyModel(), enable_cache=True)
    image = _make_image(48, 48, 99)

    first = manager.process_images_to_features([image], request_idx=0)
    assert len(first) == 1
    manager.cache.put(first[0].mm_hash, np.asarray([1, 2, 3], dtype=np.int32))

    second = manager.process_images_to_features([image], request_idx=1)
    assert len(second) == 1
    assert second[0].has_cached_embeddings


def test_process_images_to_features_skips_hashing_when_cache_disabled() -> None:
    manager = MultimodalManager(processor=None, model=_DummyModel(), enable_cache=False)
    image = _make_image(48, 48, 99)

    features = manager.process_images_to_features([image], request_idx=0)

    assert len(features) == 1
    assert features[0].mm_hash == ""
    assert isinstance(features[0].pixel_values, mx.array)


def test_prepare_builds_batched_multimodal_input() -> None:
    manager = MultimodalManager(processor=None, model=_DummyModel())
    image = _make_image(64, 64, 180)
    data_url = _as_data_url(image)
    video = np.zeros((3, 32, 32, 3), dtype=np.uint8)

    payload = MultimodalInput(
        text="describe",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": data_url},
                    {"type": "video", "video": video},
                ],
            }
        ],
    )

    prepared = manager.prepare(payload)
    assert prepared.images is not None and len(prepared.images) == 1
    assert prepared.videos is not None and len(prepared.videos) == 1
    assert prepared.features is not None and len(prepared.features) == 2
    assert isinstance(prepared.batched, BatchedMultiModalInputs)
    assert prepared.batched.has_vision
    assert prepared.batched.num_images == 1
    assert prepared.batched.num_videos == 1
    assert isinstance(prepared.batched.pixel_values, mx.array)
    assert isinstance(prepared.batched.pixel_values_videos, mx.array)
