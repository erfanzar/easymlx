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

"""LongcatFlashNgram configuration for EasyMLX inference."""

from __future__ import annotations

from easymlx.infra.factory import register_config

from ..longcat_flash.longcat_flash_configuration import LongcatFlashConfig


@register_config("longcat_flash_ngram")
class LongcatFlashNgramConfig(LongcatFlashConfig):
    """Configuration for the LongcatFlashNgram transformer model.

    Extends LongcatFlashConfig with ngram embedding parameters for
    enhanced local context modeling using hash-based ngram features.

    Attributes:
        model_type: Identifier string (``"longcat_flash_ngram"``).
        ngram_vocab_size_ratio: Ratio used to compute the ngram hash modulus.
        emb_neighbor_num: Number of neighboring tokens to consider for ngram.
        emb_split_num: Number of embedding splits per ngram level.
    """

    model_type = "longcat_flash_ngram"

    def __init__(
        self,
        *,
        ngram_vocab_size_ratio: int = 78,
        emb_neighbor_num: int = 4,
        emb_split_num: int = 4,
        **kwargs,
    ):
        """Initialize LongcatFlashNgram configuration.

        Args:
            ngram_vocab_size_ratio: Ratio used to compute the ngram hash
                modulus (``ratio * vocab_size``). Defaults to ``78``.
            emb_neighbor_num: Number of neighboring tokens to include in
                ngram computation. Defaults to ``4``.
            emb_split_num: Number of embedding splits per ngram level.
                Defaults to ``4``.
            **kwargs: Additional keyword arguments passed to
                ``LongcatFlashConfig.__init__``.
        """
        super().__init__(**kwargs)
        self.ngram_vocab_size_ratio = int(ngram_vocab_size_ratio)
        self.emb_neighbor_num = int(emb_neighbor_num)
        self.emb_split_num = int(emb_split_num)


__all__ = ("LongcatFlashNgramConfig",)
