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

"""GPT-OSS model family.

This module provides the MLX implementation of the GPT-OSS Mixture-of-Experts
model for serving and inference. GPT-OSS features sliding window attention,
SwiGLU-based MoE routing, and YaRN rope scaling.

Classes:
    GptOssConfig: Configuration for the GPT-OSS model.
    GptOssForCausalLM: GPT-OSS causal language model with LM head.
    GptOssModel: Base GPT-OSS transformer model.
"""

from .gpt_oss_configuration import GptOssConfig
from .modeling_gpt_oss import GptOssForCausalLM, GptOssModel

__all__ = ("GptOssConfig", "GptOssForCausalLM", "GptOssModel")
