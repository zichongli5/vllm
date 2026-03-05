# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeTextConfig


@pytest.mark.parametrize(
    "config_cls",
    [Qwen3_5TextConfig, Qwen3_5MoeTextConfig],
)
def test_qwen3_5_full_attention_sliding_window_default(config_cls):
    config = config_cls(num_hidden_layers=4)
    assert config.full_attention_sliding_window is None


@pytest.mark.parametrize(
    "config_cls",
    [Qwen3_5TextConfig, Qwen3_5MoeTextConfig],
)
def test_qwen3_5_full_attention_sliding_window_settable(config_cls):
    config = config_cls(
        num_hidden_layers=4,
        full_attention_sliding_window=128,
    )
    assert config.full_attention_sliding_window == 128


def test_qwen3_next_attention_resolves_window_for_full_attention_only():
    config = Qwen3_5TextConfig(
        num_hidden_layers=4,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        full_attention_sliding_window=256,
    )

    assert Qwen3NextAttention._resolve_per_layer_sliding_window(config, 0) is None
    assert Qwen3NextAttention._resolve_per_layer_sliding_window(config, 1) == 256
    assert Qwen3NextAttention._resolve_per_layer_sliding_window(config, 2) is None
    assert Qwen3NextAttention._resolve_per_layer_sliding_window(config, 3) == 256
