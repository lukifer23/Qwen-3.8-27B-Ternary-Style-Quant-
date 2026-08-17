"""Shard loader + architecture discovery. Reconstruction tests land with the pilot slice."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from q38ternary.architecture import build_layer_map, classify_tensor, layer_types_from_config
from q38ternary.safetensors_io import ShardIndex


def test_layer_types_from_qwen38_interval() -> None:
    types = layer_types_from_config(
        {"num_hidden_layers": 64, "full_attention_interval": 4}
    )
    assert len(types) == 64
    assert types[0] == "linear_attention"
    assert types[3] == "full_attention"
    assert types[63] == "full_attention"
    assert types.count("full_attention") == 16
    assert types.count("linear_attention") == 48


def test_classify_tensor_categories() -> None:
    assert classify_tensor("model.embed_tokens.weight") == "embedding"
    assert classify_tensor("lm_head.weight") == "lm_head"
    assert classify_tensor("model.layers.0.mlp.up_proj.weight") == "mlp"
    assert classify_tensor("model.layers.0.input_layernorm.weight") == "norm"
    assert classify_tensor("model.layers.3.self_attn.q_proj.weight", "full_attention") == "full_attention"
    assert classify_tensor("model.layers.0.linear_attn.in_proj_qkv.weight", "linear_attention") == "gated_deltanet"
    assert classify_tensor("model.visual.patch_embed.weight") == "vision"
    assert classify_tensor("mtp.layers.0.weight") == "mtp"


def test_build_layer_map_reads_declared_types() -> None:
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 5120,
            "num_hidden_layers": 4,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        },
    }
    payload = build_layer_map(config)
    assert payload["layers"][3]["is_full_attention"] is True
    assert payload["layers"][0]["is_gated_deltanet"] is True


@pytest.mark.skipif(
    importlib.util.find_spec("safetensors") is None,
    reason="safetensors not installed",
)
def test_shard_loader_roundtrip(tmp_path: Path) -> None:
    from safetensors.numpy import save_file

    t0 = np.arange(12, dtype=np.float32).reshape(3, 4)
    t1 = np.ones((2, 2), dtype=np.float32) * 7
    shard_a = tmp_path / "a.safetensors"
    shard_b = tmp_path / "b.safetensors"
    save_file({"model.layers.0.foo.weight": t0}, str(shard_a))
    save_file({"model.layers.1.bar.weight": t1}, str(shard_b))
    index = {
        "weight_map": {
            "model.layers.0.foo.weight": "a.safetensors",
            "model.layers.1.bar.weight": "b.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    with ShardIndex(tmp_path) as store:
        loaded = store.load_layer(0)
        np.testing.assert_array_equal(loaded["model.layers.0.foo.weight"], t0)
        store.release_layer(0)
        assert 0 not in store._layer_cache
        other = store.load_tensor("model.layers.1.bar.weight")
        np.testing.assert_array_equal(other, t1)
