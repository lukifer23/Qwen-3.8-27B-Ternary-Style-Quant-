"""Write a local qwen35 GGUF from reconstructed checkpoints + the official local shards.

No Hugging Face Hub. No student HF tree. Tokenizer and weights come from
models/source/Qwen3.8-27B on disk.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from q38ternary.config import AppConfig
from q38ternary.gguf.prism_format import PRISM_Q2_GROUP, pack_q2_0
from q38ternary.quant.grouping import group_axis, ungroup_axis
from q38ternary.safetensors_io import ShardIndex

log = logging.getLogger("q38ternary.gguf.writer")


def _prism_gguf():
    root = Path(__file__).resolve().parents[3] / "third_party" / "prism-llama.cpp" / "gguf-py"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import gguf  # type: ignore

    return gguf


def _canonical(name: str) -> str:
    return name.replace("model.language_model.", "model.")


def _load_recon_index(ckpt_root: Path) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for layer_dir in sorted(ckpt_root.glob("layer_*")):
        meta = layer_dir / "tensors.json"
        if not meta.is_file():
            continue
        for item in json.loads(meta.read_text(encoding="utf-8")):
            item = dict(item)
            item["dir"] = str(layer_dir)
            found[item["name"]] = item
    return found


def _dequant(item: dict) -> np.ndarray:
    folder = Path(item["dir"])
    codes = np.load(folder / item["codes"])
    scales = np.load(folder / item["scales"])
    grouped, _ = group_axis(codes.astype(np.float32), int(item["group_size"]))
    n_groups = grouped.shape[-2]
    scale = scales.reshape(*grouped.shape[:-2], n_groups, 1)
    recon = grouped * scale.astype(np.float32)
    return ungroup_axis(recon, int(item["last_dim"]))


def _reorder_v(tensor: np.ndarray, dim: int, n_k: int, n_v_per_k: int, head_dim: int) -> np.ndarray:
    if dim < 0:
        dim += tensor.ndim
    shape = list(tensor.shape)
    new_shape = shape[:dim] + [n_k, n_v_per_k, head_dim] + shape[dim + 1 :]
    t = tensor.reshape(new_shape)
    perm = list(range(t.ndim))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return np.transpose(t, perm).reshape(shape)


def _apply_qwen35_layout(name: str, data: np.ndarray, text: dict) -> np.ndarray:
    n_k = int(text.get("linear_num_key_heads") or 16)
    n_v = int(text.get("linear_num_value_heads") or 48)
    head_k = int(text.get("linear_key_head_dim") or 128)
    head_v = int(text.get("linear_value_head_dim") or 128)
    n_v_per_k = n_v // n_k
    if n_k == n_v or "linear_attn." not in name:
        return data
    if name.endswith(".in_proj_qkv.weight"):
        q_dim = head_k * n_k
        k_dim = head_k * n_k
        q, k, v = data[:q_dim], data[q_dim : q_dim + k_dim], data[q_dim + k_dim :]
        v = _reorder_v(v, 0, n_k, n_v_per_k, head_v)
        return np.concatenate([q, k, v], axis=0)
    if name.endswith(".in_proj_z.weight"):
        return _reorder_v(data, 0, n_k, n_v_per_k, head_v)
    if name.endswith(".in_proj_a.weight") or name.endswith(".in_proj_b.weight"):
        return _reorder_v(data, 0, n_k, n_v_per_k, 1)
    if name.endswith(".A_log") or name.endswith(".dt_bias"):
        if data.ndim == 1:
            return _reorder_v(data.reshape(-1, 1), 0, n_k, n_v_per_k, 1).reshape(-1)
        return _reorder_v(data, -1, n_k, n_v_per_k, 1)
    if "conv1d" in name:
        squeezed = np.squeeze(data)
        qk = head_k * n_k * 2
        qk_part, v_part = squeezed[:qk], squeezed[qk:]
        v_part = _reorder_v(v_part, 0, n_k, n_v_per_k, head_v)
        return np.concatenate([qk_part, v_part], axis=0)
    if name.endswith(".out_proj.weight"):
        return _reorder_v(data, 1, n_k, n_v_per_k, head_v)
    return data


def _prepare_tensor(hf_name: str, data: np.ndarray, text: dict) -> np.ndarray:
    data = np.asarray(data)
    if hf_name.endswith(".A_log"):
        data = -np.exp(data.astype(np.float32))
    elif hf_name.endswith("norm.weight") and "linear_attn.norm.weight" not in hf_name:
        data = data.astype(np.float32) + 1.0
    if "conv1d" in hf_name:
        data = np.squeeze(data)
    data = _apply_qwen35_layout(hf_name, data, text)
    return np.ascontiguousarray(data)


def _write_vocab(gguf, writer, model_dir: Path, vocab_size: int) -> None:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    vocab = tok.get_vocab()
    tokens = [f"[PAD{i}]" for i in range(vocab_size)]
    types = [gguf.TokenType.UNUSED] * vocab_size
    for piece, idx in vocab.items():
        if 0 <= idx < vocab_size:
            tokens[idx] = piece
            types[idx] = gguf.TokenType.CONTROL if piece.startswith("<|") and piece.endswith("|>") else gguf.TokenType.NORMAL
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen35")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    special = gguf.SpecialVocab(str(model_dir), load_merges=True)
    special.add_to_gguf(writer)


def write_local_gguf(cfg: AppConfig, ckpt_root: Path, outfile: Path) -> Path:
    gguf = _prism_gguf()
    src = cfg.model_local_dir
    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    n_layer = int(text["num_hidden_layers"])
    hidden = int(text["hidden_size"])
    vocab = int(text["vocab_size"])
    recon = _load_recon_index(ckpt_root)
    name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35, n_layer)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    tmp = outfile.with_suffix(".gguf.partial")
    if tmp.is_file():
        tmp.unlink()

    writer = gguf.GGUFWriter(str(tmp), arch="qwen35", use_temp_file=True)
    writer.add_name("qwen38-ternary-v01")
    writer.add_block_count(n_layer)
    writer.add_context_length(int(text.get("max_position_embeddings") or 32768))
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(int(text["intermediate_size"]))
    writer.add_head_count(int(text["num_attention_heads"]))
    writer.add_head_count_kv(int(text["num_key_value_heads"]))
    writer.add_layer_norm_rms_eps(float(text.get("rms_norm_eps") or 1e-6))
    writer.add_full_attention_interval(int(text.get("full_attention_interval") or 4))
    writer.add_ssm_conv_kernel(int(text.get("linear_conv_kernel_dim") or 4))
    writer.add_ssm_state_size(int(text.get("linear_key_head_dim") or 128))
    writer.add_ssm_group_count(int(text.get("linear_num_key_heads") or 16))
    writer.add_ssm_time_step_rank(int(text.get("linear_num_value_heads") or 48))
    v_heads = int(text.get("linear_num_value_heads") or 48)
    v_dim = int(text.get("linear_value_head_dim") or 128)
    writer.add_ssm_inner_size(v_heads * v_dim)
    head_dim = int(text.get("head_dim") or 256)
    rope = text.get("rope_parameters") or {}
    writer.add_rope_dimension_count(int(head_dim * float(rope.get("partial_rotary_factor") or 0.25)))
    writer.add_rope_freq_base(float((rope.get("rope_theta") if isinstance(rope, dict) else None) or 10_000_000))
    writer.add_rope_dimension_sections([11, 11, 10, 0])
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q2_0)
    _write_vocab(gguf, writer, src, vocab)

    index = json.loads((src / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = dict(index.get("weight_map") or {})
    skip_prefixes = ("model.visual", "mtp.", "visual")

    with ShardIndex(src) as store:
        for hf_name in weight_map:
            if any(hf_name.startswith(p) or p in hf_name for p in skip_prefixes):
                continue
            if hf_name.startswith("model.visual") or hf_name.startswith("mtp."):
                continue
            key = _canonical(hf_name)
            if key.endswith(".dt_bias"):
                key = key[: -len(".dt_bias")] + ".dt_proj.bias"
            mapped = name_map.get_name(key, try_suffixes=(".weight", ".bias"))
            if mapped is None:
                log.warning("skip unmapped tensor %s", hf_name)
                continue

            item = recon.get(hf_name)
            if item is not None:
                data = _dequant(item)
            else:
                data = store.load_tensor(hf_name)
            data = _prepare_tensor(hf_name, np.asarray(data), text)

            is_embed = "embed_tokens" in hf_name or hf_name == "lm_head.weight"
            is_matrix = (
                data.ndim >= 2
                and data.size >= 4096
                and not hf_name.endswith("norm.weight")
                and not is_embed
            )
            if is_matrix:
                data_f = data.astype(np.float32)
                grouped, _ = group_axis(data_f, PRISM_Q2_GROUP)
                q = np.sign(grouped)
                q[np.abs(grouped) < 1e-12] = 0
                num = np.sum(grouped * q, axis=-1, keepdims=True)
                den = np.sum(q * q, axis=-1, keepdims=True)
                scales = np.divide(num, den, out=np.zeros_like(num), where=den != 0).astype(np.float32)
                codes = ungroup_axis(q.astype(np.int8), data.shape[-1]).reshape(data.shape)
                blob = pack_q2_0(codes, scales.reshape(-1), PRISM_Q2_GROUP)
                packed = np.frombuffer(blob, dtype=np.uint8)
                writer.add_tensor(mapped, packed, raw_shape=tuple(data.shape), raw_dtype=gguf.GGMLQuantizationType.Q2_0)
            elif is_embed:
                q8 = gguf.quants.quantize(data.astype(np.float32), gguf.GGMLQuantizationType.Q8_0)
                writer.add_tensor(mapped, q8, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            else:
                writer.add_tensor(mapped, np.ascontiguousarray(data.astype(np.float32)))
            log.info("gguf + %s -> %s %s", hf_name, mapped, data.shape)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    if outfile.is_file():
        outfile.unlink()
    tmp.replace(outfile)
    log.info("wrote %s (%.2f GB)", outfile, outfile.stat().st_size / 1024**3)
    return outfile
