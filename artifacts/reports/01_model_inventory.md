# Model inventory

- repo: `Qwen/Qwen3.8-27B`
- tensors: 1199
- shards: 18
- index total_size: 51.747 GB
- language parameters (sum of known shapes): 26,895,998,464

| category | tensors | parameters | BF16 GB |
|---|---:|---:|---:|
| embedding | 1 | 1,271,398,400 | 2.368 |
| lm_head | 1 | 1,271,398,400 | 2.368 |
| full_attention | 64 | 1,677,721,600 | 3.125 |
| gated_deltanet | 384 | 5,562,044,928 | 10.36 |
| mlp | 192 | 17,112,760,320 | 31.875 |
| norm | 209 | 674,816 | 0.001 |
| mtp | 15 | 424,699,392 | 0.791 |
| vision | 333 | 460,730,096 | 0.858 |
| other | 0 | 0 | 0.0 |

Shapes come from Hugging Face safetensors metadata, not from loading shards.
