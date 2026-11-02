![Popeye working through a transformer block in his notebook](assets/popeye-gpt2-sketch.png)

# GPT-2, piece by piece

I’m building the pieces of GPT-2 in PyTorch to understand what happens inside a transformer. Right now, this folder has the attention layer, feed-forward network, and a transformer block wired together. The full language model is still to come.

## Inside `GPT-2/`

| File | What’s in it |
| --- | --- |
| [`config.py`](GPT-2/config.py) | Model dimensions, context length, dropout, and attention bias settings. |
| [`model.py`](GPT-2/model.py) | `TransformerBlock`: attention and feed-forward layers, each preceded by layer normalization and followed by a residual connection. |
| [`utils.py`](GPT-2/utils.py) | Custom GELU and layer normalization, causal multi-head attention, and the feed-forward network. |

The block is arranged like this:

```text
input
  ├─────────────────────────────┐
  ↓                             │
LayerNorm → causal attention → add
                                ├───────────────────────────┐
                                ↓                           │
                       LayerNorm → feed-forward → add → output
```

The attention layer splits queries, keys, and values across heads and masks future positions. The feed-forward network expands the embedding dimension by four, applies GELU and dropout, then projects back down.

## Current configuration

| Setting | Value |
| --- | ---: |
| Vocabulary size | 50,257 |
| Context length | 1,024 |
| Embedding dimension | 768 |
| Attention heads | 12 |
| Configured transformer layers | 12 |
| Dropout | 0.1 |

Query, key, value, and attention output projection biases are disabled in the config. The layer count is configured here; the stack of 12 blocks hasn’t been assembled yet.

## Work in progress

`GPT-2/` doesn’t yet include token or position embeddings, a full model wrapper, an output head, training, or text generation. These are building blocks I’m still working through, rather than a finished GPT-2 implementation.

## License

This project is licensed under the [MIT License](LICENSE).
