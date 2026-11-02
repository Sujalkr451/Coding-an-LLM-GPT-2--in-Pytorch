![Popeye working through a transformer block in his notebook](assets/popeye-gpt2-sketch.png)

# GPT-2, piece by piece

I’m building a GPT-2-style language model in PyTorch to understand how the pieces work together. The `gpt2/` folder now has the full forward pass: token and position embeddings, 12 transformer blocks, final normalization, and an output head that produces vocabulary logits.

## Inside `gpt2/`

| File | What’s in it |
| --- | --- |
| [`config.py`](gpt2/config.py) | Model dimensions, context length, dropout, and bias settings. |
| [`GPTModel.py`](gpt2/GPTModel.py) | `TransformerBlock` and `GPT2Model`, including embeddings, the block stack, and output head. |
| [`utils.py`](gpt2/utils.py) | Custom GELU, layer normalization, causal multi-head attention, and feed-forward layers. |

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

**Built from scratch:** causal multi-head attention (Q/K/V projections, head splitting, masking, and attention weights), layer normalization, GELU, the feed-forward network, and transformer blocks with residual connections. The model is assembled manually using PyTorch tensors and basic layers such as `nn.Linear`, `nn.Embedding`, and `nn.Dropout`.

The feed-forward network expands 768 features to 3,072, applies GELU, then projects back down. Dropout is applied around the embeddings and block residual branches.

## Current configuration

| Setting | Value |
| --- | ---: |
| Vocabulary size | 50,257 |
| Context length | 1,024 |
| Embedding dimension | 768 |
| Attention heads | 12 |
| Transformer layers | 12 |
| Dropout | 0.1 |

Query, key, value, and attention output projection biases are disabled. Token embeddings and the output head have separate weights.

## Parameters and space

The current model has **163,000,320 trainable parameters (163M)**. Its FP32 weights take about **652 MB (622 MiB)**. The 12 stored causal masks add another **48 MiB**, bringing weights plus buffers to roughly **670 MiB**, before checkpoint overhead. Training needs more memory for activations, gradients, and optimizer state.

## Work in progress

The architecture is wired up, but `gpt2/` doesn’t yet include a training loop, pretrained weight loading, or a text-generation loop. This is a learning implementation, with randomly initialized weights.

## License

This project is licensed under the [MIT License](LICENSE).
