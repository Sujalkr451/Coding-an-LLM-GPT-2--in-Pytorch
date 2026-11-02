import torch 
import torch.nn as nn 

from .utils import CausalMultiHeadAttention, FeedForward
from .utils import LayerNorm 
from .config import GPT_CONFIG


class TransformerBlock(nn.Module):
    """
    Transformer Block Module.
    This module implements a single transformer block, consisting of a causal multi-head attention layer followed by a feed-forward neural network, with layer normalization and residual connections.
    """
    def __init__(self,config):
        super().__init__()
        self.mha=CausalMultiHeadAttention(config)
        self.feedforward=FeedForward(config)
        self.layernorm1=LayerNorm(config)
        self.layernorm2=LayerNorm(config)
        self.dropout=nn.Dropout(config['dropout'])

    def forward(self,x):
        residual1=x
        x=self.layernorm1(x)
        x=self.mha(x)
        x=self.dropout(x)
        x=x+residual1
        residual2=x
        x=self.layernorm2(x)
        x=self.feedforward(x)
        x=self.dropout(x)
        x=x+residual2
        return x

class GPT2Model(nn.Module):
    """
    GPT-2 Model Module.
    This module implements the GPT-2 model, consisting of an embedding layer, multiple transformer blocks, and a final linear layer for output generation.
    """
    def __init__(self, config):
        super().__init__()

        self.token_embedding=nn.Embedding(config['vocab_size'], config['embedding_dim'])
        self.postional_embedding=nn.Embedding(config['context_length'], config['embedding_dim'])
        self.dropout=nn.Dropout(config['dropout'])
        self.transformer_blocks=nn.Sequential(*[TransformerBlock(config) for _ in range(config['num_layers'])])
        
        self.final_layernorm=LayerNorm(config)
        self.output_head=nn.Linear(config['embedding_dim'], config['vocab_size'], bias=False)


    def forward(self,input_ids):
        batch_size, seq_length=input_ids.shape

        token_embedding=self.token_embedding(input_ids)

        positions_embed=self.postional_embedding(torch.arange(0, seq_length, device=input_ids.device))

        x=token_embedding+positions_embed
        
        x=self.dropout(x)

        x=self.transformer_blocks(x)
        
        x=self.final_layernorm(x)

        logits=self.output_head(x)

        return logits
        