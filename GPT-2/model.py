import torch 
import torch.nn as nn 

from .utils import CausalMultiHeadAttention, FeedForward
from .utils import LayerNorm 


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

    def forward(self,x):
        residual1=x
        x=self.layernorm1(x)
        x=self.mha(x)
        x=x+residual1
        residual2=x
        x=self.layernorm2(x)
        x=self.feedforward(x)
        x=x+residual2
        return x
    