import torch
import torch.nn as nn
import math


"""
This file contains utility classes and functions for the GPT-2 model implementation. These utilities include custom activation functions, layer normalization, and other helper functions that are used throughout the model architecture.

"""

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    This activation function is used in the GPT-2 model for non-linear transformations.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (0.5* x* (1+ torch.tanh(math.sqrt(2/math.pi)*(x+0.044715*torch.pow(x,3)))))

class LayerNorm(nn.Module):
    """
    Layer Normalization Module. 
    Helps in normalizing the inputs across the features to stabilize and accelerate training. 
    Done to prevent internal covariate Shift and to make the model more robust to different input distributions.
    Also prevent exploding or vanishing gradients by normalizing the inputs to each layer.
    
    """

    def __init__(self,cfg, eps=1e-5):
        super().__init__()
        self.eps=eps
        self.gamma=nn.Parameter(torch.ones(cfg['embedding_dim']))
        self.beta=nn.Parameter(torch.zeros(cfg['embedding_dim']))

    def forward(self,x):
        mean=x.mean(dim=-1, keepdim=True)
        var=x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm=(x-mean)/torch.sqrt(var+self.eps)

        return self.gamma*x_norm+self.beta

class CausalMultiHeadAttention(nn.Module):
    """
    Causal Multi-Head Attention Module.
    This module implements the multi-head attention mechanism with causal masking, allowing the model to attend to previous tokens in the sequence while preventing information leakage from future tokens.
    """
    def __init__(self,config):
        super().__init__()
        self.num_heads=config['num_heads']
        self.num_layers=config['num_layers']
        self.embedding_dim=config['embedding_dim']
        self.layer_dim=self.embedding_dim//self.num_heads
        self.W_k=nn.Linear(self.embedding_dim, self.embedding_dim, bias=config['qkv_bias'])
        self.W_v=nn.Linear(self.embedding_dim, self.embedding_dim, bias=config['qkv_bias'])
        self.W_q=nn.Linear(self.embedding_dim, self.embedding_dim, bias=config['qkv_bias'])
        self.projection_layer=nn.Linear(self.embedding_dim, self.embedding_dim, bias=config['mha_projection_bias'])
        self.register_buffer('mask', torch.triu(torch.ones(config['context_length'], config['context_length']), diagonal=1))

    def forward(self,x):
        batch_size, seq_length,_ =x.shape

        keys=self.W_k(x)
        values= self.W_v(x)
        quries=self.W_q(x)

        keys=keys.view(batch_size, seq_length, self.num_heads, self.layer_dim).transpose(1,2)
        values=values.view(batch_size, seq_length, self.num_heads, self.layer_dim).transpose(1,2)
        quries=quries.view(batch_size, seq_length, self.num_heads, self.layer_dim).transpose(1,2)

        attention_scores= (quries @ keys.transpose(-2,-1))//math.sqrt(self.layer_dim)

        attention_scores=attention_scores.masked_fill(self.mask[:seq_length,:seq_length]==1, float('-inf'))

        attention_weights=torch.softmax(attention_scores, dim=-1)
        context_vectors= attention_weights@ values

        context_vectors=context_vectors.transpose(1,2).contiguous().view(batch_size, seq_length, self.embedding_dim)
        context_vectors=self.projection_layer(context_vectors)

        return context_vectors

class FeedForward(nn.Module):
    """
    Feed Forward Neural Network Module.
    This module implements a two-layer feed-forward neural network with GELU activation and dropout, used in the GPT-2 model for non-linear transformations of the input representations.
    """
    def __init__(self,config):
        super().__init__()
        self.linear1=nn.Linear(config['embedding_dim'], 4*config['embedding_dim'])
        self.linear2=nn.Linear(4*config['embedding_dim'], config['embedding_dim'])
        self.activation=GELU()

    def forward(self,x):
        x=self.linear1(x)
        x=self.activation(x)
        x=self.linear2(x)

        return x