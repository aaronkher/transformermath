import math
from typing import Any
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

# positional_embeddings  # token embeddings + positional encodings
 

@dataclass
class ModelComponents:
    input_batch: Any
    positional_encodings: Any
    positional_embeddings: Any
    token_embeddings: Any
    tok_pos_embd: Any
    attn_outputs: Any # output of attention applied to input
    block_outputs: Any  # output of each block
    output: Any  # final output of transformer (one before logits)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        # attention maps
        self.attention_weights = att.detach()

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y, self.attention_weights

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.nonlin = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.nonlin(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        attn_output, attn_weights = self.attn(self.ln_1(x)) # x -> LayerNorm -> Attn -> x_out
        x = x + attn_output # x + x_out // residual connection 
        x = x + self.mlp(self.ln_2(x)) # x -> LayerNorm -> MLP -> x_out
        return x, attn_weights, attn_output # x + x_out  // residual connection 

@dataclass
class GPTConfig:
    # these are default GPT-2 hyperparameters
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = False

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %d" % (sum(p.nelement() for p in self.parameters()),))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        # import pdb; pdb.set_trace()
        device = idx.device
        #NOTE:
        #print(f"device: {device}")
        try:
            b, t = idx.size()
            
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

            # forward the GPT model itself
            tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
            x = tok_emb + pos_emb

            tok_pos_embd = x  # store this
            attention_maps = [] # attention weights
            block_outputs = [] # output of block
            attn_outputs = [] # output of attention applied to input + LayerNorm
            for block in self.transformer.h:
                x, attn_weights, attn_output = block(x)
                attention_maps.append(attn_weights) 
                block_outputs.append(x)   
                attn_outputs.append(attn_output)

            output = self.transformer.ln_f(x)
            logits = self.lm_head(output[:, -1, :]) # note: only returning logits at the last time step (-1), output is 2D (b, vocab_size)
            

            model_components = ModelComponents(input_batch=idx, positional_encodings=pos, positional_embeddings=pos_emb,
                                               token_embeddings=tok_emb, tok_pos_embd=tok_pos_embd, attn_outputs=attn_outputs, 
                                               block_outputs=block_outputs, output=output)
            
            return logits, attention_maps, model_components
        
        except Exception as e:
            print(f"Error during the forward pass of the GPT model: {e}")
            raise

"""Config and create GPT"""

# TODO: cite this in dissertation report
# citation: https://github.com/H-TayyarMadabushi
