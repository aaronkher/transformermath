import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List, Tuple, Any
from transformermath.minGPT    import *
from transformermath.run_device import RunDevice

@dataclass
class TrainModel:
    model: GPT
    epoch: int

class Trainer:

    def __init__(self, vocab_size: int, context_length: int = 50, seed=123) -> None:
        self.seed = seed        
        run_device = RunDevice()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.device = run_device.get_device()
        torch.backends.cudnn.benchmark = True 


    def create_gpt_model(self, layers, heads, embd_dim) -> GPT:
        # get vocab size from tokens
        if embd_dim % heads != 0 :
            raise Exception( "Embd dim should be divisible by number of heads!" ) 
        
        config = GPTConfig(
            block_size = self.context_length,
            vocab_size = self.vocab_size,
            n_layer = layers,
            n_head = heads,
            n_embd = embd_dim,
            bias = False,
        )
        gpt_model = GPT(config)
        return gpt_model
    
    """
    Main function for running sequence experiment.
    """
    def run(self, epochs: int, gpt: GPT, Xs: List[List[int]], Ys: List[int], 
            batch_size: int = -1, training_optimiser: Any = None):
        
        sys.stdout.flush()
        if batch_size == -1:
            batch_size = len(Ys)
        
        print(f"|Batch Size: {batch_size}|")
        print(f"|Data Samples: {len(Ys)}|")
        print(f"|Training Device: {self.device}|")

        X = torch.tensor(Xs, dtype=torch.long)
        Y = torch.tensor(Ys, dtype=torch.long)
        X = X.to( self.device ) 
        Y = Y.to( self.device ) 
    
        gpt.to(self.device) 
        if not training_optimiser:
            optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3, weight_decay=1e-1)  
        else:
            optimizer = training_optimiser

        losses = list()
        models = list()
        try: 
            for epoch in range( epochs ):
                # save 0 epoch before the first forward pass
                if epoch == 0:
                    models.append(TrainModel(model=gpt,epoch=epoch))

                # shuffle data
                indices = torch.randperm(X.size(0), device=self.device)
                X_shuffled = X[indices]
                Y_shuffled = Y[indices]

                total_loss, num_batches = 0,0
                for i in range(0, X.size(0), batch_size):
                    batch_X = X_shuffled[i:i+batch_size]
                    batch_Y = Y_shuffled[i:i+batch_size]

                    optimizer.zero_grad()
                    try:
                        logits, _, _ = gpt(batch_X)
                        loss = F.cross_entropy(logits, batch_Y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1)
                        optimizer.step()

                        total_loss += loss.item()
                        num_batches += 1

                    except Exception as e:
                        print(f"Error during forward pass OR loss computation: {e}")
                        raise
                
                avg_loss = total_loss / num_batches
                losses.append(avg_loss)
            
                if epoch == epochs - 1: # last checkpoint (model after, x epochs of training)
                    models.append(TrainModel(model=gpt,epoch=epoch))
                    print(f"|Final Loss: {losses[-1]}|")
                    break

                if epoch % 50 == 0:
                    print(f"|Epoch {epoch}: Avg Loss = {avg_loss}|")
                    
        except RuntimeError as e:
            print("RuntimeError:", e)

        return models, losses

    def train(self, trainX: List[List[int]], trainY: List[int], model_architecture: GPT, epochs: int,
              batch_size: int = -1, optimizer: Any = None):
        
        models, losses = self.run(epochs=epochs, gpt=model_architecture, Xs=trainX, Ys=trainY, batch_size=batch_size, training_optimiser=optimizer)

        return models,losses




"""
    ## Notes

    Adapted from [HERE](https://colab.research.google.com/drive/1SiF0KZJp75rUeetKOWqpsA8clmHP6jMg?usp=sharing#scrollTo=xQmrWAhT6mkK
    )
"""