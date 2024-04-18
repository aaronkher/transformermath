from typing import List
import torch
from transformermath.minGPT import GPT
from transformermath.run_device import RunDevice


"""
returns...
# sequences: data-points x tokens 
# attention_weights: data-points x layers x heads x tokens x tokens
# block_outputs: data-points x layers x tokens x embedding dimension
# attention_outputs: data-points x layers x tokens x embedding dimension
# attention_weights: data-points x 
# predicitons: 1 x data-points
"""
class Prober():
    def __init__(self) -> None:
        run_device = RunDevice()
        self.device = run_device.get_device()
    
    """
    Performs inference of model to get internals
    """
    def get_model_internals(self, model: GPT, inferenceX: List[List[int]]):
        examples = []
        predictions = []
        attention_maps = []
        attention_outputs = []
        block_outputs = []
        
        with torch.no_grad():
            X = inferenceX
            tensorX = torch.tensor(X, dtype=torch.long).to(self.device)
            logits, attn_maps, model_comps = model(tensorX)
            pred = torch.argmax(logits, dim=1)
            
            for batch_index in range(tensorX.size(0)):  
                examples.append(X[batch_index])    
                predictions.append(pred[batch_index].cpu().item()) 
                attention_maps.append([attn_map[batch_index].cpu().numpy() for attn_map in attn_maps])                
                attention_outputs.append([attn_out[batch_index].cpu().numpy() for attn_out in model_comps.attn_outputs])
                block_outputs.append([block_out[batch_index].cpu().numpy() for block_out in model_comps.block_outputs])

        return examples, predictions, attention_maps, attention_outputs, block_outputs
