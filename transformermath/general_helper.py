import sys
import torch
from transformermath.run_device import RunDevice
from typing import List, Tuple

class GeneralHelper:
    def __init__(self) -> None:
        self.device = RunDevice().get_device()

    def load_model(self, model_path:str):
        model = torch.load(model_path)
        model.eval()
        return model
    
    def tensor_data(self, input_data: List, labels: List) -> Tuple:
        TX = torch.tensor(input_data, dtype=torch.long).to(self.device)
        TY = torch.tensor(labels, dtype=torch.long).to(self.device)
        return TX, TY
    
    def decode_examples(self, examples, replacer):
        decoded_examples = []
        inv_token_encodings = {v: k for k, v in replacer.items()}
        for example in examples:
            decoded_example = [inv_token_encodings.get(token, '') for token in example]
            decoded_examples.append(decoded_example)
        return decoded_examples
    
    def decode_labels(self, labels, replacer):
        inv_token_encodings = {v: k for k, v in replacer.items()}
        decoded_labels = [inv_token_encodings.get(label, '') for label in labels]
        return decoded_labels