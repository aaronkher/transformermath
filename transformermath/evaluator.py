import pandas as pd
import torch
import torch.nn as nn
from typing import Any, List, Tuple, Dict
from pathlib import Path
from transformermath.run_device import RunDevice

class Evaluator:
    def __init__(self):
        self.device = RunDevice().get_device()
        self.DATA_COLUMN_NAME = "data"
        self.DATA_INDEX_NAME = "data_id_mappings"
        self.DATA_LABEL_NAME = "labels"
        self.PAD = 13

    def model_prediction(self, model, input_tensor):
        logits, _, _ = model(input_tensor)
        probs = nn.functional.softmax(logits, dim=-1)
        _, predicted_label = probs.max(dim=1)
        predicted_label = predicted_label.item()
        return predicted_label 

    def token_level_accuracy_evaluate(self, probs, Y):
        top_preds = probs.argmax(dim=1)
        correct = (top_preds == Y).sum().item()
        return correct / Y.size(0)

    def evaluate_model_grouped_df(self, data, model):
        grouped_data = data.groupby(self.DATA_INDEX_NAME)
        correctly_predicted = 0
        for _, group in grouped_data:
            all_correct = True
            for _, row in group.iterrows():
                input_tensor = torch.tensor([row[self.DATA_COLUMN_NAME]], dtype=torch.long).to(self.device)
                predicted_label = self.model_prediction(model, input_tensor)
                if predicted_label != row[self.DATA_LABEL_NAME]:
                    all_correct = False
                    break
            if all_correct:
                correctly_predicted += 1
        return correctly_predicted / len(grouped_data)

    def evaluate(self, model, evaluation_data):
        model.eval()
        
        # Preparing the data
        data_next_token_pred = evaluation_data
        grouped_data = data_next_token_pred.groupby(self.DATA_INDEX_NAME)[[self.DATA_COLUMN_NAME, self.DATA_LABEL_NAME]].agg(list).reset_index()

        # Evaluate token-level accuracy
        all_inputs = [item for sublist in grouped_data[self.DATA_COLUMN_NAME].tolist() for item in sublist]
        all_labels = [item for sublist in grouped_data[self.DATA_LABEL_NAME].tolist() for item in sublist]
        
        input_tensor = torch.tensor(all_inputs, dtype=torch.long).to(self.device)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long).to(self.device)

        logits, _, _ = model(input_tensor)
        probs = nn.functional.softmax(logits, dim=-1)
        
        token_accuracy = self.token_level_accuracy_evaluate(probs, labels_tensor)

        # Evaluate grouped accuracy
        overall_accuracy = self.evaluate_model_grouped_df(data_next_token_pred, model)

        return overall_accuracy, token_accuracy