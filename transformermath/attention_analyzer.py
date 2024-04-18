import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
from scipy.stats import ks_2samp
from scipy.stats import energy_distance
from scipy.spatial.distance import cosine
import scipy.stats as stats

# citations:
# 1. Energy distance: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.energy_distance.html
# 2. KS: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
# 3. Cosine sim: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html

class AttentionAnalyzer:

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

        self.REVERSE_REPLACER = {v: k for k, v in tokenizer.items()}


    def display_average_attention_map(self, attention_data, layer, head):
        
        def _compute_average_attention_map(attention_data, layer, head):
            all_maps = [data[layer][head] for data in attention_data]
            return np.mean(all_maps, axis=0)
        
        average_map = _compute_average_attention_map(attention_data, layer, head)
        plt.figure(figsize=(10, 8))
        sns.heatmap(average_map, annot=True, cmap='Blues', fmt=".2f")
        plt.title(f"Average Attention Map for Layer {layer+1}, Head {head+1}")
        plt.xlabel("Token Position")
        plt.ylabel("Token Position")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.show()

    """
    Displays attention map for a single sequence
    """
    def display_attention_map(self, sequence, attention_weights, layer, head):
        attention_map = attention_weights[layer][head]  

        tokens = [self.REVERSE_REPLACER[token] for token in sequence]

        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_map, annot=True, cmap='Blues', fmt=".2f", xticklabels=tokens, yticklabels=tokens)
        plt.title(f"Attention Map for Layer {layer + 1}, Head {head + 1}")
        plt.xlabel("Tokens")
        plt.ylabel("Tokens")
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.show()

    """
    Flattens the attention weights
    """
    def flatten_attention_weights(self, attention_maps, layer, head):
        flattened_weights = []
        for _, attention_map in attention_maps:
            flattened_weights.extend(attention_map[layer][head].flatten())
        return flattened_weights

    def compare_attention_maps_energy_distance(self, values_1, values_2):
        return stats.energy_distance(values_1, values_2)
    
    def compare_attention_maps_ks(self, values_1, values_2):
        ks_statistic, p_value = ks_2samp(values_1, values_2)
        return ks_statistic, p_value
    
    def compare_attention_maps_cosine_similarity(self, values_1, values_2):
        values_1 = np.asarray(values_1)
        values_2 = np.asarray(values_2)
        similarity = 1 - cosine(values_1, values_2)
        return similarity

    def plot_attention_weight_histogram(self, ID_values, OOD_values, bins=30, title="Comparison of Attention Weights"):
        plt.figure(figsize=(10, 6))
        sns.histplot(ID_values, color='blue', label='ID', bins=bins, kde=False, stat="density", edgecolor='black')
        sns.histplot(OOD_values, color='green', label='OOD', bins=bins, kde=False, stat="density", edgecolor='black', alpha=0.75)
        
        plt.title(title)
        plt.xlabel("Attention Weights")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.show()