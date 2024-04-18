
from dataclasses import dataclass
from typing import List

@dataclass
class TokenRepresentation:
    encoded_token: int
    token_embedding_representation: List[int]
    encoded_sequence: List[int]
    token_position_in_sequence: int

@dataclass
class SequenceRepresentation:
    pooled_representation: List[float]
    encoded_sequence: List[int]

class EncodedRepresentations:
    def __init__(self) -> None:
        pass

    """
    Gets the transformed output pooled-sequence
    """
    def get_transformed_output_sequence_pooled(self, outputs, sequences, layer):
        
        def _mean_pool_sequence(transformed_sequence):
            if len(transformed_sequence) == 0 or len(transformed_sequence[0]) == 0:
                raise ValueError("Error in sequence")
            
            num_features = len(transformed_sequence[0])
            mean_pooled_sequence = [0.0] * num_features
            
            for feature_index in range(num_features):
                sum_value = sum(embedding[feature_index] for embedding in transformed_sequence)
                mean_pooled_sequence[feature_index] = sum_value / len(transformed_sequence)

            return mean_pooled_sequence
        
        sequence_representations: list[SequenceRepresentation] = []

        for sequence, sequence_output in zip(sequences, outputs):
            transformed_sequence = sequence_output[layer]
            
            pooled_sequence = _mean_pool_sequence(transformed_sequence)
            
            sequence_representations.append(SequenceRepresentation(
                pooled_representation=pooled_sequence,
                encoded_sequence=sequence
            ))

        return sequence_representations

       

    """
    Gets the token embedding representation at various.
    outputs: data-points x layers x tokens x embedding dimension
    sequences: data-points x tokens
    layer: int = the layer in the model
    """
    def get_transformed_output_tokens(self, outputs, sequences, layer):
        # digit encodings
        DIGITS = set([0,1,2,3,4,5,6,7,8,9])
        transformed_output_tokens: list[TokenRepresentation] = []

        # adds the embedding for each token after being transformed by component
        for sequence_index, sequence_output_for_all_layers in enumerate(outputs):
            sequence_output_per_layer = sequence_output_for_all_layers[layer]
            for j, token_representation in enumerate(sequence_output_per_layer):
                sequence = sequences[sequence_index]
                token = sequence[j]
                if token not in DIGITS:
                    continue

                transformed_output_tokens.append(TokenRepresentation(encoded_token=token,
                                                            token_embedding_representation=token_representation, 
                                                            encoded_sequence=sequence, 
                                                            token_position_in_sequence=j))

        return transformed_output_tokens