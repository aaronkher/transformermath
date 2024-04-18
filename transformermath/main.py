from transformermath.attention_analyzer import AttentionAnalyzer
from transformermath.data_creator import DataCreator
from transformermath.encoded_representations import EncodedRepresentations
from transformermath.evaluator import Evaluator
from transformermath.prober import Prober
from transformermath.tokenizer import Tokenizer, AdditionTokenizer, SequencesTokenizer, SubtractionTokenizer
from transformermath.trainer import Trainer
from transformermath.visualiser import Visualiser


"""
Data creation methods
"""
def generate_addition_data(num_equations, n_digits, maximum_sequence_length = None, exclude_items = []):
    data_creator = DataCreator()
    trainX, trainY, train_items, train_data = data_creator.generate_addition_data(num_equations=num_equations, n_digits=n_digits, 
                                                                                  maximum_sequence_length=maximum_sequence_length,
                                                                                  exclude_items=exclude_items)        
    return trainX, trainY, train_items, train_data

def generate_subtraction_data(num_equations, n_digitsA, n_digitsB, maximum_sequence_length = None, exclude_items = []):
    data_creator = DataCreator()
    trainX, trainY, train_items, train_data = data_creator.generate_subtraction_data(num_equations=num_equations, digitsA=n_digitsA, digitsB=n_digitsB, 
                                                                                     maximum_sequence_length=maximum_sequence_length,
                                                                                     exclude_items=exclude_items)       
    return trainX, trainY, train_items, train_data

def generate_sequence_data(num_equations, n_digits, maximum_sequence_length = None, exclude_items = []):
    data_creator = DataCreator()
    trainX, trainY, train_items, train_data = data_creator.generate_addition_data(num_equations=num_equations, n_digits=n_digits, 
                                                                                  maximum_sequence_length=maximum_sequence_length, 
                                                                                  exclude_items=exclude_items)        
    return trainX, trainY, train_items, train_data


"""
Train model
"""
def train(X, Y, epochs, layers, heads, embd_dim, vocab_size=14):
    trainer = Trainer(vocab_size=vocab_size)
    model_architecture = trainer.create_gpt_model(layers=layers, heads=heads, embd_dim=embd_dim)
    models, losses = trainer.train(trainX=X, trainY=Y, model_architecture=model_architecture, epochs=epochs)
    return models, losses

"""
Evaluate model
"""
def evaluate(model, evaluation_df):
    evaluator = Evaluator()
    overall_accuracy, token_accuracy = evaluator.evaluate(model=model, evaluation_data=evaluation_df)
    return overall_accuracy, token_accuracy


"""
Retrieve encoded representations
"""
def probe_encoded_representations(model, X):
    prober = Prober()
    examples, predictions, attention_maps, attention_outputs, block_outputs = prober.get_model_internals(model=model, inferenceX=X)
    return examples, predictions, attention_maps, attention_outputs, block_outputs

"""
Embedding analysis methods
"""
def get_token_embedding_representations(layer_outputs, layer, sequences):
    represent = EncodedRepresentations()
    token_representation_data = represent.get_transformed_output_tokens(outputs=layer_outputs, sequences=sequences,layer=layer)
    return token_representation_data

def get_sequence_embedding_representations(layer_outputs, layer, sequences):
    represent = EncodedRepresentations()
    sequence_representation_data = represent.get_transformed_output_sequence_pooled(outputs=layer_outputs, sequences=sequences,layer=layer)
    return sequence_representation_data

"""
Visualisation via dimensionality
reduction method
"""
def visualise_sequence_embeddings(sequence_embeddings_data, label_category, visualise_method):
    vis = Visualiser()
    sequence_embeddings, sequences, labels = [], [], []
    for representation in sequence_embeddings_data:
        sequence_embeddings.append(representation.pooled_representation)
        sequences.append(representation.encoded_sequence)
        labels.append(label_category)
    vis.visualize_data_interactive_2d(X=sequence_embeddings, labels=labels, method=visualise_method, metadata=sequences)


def visualise_token_embeddings(token_embeddings_data, visualise_method):
    vis = Visualiser()
    token_embeddings, tokens, sequences = [],[],[]
    for representation in token_embeddings_data:
        token_embeddings.append(representation.token_embedding_representation)
        tokens.append(representation.encoded_token)
        sequences.append(representation.encoded_sequence)
    vis.visualize_data_interactive_2d(X=token_embeddings, labels=tokens, method=visualise_method, metadata=sequences)


def plot_explained_variance(X):
    vis = Visualiser()
    vis.plot_explained_variance(X=X)


"""
Attention weight analysis methods
"""
def display_attention_map(sequences, attention_weights, layer, head, sequence_index, tokenizer):
    attention_analyzer = AttentionAnalyzer(tokenizer=tokenizer) 
    s,a = sequences[sequence_index], attention_weights[sequence_index]
    attention_analyzer.display_attention_map(sequence=s,attention_weights=a,layer=layer, head=head)


def display_average_attention_map(attention_weights, layer, head, tokenizer):
    attention_analyzer = AttentionAnalyzer(tokenizer=tokenizer) 
    attention_analyzer.display_average_attention_map(attention_data=attention_weights, layer=layer, head=head)

def display_attention_distributional_histogram(attention_weights_ID, attention_weights_OOD, layer, head, tokenizer):
    attention_analyzer = AttentionAnalyzer(tokenizer=tokenizer) 
    ID_weights = attention_analyzer.flatten_attention_weights(attention_maps=attention_weights_ID, layer=layer,head=head)
    OOD_weights = attention_analyzer.flatten_attention_weights(attention_maps=attention_weights_OOD,layer=layer,head=head)
    attention_analyzer.plot_attention_weight_histogram(ID_values=ID_weights,OOD_values=OOD_weights)


def perform_distributional_significance_tests(attention_weights_ID, attention_weights_OOD, layer, head, tokenizer):
    attention_analyzer = AttentionAnalyzer(tokenizer=tokenizer) 
    ID_weights = attention_analyzer.flatten_attention_weights(attention_maps=attention_weights_ID,layer=layer,head=head)
    OOD_weights = attention_analyzer.flatten_attention_weights(attention_maps=attention_weights_OOD,layer=layer,head=head)

    ks_statistic, ks_p_value = attention_analyzer.compare_attention_maps_ks(values_1=ID_weights,values_2=OOD_weights)
    cosine_sim = attention_analyzer.compare_attention_maps_cosine_similarity(values_1=ID_weights,values_2=OOD_weights)
    energy_distance = attention_analyzer.compare_attention_maps_energy_distance(values_1=ID_weights,values_2=OOD_weights)

    return ks_statistic, ks_p_value, cosine_sim, energy_distance


def get_addition_tokenizer():
    return AdditionTokenizer().REPLACER

def get_sequence_tokenizer():
    return SequencesTokenizer().REPLACER

def get_subtraction_tokenizer():
    return SubtractionTokenizer().REPLACER
