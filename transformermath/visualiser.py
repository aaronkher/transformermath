from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objs as go
from sklearn.manifold import TSNE
import umap
import plotly.express as px


# Citations:
# https://saturncloud.io/blog/what-is-sklearn-pca-explained-variance-and-explained-variance-ratio-difference/#interpreting-explained-variance-and-explained-variance-ratio
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# https://plotly.com/python/pca-visualization/
# https://umap-learn.readthedocs.io/en/latest/plotting.html
# https://saturncloud.io/blog/what-is-sklearn-pca-explained-variance-and-explained-variance-ratio-difference/#interpreting-explained-variance-and-explained-variance-ratio


class Visualiser:

    def __init__(self):
        pass

    def plot_explained_variance(self, X):
        X_std = StandardScaler().fit_transform(X)
        pca = PCA()
        pca.fit(X_std)
        
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), explained_variance_ratio)
        plt.xlabel('Num Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Component')

        plt.subplot(1, 2, 2)
        plt.step(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, where='mid')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Cumulative Explained Variance')

        plt.show()


    def visualize_data_interactive_2d(self, X, labels, method, metadata = None):
        X_std = StandardScaler().fit_transform(X)

        if method == 'pca':
            model = PCA(n_components=2)
        elif method == 'tsne':
            model = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42, early_exaggeration=18)
        elif method == 'umap':
            model = umap.UMAP(n_neighbors=15, min_dist=0.1, spread=1.0, random_state=42)
        else:
            raise ValueError("Invalid dimensionality reduction algorithm")

        X_reduced = model.fit_transform(X_std)

        df = pd.DataFrame(X_reduced, columns=[f'{method.upper()} 1', f'{method.upper()} 2'])
        sorted_labels = sorted(set(labels))  
        df['Label'] = pd.Categorical(labels, categories=sorted_labels, ordered=True)

        if metadata is not None:
            df['Metadata'] = metadata

        fig = px.scatter(
            df, x=f'{method.upper()} 1', y=f'{method.upper()} 2', 
            color='Label',
            hover_data=['Metadata'] if metadata is not None else None,
            labels={'color': 'Label'},
            title='Visualisation',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        fig.update_layout({
            'plot_bgcolor': 'white',  
            'paper_bgcolor': 'white', 
            'xaxis': {'showgrid': False, 'zeroline': False, 'showline': True, 'linewidth': 1, 'linecolor': '#000000'},  
            'yaxis': {'showgrid': False, 'zeroline': False, 'showline': True, 'linewidth': 1, 'linecolor': '#000000'},  
        })

        fig.update_layout(legend_title_text='Label')
        fig.update_traces(marker=dict(size=5, opacity=0.8))
        fig.show()


