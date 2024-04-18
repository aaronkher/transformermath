from setuptools import setup, find_packages

setup(
    name='transformermath',
    version='0.1.13',  
    packages=find_packages(),
    install_requires=[
        'matplotlib==3.4.3',
        'numpy>=1.22',
        'pandas==1.3.5',
        'scikit-learn==1.0.2',
        'plotly==5.6.0',
        'umap-learn>=0.5.3',  
        'torch>=1.10.2',       
        'seaborn==0.11.2',
        'scipy==1.7.3',
    ],
)
