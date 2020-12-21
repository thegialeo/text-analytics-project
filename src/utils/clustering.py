from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_score, silhouette_score
import matplotlib.pyplot as plt

def clustering_wrapper(features, cluster_method='kmeans', dim_reduc='PCA', save_name=None):
    """Performs clustering, dimension reduction to 2d space and plots the result (see folder figures). 
       Evaluate homogeneity and silhouette score (see folder results). 

    Args:
        features ([array-like, sparse matrix]): matrix with dimension (number samples, number features)
        cluster_method (str, optional): Select clustering method. Implemented so far are: 'kmeans'. Defaults to 'kmeans'.
        dim_reduc (str, optional): Select dimension reduction method. Implemented so far are: 'PCA'. Defaults to 'PCA'.
        save_name ([type], optional): name to save plots and result text file under. Defaults to None (plot and result will not be saved!).
    """
    pass 