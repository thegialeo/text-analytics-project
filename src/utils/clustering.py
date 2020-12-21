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
        save_name (str, optional): name to save plots and result text file under. Defaults to None (plot and result will not be saved!).
    """
    
    # perform selected clustering method
    if cluster_method == 'kmeans':
        clst = MiniBatchKMeans(n_clusters=6, random_state=0)
        clst.fit(features)
        clst.predict(features)
    else:
        print("Clustering method {} is not implemented yet. Please select one of the following options: 'kmeans'".format(cluster_method))
        exit()

    # perform selected dimension reduction
    if dim_reduc == 'PCA':
        pass
    else:
        