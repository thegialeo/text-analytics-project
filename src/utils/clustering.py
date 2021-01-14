from sklearn.cluster import (AffinityPropagation, MeanShift, MiniBatchKMeans,
                             SpectralClustering)
from sklearn.decomposition import PCA


def clustering_wrapper(features, cluster_method='kmeans', dim_reduc='PCA'):
    """Performs clustering, dimension reduction to 2d space and plots the result (see folder figures).
       Evaluate homogeneity and silhouette score (see folder results).

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.

    Args:
        features (array-like): matrix with dimension (number samples, number features)
        cluster_method (str, optional): Select clustering method. Implemented so far are: 'kmeans', 'AP'. Defaults to 'kmeans'.
        dim_reduc (str, optional): Select dimension reduction method. Implemented so far are: 'PCA'. Defaults to 'PCA'.

    Return:
        sklearn_cls (sklearn.cluster class object): sklearn object (see documentation for sklearn.cluster)
        reduced_features (numpy array): features after applying dimension reduction
        reduced_cluster_centers (numpy array): cluster centers after applying dimension reduction
    """

    # perform selected clustering method
    if cluster_method == 'kmeans':
        sklearn_cls = MiniBatchKMeans(n_clusters=6, random_state=0)
        sklearn_cls.fit(features)
        sklearn_cls.predict(features)
        centroid_method = True
    elif cluster_method == 'AP':
        sklearn_cls = AffinityPropagation(random_state=0)
        sklearn_cls.fit(features)
        sklearn_cls.predict(features)
        centroid_method = True
    elif cluster_method == 'mean_shift':
        sklearn_cls = MeanShift()
        sklearn_cls.fit(features)
        sklearn_cls.predict(features)
        centroid_method = True
    elif cluster_method == 'spectral':
        sklearn_cls = SpectralClustering(n_clusters=6, assign_labels='discretize', random_state=0)
        sklearn_cls.fit(features)
        centroid_method = False
    else:
        print("Clustering method {} is not implemented yet. Please select one of the following options: 'kmeans'".format(
            cluster_method))
        exit()

    # perform selected dimension reduction
    if dim_reduc == 'PCA':
        pca = PCA(n_components=2, random_state=0)
        reduced_features = pca.fit_transform(features)
        if centroid_method:
            reduced_cluster_centers = pca.transform(sklearn_cls.cluster_centers_)
    else:
        print("Dimension Reduction method {} is not implemented yet. Please select one the folling options: 'PCA'".format(dim_reduc))
        exit()

    if centroid_method:
        return sklearn_cls, reduced_features, reduced_cluster_centers
    else:
        return sklearn_cls, reduced_features
