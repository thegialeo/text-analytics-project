from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_dim(features, method='PCA'):
    """Dimension reduction to 2d space.

    Args:
        features (array-like): matrix with dimension (number samples, number features)
        method (str, optional): Select dimension reduction method. (options: 'PCA', 'TSNE'). Defaults to 'PCA'.

    Return: 
        reduced_features: features reduce to 2d space with dimension (number samples, 2)
    """

    if method == 'PCA':
        pca = PCA(n_components=2, random_state=0)
        reduced_features = pca.fit_transform(features)
    elif method == 'TSNE':
        tsne = TSNE(n_components=2, random_state=0)
        reduced_features = tsne.fit_transform(features)
    else:
        print("Dimension Reduction method {} is not implemented yet. Please select one the folling options: 'PCA', 'TSNE'".format(method))
        exit()

    return reduced_features
