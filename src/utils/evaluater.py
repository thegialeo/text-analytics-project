from os.path import abspath, dirname, join

import clustering
import pandas as pd
import vectorizer
from nltk.corpus import stopwords
from sklearn.metrics import homogeneity_score, silhouette_score


def evaluate_clustering(vec='tfidf', cluster='kmeans', dim_reduc='PCA'):
    """Perform clustering, dimension reduction on TextComplexityDE19 data.
       Evaluate clustering by homogeneity and silhouette score.
       
       Written by Leo Nguyen. Contact Xenovortex, if problems arises.
    
    Args:
        vec (str, optional): vectorizer method to used (options: 'tfidf', 'count', 'hash'), default: 'tfidf'
        cluster (str, optional): clustering method to used (options: 'kmeans', 'AP', 'mean_shift', 'spectral', 'Agg', 'DBSCAN', 'OPTICS', 'Birch'), default: 'kmeans'
        dim_reduc (str, optional): dimension reduction method to used (options: 'PCA', 'TSNE'), default: 'PCA'

    Return:
        homo_score: homogeneity score
        sil_score: silhouette score
    """

    # centroid methods
    centroid_methods = ['kmeans', 'AP', 'mean_shift']

    # read data
    data_path = join(dirname(dirname(dirname(abspath(__file__)))), "data", "TextComplexityDE19")
    df_ratings = pd.read_csv(
     join(data_path, "ratings.csv"),
     sep = ",", encoding = "ISO-8859-1")
     
    # feature extraction
    german_stopwords = stopwords.words('german')
    features = vectorizer.vectorizer_wrapper(df_ratings.Sentence.values, vec, german_stopwords)
    features = features.toarray()
  
    # Clustering and Dimension Reduction
    if cluster in centroid_methods:
        cls_object, reduced_features, reduced_cluster_centers = clustering.clustering_wrapper(features, cluster, dim_reduc)
    else: 
        cls_object, reduced_features = clustering.clustering_wrapper(features, cluster, dim_reduc)

    # Evaluate homogeneity score
    homo_score = homogeneity_score(df_ratings.MOS_Complexity.values.round(0), cls_object.labels_)

    # Evaluate silhouette score
    sil_score = silhouette_score(features, labels=cls_object.labels_)

    return homo_score, sil_score



def evaluate_baseline(method='linear'):
    """Perform baseline regression on TextComplexityDE19 data.
       Evaluate RMSE, MAE and R squares

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.

    Args:
        method (str, optional): [description]. Defaults to 'linear'.
    """

    # read data
    data_path = join(dirname(dirname(dirname(abspath(__file__)))), "data", "TextComplexityDE19")
    df_ratings = pd.read_csv(join(data_path, "ratings.csv"), sep = ",", encoding = "ISO-8859-1")

    # feature extration