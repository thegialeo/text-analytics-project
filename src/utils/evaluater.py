from os.path import abspath, dirname, join

import clustering
import pandas as pd
import preprocessing
import regression
import vectorizer
from sklearn.metrics import (homogeneity_score, mean_absolute_error, mean_squared_error,
                             r2_score, silhouette_score)
from sklearn.model_selection import train_test_split


def evaluate_clustering(vec='tfidf', cluster='kmeans', dim_reduc='PCA', stopword='nltk'):
    """Perform clustering, dimension reduction on TextComplexityDE19 data.
       Evaluate clustering by homogeneity and silhouette score.
       
       Written by Leo Nguyen. Contact Xenovortex, if problems arises.
    
    Args:
        vec (str, optional): vectorizer method to used (options: 'tfidf', 'count', 'hash'), default: 'tfidf'
        cluster (str, optional): clustering method to used (options: 'kmeans', 'AP', 'mean_shift', 'spectral', 'Agg', 'DBSCAN', 'OPTICS', 'Birch'), default: 'kmeans'
        dim_reduc (str, optional): dimension reduction method to used (options: 'PCA', 'TSNE'), default: 'PCA'
        stopword (str, optional): source to load stopwords from (options: "spacy", "nltk", "stop_words", "german_plain", "german_full"). Defaults to "nltk".

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
    german_stopwords = preprocessing.get_stopwords(stopword)
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



def evaluate_baseline(vec='tfidf', method='linear', stopword='nltk'):
    """Perform baseline regression on TextComplexityDE19 data.
       Evaluate RMSE, MSE, MAE and R squares

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.

    Args:
        vec (str, optional): vectorizer method to used (options: 'tfidf', 'count', 'hash'), default: 'tfidf'
        method (str, optional): regression method to use (options: 'linear', 'lasso', 'ridge', 'elastic-net', 'random-forest'). Defaults to 'linear'.
        stopword (str, optional): source to load stopwords from (options: "spacy", "nltk", "stop_words", "german_plain", "german_full"). Defaults to "nltk".

    Return:
        MSE (double): Mean Square Error
        RMSE (double): Root Mean Square Error
        MAE (double): Mean Absolute Error
        r_square (double): R Square 
    """

    # read data
    data_path = join(dirname(dirname(dirname(abspath(__file__)))), "data", "TextComplexityDE19")
    df_ratings = pd.read_csv(join(data_path, "ratings.csv"), sep = ",", encoding = "ISO-8859-1")

    # feature extraction
    german_stopwords = preprocessing.get_stopwords(stopword)
    features = vectorizer.vectorizer_wrapper(df_ratings.Sentence.values, vec, german_stopwords)
    features = features.toarray()

    # labels
    labels = df_ratings.MOS_Complexity.values

    # split into train- and testset
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0, shuffle=False)

    # training
    reg = regression.baseline(X_train, y_train, method)
    
    # testing
    pred = reg.predict(X_test)

    # evaluation
    r_square = r2_score(y_test, pred)
    MSE = mean_squared_error(y_test, pred)
    RMSE = mean_squared_error(y_test, pred, squared = False)
    MAE = mean_absolute_error(y_test, pred)

    return MSE, RMSE, MAE, r_square
