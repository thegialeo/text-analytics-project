from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    homogeneity_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from utils import (
    clustering,
    preprocessing,
    regression,
    sentencestats,
    to_dataframe,
    vectorizer,
    gpu
)


def evaluate_clustering(
    vec="tfidf", cluster="kmeans", dim_reduc="PCA", stopword="nltk"
):
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
    centroid_methods = ["kmeans", "AP", "mean_shift"]

    # read data
    data_path = join(
        dirname(dirname(dirname(abspath(__file__)))), "data", "TextComplexityDE19"
    )
    df_ratings = pd.read_csv(
        join(data_path, "ratings.csv"), sep=",", encoding="ISO-8859-1"
    )

    # feature extraction
    german_stopwords = preprocessing.get_stopwords(stopword)
    features = vectorizer.vectorizer_wrapper(
        df_ratings.Sentence.values, vec, german_stopwords
    )
    features = features.toarray()

    # Clustering and Dimension Reduction
    if cluster in centroid_methods:
        (
            cls_object,
            reduced_features,
            reduced_cluster_centers,
        ) = clustering.clustering_wrapper(features, cluster, dim_reduc)
    else:
        cls_object, reduced_features = clustering.clustering_wrapper(
            features, cluster, dim_reduc
        )

    # Evaluate homogeneity score
    homo_score = homogeneity_score(
        df_ratings.MOS_Complexity.values.round(0), cls_object.labels_
    )

    # Evaluate silhouette score
    sil_score = silhouette_score(features, labels=cls_object.labels_)

    return homo_score, sil_score


def evaluate_baseline(
    vec="tfidf",
    method="linear",
    filename="all_data.h5",
    engineered_features=False,
    return_pred=False,
):
    """Perform baseline regression on TextComplexityDE19 data. Will be extended to all datasets, when raouls dataloader finished.
       Evaluate RMSE, MSE, MAE and R squares or return prediction

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.

    Args:
        vec (str, optional): vectorizer method to used (options: 'tfidf', 'count', 'hash'), default: 'tfidf'
        method (str, optional): regression method to use (options: 'linear', 'lasso', 'ridge', 'elastic-net', 'random-forest'). Defaults to 'linear'.
        filename (str, optional): name of h5 file to load (run augmentation first)
        return (bool, optional): return predictions, instead of metrics

    Return:
        MSE (double): Mean Square Error
        RMSE (double): Root Mean Square Error
        MAE (double): Mean Absolute Error
        r_square (double): R Square
    """

    # read data
    df_train, df_test = to_dataframe.read_augmented_h5(filename)
    df_train = df_train[
        df_train["source"] == "text_comp19"
    ]  # TODO: remove once Raoul fixes his dataloader
    df_test = df_test[
        df_test["source"] == "text_comp19"
    ]  # TODO: remove once Raoul fixes his dataloader

    # feature extraction
    X_train, vec_object = vectorizer.vectorizer_wrapper(
        df_train.raw_text.values, vec, None, True
    )
    X_test = vec_object.transform(df_test.raw_text.values)

    # add engineered features
    if engineered_features:
        extra_train_feat = sentencestats.construct_features(df_train.raw_text)
        X_train = np.concatenate((X_train.toarray(), extra_train_feat), axis=1)
        extra_test_feat = sentencestats.construct_features(df_test.raw_text)
        X_test = np.concatenate((X_test.toarray(), extra_test_feat), axis=1)

    # labels
    y_train = df_train.rating.values
    y_test = df_test.rating.values

    # training
    reg = regression.baseline(X_train, y_train, method)

    # testing
    pred = reg.predict(X_test)

    # evaluation
    r_square = r2_score(y_test, pred)
    MSE = mean_squared_error(y_test, pred)
    RMSE = mean_squared_error(y_test, pred, squared=False)
    MAE = mean_absolute_error(y_test, pred)

    if return_pred:
        return pred
    else:
        return MSE, RMSE, MAE, r_square


def evaluate(label, pred):
    """Generic evaluation of regression metrics

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.

    Args:
        label (array-like): ground truth label
        pred (array-like): prediction made by model

    Return:
        MSE (double): Mean Square Error
        RMSE (double): Root Mean Square Error
        MAE (double): Mean Absolute Error
        r_square (double): R Square
    """

    r_square = r2_score(label, pred)
    MSE = mean_squared_error(label, pred)
    RMSE = mean_squared_error(label, pred, squared=False)
    MAE = mean_absolute_error(label, pred)

    return MSE, RMSE, MAE, r_square


def evaluate_model(model, bert_model, dataloader):
    """Evaluate regression metrics of a model on a dataset

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.

    Args:
        model (torch.nn.Module): PyTorch model of a Regression Neural Network
        bert_model (torch.nn.Module): BERT PyTorch model for feature extraction
        dataloader (PyTorch dataloader): PyTorch dataloader of dataset

    Return:
        MSE_mean (double): Mean Square Error
        RMSE_mean (double): Root Mean Square Error
        MAE_mean (double): Mean Absolute Error
        r_square_mean (double): R Square
    """

    # check if GPU available
    device = gpu.check_gpu()

    # move model to device
    model = model.to(device)
    
    # record metrics per batch
    MSE_lst = []
    RMSE_lst = []
    MAE_lst = []
    r_square_lst = []

    # iterate through dataset
    with torch.no_grad():
        for i, (input_id, segment, label) in enumerate(dataloader):
            # move batch to device 
            input_id = input_id.to(device)
            segment = segment.to(device)
            label = label.to(device)

            # BERT feature extraction
            features = bert_model.get_features(input_id, segment)
            
            # prediction
            output = model(features)

            # evaluate
            MSE, RMSE, MAE, r_square = evaluate(label.cpu(), output.cpu())
            MSE_lst.append(MSE)
            RMSE_lst.append(RMSE)
            MAE_lst.append(MAE)
            r_square_lst.append(r_square)

    # compute mean over all batches
    MSE_mean = sum(MSE_lst) / len(MSE_lst)
    RMSE_mean = sum(RMSE_lst) / len(RMSE_lst)
    MAE_mean = sum(MAE_lst) / len(MAE_lst)
    r_square_mean = sum(r_square_lst) / len(r_square_lst)

    return MSE_mean, RMSE_mean, MAE_mean, r_square_mean



         



