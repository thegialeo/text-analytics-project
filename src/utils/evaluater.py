from os.path import abspath, dirname, join
from tqdm import tqdm
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
    gpu,
    normalization
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
        filename (str, optional): name of h5 file to load (run preprocessing first)
        engineered_features (bool, optional): contenate engineered features to vectorized sentence
        return_pred (bool, optional): return predictions, instead of metrics

    Return:
        MSE (double): Mean Square Error
        RMSE (double): Root Mean Square Error
        MAE (double): Mean Absolute Error
        r_square (double): R Square
    """

    # read data
    df_train, df_test = to_dataframe.read_augmented_h5(filename)

    # stopwords
    stopword_lst = preprocessing.get_stopwords()

    # feature extraction
    X_train, vec_object = vectorizer.vectorizer_wrapper(
        df_train.raw_text, vec, stopword_lst, True
    )
    X_test = vec_object.transform(df_test.raw_text)

    # add engineered features
    if engineered_features:
        extra_train_feat = sentencestats.construct_features(df_train.raw_text)
        extra_test_feat = sentencestats.construct_features(df_test.raw_text)
        if vec == "word2vec" or vec == "pretrained_word2vec":
            X_train = np.concatenate((np.array(X_train), extra_train_feat), axis=1)
            X_test = np.concatenate((np.array(X_test), extra_test_feat), axis=1)
        else:
            X_train = np.concatenate((X_train.toarray(), extra_train_feat), axis=1)
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


def evaluate_model(
    model, bert_model, dataloader, engineered_features=False, multiple_dataset=False
):
    """Evaluate regression metrics of a model on a dataset

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.

    Args:
        model (torch.nn.Module): PyTorch model of a Regression Neural Network
        bert_model (torch.nn.Module): BERT PyTorch model for feature extraction
        dataloader (PyTorch dataloader): PyTorch dataloader of dataset
        engineered_features (bool, optional): contenate engineered features to vectorized sentence
        multiple_dataset (bool, optional): use multiple datasets

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
        for i, data in enumerate(tqdm(dataloader)):
            # move batch and model to device
            model.to(device)
            input_id = data[0].to(device)
            segment = data[1].to(device)
            label = data[2].to(device)
            if engineered_features and multiple_dataset:
                extra_feat = data[3].to(device)
                dataset_label = data[4].to(device)
            elif engineered_features:
                extra_feat = data[3].to(device)
            elif multiple_dataset:
                dataset_label = data[3].to(device)

            # BERT feature extraction
            features = bert_model.get_features(input_id, segment)

            # add engineered features
            if engineered_features:
                features = torch.cat((features, extra_feat), 1)

            # add dataset conditional label (always 0)
            if multiple_dataset:
                features = torch.cat(
                    (features, torch.tensor(np.zeros(dataset_label.shape), dtype=torch.float).to(device)), 1
                )

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


def evaluate_acc(model, bert_model, dataloader, engineered_features=False):
    """Evaluate accuracy of a model.

    Written by Leo Nguyen. Contact Xenovortex, if problems arises.

    Args:
        model (torch.nn.Module): PyTorch model of a Regression Neural Network
        bert_model (torch.nn.Module): BERT PyTorch model for feature extraction
        dataloader (PyTorch dataloader): PyTorch dataloader of dataset
        engineered_features (bool, optional): contenate engineered features to vectorized sentence
    
    Return: 
        accuracy of the model on dataset
    """

    # check if GPU available
    device = gpu.check_gpu()

    # move model to device
    model = model.to(device)

    # log
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            # move batch and model to device
            model.to(device)
            input_id = data[0].to(device)
            segment = data[1].to(device)
            label = data[2].to(device)
            if engineered_features:
                extra_feat = data[3].to(device)

            # BERT feature extraction
            features = bert_model.get_features(input_id, segment)

            # add engineered features
            if engineered_features:
                features = torch.cat((features, extra_feat), 1)

            # prediction
            output = torch.nn.Sigmoid(model(features))
            _, pred = torch.max(out.data, 1)
            
            # count correct predictions
            total += label.size(0)
            correct += (pred == label).sum().item()

    return 100 * correct / total