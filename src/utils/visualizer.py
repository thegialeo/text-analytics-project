import os
from os.path import abspath, dirname, exists, join

import matplotlib.pyplot as plt
import pandas as pd
from utils import clustering, dimension_reduction, preprocessing, vectorizer

#import to_dataframe


def visualize_vectorizer(vec='tfidf', dim_reduc='PCA', stopword=None):
    """Apply vectorizer on TextComplexityDE19 data and visualize the vectorization.

    Written by Leo Nguyen. Contact Xenovortex, if problems arises.

    Args:
        vec (str, optional): vectorizer method to used (options: 'tfidf', 'count', 'hash'), default: 'tfidf'
        dim_reduc (str, optional): dimension reduction method to used (options: 'PCA', 'TSNE'), default: 'PCA'
        stopword (str, optional): source to load stopwords from (options: "spacy", "nltk", "stop_words", "german_plain", "german_full"). Defaults to None.
    """
    
    print("Visualize {} vectorizer with {} projection".format(vec, dim_reduc))
    
    # read data
    data_path = join(dirname(dirname(dirname(abspath(__file__)))), "data", "TextComplexityDE19")
    df_ratings = pd.read_csv(join(data_path, "ratings.csv"), sep = ",", encoding = "ISO-8859-1")
    
    # feature extraction
    if stopword is not None:
        german_stopwords = preprocessing.get_stopwords(stopword)
    else: 
        german_stopwords = None
    features = vectorizer.vectorizer_wrapper(df_ratings.Sentence.values, vec, german_stopwords)
    features = features.toarray()

    # dimension reduction
    reduced_features = dimension_reduction.reduce_dim(features, dim_reduc) 

    # plotting
    fig, ax = plt.subplots(1, 1, figsize = (15, 10))
    data = pd.DataFrame({"X value": reduced_features[:, 0], "Y value": reduced_features[:, 1], "Label": df_ratings.MOS_Complexity.values.round(0)})
    groups = data.groupby("Label")
    classes = list(set(df_ratings.MOS_Complexity.values.round(0)))
    colors = [plt.cm.jet(float(i)/max(classes)) for i in classes]
    for i, (name, group) in enumerate(groups):
        ax.plot(group["X value"], group["Y value"], marker='o', linestyle='', label=name, c=colors[i], alpha=0.5)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_title("{} vectorizer (projection: {})".format(vec, dim_reduc))
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    # save
    if stopword is None:
        save_path = join(dirname(dirname(dirname(abspath(__file__)))), "figures", "vectorizer", "{}_{}.png".format(vec, dim_reduc))
    else:
        save_path = join(dirname(dirname(dirname(abspath(__file__)))), "figures", "vectorizer", "{}_{}_{}.png".format(vec, dim_reduc, stopword))

    if not exists(dirname(dirname(save_path))):
        os.makedirs(dirname(dirname(save_path)))

    if not exists(dirname(save_path)):
        os.makedirs(dirname(save_path))
    
    fig.savefig(save_path)
    print("Save results to: {}".format(save_path))
    


def visualize_clustering(vec='tfidf', cluster='kmeans', dim_reduc='PCA'):
    """Perform clustering, dimension reduction on TextComplexityDE19 data and plot the result.

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.

    Args:
        vec (str, optional): vectorizer method to used (options: 'tfidf', 'count', 'hash'), default: 'tfidf'
        cluster (str, optional): clustering method to used (options: 'kmeans', 'AP', 'mean_shift', 'spectral', 'Agg', 'DBSCAN', 'OPTICS', 'Birch'), default: 'kmeans'
        dim_reduc (str, optional): dimension reduction method to used (options: 'PCA', 'TSNE'), default: 'PCA'
    """

    # centroid methods
    centroid_methods = ['kmeans', 'AP', 'mean_shift']

    # read data
    data_path = join(dirname(dirname(dirname(abspath(__file__)))), "data", "TextComplexityDE19")
    df_ratings = pd.read_csv(join(data_path, "ratings.csv"), sep=",", encoding="ISO-8859-1")

    # feature extraction
    german_stopwords = stopwords.words('german')
    features = vectorizer.vectorizer_wrapper(df_ratings.Sentence.values, vec, german_stopwords)
    features = features.toarray()

    # Clustering and Dimension Reduction
    if cluster in centroid_methods:
        cls_object, reduced_features, reduced_cluster_centers = clustering.clustering_wrapper(features, cluster, dim_reduc)
    else: 
        cls_object, reduced_features = clustering.clustering_wrapper(features, cluster, dim_reduc)

    # plotting
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 10))
    ax[0].scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=cls_object.labels_,
        alpha=0.5)
    ax[1].scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=df_ratings.MOS_Complexity.values.round(0),
        alpha=0.5)

    if cluster in centroid_methods:
        ax[0].scatter(
            reduced_cluster_centers[:, 0],
            reduced_cluster_centers[:, 1],
            marker='x', s=100, c='r')
        ax[1].scatter(
            reduced_cluster_centers[:, 0],
            reduced_cluster_centers[:, 1],
            marker='x', s=100, c='r')
        
    ax[0].set_xlabel("feature 1")
    ax[0].set_ylabel("feature 2")
    ax[1].set_xlabel("feature 1")
    ax[1].set_ylabel("feature 2")
    ax[0].set_title("{} clustering result (projection. {})".format(cluster, dim_reduc))
    ax[1].set_title("rounded MOS_Complexity label")
    ax[0].grid(True)
    ax[1].grid(True)
    plt.tight_layout()
    fig.savefig(join(dirname(dirname(dirname(abspath(__file__)))), "figures", "{}_{}_{}.png".format(cluster, vec, dim_reduc)))
    
"""
def basic_stats():

    """"""
    This function is supposed to illustrate the distribution of the data.
    The number of words, sentences and other important statistics
    are being displayed. This will further be used to see the effect of the
    augmentation step.
    Written by Raoul Berger.
    """"""
    # define path where figures are saved
    save_path = join(dirname(dirname(abspath(__file__))), "figures")

    # read in data
    all_data = to_dataframe.all_data()

    # plot the original distribution of rows across the datasets
    all_data["source"].hist()
    plt.grid()
    plt.ylabel("entries")
    plt.title("distribution of entries across the datasets")
    plt.savefig(join(save_path,"Original distribution of entries across datasets")

    # plot the original distribution of word count per sentence for every dataset
    plt.title("Mean word count per entry across datasets")
    plt.ylabel("Mean word count per entry")
    plt.bar(["text_comp19",
             "Weebit",
             "dw"],
            [all_data[all_data["source"] == "text_comp19"]["word_count"].mean(),
             all_data[all_data["source"] == "Weebit"]["word_count"].mean(),
             all_data[all_data["source"] == "dw"]["word_count"].mean()
             ])
    plt.savefig(join(save_path, "Original distribution of word count per entry across datasets")

    # plot the distribution of sentences per dataset

    # plot the distribution of sentences per row for each dataset

    # plot flesch reading ease index and compare to rating

    # plot word cloud with most important words for each dataset


    return all_data

"""

if __name__ == "__main__":
    visualize_vectorizer()
    #basic_stats()
