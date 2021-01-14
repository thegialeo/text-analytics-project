from os.path import abspath, dirname, join

import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics import homogeneity_score, silhouette_score
from utils import clustering, vectorizer


def visualize_data():
    """Perform clustering, dimension reduction on TextComplexityDE19 data and plot the result.
       Evaluate clustering by homogeneity and silhouette score.

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.
    """

    # read data
    data_path = join(dirname(dirname(abspath(__file__))), "data", "TextComplexityDE19")
    df_ratings = pd.read_csv(
        join(data_path, "ratings.csv"),
        sep=",", encoding="ISO-8859-1")

    # feature extraction
    german_stopwords = stopwords.words('german')
    features = vectorizer.vectorizer_wrapper(
        df_ratings.Sentence.values, 'tfidf', german_stopwords)
    features = features.toarray()

    # KMeans Clustering and PCA
    #cls_kmeans, reduced_features, reduced_cluster_centers = clustering.clustering_wrapper(
     #   features, 'spectral', 'PCA')
    cls_kmeans, reduced_features = clustering.clustering_wrapper(
         features, 'OPTICS', 'PCA')

    # Plot cluster result against targets
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 10))
    ax[0].scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=cls_kmeans.labels_,
        alpha=0.5)
 #   ax[0].scatter(
  #      reduced_cluster_centers[:, 0],
   #     reduced_cluster_centers[:, 1],
    #    marker='x', s=100, c='r')
    ax[1].scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=df_ratings.MOS_Complexity.values.round(0),
        alpha=0.5)
  #  ax[1].scatter(
   #     reduced_cluster_centers[:, 0],
    #    reduced_cluster_centers[:, 1],
     #   marker='x', s=100, c='r')
    ax[0].set_xlabel("feature 1")
    ax[0].set_ylabel("feature 2")
    ax[1].set_xlabel("feature 1")
    ax[1].set_ylabel("feature 2")
    ax[0].set_title("KMeans clustering result (Projection with PCA)")
    ax[1].set_title("PCA features colorcoded by rounded MOS_Complexity")
    ax[0].grid(True)
    ax[1].grid(True)
    plt.tight_layout()
    fig.savefig(
        join(
            dirname(dirname(abspath(__file__))),
            "figures", "KMeans_clustering_MOS_Complexity"))
    # Evaluate homogeneity score
    print(
        homogeneity_score(
            df_ratings.MOS_Complexity.values.round(0),
            cls_kmeans.labels_))
            #cls_kmeans.predict(features)))
    # Evaluate silhouette score
    #print(silhouette_score(features, labels=cls_kmeans.predict(features)))


if __name__ == "__main__":
    visualize_data()
