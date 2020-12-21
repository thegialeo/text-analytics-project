from os.path import dirname, abspath, join
import pandas as pd
from nltk.corpus import stopwords
from utils import vectorizer


if __name__ == "__main__":
    # read data
    data_path = join(dirname(dirname(abspath(__file__))), "data", "TextComplexityDE19")
    df_ratings = pd.read_csv(join(data_path, "ratings.csv"), sep=",", encoding="ISO-8859-1")
    df_complexity = df_ratings[["Sentence", "MOS_Complexity"]]
    
    # feature extraction
    german_stopwords = stopwords.words('german')
    features = vectorizer.vectorizer_wrapper(df_complexity.Sentence.values, 'tfidf', german_stopwords)

    # clustering with KMeans
    cls_kmeans = MiniBatchKMeans(n_clusters=6, random_state=0)
    cls_kmeans.fit(features)
    cls_kmeans.predict(features)

    # Dimension reduction of feature space with PCA
    pca = PCA(n_components=2, random_state=0)
    reduced_features = pca.fit_transform(features.toarray())
    reduced_cluster_centers = pca.transform(cls_kmeans.cluster_centers_)
    
    # Plot cluster result against targets 
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15,10))
    ax[0].scatter(reduced_features[:, 0], reduced_features[:, 1], c=cls_kmeans.predict(features), alpha=0.5)
    ax[0].scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='x', s=100, c='r')
    ax[1].scatter(reduced_features[:, 0], reduced_features[:, 1], c=df_complexity.MOS_Complexity.values.round(0), alpha=0.5)
    ax[1].scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='x', s=100, c='r')
    ax[0].set_xlabel("feature 1")
    ax[0].set_ylabel("feature 2")
    ax[1].set_xlabel("feature 1")
    ax[1].set_ylabel("feature 2")
    ax[0].set_title("KMeans clustering result (Projection with PCA)")
    ax[1].set_title("PCA features colorcoded by rounded MOS_Complexity")
    ax[0].grid(True)
    ax[1].grid(True)
    plt.tight_layout()
    fig.savefig(join(dirname(dirname(abspath(__file__))), "figures", "KMeans_clustering_MOS_Complexity"))

    # Evaluate homogeneity score
    print(homogeneity_score(df_complexity.MOS_Complexity.values.round(0), cls_kmeans.predict(features)))

    # Evaluate silhouette score
    print(silhouette_score(features, labels=cls_kmeans.predict(features)))   
