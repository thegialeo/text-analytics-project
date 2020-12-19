from os.path import dirname, abspath, join
import pandas as pd
from nltk.corpus import stopwords
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # read data
    data_path = join(dirname(dirname(abspath(__file__))), "data", "TextComplexityDE19")
    df_ratings = pd.read_csv(join(data_path, "ratings.csv"), sep=",", encoding="ISO-8859-1")
    df_complexity = df_ratings[["Sentence", "MOS_Complexity"]]
    
    # feature extraction
    german_stopwords = stopwords.words('german')
    tfidf = TfidfVectorizer(encoding="ISO-8859-1", stop_words=german_stopwords)
    tfidf.fit(df_complexity.Sentence.values)
    features = tfidf.transform(df_complexity.Sentence.values)

    # clustering with KMeans
    cls_kmeans = MiniBatchKMeans(n_clusters=6, random_state=0)
    cls_kmeans.fit(features)
    cls_kmeans.predict(features)

    # Dimension reduction of feature space with PCA
    pca = PCA(n_components=2, random_state=0)
    reduced_features = pca.fit_transform(features.toarray())
    reduced_cluster_centers = pca.transform(cls_kmeans.cluster_centers_)
    
    
    

    
    
    

    
    
