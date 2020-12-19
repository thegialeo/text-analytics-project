from os.path import dirname, abspath, join
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


if __name__ == "__main__":
    # read data
    data_path = join(dirname(dirname(abspath(__file__))), "data", "TextComplexityDE19")
    df_ratings = pd.read_csv(join(data_path, "ratings.csv"), sep = ",", encoding = "ISO-8859-1")
    df_complexity = df_ratings[["Sentence", "MOS_Complexity"]]
    
    # feature extraction
    german_stopwords = stopwords.words('german')
    tfidf = TfidfVectorizer(encoding="ISO-8859-1", stop_words=german_stopwords)
    tfidf.fit(df_complexity.Sentence.values)
    features = tfidf.transform(df_complexity.Sentence.values)
    
