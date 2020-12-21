from sklearn.feature_extraction.text import TfidfVectorizer


def vectorizer_wrapper(data, vectorizer='tfidf', stop_words=None):
    """Takes in a numpy array of sentences and perform the selected vectorizer on the data. 
       Returns a numpy array of sentence features represented by number vectors.   

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.

    Args:
        data (numpy array): 1d array containing sentences
        vectorizer (str, optional): Select the vectorizer type. Implemented so far are: 'tfidf'. Defaults to 'tfidf'.
        stop_words (list, optional): List of stopwords. Defaults to None.

    Returns:
        features [scipy sparse matrix (csr)]: document-term matrix with dimension (number of sentences, features per sentence)
    """

    # apply selected vectorizer
    if vectorizer == 'tfidf':
        tfidf = TfidfVectorizer(encoding='ISO-8859-1', stop_words=stop_words)
        tfidf.fit(data)
        features = tfidf.transform(data)
    else:
        print("Vectorizer {} is not implemented yet. Please select one of the following options: 'tfidf'".format(vectorizer))
        exit()

    return features
