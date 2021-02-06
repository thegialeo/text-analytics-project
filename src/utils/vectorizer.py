from sklearn.feature_extraction.text import (CountVectorizer, HashingVectorizer,
                                             TfidfVectorizer)
from utils import word2vec


def vectorizer_wrapper(data, vectorizer='tfidf', stopwords=None, return_vectorizer=False):
    """Takes in a numpy array of sentences and perform the selected vectorizer on the data.
       Returns a numpy array of sentence features represented by number vectors.

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.

    Args:
        data (numpy array): 1d array containing sentences
        vectorizer (str, optional): Select the vectorizer type. Implemented so far are: 'tfidf', 'count', 'hash'. Defaults to 'tfidf'.
        stop_words (list, optional): List of stopwords. Defaults to None.

    Returns:
        features [scipy sparse matrix (csr)]: document-term matrix with dimension (number of sentences, features per sentence)
    """

    # apply selected vectorizer
    if vectorizer == 'tfidf':
        vec = TfidfVectorizer(encoding='ISO-8859-1', stop_words=stopwords)
        features = vec.fit_transform(data)
    elif vectorizer == 'count':
        vec = CountVectorizer(encoding='ISO-8859-1', stop_words=stopwords)
        features = vec.fit_transform(data)
    elif vectorizer == 'hash':
        vec = HashingVectorizer(encoding='ISO-8859-1', stop_words=stopwords)
        features = vec.fit_transform(data)
    else:
        except ValueError:
            print("Vectorizer {} not implemented. Please select one of the following options: 'tfidf', 'count', 'hash'.".format(vectorizer))

    if return_vectorizer:
        return features, vec
    else:
        return features


def NN_vectorizer_wrapper(corpus, epochs, lr, min_lr, num_features, window_size=5, min_count=5, algorithm="skip-gram", vectorizer='word2vec', mode='train'):
    """Takes in a 2d list of sentences and perform the selected vectorizer on the data.
       Returns an array of sentence features represented by number vectors.

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.

    Args:
        corpus (2d list): 2d list of tokens for each sentence
        epochs (int): number of epochs 
        lr (double): start learning rate
        min_lr (double): end learning rate
        num_features (int): dimension of feature space
        window_size (int, optional): window size of word2vec. Defaults to 5.
        min_count (int, optional): ignore words that occur less than min_count. Defaults to 5.
        algorithm (str, optional): choose between "CBOW" and "skip-gram". Defaults to "skip-gram".
        vectorizer (str, optional): Select the vectorizer type. Implemented so far are: 'word2vec'. Defaults to 'word2vec'.
        mode (str, optional): train new word2vec model or load existing model (options: 'train' or 'load')

    Return: 
        features [2d array]: document-term matrix with dimension (number of sentences, features per sentence)
    """
   
    # apply selected vectorizer
    if vectorizer == 'word2vec':
        model = word2vec.word2vec(corpus, epochs, lr, min_lr, num_features, window_size, min_count, algorithm)
        if mode == 'train':
            model.train()
        elif mode == 'load':
            model.load_wv()
        else:
            print("mode {} unknown. Please choose 'train' or 'load'".format(mode))
        model.vectorize()
        features = model.features
    else:
        print("Vectorizer {} not implemented. Please one of the following options: 'word2vec'.".format(vectorizer))
        exit()

    return features
