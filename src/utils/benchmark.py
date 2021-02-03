


def benchmark_baseline():
    """Run benchmark on all baseline regressions.

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.
    """

    # set all benchmark parameters
    stopwords = ["spacy", "nltk", "stop_words", "german_plain", "german_full"]
    vec = ['tfidf', 'count', 'hash']
    reg_method = ['linear', 'lasso', 'ridge', 'elastic-net', 'random-forest']