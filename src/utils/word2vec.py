
import multiprocessing
import gensim
from gensim.models.word2vec import Word2Vec

class word2vec:

    def __init__(self, corpus, epochs, lr, min_lr, num_features, window_size=5, min_count=5):
        self.corpus = corpus
        self.epochs = epochs
        self.lr = lr
        self.min_lr = min_lr
        self.num_features = num_features
        self.window_size = window_size
        self.min_count = min_count
        self.model = Word2Vec(corpus,
                                sg=1,
                                epochs=epochs,
                                alpha=lr,
                                min_alpha=min_lr,
                                size=num_features,
                                window=window_size,
                                min_count = min_count,
                                workers=multiprocessing.cpu_count())
