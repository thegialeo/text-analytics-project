import multiprocessing
from os.path import abspath, dirname, exists, join

from gensim.models.word2vec import Word2Vec


def word2vec()

class word2vec:

    def __init__(self, corpus, epochs, lr, min_lr, num_features, window_size=5, min_count=5, algorithm="skip-gram"):
        """Gensim Word2Vec wrapper class

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
        """
        self.corpus = corpus
        self.epochs = epochs
        self.lr = lr
        self.min_lr = min_lr
        self.num_features = num_features
        self.window_size = window_size
        self.min_count = min_count
        self.data_path = join(dirname(dirname(dirname(abspath(__file__)))), "model", "word2vec.model")

        if algorithm == "skip-gram":
            self.algorithm = 1
        elif algorithm == "CBOW":
            self.algorithm = 0
        else:
            print("algorithm {} unknown. Please choose 'skip-gram' or 'CBOW'".format(algorithm))
            exit()
        

    def train(self):
        self.model = Word2Vec(corpus,
                                sg = 1,
                                epochs = self.epochs,
                                alpha = self.lr,
                                min_alpha = self.min_lr,
                                size = self.num_features,
                                window = self.window_size,
                                min_count = semin_count,
                                workers=multiprocessing.cpu_count())
        
        self.model.init_sims(replace=True)
        self.save_model()


    def save_model(self):
        if not exists(self.data_path):
            os.makedirs(self.data_path)
        
        self.model.save(self.data_path)


    def load_model(self):
        self.model = Word2Vec.load(self.data_path)
