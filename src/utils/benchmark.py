import gc
import os
from os.path import abspath, dirname, exists, join

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import evaluater, preprocessing, vectorizer, visualizer


def benchmark_baseline():
    """Run benchmark on all baseline regressions.

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.
    """

    # set all benchmark parameters
    stopwords = ["nltk", "spacy", "stop_words", "german_plain", "german_full"] 
    vec_lst = ['tfidf', 'count'] # hashing vectorizer omitted, because out of memory
    reg_lst = ['linear', 'lasso', 'ridge', 'elastic-net', 'random-forest']
 
    # run benchmark
    for stopword in tqdm(stopwords):
        for vec in vec_lst:
            
            # keep track of results
            results = np.zeros((len(reg_lst), 4))

            # evaluation
            for i, method in enumerate(reg_lst):
                MSE, RMSE, MAE, r_square = evaluater.evaluate_baseline(vec, method, stopword)
                results[i][0] = MSE
                results[i][1] = RMSE
                results[i][2] = MAE
                results[i][3] = r_square
                
            # save results
            df = pd.DataFrame(results, index=reg_lst, columns=['MSE', 'RMSE', 'MAE', 'r_square'])

            path = join(dirname(dirname(dirname(abspath(__file__)))), "result", "vectorizer")

            if not exists(dirname(path)):
                os.makedirs(dirname(path))

            if not exists(path):
                os.makedirs(path)

            df.to_csv(join(path, "{}_{}.csv".format(vec, stopword)))

            # visualize vectorization
            visualizer.visualize_vectorizer(vec, 'PCA', stopword)
            visualizer.visualize_vectorizer(vec, 'TSNE', stopword)

            # release memory
            gc.collect()


def traverser_feature_dim(start, end, step, model="word2vec"):
    """Find optimal feature dimension

    Args:
        start (int): starting feature dimension
        end (int): final feature dimension
        step (int): step size to traverse from start to end
        model (str, optional): vectorization model. Defaults to "word2vec".
    """

    # read data
    data_path = join(dirname(dirname(dirname(abspath(__file__)))), "data", "TextComplexityDE19")
    df_ratings = pd.read_csv(join(data_path, "ratings.csv"), sep = ",", encoding = "ISO-8859-1")

    # labels
    labels = df_ratings.MOS_Complexity.values

    # tokenization
    corpus = preprocessing.tokenizer(df_ratings.Sentence, method='spacy')
 
    # hyperparameters
    epochs = 10
    lr = 0.25
    min_lr = 0.0001

    # track performance
    MSE_skip = []
    RMSE_skip = []
    MAE_skip = []
    R2_skip = []
    MSE_CBOW = []
    RMSE_CBOW = []
    MAE_CBOW = []
    R2_CBOW = []

    # traversal
    for algorithm in ["skip-gram", "CBOW"]:
        for i in range(start, end, step):

            # init model
            model = vectorizer.NN_vectorizer_wrapper(corpus,
                                                     epochs,
                                                     lr,
                                                     min_lr,
                                                     num_features = i,
                                                     window_size = 5,
                                                     min_count = 5,
                                                     algorithm = algorithm,
                                                     vectorizer = model,
                                                     mode='train')         
                         
            # train model
            model.train()

            # feature extraction
            model.vectorize()
            features = model.features

            # split into train- and testset
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0, shuffle=False)

            # train linear regression
            reg = regression.baseline(X_train, y_train, "linear")

            # testing
            pred = reg.predict(X_test)

            # evaluation
            MSE = mean_squared_error(y_test, pred)
            RMSE = mean_squared_error(y_test, pred, squared = False)
            MAE = mean_absolute_error(y_test, pred)
            r_square = r2_score(y_test, pred)

            # track results
            if algorithm == "skip-gram":
                MSE_skip.append(MSE)
                RMSE_skip.append(RMSE)
                MAE_skip.append(MAE)
                R2_skip.append(r_square)

            if algorithm == "CBOW":
                MSE_CBOW.append(MSE)
                RMSE_CBOW.append(RMSE)
                MAE_CBOW.append(MAE)
                R2_CBOW.append(r_square)

    # plot
    fig, ax = plt.subplots(1, 1, figsize = (15, 10))
    ax.plot([x for x in range(start, end, step)], MSE_skip, label="MSE skip-gram")
    ax.plot([x for x in range(start, end, step)], RMSE_skip, label="RMSE skip-gram")
    ax.plot([x for x in range(start, end, step)], MAE_skip, label="MAE skip-gram")
    ax.plot([x for x in range(start, end, step)], R2_skip, label="R2 skip-gram")
    ax.plot([x for x in range(start, end, step)], MSE_CBOW, label="MSE CBOW")
    ax.plot([x for x in range(start, end, step)], RMSE_CBOW, label="RMSE CBOW")
    ax.plot([x for x in range(start, end, step)], MAE_CBOW, label="MAE CBOW")
    ax.plot([x for x in range(start, end, step)], R2_CBOW, label="R2 CBOW")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_title("{} vectorizer (projection. {})".format(vec, dim_reduc))
    ax.grid(True)
    plt.tight_layout()

    # save
    save_path = join(dirname(dirname(dirname(abspath(__file__)))), "figures", "hyperparameter", "{}_feature_dimension.png".format(model))
    if not exists(dirname(dirname(save_path))):
        os.makedirs(dirname(dirname(save_path)))
    if not exists(dirname(save_path)):
        os.makedirs(dirname(save_path))
    
    fig.savefig(save_path)
