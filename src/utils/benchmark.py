from os.path import abspath, dirname, exists, join

import evaluater
import numpy as np
import pandas as pd


def benchmark_baseline():
    """Run benchmark on all baseline regressions.

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.
    """

    # set all benchmark parameters
    stopwords = ["spacy", "nltk", "stop_words", "german_plain", "german_full"]
    vec_lst = ['tfidf', 'count', 'hash']
    reg_lst = ['linear', 'lasso', 'ridge', 'elastic-net', 'random-forest']

    for stopword in stopwords:

        # keep track of results
        MSE_result = np.zeros((len(reg_lst), len(vec_lst)))
        RMSE_result = np.zeros((len(reg_lst), len(vec_lst)))
        MAE_result = np.zeros((len(reg_lst), len(vec_lst)))
        R2_result = np.zeros((len(reg_lst), len(vec_lst)))

        # run benchmark
        for i, method in enumerate(reg_lst):
            for j, vec in enumerate(vec_lst):
                MSE, RMSE, MAE, r_square = evaluate_baseline(vec, method, stopword)
                MSE_result[i][j] = MSE
                RMSE_result[i][j] = RMSE
                MAE_result[i][j] = MAE
                R2_result[i][j] = r_square

        # save results
        MSE_df = pd.DataFrame(MSE_result, index=reg_lst, columns=vec_lst)
        RMSE_df = pd.DataFrame(RMSE_result, index=reg_lst, columns=vec_lst)
        MAE_df = pd.DataFrame(MAE_result, index=reg_lst, columns=vec_lst)
        R2_df = pd.DataFrame(R2_result, index=reg_lst, columns=vec_lst)

        path = join(dirname(dirname(dirname(abspath(__file__)))), "result", "compare-vectorizer")

        if not exists(dirname(path)):
            os.makedirs(dirname(path))

        if not exists(path):
            os.makedirs(path)

        MSE_df.to_csv(join(path, "MSE_{}.csv".format(stopword)))
        RMSE_df.to_csv(join(path, "RMSE_{}.csv".format(stopword)))
        MAE_df.to_csv(join(path, "MAE_{}.csv".format(stopword)))
        R2_df.to_csv(join(path, "R2_{}.csv".format(stopword)))
