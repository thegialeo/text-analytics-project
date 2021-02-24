import gc
import os
from os.path import abspath, dirname, exists, join

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import evaluater, preprocessing, vectorizer, visualizer


def benchmark_all(filename, engineered_features=False):
    """Run benchmark on all baseline regressions and all vectorization methods.

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.

    Args:
        filename (string): name of h5 file to load (run preprocessing first)
        engineered_features (bool, optional): contenate engineered features to vectorized sentence
    """

    # set all benchmark parameters
    vec_lst = ['word2vec'] #['tfidf', 'count', 'word2vec', 'pretrained_word2vec'] # hashing vectorizer omitted, because out of memory
    reg_lst = ['linear', 'lasso', 'ridge', 'elastic-net', 'random-forest']
 
    # run benchmark
    for vec in vec_lst:
        print("Benchmark {} vectorizer:".format(vec))
        
        # keep track of results
        results = np.zeros((len(reg_lst), 4))
        
        # evaluation
        for i, method in enumerate(tqdm(reg_lst)):
            MSE, RMSE, MAE, r_square = evaluater.evaluate_baseline(vec, method, filename, engineered_features)
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
        if engineered_features:
            df.to_csv(join(path, "{}_{}_extra_feat.csv".format(filename, vec)))
        else:
            df.to_csv(join(path, "{}_{}.csv".format(filename, vec)))
        print("Save results to: {}".format(join(path, "{}.csv".format(vec))))
        
        # visualize vectorization
        visualizer.visualize_vectorizer(vec, 'PCA', filename=filename)
        visualizer.visualize_vectorizer(vec, 'TSNE', filename=filename)
        
        # release memory
        gc.collect()

