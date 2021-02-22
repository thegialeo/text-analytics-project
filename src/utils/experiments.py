import gc
import os
from os.path import abspath, dirname, exists, join

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import evaluater, preprocessing, vectorizer, visualizer


def benchmark_vectorizer():
    """Run benchmark on all baseline regressions.

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.
    """

    # set all benchmark parameters
    vec_lst = ['tfidf', 'count'] # hashing vectorizer omitted, because out of memory
    reg_lst = ['linear', 'lasso', 'ridge', 'elastic-net', 'random-forest']
 
    # run benchmark
    for vec in vec_lst:
        print("Benchmark {} vectorizer:".format(vec))
        
        # keep track of results
        results = np.zeros((len(reg_lst), 4))
        
        # evaluation
        for i, method in enumerate(tqdm(reg_lst)):
            MSE, RMSE, MAE, r_square = evaluater.evaluate_baseline(vec, method)
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
        df.to_csv(join(path, "{}.csv".format(vec)))
        print("Save results to: {}".format(join(path, "{}.csv".format(vec))))
        
        # visualize vectorization
        visualizer.visualize_vectorizer(vec, 'PCA')
        visualizer.visualize_vectorizer(vec, 'TSNE')
        
        # release memory
        gc.collect()

