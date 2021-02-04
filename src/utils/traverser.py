import os
from os.path import abspath, dirname, exists, join

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import preprocessing, regression, vectorizer


def traverser(hyperparameter, start, end, step, model="word2vec"):
    """Find optimal feature dimension

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.

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
        print("Feature Dimension Traversal for {}".format(algorithm))
        for i in tqdm(range(start, end, step)):

            # train model + feature extraction
            if hyperparameter == 'feature':
                features = vectorizer.NN_vectorizer_wrapper(corpus,
                                                            epochs,
                                                            lr,
                                                            min_lr,
                                                            num_features = i,
                                                            window_size = 5,
                                                            min_count = 5,
                                                            algorithm = algorithm,
                                                            vectorizer = model,
                                                            mode='train')
            elif hyperparameter == 'window':    
                features = vectorizer.NN_vectorizer_wrapper(corpus,
                                                            epochs,
                                                            lr,
                                                            min_lr,
                                                            num_features = 120,
                                                            window_size = i,
                                                            min_count = 5,
                                                            algorithm = algorithm,
                                                            vectorizer = model,
                                                            mode='train')
            elif hyperparameter == 'count':
                features = vectorizer.NN_vectorizer_wrapper(corpus,
                                                            epochs,
                                                            lr,
                                                            min_lr,
                                                            num_features = 120,
                                                            window_size = 5,
                                                            min_count = i,
                                                            algorithm = algorithm,
                                                            vectorizer = model,
                                                            mode='train')
            else:
                print("hyperparameter {} unknown. Options: 'feature', 'window', 'count'".format(hyperparameter))
                exit()

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
    fig, ax = plt.subplots(2, 2, figsize = (15, 10))
    ax[0, 0].plot([x for x in range(start, end, step)], MSE_skip, label="MSE skip-gram")
    ax[0, 1].plot([x for x in range(start, end, step)], RMSE_skip, label="RMSE skip-gram")
    ax[1, 0].plot([x for x in range(start, end, step)], MAE_skip, label="MAE skip-gram")
    ax[1, 1].plot([x for x in range(start, end, step)], R2_skip, label="R2 skip-gram")
    ax[0, 0].plot([x for x in range(start, end, step)], MSE_CBOW, label="MSE CBOW")
    ax[0, 1].plot([x for x in range(start, end, step)], RMSE_CBOW, label="RMSE CBOW")
    ax[1, 0].plot([x for x in range(start, end, step)], MAE_CBOW, label="MAE CBOW")
    ax[1, 1].plot([x for x in range(start, end, step)], R2_CBOW, label="R2 CBOW")
    for i in range(2):
        for j in range(2):
            ax[i, j].set_xlabel("feature dimension")
            ax[i, j].set_ylabel("metric")
            ax[i, j].grid(True)
            ax[i, j].legend()
    plt.tight_layout()

    # save
    save_path = join(dirname(dirname(dirname(abspath(__file__)))), "figures", "hyperparameter", "{}_feature_dimension.png".format(model))
    if not exists(dirname(dirname(save_path))):
        os.makedirs(dirname(dirname(save_path)))
    if not exists(dirname(save_path)):
        os.makedirs(dirname(save_path))
    
    fig.savefig(save_path)


