import os
from os.path import abspath, dirname, exists, join

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import preprocessing, regression, vectorizer, to_dataframe


def traverser(hyperparameter, start, end, step, model="word2vec", filename="all_data.h5", pretrained=False):
    """Traverse along a hyperparameter

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.

    Args:
        hyperparameter (str): choose hyperparameter to traverse. Options: 'feature', 'window', 'count', 'epochs', 'lr', 'min_lr'
        start (int): starting feature dimension
        end (int): final feature dimension
        step (int): step size to traverse from start to end
        model (str, optional): vectorization model. Defaults to "word2vec"
        filename (str, optional): name of h5 file to load (run augmentation first)
        pretrained (bool, optional): if True, finetune the pretrained model instead of training from scratch
    """

    # read data
    df_train, df_test = to_dataframe.read_augmented_h5(filename)
    df_train = df_train[df_train["source"] == "text_comp19"]  # TODO: remove once Raoul fixes his dataloader
    
    # labels
    y_train = df_train.rating.values
    y_test = df_test.rating.values
    
    # tokenization
    corpus = preprocessing.tokenizer(df_train.raw_text, method='spacy')

    # track performance
    MSE_skip = []
    RMSE_skip = []
    MAE_skip = []
    R2_skip = []
    MSE_CBOW = []
    RMSE_CBOW = []
    MAE_CBOW = []
    R2_CBOW = []

    # type cast
    start = int(start)
    end = int(end)
    step = int(step)

    # lr and lr_min use 10e5 factor warning
    if hyperparameter == 'lr' or hyperparameter == 'lr_min':
        start *= 10e10
        end *= 10e10
        step *= 10e10

    # traversal
    for algorithm in ["skip-gram", "CBOW"]:
        print("{} Traversal for hyperparameter: {}".format(algorithm, hyperparameter))
        for i in tqdm(range(int(start), int(end), int(step))):

            # train model + feature extraction
            if hyperparameter == 'feature':
                features, vec_object = vectorizer.NN_vectorizer_wrapper(corpus,
                                                                        epochs = 10,
                                                                        lr = 0.05,
                                                                        min_lr = 0.0001,
                                                                        num_features = i,
                                                                        window_size = 10,
                                                                        min_count = 7,
                                                                        algorithm = algorithm,
                                                                        vectorizer = model,
                                                                        mode='train',
                                                                        return_vectorizer=True,
                                                                        pretrained=pretrained)
            elif hyperparameter == 'window':    
                features, vec_object = vectorizer.NN_vectorizer_wrapper(corpus,
                                                                        epochs = 10,
                                                                        lr = 0.05,
                                                                        min_lr = 0.0001,
                                                                        num_features = 100,
                                                                        window_size = i,
                                                                        min_count = 7,
                                                                        algorithm = algorithm,
                                                                        vectorizer = model,
                                                                        mode='train',
                                                                        return_vectorizer=True,
                                                                        pretrained=pretrained)
            elif hyperparameter == 'count':         
                features, vec_object = vectorizer.NN_vectorizer_wrapper(corpus,
                                                                        epochs = 10,
                                                                        lr = 0.05,
                                                                        min_lr = 0.0001,
                                                                        num_features = 100,
                                                                        window_size = 10,
                                                                        min_count = i,
                                                                        algorithm = algorithm,
                                                                        vectorizer = model,
                                                                        mode='train',
                                                                        return_vectorizer=True,
                                                                        pretrained=pretrained)
            elif hyperparameter == 'epochs':
                features, vec_object = vectorizer.NN_vectorizer_wrapper(corpus,
                                                                        epochs = i,
                                                                        lr = 0.05,
                                                                        min_lr = 0.0001,
                                                                        num_features = 100,
                                                                        window_size = 10,
                                                                        min_count = 7,
                                                                        algorithm = algorithm,
                                                                        vectorizer = model,
                                                                        mode='train',
                                                                        return_vectorizer=True,
                                                                        pretrained=pretrained)
            elif hyperparameter == 'lr':
                features, vec_object = vectorizer.NN_vectorizer_wrapper(corpus,
                                                                        epochs = 10,
                                                                        lr = i*10e-10,
                                                                        min_lr = 0.0001,
                                                                        num_features = 100,
                                                                        window_size = 10,
                                                                        min_count = 7,
                                                                        algorithm = algorithm,
                                                                        vectorizer = model,
                                                                        mode='train',
                                                                        return_vectorizer=True,
                                                                        pretrained=pretrained)                
            elif hyperparameter == 'min_lr':
                features, vec_object = vectorizer.NN_vectorizer_wrapper(corpus,
                                                                        epochs = 10,
                                                                        lr = 0.05,
                                                                        min_lr = i*10e-10,
                                                                        num_features = 100,
                                                                        window_size = 10,
                                                                        min_count = 7,
                                                                        algorithm = algorithm,
                                                                        vectorizer = model,
                                                                        mode='train',
                                                                        return_vectorizer=True,
                                                                        pretrained=pretrained)                

            else:
                raise ValueError("hyperparameter {} unknown. Options: 'feature', 'window', 'count'".format(hyperparameter))

            # apply trained word2vec on testset
            X_test = vec_object.vectorize(df_test.raw_text.values)

            # train linear regression
            reg = regression.baseline(features, y_train, "linear")

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
    save_path = join(dirname(dirname(dirname(abspath(__file__)))), "figures", "hyperparameter", model, "{} ({} to {}).png".format(hyperparameter, start, end))
    if not exists(dirname(dirname(dirname(save_path)))):
        os.makedirs(dirname(dirname(dirname(save_path))))
    if not exists(dirname(dirname(save_path))):
        os.makedirs(dirname(dirname(save_path)))
    if not exists(dirname(save_path)):
        os.makedirs(dirname(save_path))
    
    fig.savefig(save_path)
    print("Save results to: {}".format(save_path))


