import argparse

from utils import experiments, downloader, evaluater, traverser, to_dataframe, trainer
from utils.sample import hello_world  # import of module from subfolder


"""
This script should serve as entrypoint to your program.
Every module or package it relies on has to be imported at the beginning.
The code that is actually executed is the one below 'if __name__ ...' (if run
as script).
"""


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", dest='download', action='store',
                        help="Download specific or all datasets. Options: 'all', 'TextComplexityDE19', 'Weebit', 'dw'")
    parser.add_argument("--experiment", dest='experiment', action='store',
                        help="Select experiment to perform. Options: 'vectorizer'")
    parser.add_argument("--search", dest='search', action='store', nargs=6,
                        help="Perform linear search for [hyperparameter, start, end, step, model, filename]. Options: hyperparameter ['feature', 'window', 'count', 'epochs', 'lr', 'min_lr'], model ['word2vec']")
    parser.add_argument("--augmentation", dest="augmentation", action='store_true',
                        help="Augmentate the downloaded datasets and save the result in a h5 file")
    parser.add_argument("--pretrained", dest='pretrained', action='store_true',
                        help="Finetune pretrained model instead of training from scratch")

    parser.set_defaults(download=None, experiment=None, search=None, augmentation=False, pretrained=False)
    args = parser.parse_args()


    # load datasets
    if args.download is not None:
        if args.download == 'all':
            downloader.download_TextComplexityDE19()
            downloader.download_Weebit()
            downloader.download_dw_set()
        elif args.download == 'TextComplexityDE19':
            downloader.download_TextComplexityDE19()
        elif args.download == 'Weebit':
            downloader.download_Weebit()
        elif args.download == 'dw':
            downloader.download_dw_set()
        else:
            raise ValueError("Input {} for --download is invalid. Choose one of the following: 'all', 'TextComplexityDE19', 'Weebit', 'dw'".format(args.download))

    # augmentation
    if args.augmentation:
        to_dataframe.store_augmented_h5("all_data.h5", test_size=0.2)

    # hyperparameter search
    if args.search is not None:
            traverser.traverser(*args.search, args.pretrained)




    # experiments
    if args.experiment is not None:
        # vectorizer
        if args.experiment == 'vectorizer':
            benchmark.benchmark_vectorizer()
        # test and debug evaluate_baseline
        if args.experiment == 'test':
            MSE, RMSE, MAE, r_square = evaluater.evaluate_baseline(engineered_features=True)
            print(r_square)
        # test BERT training
        if args.experiment == 'BERT':
            trainer.train_model("all_data.h5", 10, [5, 8, 10], 1024, 1e-3, "test")




