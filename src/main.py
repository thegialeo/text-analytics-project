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
    parser.add_argument("--create_h5", dest="create_h5", action='store_true',
                        help="Preprocess the downloaded datasets and save the result in a h5 file")
    parser.add_argument("--backtranslation", dest="backtrans", action='store_true',
                        help="Use backtranslation during --create_h5")
    parser.add_argument("--lemmatization", dest="lemma", action='store_true',
                        help="Use lemmatization during --create_h5")
    parser.add_argument("--stemming", dest="stemm", action='store_true',
                        help="Use stemming during --create_h5")
    parser.add_argument("--random_swap", dest="swap", action='store_true',
                        help="Use random swap during --create_h5")
    parser.add_argument("--random_deletion", dest="delete", action='store_true',
                        help="Use random deletion during --create_h5")                                                                              
    parser.add_argument("--filename", dest="filename", action='store',
                        help="Name of h5 file to load or save to")
    parser.add_argument("--search", dest='search', action='store', nargs=6,
                        help="Perform linear search for [hyperparameter, start, end, step, model, filename]. Options: hyperparameter ['feature', 'window', 'count', 'epochs', 'lr', 'min_lr'], model ['word2vec']")
    parser.add_argument("--experiment", dest='experiment', action='store',
                        help="Select experiment to perform. Options: 'compare_all', 'evaluate'")
    

    parser.set_defaults(download=None, create_h5=False, backtrans=False, lemma=False, stem=False, swap=False, delete=False, filename=None, search=None, experiment=None)
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

    # preprocessing + augmentation
    if args.create_h5:
        to_dataframe.store_augmented_h5(args.filename, args.backtrans, args.lemma, args.stem, args.swap, args.delete)

    # hyperparameter search
    if args.search is not None:
            traverser.traverser(*args.search, args.pretrained)

    # experiments
    if args.experiment is not None:
        # compare all regression and vectorization methods
        if args.experiment == 'compare_all':
            experiments.benchmark_all(args.filename, False)
            experiments.benchmark_all(args.filename, True)
        # test and debug evaluate_baseline
        if args.experiment == 'test':
            MSE, RMSE, MAE, r_square = evaluater.evaluate_baseline(vec="tfidf")
            print(r_square)

        # test BERT training
        if args.experiment == 'BERT':
            trainer.train_model(args.filename, 20, [10, 15, 18, 20], 128, 1e-3, "test")




