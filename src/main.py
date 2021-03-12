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
    parser.add_argument(
        "--download",
        dest="download",
        action="store",
        help="Download specific or all datasets. Options: 'all', 'TextComplexityDE19', 'Weebit', 'dw'",
    )
    parser.add_argument(
        "--dset",
        dest="dset",
        action="store",
        help="Specify which datasets are used. Options: '0' = 'TextComplexityDE19','1' = 'Weebit','2' = 'dw'. Enter eg. '012' for all.",
    )
    parser.add_argument(
        "--create_h5",
        dest="create_h5",
        action="store_true",
        help="Preprocess the downloaded datasets and save the result in a h5 file",
    )
    parser.add_argument(
        "--backtranslation",
        dest="backtrans",
        action="store_true",
        help="Use backtranslation during --create_h5",
    )
    parser.add_argument(
        "--lemmatization",
        dest="lemma",
        action="store_true",
        help="Use lemmatization during --create_h5",
    )
    parser.add_argument(
        "--stemming",
        dest="stem",
        action="store_true",
        help="Use stemming during --create_h5",
    )
    parser.add_argument(
        "--random_swap",
        dest="swap",
        action="store_true",
        help="Use random swap during --create_h5",
    )
    parser.add_argument(
        "--random_deletion",
        dest="delete",
        action="store_true",
        help="Use random deletion during --create_h5",
    )
    parser.add_argument(
        "--filename",
        dest="filename",
        action="store",
        help="Name of h5 file to load or save to",
    )
    parser.add_argument(
        "--search",
        dest="search",
        action="store",
        nargs=6,
        help="Perform linear search for [hyperparameter, start, end, step, model, filename]. Options: hyperparameter ['feature', 'window', 'count', 'epochs', 'lr', 'min_lr'], model ['word2vec']",
    )
    parser.add_argument(
        "--experiment",
        dest="experiment",
        action="store",
        help="Select experiment to perform. Options: 'compare_all', 'evaluate', 'train_net'",
    )
    parser.add_argument(
        "--engineered_features",
        dest="extra_feat",
        action="store_true",
        help="Concatenate engineered features to features obtained by vectorizer",
    )
    parser.add_argument(
        "--vectorizer",
        dest="vectorizer",
        action="store",
        help="Specify which vectorizer method to use. Options: 'tfidf', 'count', 'hash', 'word2vec', 'pretrained_word2vec'",
    )
    parser.add_argument(
        "--method",
        dest="method",
        action="store",
        help="Specify which regression method to use. Options: 'linear', 'lasso', 'ridge', 'elastic-net', 'random-forest'",
    )
    parser.add_argument(
        "--save_name",
        dest="save_name",
        action="store",
        help="Name to save train model under. Only available for --experiment train_net (used for prototyping and hyperparameter tuning)",
    )
    parser.add_argument(
        "--multiple_datasets",
        dest="conditional",
        action="store_true",
        help="If multiple datasets are used, to conditional training",
    )
    parser.add_argument(
        "--pretask",
        dest="pretask",
        action="store",
        nargs=2,
        help="Provide number of epochs to train pretask and filename of dataset to perform the pretask on"
    )
    parser.add_argument(
        "--dropout",
        dest="dropout",
        action="store_true",
        help="Use network architecture that has dropout layers"
    )
    parser.add_argument(
        "--batchnorm",
        dest="batchnorm",
        action="store_true",
        help="Use network architecture that has batch normalization layers"
    )
    parser.add_argument(
        "--no_freeze",
        dest="no_freeze",
        action="store_true",
        help="Don't freeze first layer in pretask training"
    )

    parser.set_defaults(
        dset="0",
        download=None,
        create_h5=False,
        backtrans=False,
        lemma=False,
        stem=False,
        swap=False,
        delete=False,
        filename=None,
        search=None,
        experiment=None,
        extra_feat=False,
        vectorizer=None,
        method=None,
        save_name=None,
        conditional=False,
        pretask=[None, None],
        dropout=False,
        batchnorm=False,
        no_freeze=False
    )
    args = parser.parse_args()


    # load datasets
    if args.download is not None:
        if args.download == "all":
            downloader.download_TextComplexityDE19()
            downloader.download_Weebit()
            downloader.download_dw_set()
        elif args.download == "TextComplexityDE19":
            downloader.download_TextComplexityDE19()
        elif args.download == "Weebit":
            downloader.download_Weebit()
        elif args.download == "dw":
            downloader.download_dw_set()
        else:
            raise ValueError(
                "Input {} for --download is invalid. Choose one of the following: 'all', 'TextComplexityDE19', 'Weebit', 'dw'".format(
                    args.download
                )
            )

    # preprocessing + augmentation
    if args.create_h5:
        use_textcomp = True if "0" in args.dset else False
        use_weebit = True if "1" in args.dset else False
        use_dw = True if "2" in args.dset else False
        to_dataframe.store_augmented_h5(
            args.filename,
            use_textcomp,
            use_weebit,
            use_dw,
            args.backtrans,
            args.lemma,
            args.stem,
            args.swap,
            args.delete,
            0.2,
        )

    # hyperparameter search
    if args.search is not None:
        traverser.traverser(*args.search)

    # experiments
    if args.experiment is not None:
        # compare all regression and vectorization methods
        if args.experiment == "compare_all":
            experiments.benchmark_all(args.filename, False)
            experiments.benchmark_all(args.filename, True)
        # evaluate a regression method with a vectorization method
        if args.experiment == "evaluate":
            MSE, RMSE, MAE, r_square = evaluater.evaluate_baseline(
                args.vectorizer, args.method, args.filename, args.extra_feat
            )
            print("MSE:", MSE)
            print("RMSE:", RMSE)
            print("MAE:", MAE)
            print("R square:", r_square)

        # pretrained BERT + regression neural network
        if args.experiment == "train_net":
            if args.save_name is None:
                args.save_name = args.filename
            if args.pretask[0] is not None:
                args.pretask[0] = int(args.pretask[0])
            trainer.train_model(
                args.filename,
                20,
                [10, 15, 18, 20],
                128,
                1e-3,
                args.save_name,
                args.extra_feat,
                args.conditional,
                args.pretask[0],
                args.pretask[1],
                args.dropout,
                args.batchnorm,
                args.no_freeze
            )
