import os
from os.path import abspath, dirname, join, exists
import multiprocessing
import time
import pickle

import torch
import numpy as np
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as opt
import matplotlib.pyplot as plt
from utils import BERT, evaluater, gpu, regression, to_dataframe, sentencestats, architectures
from tqdm import tqdm


def train_model(
    filename,
    num_epoch,
    step_epochs,
    batch_size,
    lr,
    save_name,
    engineered_features=False,
    multiple_dataset=False,
    pretask_epoch=None,
    pretask_file=None,
    dropout=False,
    batchnorm=False
):
    """Train a model on the given dataset

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.

    Args:
        filename (string): name of h5 file containing dataset
        num_epoch (int): number of epochs
        step_epochs (list): list of epoch number, where learning rate will be reduce by a factor of 10
        batch_size (int): batch size
        lr (float): learning rate
        save_name (string): name under which to save trained model and results
        engineered_features (bool, optional): contenate engineered features to vectorized sentence
        multiple_dataset (bool, optional): use multiple datasets for conditional training
        pretask_epoch (int, optional): integer provided will be the number of epochs spent on the pretask
        pretask_file (str, optional): filename of dataset for pretask
        dropout (bool, optional): use network architecture with dropout
        batchnorm (bool, optional): use network architecture with batch normalization
    """

    # save paths
    model_path = join(dirname(dirname(dirname(abspath(__file__)))), "model", "BERT")
    log_path = join(dirname(dirname(dirname(abspath(__file__)))), "result", "BERT")
    fig_path = join(dirname(dirname(dirname(abspath(__file__)))), "figures", "BERT")

    # create directories
    for path in [model_path, log_path, fig_path]:
        if not exists(dirname(path)):
            os.makedirs(dirname(path))
        if not exists(path):
            os.makedirs(path)

    # set device
    device = gpu.check_gpu()
    num_workers = multiprocessing.cpu_count()
    print("Training on:", device)
    print("Number of CPU cores detected:", num_workers)

    # read data
    df_train, df_test = to_dataframe.read_augmented_h5(filename)

    # setup BERT model
    bert_model = BERT.BERT()

    # prepare BERT input
    train_sentences = df_train.raw_text.values
    test_sentences = df_test.raw_text.values
    train_input_tensor, train_segment_tensor = bert_model.preprocessing(train_sentences)
    test_input_tensor, test_segment_tensor = bert_model.preprocessing(test_sentences)

    # extract labels and cast to PyTorch tensor
    train_labels = torch.tensor(
        list(df_train.rating.values), dtype=torch.float
    ).unsqueeze_(1)
    test_labels = torch.tensor(
        list(df_test.rating.values), dtype=torch.float
    ).unsqueeze_(1)

    # prepare dataset
    if engineered_features and multiple_dataset:
        extra_train_feat = torch.from_numpy(
            np.nan_to_num(sentencestats.construct_features(df_train.raw_text).values)
        ).float()
        extra_test_feat = torch.from_numpy(
            np.nan_to_num(sentencestats.construct_features(df_test.raw_text).values)
        ).float()
        train_dataset_label = torch.tensor(
            list(df_train.source.values), dtype=torch.float
        ).unsqueeze_(1)
        test_dataset_label = torch.tensor(
            list(df_test.source.values), dtype=torch.float
        ).unsqueeze_(1)
        trainset = TensorDataset(
            train_input_tensor,
            train_segment_tensor,
            train_labels,
            extra_train_feat,
            train_dataset_label,
        )
        testset = TensorDataset(
            test_input_tensor,
            test_segment_tensor,
            test_labels,
            extra_test_feat,
            test_dataset_label,
        )
    elif engineered_features:
        extra_train_feat = torch.from_numpy(
            np.nan_to_num(sentencestats.construct_features(df_train.raw_text).values)
        ).float()
        extra_test_feat = torch.from_numpy(
            np.nan_to_num(sentencestats.construct_features(df_test.raw_text).values)
        ).float()
        trainset = TensorDataset(
            train_input_tensor, train_segment_tensor, train_labels, extra_train_feat
        )
        testset = TensorDataset(
            test_input_tensor, test_segment_tensor, test_labels, extra_test_feat
        )
    elif multiple_dataset:
        train_dataset_label = torch.tensor(
            list(df_train.source.values), dtype=torch.float
        ).unsqueeze_(1)
        test_dataset_label = torch.tensor(
            list(df_test.source.values), dtype=torch.float
        ).unsqueeze_(1)
        trainset = TensorDataset(
            train_input_tensor, train_segment_tensor, train_labels, train_dataset_label
        )
        testset = TensorDataset(
            test_input_tensor, test_segment_tensor, test_labels, test_dataset_label
        )
    else:
        trainset = TensorDataset(train_input_tensor, train_segment_tensor, train_labels)
        testset = TensorDataset(test_input_tensor, test_segment_tensor, test_labels)

    # dataloader
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # setup pretask
    if pretask_epoch is not None and pretask_file is not None:
        # read data
        df_pretask = to_dataframe.read_augmented_h5(pretask_file)
        
        # prepare BERT input
        pretask_sentences = df_pretask.raw_text.values
        pretask_input_tensor, pretask_segment_tensor = bert_model.preprocessing(pretask_sentences)
        
        # extract labels + cast to PyTorch tensors
        pretask_labels = torch.tensor(
            list(df_pretask.rating.values), dtype=torch.float
        ).unsqueeze_(1)
        
        # prepare dataset
        if engineered_features:
            extra_pretask_feat = torch.from_numpy(
                sentencestats.construct_features(pretask_sentences)
            )
            pretask_set = TensorDataset(
                pretask_input_tensor, pretask_segment_tensor, pretask_labels, extra_pretask_feat
            )
        else:
            pretask_set = TensorDataset(pretask_input_tensor, pretask_segment_tensor, pretask_labels)
        
        # dataloader 
        pretask_loader = DataLoader(
            pretask_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )


    # prepare regression model
    feat_size = 768
    if engineered_features:
        feat_size += 21
    if multiple_dataset:
        feat_size += 1

    if dropout and batchnorm:
        reg_model = architectures.BN_Dropout_Net(feat_size, 512, 1)
    elif batchnorm:
        reg_model = architectures.BN_Net(feat_size, 512, 1)
    elif dropout:
        reg_model = architectures.Dropout_Net(feat_size, 512, 1)
    else:
        reg_model = architectures.Net(feat_size, 512, 1)
    reg_model = reg_model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(reg_model.parameters(), lr=lr)

    # criterion
    criterion = torch.nn.MSELoss()
    pretask_criterion = torch.nn

    # scheduler
    scheduler = opt.lr_scheduler.MultiStepLR(optimizer, step_epochs, 0.1)

    # log
    loss_log = []
    train_MSE_log = []
    train_RMSE_log = []
    train_MAE_log = []
    train_r2_log = []
    test_MSE_log = []
    test_RMSE_log = []
    test_MAE_log = []
    test_r2_log = []

    for epoch in range(num_epoch):
        start = time.time()
        reg_model.train()

        print("Training:")

        # training
        for i, data in enumerate(tqdm(trainloader)):
            # move batch and model to device
            reg_model.to(device)
            input_id = data[0].to(device)
            segment = data[1].to(device)
            label = data[2].to(device)
            if engineered_features and multiple_dataset:
                extra_feat = data[3].to(device)
                dataset_label = data[4].to(device)
            elif engineered_features:
                extra_feat = data[3].to(device)
            elif multiple_dataset:
                dataset_label = data[3].to(device)

            # clear gradients
            optimizer.zero_grad()

            # BERT feature extraction
            features = bert_model.get_features(input_id, segment)

            # add engineered features
            if engineered_features:
                features = torch.cat((features, extra_feat), 1)

            # add dataset conditional label
            if multiple_dataset:
                features = torch.cat((features, dataset_label), 1)

            # prediction
            output = reg_model(features)

            # loss evaluation
            loss = criterion(output, label)
            loss.backward()

            # backpropagation
            optimizer.step()

            # record loss
            curr_loss = torch.mean(loss).item()
            running_loss = (
                curr_loss if ((i == 0) and (epoch == 0)) else running_loss + curr_loss
            )

        # update training schedule
        scheduler.step()

        # evaluation
        print("Evaluation:")
        reg_model.eval()
        running_loss /= len(trainloader)
        MSE_train, RMSE_train, MAE_train, r2_train = evaluater.evaluate_model(
            reg_model, bert_model, trainloader, engineered_features, multiple_dataset
        )
        MSE_test, RMSE_test, MAE_test, r2_test = evaluater.evaluate_model(
            reg_model, bert_model, testloader, engineered_features, multiple_dataset
        )

        # log evaluation result
        loss_log.append(running_loss)
        train_MSE_log.append(MSE_train)
        train_RMSE_log.append(RMSE_train)
        train_MAE_log.append(MAE_train)
        train_r2_log.append(r2_train)
        test_MSE_log.append(MSE_test)
        test_RMSE_log.append(RMSE_test)
        test_MAE_log.append(MAE_test)
        test_r2_log.append(r2_test)

        # save logs
        file = open(join(log_path, save_name + ".txt"), "w")
        print("Last Epoch:", epoch + 1, file=file)
        print("Final Loss:", loss_log[-1], file=file)
        print("Final Train MSE:", train_MSE_log[-1], file=file)
        print("Final Train RMSE:", train_RMSE_log[-1], file=file)
        print("Final Train MAE:", train_MAE_log[-1], file=file)
        print("Final Train R2:", train_r2_log[-1], file=file)
        print("Final Test MSE:", test_MSE_log[-1], file=file)
        print("Final Test RMSE:", test_RMSE_log[-1], file=file)
        print("Final Test MAE:", test_MAE_log[-1], file=file)
        print("Final Test R2:", test_r2_log[-1], file=file)

        # save variables
        with open(join(log_path, save_name + ".pkl"), "wb") as f:
            pickle.dump(
                [
                    loss_log,
                    train_MSE_log,
                    train_RMSE_log,
                    train_MAE_log,
                    train_r2_log,
                    test_MSE_log,
                    test_RMSE_log,
                    test_MAE_log,
                    test_r2_log,
                ],
                f,
            )

        # save model weights
        torch.save(
            reg_model.to("cpu").state_dict(), join(model_path, save_name + ".pt")
        )

        # print stats
        print(
            "epoch {} \t loss {:.5f} \t train_r2 {:.3f} \t test_r2 {:.3f} \t time {:.1f} sec".format(
                epoch + 1, running_loss, r2_train, r2_test, time.time() - start
            )
        )

    # plots
    plot_names = [
        "loss",
        "train_MSE",
        "train_RMSE",
        "train_MAE",
        "train_r2",
        "test_MSE",
        "test_RMSE",
        "test_MAE",
        "test_r2",
    ]
    for i, log in enumerate(
        [
            loss_log,
            train_MSE_log,
            train_RMSE_log,
            train_MAE_log,
            train_r2_log,
            test_MSE_log,
            test_RMSE_log,
            test_MAE_log,
            test_r2_log,
        ]
    ):
        plt.figure(num=None, figsize=(15, 10))
        plt.plot(log)
        plt.grid(True, which="both")
        plt.xlabel("epoch", fontsize=14)
        plt.ylabel(plot_names[i], fontsize=14)
        plt.savefig(join(fig_path, save_name + "_" + plot_names[i] + ".png"))
        plt.close("all")


def train_pretask(pretask_epoch, model, bert_model, dataloader, criterion, optimizer, engineered_features=False):
    """Train a model on a pretask

    Written by Leo Nguyen. Contact Xenovortex, if problems arises.

    Args:
        pretask_epoch (int): integer provided will be the number of epochs spent on the pretask
        model (torch.nn.Module): PyTorch model of a Regression Neural Network
        bert_model (torch.nn.Module): BERT PyTorch model for feature extraction
        dataloader (PyTorch dataloader): PyTorch dataloader of dataset
        criterion (function): loss function
        optimizer (PyTorch optimizer): optimizer of model parameters
        engineered_features (bool, optional): contenate engineered features to vectorized sentence
    """

    # log
    pretask_loss_log = []
    pretask_train_MSE_log = []
    pretask_train_RMSE_log = []
    pretask_train_MAE_log = []
    pretask_train_r2_log = []
   
    # set device
    device = gpu.check_gpu()
    print("Pretask Training on:", device)

    for epoch in range(pretask_epoch):
        start = time.time()
        model.train()

        print("Training:")

        # training
        for i, data in enumerate(tqdm(dataloader)):
            # move batch and model to device
            model.to(device)
            input_id = data[0].to(device)
            segment = data[1].to(device)
            label = data[2].to(device)
            if engineered_features:
                extra_feat = data[3].to(device)

            # clear gradients
            optimizer.zero_grad()

            # BERT feature extraction
            features = bert_model.get_features(input_id, segment)

            # add engineered features
            if engineered_features:
                features = torch.cat((features, extra_feat), 1)
            
            # prediction
            output = model(features)

            # loss evaluation
            loss = criterion(output, label)
            loss.backward()

            # backpropagation
            optimizer.step()

            # record loss 
            curr_loss = torch.mean(loss).item()
            running_loss = (
                curr_loss if ((i == 0) and (epoch == 0)) else running_loss + curr_loss
            )

       