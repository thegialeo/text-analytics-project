from os.path import abspath, dirname, join
import multiprocessing
import time

import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from utils import BERT, evaluater, gpu, regression, to_dataframe


def train_model(filename, num_epoch, step_epochs, batch_size, lr, save_name):
    """Train a model on the given dataset

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.

    Args:
        filename (string): name of h5 file containing dataset
        num_epoch (int): number of epochs
        step_epochs (list): list of epoch number, where learning rate will be reduce by a factor of 10
        batch_size (int): batch size
        lr (float): learning rate
        save_name (string): name under which to save trained model and results
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
    print("Number of CPUs detected:", num_workers)

    # read data
    df_train, df_test = to_dataframe.read_augmented_h5("all_data.h5")
    df_train = df_train[
        df_train["source"] == "text_comp19"
    ]  # TODO: remove once Raoul fixes his dataloader
    df_test = df_test[
        df_test["source"] == "text_comp19"
    ]  # TODO: remove once Raoul fixes his dataloader

    # setup BERT model
    bert_model = BERT.BERT()

    # prepare BERT input
    train_sentences = df_train.raw_text.values
    test_sentences = df_test.raw_text.values
    train_input_tensor, train_segment_tensor = bert_model.preprocessing(train_sentences)
    test_input_tensor, test_segment_tensor = bert_model.preprocessing(test_sentences)

    # feature extraction
    train_features = bert_model.get_features(train_input_tensor, train_segment_tensor)
    test_features = bert_model.get_features(test_input_tensor, test_segment_tensor)

    # extract labels and cast to PyTorch tensor
    train_labels = torch.tensor(list(df_train.rating.values))
    test_labels = torch.tensor(list(df.test.rating.values))

    # prepare dataset
    trainset = TensorDataset(train_features, train_labels)
    testset = TensorDataset(test_features, test_labels)

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

    # prepare regression model
    reg_model = regression.Net()
    reg_model = reg_model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(reg_model.parameters(), lr=lr)

    # criterion
    criterion = torch.nn.MSELoss()

    # scheduler
    scheduler = opt.lr_scheduler.MultiStepLR(optimizer, steps_epochs, 0.1)

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

        # training
        for i, (feature, label) in enumerate(trainloader):
            # move batch to device
            feature = feature.to(device)
            label = label.to(device)

            # clear gradients
            optimizer.zero_grad()

            # prediction
            output = reg_model(feature)

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
        reg_model.eval()
        running_loss /= len(trainloader)
        MSE_train, RMSE_train, MAE_train, r2_train = evaluater.evaluate_model(
            reg_model, trainloader
        )
        MSE_test, RMSE_test, MAE_test, r2_test = evaluater.evaluate_model(
            reg_model, testloader
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

        # print stats
        print(
            "epoch {} \t loss {:.5f} \t train_r2 {:.3f} \t test_r2 {:.3f} \t time {:.1f} sec".format(
                epoch + 1, running_loss, r2_train, r2_test, time.time() - start
            )
        )

        # save logs
        file = open(join(log_path, save_name + '.txt'), 'w')
        print('Last Epoch:', epoch + 1, file=file)
        print('Final Loss:', loss_log[-1], file=file)
        print('Final Train MSE:', train_MSE_log[-1], file=file)
        print('Final Train RMSE:', train_RMSE_log[-1], file=file)
        print('Final Train MAE:', train_MAE_log[-1], file=file)
        print('Final Train R2:', train_r2_log[-1], file=file)
        print('Final Test MSE:', test_MSE_log[-1], file=file)
        print('Final Test RMSE:', test_RMSE_log[-1], file=file)
        print('Final Test MAE:', test_MAE_log[-1], file=file)
        print('Final Test R2:', test_r2_log[-1], file=file)


