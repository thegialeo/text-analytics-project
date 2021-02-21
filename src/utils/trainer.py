import torch
from torch.utils.data import TensorDataset
from utils import to_dataframe, BERT, regression, gpu



def train_model(filename, num_epoch, batch_size, lr, save_name):
    """Train a model on the given dataset

    Args:
        filename (string): name of h5 file containing dataset
        num_epoch (int): number of epochs
        batch_size (int): batch size 
        lr (float): learning rate
        save_name (string): name under which to save trained model and results
    """

    # set device
    device = gpu.check_gpu()
    
    # read data
    df_train, df_test = to_dataframe.read_augmented_h5("all_data.h5")
    df_train = df_train[df_train["source"] == "text_comp19"] # TODO: remove once Raoul fixes his dataloader
    df_test = df_test[df_test["source"] == "text_comp19"]  # TODO: remove once Raoul fixes his dataloader

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

    # prepare regression model
    reg_model = regression.Net()
    reg_model = reg_model.to(device)
    reg_model.train()

    # optimizer
    optimizer = torch.optim.Adam(reg_model.parameters(), lr = lr)
    
    # criterion
    criterion = torch.nn.MSELoss()
