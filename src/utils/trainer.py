import torch
from utils import to_dataframe



def train_model(filename, model, optimizer, criterion, num_epoch, batch_size, save_name, use_BERT=False):
    """Train a model on the given dataset

    Args:
        filename (string): name of h5 file containing dataset
        model (PyTorch nn.Module): PyTorch neural network architecture
        optimizer (PyTorch optimizer): optimization method
        criterion (function): loss function
        num_epoch (int): number of epochs
        batch_size (int): batch size 
        save_name (string): name under which to save trained model and results
        use_BERT (bool): use pretrained BERT for feature extraction
    """
    
    # read data
    df_train, df_test = to_dataframe.read_augmented_h5("all_data.h5")
    df_train = df_train[df_train["source"] == "text_comp19"] # TODO: remove once Raoul fixes his dataloader
    df_test = df_test[df_test["source"] == "text_comp19"]  # TODO: remove once Raoul fixes his dataloader

    # extract labels and cast to PyTorch tensor
    train_labels = torch.tensor(list(df_train.rating.values))
    test_labels = torch.tensor(list(df.test.rating.values))
