

def train_model(filename, model, optimizer, criterion, num_epoch, save_name, use_BERT=False):
    """Train a model on the given dataset

    Args:
        filename (string): name of h5 file containing dataset
        model (PyTorch nn.Module): PyTorch neural network architecture
        optimizer (PyTorch optimizer): optimization method
        criterion (function): loss function
        num_epoch (int): number of epochs
        save_name (string): name under which to save trained model and results
        use_BERT (bool): use pretrained BERT for feature extraction
    """
    pass