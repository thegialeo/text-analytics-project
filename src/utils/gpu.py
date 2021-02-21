import torch


def check_gpu():
    """Get device on which to train on. 

       Written by Leo Nguyen. Contact Xenovortex, if problems arises.

    Return:
        device(object): PyTorch Device object that decides on which device to train/evaluate on 
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Device used for training and evaluation: {}".format(device))
    
    return device
        
