from torch import nn


class Net(nn.Module):
    """3-layer Neural Network

    Written by Leo Nguyen. Contact Xenovortex, if problems arises.
    """

    def __init__(self, num_features, num_hidden, num_output):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, num_output),
        )

    def forward(self, x):
        return self.model(x)


class BN_Net(nn.Module):
    """3-layer Neural Network with Batch Normalization

    Written by Leo Nguyen. Contact Xenovortex, if problems arises.
    """

    def __init__(self, num_features, num_hidden, num_output):
        super(BN_Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, num_output)
        )

    def forward(self, x):
        return self.model(x)


class Dropout_Net(nn.Module):
    """3-layer Neural Network with Dropout

    Written by Leo Nguyen. Contact Xenovortex, if problems arises.
    """

    def __init__(self, num_features, num_hidden, num_output):
        super(Dropout_Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True), 
            nn.Dropout(0.2),
            nn.Linear(num_hidden, num_output)
        )

    def forward(self, x):
        return self.model(x)


