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
