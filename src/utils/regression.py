from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from torch import nn


def baseline(data, labels, method='linear'):
    """Various ML baseline regressions.

        Written by Leo Nguyen. Contact Xenovortex, if problems arises.

    Args:
        data (array-like): training data to train the model on
        labels (array-like): correspoding labels for the training data
        method (str, optional): regression method to use (options: 'linear', 'lasso', 'ridge', 'elastic-net', 'random-forest'). Defaults to 'linear'.

    Return: 
        reg (object): returns a regression model trained on the given data and labels
    """

    if method == 'linear':
        reg = LinearRegression().fit(data, labels)
    elif method == 'lasso':
        reg = Lasso(random_state=0).fit(data, labels)
    elif method == 'ridge':
        reg = Ridge(random_state=0).fit(data, labels)
    elif method == 'elastic-net':
        reg = ElasticNet(random_state=0).fit(data, labels)
    elif method == 'random-forest':
        reg = RandomForestRegressor(random_state=0).fit(data, labels)
    else:
        raise ValueError("Regression {} is unknown. Please choose: 'linear', 'lasso', 'ridge', 'elastic-net', 'random-forest'".format(method))
    
    return reg



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
            nn.Linear(num_hidden, num_output))

    def forward(self, x):
        return self.model(x)
