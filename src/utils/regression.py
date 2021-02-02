from sklearn.linear_model import LinearRegression, Lasso, Ridge


def baseline(data, labels, method='linear'):
    """Various ML baseline regressions.

    Args:
        data (array-like): training data to train the model on
        labels (array-like): correspoding labels for the training data
        method (str, optional): [description]. Defaults to 'linear'.
    """

    if method == 'linear':
        reg = LinearRegression().fit(data, labels)
    elif method == 'lasso':
        reg = Lasso(random_state=0).fit(data, labels)
    elif method == 'ridge':
        reg = Ridge(random_state=0).fit(data, labels)
    else:
        print("Regression {} is unknown. Please choose: 'linear'".format(method))
        exit()
    
    
    return reg
