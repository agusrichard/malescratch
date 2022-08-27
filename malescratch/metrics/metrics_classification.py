import numpy as np



def accuracy_score(predictions, labels):

    """Calculate the accuracy score between predictions and labels. \
    The percentage of correct predictions.

    Parameters:
    ----------

    predictions : array-like
    The predictions of an estimator

    labels : array-like
    The true labels of data

    Raise:
    -----

    ValueError: raise value error when the length of predictions and labels are not the same

    """


    if len(predictions) == len(labels):
        return np.sum(np.equal(predictions, labels)) / len(predictions)
    else:
        raise ValueError("The length of predictions and labels are not the same")
