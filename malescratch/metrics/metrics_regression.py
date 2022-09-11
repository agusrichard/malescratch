import numpy as np


def mean_absolute_error(predictions, labels):

    """Calculate Mean Absolute Error between predictions and labels

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
        return np.sum(np.abs(predictions - labels)) / len(labels)
    else:
        raise ValueError("The length of predictions and labels are not the same")


# ======================================================================================================================


def mean_absolute_percentage_error(predictions, labels):

    """Calculate Mean Absolute Percentage Error between predictions and labels.
    It will be problematic if some elements inside labels are zero.

    Parameters:
    ----------

    predictions : array-like
    The predictions of an estimator

    labels : array-like
    The true labels of data

    Raises:
    -----

    ValueError: get raised when the length of predictions and labels are not the same

    ZeroDivisionError : get raised when some of the labels contain zero

    """

    if len(predictions) == len(labels):
        return np.sum(np.abs((labels - predictions) / labels)) * (
            100 / len(predictions)
        )
    else:
        raise ValueError("The length of predictions and labels are not the same")


# ======================================================================================================================


def mean_squared_error(predictions, labels):

    """Calculate Mean Squared Error between predictions and labels.

    Parameters:
    ----------

    predictions : array-like
    The predictions of an estimator

    labels : array-like
    The true labels of data

    Raise:
    -----

    ValueError: get raised when the length of predictions and labels are not the same

    """

    if len(predictions) == len(labels):
        return np.sum(np.power(labels - predictions, 2)) / len(predictions)
    else:
        raise ValueError("The length of predictions and labels are not the same")


# ======================================================================================================================


def root_mean_square_error(predictions, labels):

    """Calculate Root Mean Squared Error between predictions and labels.

    Parameters:
    ----------

    predictions : array-like
    The predictions of an estimator

    labels : array-like
    The true labels of data

    Raise:
    -----

    ValueError: get raised when the length of predictions and labels are not the same

    """

    if len(predictions) == len(labels):
        return np.sqrt(np.sum(np.power(labels - predictions, 2)) / len(predictions))
    else:
        raise ValueError("The length of predictions and labels are not the same")


# ======================================================================================================================


def r2_score(predictions, labels):

    """Calculate R-Squared (Coefficients of Determination) between predictions and labels.

    Parameters:
    ----------

    predictions : array-like
    The predictions of an estimator

    labels : array-like
    The true labels of data

    Raise:
    -----

    ValueError: get raised when the length of predictions and labels are not the same

    """

    if len(predictions) == len(labels):
        ybar = np.mean(labels)
        sres = np.sum(np.power(labels - predictions, 2))
        stot = np.sum(np.power(labels - ybar, 2))
        return 1 - (sres / stot)
    else:
        raise ValueError("The length of predictions and labels are not the same")


# ======================================================================================================================


def adjusted_r2_score(predictions, labels, num_features):

    """Calculate R-Squared (Coefficients of Determination) between predictions and labels.

    Parameters:
    ----------

    predictions : array-like
    The predictions of an estimator

    labels : array-like
    The true labels of data

    Raise:
    -----

    ValueError: get raised when the length of predictions and labels are not the same

    """

    def r2_score(predictions, labels):
        if len(predictions) == len(labels):
            ybar = np.mean(labels)
            sres = np.sum(np.power(labels - predictions, 2))
            stot = np.sum(np.power(labels - ybar, 2))
            return 1 - (sres / stot)
        else:
            raise ValueError("The length of predictions and labels are not the same")

    r2 = r2_score(predictions, labels)
    n = len(predictions)
    k = num_features
    cons = (n - 1) / (n - (k + 1))

    return 1 - (1 - r2) * cons


# ======================================================================================================================


def max_error(predictions, labels):

    """Calculate the maximum residual error (absolute) between predictions and labels.

    Parameters:
    ----------

    predictions : array-like
    The predictions of an estimator

    labels : array-like
    The true labels of data

    Raise:
    -----

    ValueError: get raised when the length of predictions and labels are not the same

    """

    if len(predictions) == len(labels):
        return np.max(np.abs(labels - predictions))
    else:
        raise ValueError("The length of predictions and labels are not the same")


# ======================================================================================================================


def mean_squared_log_error(predictions, labels):

    """Calculate Mean Squared Logarithmic Error between predictions and labels.

    Parameters:
    ----------

    predictions : array-like
    The predictions of an estimator

    labels : array-like
    The true labels of data

    Raise:
    -----

    ValueError: 
        Get raised when the length of predictions and labels are not the same or \
        when the predictions or labels contain negative values.

    """

    if len(predictions) == len(labels):
        if (predictions < 0).any() or (labels < 0).any():
            raise ValueError(
                "Mean Squared Logarithmic Error cannot be used when targets contain negative values"
            )
        else:
            log_labels = np.log(1 + labels)
            log_preds = np.log(1 + predictions)
            return np.sum(np.power(log_labels - log_preds, 2)) / len(predictions)
    else:
        raise ValueError("The length of predictions and labels are not the same")
