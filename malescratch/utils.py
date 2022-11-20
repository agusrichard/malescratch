import numpy as np
from typing import Tuple, List


def train_test_split(
    *arrays: List[np.ndarray], train_size=None, test_size=None, random_state=42
) -> Tuple[np.ndarray]:
    """Split the data into train set and test set. This function shuffle the data
        before splitting it

        Parameters:
        ----------

        *arrays: array-like
            Sequence of indexables with same length / shape[0]

        train_size: float or int, default 0.9


        test_size : float or int, default 0.1
    If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples.

        random_state : integer
            Random state seed

        Returns:
        -------

        splitting : list, length=2 * len(arrays)
            List containing train-test split of inputs.

        Examples
        --------
        >>> from malescratch.utils import train_test_split
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    """

    seed = np.random.RandomState(random_state)
    index = seed.permutation(np.arange(len(arrays[0])))
    test_size = int(len(arrays[0]) * float(test_ratio))

    def wrapper():
        for array in arrays:
            array = np.array(array)
            test_index = index[:test_size]
            train_index = index[test_size:]
            test = array[test_index]
            train = array[train_index]
            yield train
            yield test

    return tuple(wrapper())


def make_batch_index(sample_size, num_batch, size, shuffle=False, random_state=42):
    """Make batch index for further batch making process

    Parameters:
    ----------

    sample_size : integer
        Sample size

    num_batch : integer
        Number of batch

    size : integer
        sample size will be created

    shuffle : boolean
        If True, using permutation to create index.
        If False, using arange to create index

    random_state : integer
        Random state seed

    Returns:
        Batch index
    """

    gen = np.random.RandomState(random_state)
    if sample_size == size:
        if shuffle:
            index_batch = np.array_split(gen.permutation(sample_size), num_batch)
        else:
            index_batch = np.array_split(np.arange(sample_size), num_batch)
    elif num_batch <= size:
        index_batch = np.array_split(gen.randint(0, sample_size, size=size), num_batch)
    else:
        raise ValueError("Size must be higher than number of batch")

    return index_batch


class MinMaxScaler(object):
    """Scaling the data to between 0 and 1"""

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)

        return self

    def transform(self, X):
        diff_X = X - self.min_
        diff_minmax = self.max_ - self.min_

        return diff_X / diff_minmax

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class StandardScaler(object):
    """Standardize the data"""

    def __init__(self):
        self.something = None
        self.mean_ = None
        self.stddev_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.stddev_ = X.std(axis=0)

        return self

    def transform(self, X):
        diff_mean = X - self.mean_

        return diff_mean / self.stddev_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def to_categorical(labels):
    sample = len(labels)
    cols = np.max(labels) + 1
    result = np.zeros(shape=(sample, cols))
    for i, row in enumerate(result):
        row[labels[i]] = 1

    return result
