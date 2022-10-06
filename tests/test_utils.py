import numpy as np

from malescratch.utils import (
    MinMaxScaler,
    StandardScaler,
    to_categorical,
    train_test_split,
    make_batch_index,
)


def test_positive_train_test_split():
    X = np.random.randint(0, 10, size=(100, 10))
    y = np.random.randint(0, 10, size=(100,))
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    assert type(X_train) == np.ndarray
    assert type(X_test) == np.ndarray
    assert type(y_train) == np.ndarray
    assert type(y_test) == np.ndarray


def test_positive_make_batch_index():
    pass


def test_to_categorical():
    pass


def test_MinMaxScaler():
    pass
