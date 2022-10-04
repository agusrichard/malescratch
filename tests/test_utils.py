import numpy as np

from malescratch.utils import (
    MinMaxScaler,
    StandardScaler,
    to_categorical,
    train_test_split,
    make_batch_index,
)


def test_positive_train_test_split():
    X1 = np.random.randint(0, 10, size=(100, 10))
    y1 = np.random.randint(0, 10, size=(100,))
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1)

    assert type(X1_train) == np.ndarray
    assert type(X1_test) == np.ndarray
    assert type(y1_train) == np.ndarray
    assert type(y1_test) == np.ndarray


def test_positive_make_batch_index():
    pass


def test_to_categorical():
    pass
