import numpy as np

from malescratch.utils import (
    MinMaxScaler,
    StandardScaler,
    to_categorical,
    train_test_split,
    make_batch_index,
)


def test_positive_assert_type_train_test_split():
    X = np.random.randint(0, 10, size=(100, 10))
    y = np.random.randint(0, 10, size=(100,))
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    assert type(X_train) == np.ndarray, "X_train should have type np.ndarray"
    assert type(X_test) == np.ndarray, "X_test should have type np.ndarray"
    assert type(y_train) == np.ndarray, "y_train should have type np.ndarray"
    assert type(y_test) == np.ndarray, "y_test should have type np.ndarray"


def test_positive_assert_length_train_test_split():
    X = np.random.randint(0, 10, size=(100, 10))
    y = np.random.randint(0, 10, size=(100,))
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    assert X_train.shape == (90, 10), "X_train should have the same of (90, 10)"
    assert X_test.shape == (10, 10), "X_test should have the same of (10, 10)"
    assert y_train.shape == (90,), "y_train should have the same of (90,)"
    assert y_test.shape == (10,), "y_test should have the same of (10,)"


def test_positive_check_all_numbers_exist():
    X = np.arange(0, 10)
    y = np.arange(0, 10)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    assert set(X) == set(X_train).union(
        X_test
    ), "X_train and X_test should have the same elements as X"
    assert set(y) == set(y_train).union(
        y_test
    ), "y_train and y_test should have the same elements as X"


def test_negative_X_and_y_have_different_length():
    pass


def test_positive_make_batch_index():
    pass


def test_to_categorical():
    pass


def test_MinMaxScaler():
    pass
