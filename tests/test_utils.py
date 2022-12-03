import numpy as np

from malescratch.utils import (
    MinMaxScaler,
    StandardScaler,
    to_categorical,
    train_test_split,
    make_batch_index,
)


def test_experiment():
    X = np.random.randint(0, 10, 1000).reshape((200, 5))
    y = np.random.randint(0, 10, 1000)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=120)

    print(X_train.shape)
    print(X_test.shape)


def test_positive_train_test_split_assert_type():
    X = np.random.randint(0, 10, size=(100, 10))
    y = np.random.randint(0, 10, size=(100,))
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    assert type(X_train) == np.ndarray, "X_train should have type np.ndarray"
    assert type(X_test) == np.ndarray, "X_test should have type np.ndarray"
    assert type(y_train) == np.ndarray, "y_train should have type np.ndarray"
    assert type(y_test) == np.ndarray, "y_test should have type np.ndarray"


def test_positive_train_test_split_assert_length_using_default_train_size():
    X = np.random.randint(0, 10, size=(100, 10))
    y = np.random.randint(0, 10, size=(100,))
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    assert X_train.shape == (90, 10), "X_train should have shape (90, 10)"
    assert X_test.shape == (10, 10), "X_test should have shape (10, 10)"
    assert y_train.shape == (90,), "y_train should have shape (90,)"
    assert y_test.shape == (10,), "y_test should have shape (10,)"


def test_positive_train_test_split_check_all_numbers_exist():
    X = np.arange(0, 10)
    y = np.arange(0, 10)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    assert set(X) == set(X_train).union(
        X_test
    ), "X_train and X_test should have the same elements as X"
    assert set(y) == set(y_train).union(
        y_test
    ), "y_train and y_test should have the same elements as X"


def test_positive_train_test_split_train_size_int():
    X = np.random.randint(0, 10, 1000).reshape((200, 5))
    y = np.random.randint(0, 10, 1000)

    X_train, _, y_train, _ = train_test_split(X, y, train_size=150)

    assert len(X_train) == 150, "length of X_train should be 150"
    assert len(y_train) == 150, "length of y_train should be 150"


def test_positive_train_test_split_test_size_int():
    X = np.random.randint(0, 10, 1000).reshape((200, 5))
    y = np.random.randint(0, 10, 1000)

    _, X_test, _, y_test = train_test_split(X, y, test_size=40)

    assert len(X_test) == 40, "length of X_test should be 40"
    assert len(y_test) == 40, "length of y_test should be 40"


def test_positive_train_test_split_train_size_float():
    X = np.random.randint(0, 10, 1000).reshape((200, 5))
    y = np.random.randint(0, 10, 1000)

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.9)

    assert len(X_train) == 180, "length of X_train should be 180"
    assert len(y_train) == 180, "length of y_train should be 180"


def test_positive_train_test_split_test_size_float():
    X = np.random.randint(0, 10, 1000).reshape((200, 5))
    y = np.random.randint(0, 10, 1000)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.1)

    assert len(X_test) == 20, "length of X_test should be 20"
    assert len(y_test) == 20, "length of y_test should be 20"


def test_negative_shpuld_accept_list():
    x = list(range(10))
    x_train, x_test = train_test_split(x)

    assert (
        type(x_train) == np.ndarray
    ), "train_test_split should accept list object and return np.ndarray object"
    assert (
        type(x_test) == np.ndarray
    ), "train_test_split should accept list object and return np.ndarray object"


def test_positive_make_batch_index_check_len_batch():
    n_batches = 5
    batches = make_batch_index(100, n_batches, 100)

    assert len(batches) == n_batches, "number of batches should be 10"


def test_positive_make_batch_index():
    print("Aun Aprendo")
    pass
