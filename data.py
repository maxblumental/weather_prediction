import random
from typing import Optional, Tuple
import numpy as np
import pandas as pd

TRAIN_SPLIT = 300000
UNIVARIATE_HISTORY_SIZE = 20
MULTIVARIATE_HISTORY_SIZE = 720
FUTURE_SIZE = 72
STEP = 6


def univariate_data(dataset: np.array,
                    start_index: int, end_index: Optional[int],
                    history_size: int, target_size: int) -> Tuple[np.array, np.array]:
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])

    return np.array(data), np.array(labels)


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])
    return np.array(data), np.array(labels)


def make_univariate_split(df: pd.DataFrame,
                          train_split: int = TRAIN_SPLIT,
                          history_size: int = UNIVARIATE_HISTORY_SIZE):
    orig_data = df['T'].values
    train_mean = orig_data[:train_split].mean()
    train_std = orig_data[:train_split].std()
    data = (orig_data - train_mean) / train_std
    x_train, y_train = univariate_data(data, 0, train_split, history_size, 0)
    x_val, y_val = univariate_data(data, train_split, None, history_size, 0)
    return x_train, y_train, x_val, y_val


def sample_tasks(x_val: np.array, y_val: np.array,
                 k: int = 5, seed: int = None) -> Tuple[np.array, np.array]:
    random.seed(seed)
    indices = random.sample(range(len(x_val)), k=k)
    return x_val[indices], y_val[indices]


def make_multivariate_split(df: pd.DataFrame,
                            train_split: int = TRAIN_SPLIT,
                            history_size: int = MULTIVARIATE_HISTORY_SIZE,
                            future_size: int = FUTURE_SIZE,
                            step: int = STEP, single_step: bool = False):
    dataset = df[['p', 'T', 'h']].values
    data_mean = dataset[:train_split].mean(axis=0)
    data_std = dataset[:train_split].std(axis=0)
    dataset = (dataset - data_mean) / data_std

    x_train, y_train = multivariate_data(dataset, dataset[:, 1], 0, train_split,
                                         history_size, future_size, step, single_step=single_step)
    x_val, y_val = multivariate_data(dataset, dataset[:, 1], train_split, None,
                                     history_size, future_size, step, single_step=single_step)

    return x_train, y_train, x_val, y_val
