""" Splitters for time-series model evaluation

This module defines functions to split time-series data for cross-validation.

The splitting functions take a timestamped dataframe as input and return
a generator function for (train, test) splits.  The train and test sets will
be views of the original dataframe.
"""

import logging

import pandas as pd


def cross_val_split(df,
                    minimum_training_period='7d',
                    maximum_training_period=None,
                    test_window='1d',
                    timestamp_column='ds'):
    """ Split a DataFrame for time-series cross-validation.

    Args:
        df (pandas.DataFrame):
            The DataFrame to split.
        minimum_training_period (str or pandas.Timedelta):
            The minimum amount of time that should be included in one training
            split. If None, then the minimum size of training data will be equal
            to `test_window`.
        maximum_training_period (str or pandas.Timedelta):
            The maxium amount of time that should be included in one training
            split. If this parameter is set, then the training window will
            extend `maximum_training_period` behind the test window.
        test_window (str or pandas.Timedelta):
            The size of the test data window to generate.
        timestamp_column (str):
            The column of df which specifies the timestamp.

    Returns:
        Generator[pd.DataFrame, pd.DataFrame]:
            Generator that emits DataFrame tuples (train, test).
            Train and test will be slices of the original DataFrame.
    """
    logger = logging.getLogger(__name__)
    tsc = timestamp_column
    _df = df.copy().sort_values(by=tsc)

    if test_window is None:
        raise ValueError('Size of test window must be specified.')

    has_max_training_period = maximum_training_period is not None
    has_min_training_period = minimum_training_period is not None
    if has_max_training_period and has_min_training_period:
        max_td = pd.Timedelta(maximum_training_period)
        min_td = pd.Timedelta(minimum_training_period)
        if max_td < min_td:
            raise ValueError('Maximum training period must be larger than '
                             'the minimum training period')

    if minimum_training_period is None:
        minimum_training_period = test_window

    max_date = _df.iloc[-1][tsc]
    train_start = _df.iloc[0][tsc]
    train_end = train_start + pd.Timedelta(minimum_training_period)
    test_end = train_end + pd.Timedelta(test_window)
    while test_end < max_date:
        train = _df[(train_start <= _df[tsc]) & (_df[tsc] < train_end)]
        test = _df[(train_end <= _df[tsc]) & (_df[tsc] < test_end)]

        # ensure train and test are non-empty
        if len(train) > 0 and len(test) > 0:
            yield train, test
        else:
            logger.warning('Empty train or test set generated. '
                           'Consider expanding test window.')

        # walk forward
        train_end += pd.Timedelta(test_window)
        test_end += pd.Timedelta(test_window)

        # enforce maximum_training_period
        if maximum_training_period is not None:
            train_start = train_end - pd.Timedelta(maximum_training_period)
