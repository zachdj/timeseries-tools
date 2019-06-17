""" Persistence Models for time-series data in pandas

Persistence models operate by using historical observations to predict values
in the future.  These models are often used as baselines for time-series
forecasting problems.

This module defines a variety of persistence models designed to work with time-
series data stored in a pandas DataFrame.  The models roughly conform to the API
used by FB Prophet.  Each model has `fit` and `predict` methods which accept
time-stamped DataFrames.
"""

from abc import abstractmethod
import pandas as pd


class PersistenceModel(object):
    @abstractmethod
    def fit(self, df):
        pass

    @abstractmethod
    def predict(self, df):
        pass


class LatestValuePersistence(PersistenceModel):
    """ Predicts future time-steps using the latest historical time-step

    Useful baseline for problems wherein the forecasting window is small and the
    sample data is dense.
    """

    def __init__(self):
        super().__init__()
        self.latest_value = None
        self.training_data = None
        self.trained = False

    def fit(self, df):
        """ Train the model

        For persistence models, this generally entails memorizing the training
        data. In-sample predictions will use the nearest past value from the
        training data, and out-of-sample predictions will use the last value
        from the training data.

        Args:
            df (pandas.DataFrame):
                DataFrame including columns `ds` and `y`.
                `ds` should be the timestamp, and `y` should be the target value

        Returns: None
        """
        self.training_data = df.copy()[['ds', 'y']] \
            .rename({'y': 'yhat'}, axis='columns')
        self.trained = True

    def predict(self, df):
        """ Make predictions on an unlabeled time-series

        Args:
            df (pandas.DataFrame):
                DataFrame with column `ds` specifying timestamps.

        Returns:
            pandas.DataFrame: input DataFrame augmented with a column `yhat`
                containing predicted values
        """
        if not self.trained:
            raise Exception('Model must be fit before predictions can be made.')

        # copy the dataframe, since we are going to be modifying it
        predictions = df.copy()

        # preserve the index through the merge
        idx_names = predictions.index.names
        idx_names = [name or 'index' for name in idx_names]  # None -> 'index'
        predictions = predictions.reset_index()

        # merges each row of predictions with the row from samples having the
        # closest timestamp less than the timestamp on `predictions`
        predictions = pd.merge_asof(
            predictions,
            self.training_data,
            on='ds',
            direction='backward',
            allow_exact_matches=False)

        # retain the original index
        predictions.set_index(idx_names, drop=True, inplace=True)

        # if there are any rows with a `ds` <= the earliest datestamp in the
        # training data, then those rows will have a NaN yhat.
        # Fill these rows will the latest `y`
        latest_value = self.training_data \
            .loc[self.training_data['ds'].idxmax()]['yhat']

        return predictions.fillna(latest_value)


class SlidingWindowPersistence(PersistenceModel):
    """ Predicts the next time-step using the mean of a window of historical obervations

    This model allows for the lookup window to be specified in two ways.
    The window can be an integer N, which will create a sliding window of the
    past N observations.
    The window can also be a string specifying an offset (e.g. "7d").
    In this case, the window will be a time period, and the number of
    observations included in the window will be variable.

    This model can be thought of as a smoothed version of LatestValuePersistence

    Args:
        window (int or str):
            Size of the historical averaging window.
            See pandas.DataFrame.rolling docs for more info:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html

        **kwargs:
            Keyword arguments to forward to pandas.DataFrame.rolling when
            creating the sliging window.
    """
    def __init__(self, window=7, **kwargs):
        super().__init__()
        self.window = window
        self.kwargs = kwargs
        self.sliding_window = None
        self.trained = False

    def fit(self, df):
        """ Train the model
    
        Args:
            df (pandas.DataFrame):
                DataFrame including columns `ds` and `y`.
                `ds` should be the timestamp, and `y` should be the target value
        """
        _df = df.copy()
        self.sliding_window = _df.set_index('ds') \
            .rename({'y': 'yhat'}, axis='columns')
        self.sliding_window = self.sliding_window[['yhat']] \
            .rolling(window=self.window, **self.kwargs).mean()

        self.trained = True

    def predict(self, df):
        """ Make predictions on an unlabeled time-series
    
        Args:
            df (pandas.DataFrame):
                DataFrame with column `ds`
    
        Returns:
            pandas.DataFrame: input DataFrame augmented with a column `yhat`
                containing predicted values
        """
        if not self.trained:
            raise Exception('Model must be fit before predictions can be made.')

        # copy the dataframe, since we are going to be modifying it
        predictions = df.copy()

        # promote index to column level so it will be preserved during the merge
        idx_names = predictions.index.names
        idx_names = [name or 'index' for name in idx_names]  # None -> 'index'
        predictions = predictions.reset_index()

        # merge with rolling window averages
        predictions = pd.merge_asof(predictions, self.sliding_window,
                                    on='ds', direction='backward')

        # retain original index
        predictions.set_index(idx_names, drop=True, inplace=True)

        # fill missing values with historical average
        return predictions.fillna(self.sliding_window['yhat'].mean())


class LaggedValuePersistence(PersistenceModel):
    def __init__(self, lag='7d'):
        """ Predicts the next value using the closest lagged historical value

        For example, using a lag of 365 days, the value for January 1st,
        2019 would be predicted using the value from January 1st, 2018.

        Args:
            lag (str):
                The offset used for the lookback.
                Any string accepted by the pd.Timedelta constructor will work.
        """
        super().__init__()
        self.lag = pd.Timedelta(lag)
        self.training_data = None
        self.trained = False

    def fit(self, df):
        """ Train the model

        Training for this model entails creating a lagged version of the
        training data that can later be joined against.

        Args:
            df (pandas.DataFrame):
                DataFrame including columns `ds` and `y`.
                `ds` should be the timestamp, and `y` should be the target value
        """
        lagged_df = df.copy()[['ds', 'y']]
        lagged_df['ds'] = lagged_df['ds'] + self.lag
        self.lagged_df = lagged_df.rename({'y': 'yhat'}, axis='columns')
        self.trained = True

    def predict(self, df):
        """ Make predictions on an unlabeled time-series

        Args:
            df (pandas.DataFrame):
                DataFrame with column `ds`

        Returns:
            pandas.DataFrame: input DataFrame augmented with a column `yhat`
                containing predicted values
        """
        if not self.trained:
            raise Exception('Model must be fit before predictions can be made.')

        # copy the dataframe, since we are going to be modifying it
        predictions = df.copy()

        # dataframes must be sorted by merging key
        predictions = predictions.sort_values(by='ds')

        # promote index to column level so it will be preserved during the merge
        idx_names = predictions.index.names
        idx_names = [name or 'index' for name in idx_names]  # None -> 'index'
        predictions = predictions.reset_index()

        # merge with lagged training data
        predictions = pd.merge_asof(predictions, self.lagged_df,
                                    on='ds', direction='backward')

        # retain original index
        predictions.set_index(idx_names, drop=True, inplace=True)

        # fill missing values with mean `y` value from training data
        return predictions.fillna(self.lagged_df['yhat'].mean())
