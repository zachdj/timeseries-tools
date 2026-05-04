import pandas as pd
import numpy as np
import unittest

from models.persistence import (
    LaggedValuePersistence,
    LatestValuePersistence,
    SlidingWindowPersistence
)


class LaggedValueTestCase(unittest.TestCase):
    def setUp(self) -> None:
        uniform_ts = pd.DataFrame(columns=('ds', 'y'))
        uniform_ts['ds'] = pd.date_range('2020-01-01', '2020-01-14')
        uniform_ts['y'] = np.arange(0, uniform_ts.shape[0])

        non_uniform_ts = pd.DataFrame(columns=('ds', 'y'))
        timestamps = pd.date_range('2020-01-01', '2020-01-14')
        # add noise to the timestamps to disrupt the even spacing
        perturbations = np.random.random(len(timestamps)) * 2
        timestamps = timestamps + pd.to_timedelta(perturbations, unit='day')
        non_uniform_ts['ds'] = timestamps
        non_uniform_ts['y'] = np.arange(0, non_uniform_ts.shape[0])

        self.uniform_ts = uniform_ts
        self.non_uniform_ts = non_uniform_ts

    def test_uniform(self):
        model = LaggedValuePersistence(lag='7d')
        model.fit(self.uniform_ts)

        # make some in-sample predictions
        in_sample = self.uniform_ts.copy()
        in_sample.set_index('ds', drop=False, inplace=True)
        in_sample.index.names = ['idx']
        idx = in_sample.index
        predictions = model.predict(in_sample)

        print(predictions)

        # in-sample predictions should be lagged by 7 days
        self.assertEqual(
            predictions.loc['2020-01-08', 'yhat'],
            predictions.loc['2020-01-01', 'y'])
        self.assertEqual(
            predictions.loc[idx[-1], 'yhat'],
            predictions.loc[idx[-8], 'y'])

        # predictions without historical data should be the mean `y` value
        self.assertEqual(
            predictions.loc['2020-01-01', 'yhat'],
            predictions['y'].mean())

        # make some out-of-sample predictions
        # out_of_sample = pd.DataFrame(columns=('ds',))
        # out_of_sample['ds'] = pd.date_range('2020-01-15', '2020-01-21')
        # out_of_sample.set_index('ds', drop=False)
        # predictions = model.predict(out_of_sample)
        #
        # # out-of-sample predictions should also be lagged by 7 days
        # self.assertEqual(
        #     predictions.loc[pd.Timestamp('2020-01-15'), 'yhat'],
        #     in_sample.loc[pd.Timestamp('2020-01-08'), 'y']
        # )



    def test_non_uniform(self):
        pass
