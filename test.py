from models.persistence import LaggedValuePersistence, LatestValuePersistence, SlidingWindowPersistence

import pandas as pd
import numpy as np

test_df = pd.DataFrame(columns=('ds', 'y'))
test_df['ds'] = pd.date_range('2019-01-01', '2019-01-31')
test_df['y'] = np.arange(test_df.shape[0])

model = SlidingWindowPersistence()
model.fit(test_df)

future_df = pd.DataFrame(columns=('ds',))
future_df['ds'] = pd.date_range('2019-02-01', '2019-02-07')
future_df = future_df.set_index(['ds', 'ds'], drop=False)
future_df.index.names = ['idx1', 'idx2']

# print(future_df.reset_index())
# print(future_df.reset_index().set_index(future_df.index.names))
#
# # print(future_df)
# # print(test_df.index.names)
#
print(model.predict(future_df))
